from casia_dataset import CASIA
from segmentation.timm_efficient_unet import get_efficientunet
from segmentation.timm_efficientnet import EfficientNet
import seg_metrics
from pytorch_toolbelt import losses
from utils import *
from apex import amp
import wandb
import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
import pickle as pkl
from sklearn import metrics
import scikitplot as skplt

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc

import albumentations
from albumentations import augmentations

torch.backends.cudnn.benchmark = True

# import neptune
# from neptunecontrib.monitoring.metrics import *


OUTPUT_DIR = "weights"
device = 'cuda'
config_defaults = {
    "epochs": 100,
    "train_batch_size": 20,
    "valid_batch_size": 100,
    "optimizer": "adam",
    "learning_rate": 0.001959,
    "weight_decay": 0.0005938,
    "schedule_patience": 3,
    "schedule_factor": 0.2569,
    "model": "Unet(end-end)",
}
VAL_FOLD = 0
TEST_FOLD = 9


def train(name, df, data_root, patch_size):
    now = datetime.now()
    dt_string = now.strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wandb.init(
        project="imanip", config=config_defaults, name=f"{name},{dt_string}",
    )
    config = wandb.config

    # neptune.init("sowmen/imanip")
    # neptune.create_experiment(name=f"{name},val_fold:{VAL_FOLD},run{run}")

    # model = EfficientNet('tf_efficientnet_b4_ns')
    model = get_efficientunet(
        'tf_efficientnet_b4_ns',
        # encoder_checkpoint='weights/256_CASIA_FULLtimm_effunet_[20_09_08_11_47].h5',
        # freeze_encoder=True
    )
    model.to(device)

    normalize = {
        "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
        "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
    }
    train_aug = albumentations.Compose([
        augmentations.transforms.Flip(p=0.5),
        augmentations.transforms.Rotate((-45, 45), p=0.4),
        augmentations.transforms.ShiftScaleRotate(p=0.3),
        augmentations.transforms.HueSaturationValue(p=0.3),
        augmentations.transforms.JpegCompression(quality_lower=70, p=0.3),
        augmentations.transforms.Resize(224, 224, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
        albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
        albumentations.pytorch.ToTensor()
    ])
    valid_aug = albumentations.Compose([
        augmentations.transforms.Resize(224, 224, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
        albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
        albumentations.pytorch.ToTensor()
    ])

    # -------------------------------- CREATE DATASET and DATALOADER --------------------------
    train_dataset = CASIA(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        equal_sample=False,
        transforms=train_aug,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8)

    valid_dataset = CASIA(
        dataframe=df,
        mode="val",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        equal_sample=False,
        transforms=valid_aug,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=8)

    test_dataset = CASIA(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        equal_sample=False,
        transforms=valid_aug,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=8)

    optimizer = get_optimizer(model, config.optimizer,config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="max",
        factor=config.schedule_factor,
    )

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    criterion = nn.BCEWithLogitsLoss()

    es = EarlyStopping(patience=20, mode="max")

    wandb.watch(model, log_freq=50, log='all')

    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, epoch)

        valid_metrics = valid_epoch(model, valid_loader, criterion,  epoch)
        # scheduler.step(valid_metrics["train_dice_tot"])

        print(
            f"TRAIN_DICE_TOT = {train_metrics['train_dice_tot']}, TRAIN_LOSS = {train_metrics['train_loss_segmentation']}"
        )
        print(
            f"VALID_DICE_TOT = {valid_metrics['valid_dice_tot']}, VALID_LOSS = {valid_metrics['valid_loss_segmentation']}"
        )

        es(
            valid_metrics["valid_dice_tot"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join(
        OUTPUT_DIR, f"{name}_[{dt_string}].h5")))

    test(model, test_loader, criterion)
    wandb.save(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter()  # Sum all batch using torch.sum
    dice_ind = AverageMeter()  # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        gt = batch["mask"].to(device)

        optimizer.zero_grad()

        out_mask = model(images)

        loss_segmentation = criterion(out_mask, gt)

        # with amp.scale_loss(loss_segmentation, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss_segmentation.backward()
        optimizer.step()

        # ---------------------Batch Loss Update-------------------------
        segmentation_loss.update(
            loss_segmentation.item(), train_loader.batch_size)

        with torch.no_grad():
            out_mask = out_mask.cpu()
            gt = gt.cpu()
            # ---------------------Batch Metrics Update------------------------
            dice_tot.update(losses.functional.soft_dice_score(
                torch.sigmoid(out_mask), gt), train_loader.batch_size)
            dice_ind.update(seg_metrics.dice_coeff_single(
                torch.sigmoid(out_mask), gt), train_loader.batch_size)

            jaccard_tot.update(losses.functional.soft_jaccard_score(
                torch.sigmoid(out_mask), gt), train_loader.batch_size)
            jaccard_ind.update(seg_metrics.jaccard_coeff_single(
                torch.sigmoid(out_mask), gt), train_loader.batch_size)

        # gc.collect()
        # torch.cuda.empty_cache()

    train_metrics = {
        "train_loss_segmentation": segmentation_loss.avg,
        "train_dice_tot": dice_tot.avg,
        "train_dice_ind": dice_ind.avg,
        "train_jaccard_tot": jaccard_tot.avg,
        "train_jaccard_ind": jaccard_ind.avg,
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter()  # Sum all batch using torch.sum
    dice_ind = AverageMeter()  # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()
    # example_images = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            gt = batch["mask"].to(device)

            out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(
                loss_segmentation.item(), valid_loader.batch_size)

            out_mask = out_mask.cpu()
            gt = gt.cpu()
            # ---------------------Batch Metrics Update------------------------
            dice_tot.update(losses.functional.soft_dice_score(
                torch.sigmoid(out_mask), gt), valid_loader.batch_size)
            dice_ind.update(seg_metrics.dice_coeff_single(
                torch.sigmoid(out_mask), gt), valid_loader.batch_size)

            jaccard_tot.update(losses.functional.soft_jaccard_score(
                torch.sigmoid(out_mask), gt), valid_loader.batch_size)
            jaccard_ind.update(seg_metrics.jaccard_coeff_single(
                torch.sigmoid(out_mask), gt), valid_loader.batch_size)

    valid_metrics = {
        "valid_loss_segmentation": segmentation_loss.avg,
        "valid_dice_tot": dice_tot.avg,
        "valid_dice_ind": dice_ind.avg,
        "valid_jaccard_tot": jaccard_tot.avg,
        "valid_jaccard_ind": jaccard_ind.avg,
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter()  # Sum all batch using torch.sum
    dice_ind = AverageMeter()  # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            gt = batch["mask"].to(device)

            out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(
                loss_segmentation.item(), test_loader.batch_size)

            out_mask = out_mask.cpu()
            gt = gt.cpu()
            # ---------------------Batch Metrics Update------------------------
            dice_tot.update(losses.functional.soft_dice_score(
                torch.sigmoid(out_mask), gt), test_loader.batch_size)
            dice_ind.update(seg_metrics.dice_coeff_single(
                torch.sigmoid(out_mask), gt), test_loader.batch_size)

            jaccard_tot.update(losses.functional.soft_jaccard_score(
                torch.sigmoid(out_mask), gt), test_loader.batch_size)
            jaccard_ind.update(seg_metrics.jaccard_coeff_single(
                torch.sigmoid(out_mask), gt), test_loader.batch_size)

    test_metrics = {
        "test_loss_segmentation": segmentation_loss.avg,
        "test_dice_tot": dice_tot.avg,
        "test_dice_ind": dice_ind.avg,
        "test_jaccard_tot": jaccard_tot.avg,
        "test_jaccard_ind": jaccard_ind.avg,
    }
    wandb.log(test_metrics)
    # wandb.log(
    #     {
    #         "test_roc_auc_curve": skplt.metrics.plot_roc(
    #             targets, expand_prediction(correct_predictions)
    #         ),
    #         "test_precision_recall_curve": skplt.metrics.plot_precision_recall(
    #             targets, expand_prediction(correct_predictions)
    #         ),
    #     }
    # )

    # y_test = targets
    # y_test_pred = expand_prediction(correct_predictions)
    # log_confusion_matrix(y_test, y_test_pred[:, 1] > 0.5)
    # log_classification_report(y_test, y_test_pred[:, 1] > 0.5)
    # log_class_metrics(y_test, y_test_pred[:, 1] > 0.5)
    # log_roc_auc(y_test, y_test_pred)
    # log_precision_recall_auc(y_test, y_test_pred)
    # log_brier_loss(y_test, y_test_pred[:, 1])
    # log_log_loss(y_test, y_test_pred)
    # log_ks_statistic(y_test, y_test_pred)
    # log_cumulative_gain(y_test, y_test_pred)
    # log_lift_curve(y_test, y_test_pred)
    # log_prediction_distribution(y_test, y_test_pred[:, 1])
    # log_class_metrics_by_threshold(y_test, y_test_pred[:, 1])


def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)


if __name__ == "__main__":
    patch_size = 'FULL'
    DATA_ROOT = f"Image_Manipulation_Dataset/CASIA_2.0"

    df = pd.read_csv(f"casia_{patch_size}.csv").sample(
        frac=1).reset_index(drop=True)

    train(
        name=f"256_CASIA_FULL" + config_defaults["model"],
        df=df,
        data_root=DATA_ROOT,
        patch_size=patch_size,
    )
