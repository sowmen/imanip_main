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
import gc

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations
from albumentations import augmentations

from torch.backends import cudnn
cudnn.benchmark = True

import wandb
# import neptune
# from neptunecontrib.monitoring.metrics import *

from utils import *
from pytorch_toolbelt import losses
import seg_metrics

from segmentation.timm_efficientnet import EfficientNet
from segmentation.timm_efficient_unet import get_efficientunet
from casia_dataset import CASIA

OUTPUT_DIR = "weights"
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_defaults = {
    "epochs": 100,
    "train_batch_size": 20,
    "valid_batch_size": 40,
    "optimizer": "adam",
    "learning_rate": 0.001959,
    "weight_decay": 0.0005938,
    "schedule_patience": 3,
    "schedule_factor": 0.2569,
    "model": "Unet",
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
    model = get_efficientunet('tf_efficientnet_b4_ns')
    model.to(device)
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device)

    # wandb.watch(model)
    
    train_aug = albumentations.Compose(
        [
            augmentations.transforms.Flip(p=0.5),
            augmentations.transforms.Rotate((-45, 45), p=0.4),
            augmentations.transforms.ShiftScaleRotate(p=0.5),
            augmentations.transforms.HueSaturationValue(p=0.3),
            augmentations.transforms.JpegCompression(quality_lower=70, p=0.3),
            augmentations.transforms.Resize(
                256, 256, interpolation=cv2.INTER_AREA, always_apply=True, p=1
            ),
        ]
    )
    valid_aug = albumentations.Compose(
        [
            augmentations.transforms.Resize(
                256, 256, interpolation=cv2.INTER_AREA, always_apply=True, p=1
            )
        ]
    )

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


    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="max",
        factor=config.schedule_factor,
    )
    criterion = nn.BCEWithLogitsLoss()

    es = EarlyStopping(patience=20, mode="max")

    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        
        valid_metrics = valid_epoch(model, valid_loader, criterion,  epoch)
        scheduler.step(valid_metrics["train_dice_tot"])

        print(
            f"TRAIN_DICE_TOT = {train_metrics['train_dice_tot']}, TRAIN_LOSS = {train_metrics['train_loss']}"
        )
        print(
            f"VALID_DICE_TOT = {valid_metrics['valid_dice_tot']}, VALID_LOSS = {valid_metrics['valid_loss']}"
        )

        es(
            valid_metrics["valid_dice_tot"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f"weights/{name}_[{dt_string}].h5"))

    test(model, test_loader, criterion)
    wandb.save(f"weights/{name}_[{dt_string}].h5")


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter() # Sum all batch using torch.sum
    dice_ind = AverageMeter() # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()
    

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        gt = batch["mask"].to(device)

        optimizer.zero_grad()

        out_mask = model(images)

        loss_segmentation = criterion(out_mask, gt)

        loss_segmentation.backward()
        optimizer.step()

        #---------------------Batch Loss Update-------------------------
        segmentation_loss.update(loss_segmentation.item(), train_loader.batch_size)
        
        #---------------------Batch Metrics Update------------------------
        dice_tot.update(losses.functional.soft_dice_score(torch.sigmoid(out_mask), gt), train_loader.batch_size)
        dice_ind.update(seg_metrics.dice_coeff_single(torch.sigmoid(out_mask), gt), train_loader.batch_size)
        
        jaccard_tot.update(losses.functional.soft_jaccard_score(torch.sigmoid(out_mask), gt), train_loader.batch_size)
        jaccard_ind.update(seg_metrics.jaccard_coeff_single(torch.sigmoid(out_mask), gt), train_loader.batch_size)       

        gc.collect()
        torch.cuda.empty_cache()
        

    train_metrics = {
        "train_loss_segmentation": segmentation_loss.avg,
        "train_dice_tot" : dice_tot.avg,
        "train_dice_ind" : dice_ind.avg,
        "train_jaccard_tot" : jaccard_tot.avg,
        "train_jaccard_ind" : jaccard_ind.avg,
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter() # Sum all batch using torch.sum
    dice_ind = AverageMeter() # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()
    example_images = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            gt = batch["mask"].to(device)

            out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            #---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), valid_loader.batch_size)

            #---------------------Batch Metrics Update------------------------
            dice_tot.update(losses.functional.soft_dice_score(torch.sigmoid(out_mask), gt), valid_loader.batch_size)
            dice_ind.update(seg_metrics.dice_coeff_single(torch.sigmoid(out_mask), gt), valid_loader.batch_size)
            
            jaccard_tot.update(losses.functional.soft_jaccard_score(torch.sigmoid(out_mask), gt), valid_loader.batch_size)
            jaccard_ind.update(seg_metrics.jaccard_coeff_single(torch.sigmoid(out_mask), gt), valid_loader.batch_size) 

    valid_metrics = {
        "valid_loss_segmentation": segmentation_loss.avg,
        "valid_dice_tot" : dice_tot.avg,
        "valid_dice_ind" : dice_ind.avg,
        "valid_jaccard_tot" : jaccard_tot.avg,
        "valid_jaccard_ind" : jaccard_ind.avg,
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    segmentation_loss = AverageMeter()
    dice_tot = AverageMeter() # Sum all batch using torch.sum
    dice_ind = AverageMeter() # Sum of individual dice for batch
    jaccard_tot = AverageMeter()
    jaccard_ind = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            gt = batch["mask"].to(device)

            out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            #---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), test_loader.batch_size)

            #---------------------Batch Metrics Update------------------------
            dice_tot.update(losses.functional.soft_dice_score(torch.sigmoid(out_mask), gt), test_loader.batch_size)
            dice_ind.update(seg_metrics.dice_coeff_single(torch.sigmoid(out_mask), gt), test_loader.batch_size)
            
            jaccard_tot.update(losses.functional.soft_jaccard_score(torch.sigmoid(out_mask), gt), test_loader.batch_size)
            jaccard_ind.update(seg_metrics.jaccard_coeff_single(torch.sigmoid(out_mask), gt), test_loader.batch_size) 

    

    test_metrics = {
        "test_loss_segmentation": segmentation_loss.avg,
        "test_dice_tot" : dice_tot.avg,
        "test_dice_ind" : dice_ind.avg,
        "test_jaccard_tot" : jaccard_tot.avg,
        "test_jaccard_ind" : jaccard_ind.avg,
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

    df = pd.read_csv(f"casia_{patch_size}.csv").sample(frac=1).reset_index(drop=True)

    train(
        name=f"256_CASIA_FULL" + config_defaults["model"],
        df=df,
        data_root=DATA_ROOT,
        patch_size=patch_size,
    )

