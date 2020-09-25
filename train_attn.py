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

import albumentations
from albumentations import augmentations

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import wandb
# import neptune
# from neptunecontrib.monitoring.metrics import *

from utils import *
from effb4_attention import Efficient_Attention
from segmentation.timm_efficientnet import EfficientNet
from casia_dataset import CASIA

OUTPUT_DIR = "weights"
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_defaults = {
    "epochs": 100,
    "train_batch_size": 26,
    "valid_batch_size": 100,
    "optimizer": "adam",
    "learning_rate": 0.001959,
    "weight_decay": 0.0005938,
    "schedule_patience": 3,
    "schedule_factor": 0.2569,
    "model": "EFFN",
    "attn_map_weight": 0,
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

    # model = Efficient_Attention()
    model = EfficientNet('tf_efficientnet_b4_ns')
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0,1])

    model.to(device)
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
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)

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
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)

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
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)


    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="max",
        factor=config.schedule_factor,
    )
    criterion = nn.BCEWithLogitsLoss()
    attn_map_criterion = nn.L1Loss()

    es = EarlyStopping(patience=20, mode="max")

    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            attn_map_criterion,
            config.attn_map_weight,
            epoch,
        )
        valid_metrics = valid_epoch(
            model, valid_loader, criterion, attn_map_criterion, config.attn_map_weight, epoch
        )
        scheduler.step(valid_metrics["valid_acc_05"])

        print(
            f"TRAIN_ACC = {train_metrics['train_acc_05']}, TRAIN_LOSS = {train_metrics['train_loss']}"
        )
        print(
            f"VALID_ACC = {valid_metrics['valid_acc_05']}, VALID_LOSS = {valid_metrics['valid_loss']}"
        )

        es(
            valid_metrics["valid_acc_05"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f"weights/{name}_[{dt_string}].h5"))

    test(model, test_loader, criterion, attn_map_criterion, config.attn_map_weight)
    wandb.save(f"weights/{name}_[{dt_string}].h5")


def train_epoch(model, train_loader, optimizer, criterion, attn_map_criterion, attn_map_weight, epoch):
    model.train()

    classification_loss = AverageMeter()
    attn_map_loss = AverageMeter()
    total_loss = AverageMeter()
    
    predictions = []
    targets = []

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        target_labels = batch["label"].to(device)
        attn_gt = batch["attn_mask"].to(device)

        optimizer.zero_grad()
        
        # out_labels, attn_map = model(images)
        out_labels = model(images)

        loss_classification = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
        # loss_attn_map = attn_map_criterion(attn_map, attn_gt)
        loss = loss_classification  #+ attn_map_weight * loss_attn_map

        loss.backward()
        optimizer.step()

        #---------------------Batch Loss Update-------------------------
        # classification_loss.update(loss_classification.item(), train_loader.batch_size)
        # attn_map_loss.update(loss_attn_map.item(), train_loader.batch_size)
        total_loss.update(loss.item(), train_loader.batch_size)
        
        targets.append((target_labels.view(-1, 1).cpu() >= 0.5) * 1.0)
        predictions.append(torch.sigmoid(out_labels).cpu().detach().numpy())

    # Epoch Logging
    with torch.no_grad():
        targets = np.vstack((targets)).ravel()
        predictions = np.vstack((predictions)).ravel()

        train_auc = metrics.roc_auc_score(targets, predictions)
        train_f1_05 = metrics.f1_score(targets, (predictions >= 0.5) * 1)
        train_acc_05 = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
        train_balanced_acc_05 = metrics.balanced_accuracy_score(targets, (predictions >= 0.5) * 1)
        
    train_metrics = {
        # "train_loss_classification": classification_loss.avg,
        # "train_loss_attn_map": attn_map_loss.avg,
        "train_loss" : total_loss.avg,
        "train_auc": train_auc,
        "train_f1_05": train_f1_05,
        "train_acc_05": train_acc_05,
        "train_balanced_acc_05": train_balanced_acc_05,
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, attn_map_criterion, attn_map_weight, epoch):
    model.eval()

    classification_loss = AverageMeter()
    attn_map_loss = AverageMeter()
    total_loss = AverageMeter()
    
    predictions = []
    targets = []
    example_images = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            target_labels = batch["label"].to(device)
            attn_gt = batch["attn_mask"].to(device)

            # out_labels, attn_map = model(images)
            out_labels = model(images)

            loss_classification = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
            # loss_attn_map = attn_map_criterion(attn_map, attn_gt)
            loss = loss_classification  #+ attn_map_weight * loss_attn_map
            
            #---------------------Batch Loss Update-------------------------
            # classification_loss.update(loss_classification.item(), valid_loader.batch_size)
            # attn_map_loss.update(loss_attn_map.item(), valid_loader.batch_size)
            total_loss.update(loss.item(), valid_loader.batch_size)
            
            batch_targets = (target_labels.view(-1, 1).cpu() >= 0.5) * 1.0
            batch_preds = torch.sigmoid(out_labels).cpu()          
            
            targets.append(batch_targets)
            predictions.append(batch_preds)

            best_batch_pred_idx = np.argmin(abs(batch_targets - batch_preds))
            worst_batch_pred_idx = np.argmax(abs(batch_targets - batch_preds))
            example_images.append(
                wandb.Image(
                    images[best_batch_pred_idx],
                    caption=f"Pred : {batch_preds[best_batch_pred_idx].item()} Label: {batch_targets[best_batch_pred_idx].item()}",
                )
            )
            example_images.append(
                wandb.Image(
                    images[worst_batch_pred_idx],
                    caption=f"Pred : {batch_preds[worst_batch_pred_idx].item()} Label: {batch_targets[worst_batch_pred_idx].item()}",
                )
            )

    # Logging
    targets = np.vstack((targets)).ravel()
    predictions = np.vstack((predictions)).ravel()

    valid_auc = metrics.roc_auc_score(targets, predictions)
    valid_f1_05 = metrics.f1_score(targets, (predictions >= 0.5) * 1)
    valid_acc_05 = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
    valid_balanced_acc_05 = metrics.balanced_accuracy_score(targets, (predictions >= 0.5) * 1)

    valid_metrics = {
        "valid_loss": total_loss.avg,
        # "valid_loss_classification": classification_loss.avg,
        # "valid_loss_attn_map": attn_map_loss.avg,
        "valid_auc": valid_auc,
        "valid_f1_05": valid_f1_05,
        "valid_acc_05": valid_acc_05,
        "valid_balanced_acc_05": valid_balanced_acc_05,
        "valid_examples": example_images[-20:],
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion, attn_map_criterion, attn_map_weight):
    model.eval()

    classification_loss = AverageMeter()
    attn_map_loss = AverageMeter()
    total_loss = AverageMeter()
    
    predictions = []
    targets = []
    example_images = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            target_labels = batch["label"].to(device)
            attn_gt = batch["attn_mask"].to(device)

            # out_labels, attn_map = model(images)
            out_labels = model(images)

            loss_classification = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
            # loss_attn_map = attn_map_criterion(attn_map, attn_gt)
            loss = loss_classification  #+ attn_map_weight * loss_attn_map
            
            #---------------------Batch Loss Update-------------------------
            # classification_loss.update(loss_classification.item(), test_loader.batch_size)
            # attn_map_loss.update(loss_attn_map.item(), test_loader.batch_size)
            total_loss.update(loss.item(), test_loader.batch_size)
            
            targets.append((target_labels.view(-1, 1).cpu() >= 0.5) * 1.0)
            predictions.append(torch.sigmoid(out_labels).cpu() )

    # Logging
    targets = np.vstack((targets)).ravel()
    predictions = np.vstack((predictions)).ravel()

    test_auc = metrics.roc_auc_score(targets, predictions)
    test_f1_05 = metrics.f1_score(targets, (predictions >= 0.5) * 1)
    test_acc_05 = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
    test_balanced_acc_05 = metrics.balanced_accuracy_score(targets, (predictions >= 0.5) * 1)

    test_metrics = {
        "test_loss": total_loss.avg,
        # "test_loss_classification": classification_loss.avg,
        # "test_loss_attn_map": attn_map_loss.avg,
        "test_auc": test_auc,
        "test_f1_05": test_f1_05,
        "test_acc_05": test_acc_05,
        "test_balanced_acc_05": test_balanced_acc_05,
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
    patch_size = 128
    DATA_ROOT = f"Image_Manipulation_Dataset/CASIA_2.0/image_patch_128"

    df = pd.read_csv(f"casia_{patch_size}.csv").sample(frac=1).reset_index(drop=True)

    train(
        name=f"256_CASIA_FULL" + config_defaults["model"],
        df=df,
        data_root=DATA_ROOT,
        patch_size=patch_size,
    )

