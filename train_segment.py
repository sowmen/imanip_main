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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

import albumentations
from albumentations import *
from albumentations import augmentations
from torchvision import transforms

torch.backends.cudnn.benchmark = True

# from apex import amp
import wandb
# import neptune
# from neptunecontrib.monitoring.metrics import *

from casia_dataset import CASIA
from segmentation.timm_efficientnet import EfficientNet
import seg_metrics
from pytorch_toolbelt import losses
from utils import *
# import segmentation_models_pytorch as smp
# from segmentation.smp_effb4 import SMP_DIY
from segmentation.timm_unetb4 import UnetB4, UnetB4_Inception
from segmentation.timm_unetpp import UnetPP
from segmentation.merged_net import SRM_Classifer 
from sim_dataset import SimDataset

OUTPUT_DIR = "weights"
device = 'cuda'
config_defaults = {
    "epochs": 60,
    "train_batch_size": 64,
    "valid_batch_size": 128,
    "optimizer": "radam",
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "schedule_patience": 3,
    "schedule_factor": 0.25,
    'sampling':'nearest',
    "model": "UnetPP(ELA)No Resize log dice",
}
TEST_FOLD = 9


def train(name, df, data_root, patch_size, VAL_FOLD=0, SRM_FLAG=1):
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
    # model = get_efficientunet(
    #     'tf_efficientnet_b4_ns',
    #     # encoder_checkpoint='weights/256_CASIA_FULLtimm_effunet_[20_09_08_11_47].h5',
    #     # freeze_encoder=True
    # )
    # model = smp.Unet('timm-efficientnet-b4', classes=6, encoder_weights='imagenet')
    # model = SMP_DIY(num_classes=6)
    
    # encoder = EfficientNet(encoder_checkpoint='64_encoder.h5', freeze_encoder=True).get_encoder()
    encoder = SRM_Classifer(encoder_checkpoint='64_ELA.h5', freeze_encoder=True)
    # model = UnetB4_Inception(encoder, in_channels=54, num_classes=1, sampling=config.sampling, layer='end')
    model = UnetPP(encoder, in_channels=54, num_classes=1, sampling=config.sampling, layer='end')
    model.to(device)
    model = nn.DataParallel(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    normalize = {
        "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
        "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
    }
    # train_aug = albumentations.Compose(
    #     [
    #         augmentations.transforms.Flip(p=0.6),
    #         augmentations.transforms.Rotate((-45, 45), p=0.4),
    #         augmentations.transforms.ShiftScaleRotate(p=0.5),
    #         augmentations.transforms.HueSaturationValue(p=0.3),
    #         augmentations.transforms.JpegCompression(quality_lower=70, p=0.3),
    #         augmentations.transforms.Resize(224, 224, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
    #         albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
    #         albumentations.pytorch.ToTensor()
    #     ],
    #     additional_targets={'ela':'image'}
    # )
    train_aug = albumentations.Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.1),
            ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=45, p=0.25),
            RandomBrightnessContrast(p=0.5),
            IAAEmboss(p=0.25),
            Blur(p=0.1, blur_limit = 3),
            OneOf([
                ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(p=0.5),
                OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
            ], p=0.8),
            augmentations.transforms.Resize(64, 64, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
            albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
            albumentations.pytorch.ToTensor()
        ],
        additional_targets={'ela':'image'}
    )
    valid_aug = albumentations.Compose(
        [
            augmentations.transforms.Resize(64, 64, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
            albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
            albumentations.pytorch.ToTensor()
        ],
        additional_targets={'ela':'image'}
    )

    #region SIMULATION
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    # ])

    # train_set = SimDataset(2000, transform = trans)
    # train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, num_workers=8)
    # val_set = SimDataset(500, transform = trans)
    # valid_loader = DataLoader(val_set, batch_size=config.valid_batch_size, shuffle=False, num_workers=8)
    #endregion

    # -------------------------------- CREATE DATASET and DATALOADER --------------------------
    train_dataset = CASIA(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        transforms=train_aug,
        segment=True
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8)

    valid_dataset = CASIA(
        dataframe=df,
        mode="val",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        transforms=valid_aug,
        segment=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=8)

    test_dataset = CASIA(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        root_dir=data_root,
        patch_size=patch_size,
        transforms=valid_aug,
        segment=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=8)

    optimizer = get_optimizer(model, config.optimizer,config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    bce = nn.BCEWithLogitsLoss()
    dice = losses.DiceLoss(mode='binary', log_loss=True)
    criterion = losses.JointLoss(bce, dice)

    es = EarlyStopping(patience=16, mode="min")

    wandb.watch(model, log_freq=50, log='all')

    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        if epoch == 3:
            model.module.encoder.unfreeze()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch, SRM_FLAG)

        valid_metrics = valid_epoch(model, valid_loader, criterion,  epoch)
        scheduler.step(valid_metrics["valid_loss_segmentation"])

        print(
            f"TRAIN_LOSS = {train_metrics['train_loss_segmentation']}, \
            TRAIN_DICE = {train_metrics['train_dice']}, \
            TRAIN_JACCARD = {train_metrics['train_jaccard']},"
        )
        print(
            f"VALID_LOSS = {valid_metrics['valid_loss_segmentation']}, \
            VALID_DICE = {valid_metrics['valid_dice']}, \
            VALID_JACCARD = {valid_metrics['valid_jaccard']},"
        )

        es(
            valid_metrics["valid_loss_segmentation"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5")))

    test(model, test_loader, criterion)
    wandb.save(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))

#region DICE TEST
####################################################################################
# def predict(X, threshold):
#     X_p = np.copy(X)
#     preds = (X_p > threshold).astype('uint8')
#     return preds

# def metric(probability, truth, threshold=0.5, reduction='none'):
#     '''Calculates dice of positive and negative images seperately'''
#     '''probability and truth must be torch tensors'''
#     batch_size = len(truth)
#     with torch.no_grad():
#         probability = probability.view(batch_size, -1)
#         truth = truth.view(batch_size, -1)
#         assert(probability.shape == truth.shape)

#         p = (probability > threshold).float()
#         t = (truth > 0.5).float()

#         t_sum = t.sum(-1)
#         p_sum = p.sum(-1)
#         neg_index = torch.nonzero(t_sum == 0)
#         pos_index = torch.nonzero(t_sum >= 1)

#         dice_neg = (p_sum == 0).float()
#         dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

#         dice_neg = dice_neg[neg_index]
#         dice_pos = dice_pos[pos_index]
#         dice = torch.cat([dice_pos, dice_neg])

# #         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
# #         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
# #         dice = dice.mean().item()

#         num_neg = len(neg_index)
#         num_pos = len(pos_index)

#     return dice, dice_neg, dice_pos, num_neg, num_pos

# def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
#     '''computes iou for one ground truth mask and predicted mask'''
#     pred[label == ignore_index] = 0
#     ious = []
#     for c in classes:
#         label_c = label == c
#         if only_present and np.sum(label_c) == 0:
#             ious.append(np.nan)
#             continue
#         pred_c = pred == c
#         intersection = np.logical_and(pred_c, label_c).sum()
#         union = np.logical_or(pred_c, label_c).sum()
#         if union != 0:
#             ious.append(intersection / union)
#     return ious if ious else [1]


# def compute_iou_batch(outputs, labels, classes=None):
#     '''computes mean iou for a batch of ground truth masks and predicted masks'''
#     ious = []
#     preds = np.copy(outputs) # copy is imp
#     labels = np.array(labels) # tensor to np
#     for pred, label in zip(preds, labels):
#         ious.append(np.nanmean(compute_ious(pred, label, classes)))
#     iou = np.nanmean(ious)
#     return iou

# class Meter:
#     '''A meter to keep track of iou and dice scores throughout an epoch'''
#     def __init__(self, phase, epoch):
#         self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
#         self.base_dice_scores = []
#         self.dice_neg_scores = []
#         self.dice_pos_scores = []
#         self.iou_scores = []

#     def update(self, targets, outputs):
#         probs = torch.sigmoid(outputs)
#         dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
#         self.base_dice_scores.extend(dice)
#         self.dice_pos_scores.extend(dice_pos)
#         self.dice_neg_scores.extend(dice_neg)
#         preds = predict(probs, self.base_threshold)
#         iou = compute_iou_batch(preds, targets, classes=[1])
#         self.iou_scores.append(iou)

#     def get_metrics(self):
#         dice = np.nanmean(self.base_dice_scores)
#         dice_neg = np.nanmean(self.dice_neg_scores)
#         dice_pos = np.nanmean(self.dice_pos_scores)
#         dices = [dice, dice_neg, dice_pos]
#         iou = np.nanmean(self.iou_scores)
#         return dices, iou
    
# def epoch_log(phase, epoch, epoch_loss, meter, start):
#     '''logging the metrics at the end of an epoch'''
#     dices, iou = meter.get_metrics()
#     dice, dice_neg, dice_pos = dices
#     print("Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (epoch_loss, dice, dice_neg, dice_pos, iou))
#     return dice, iou
#######################################################################################
#endregion
    
def train_epoch(model, train_loader, optimizer, criterion, epoch, SRM_FLAG):
    model.train()

    segmentation_loss = AverageMeter()
    targets = []
    outputs = []

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        gt = batch["mask"].to(device)

        optimizer.zero_grad()
        out_mask = model(images, elas)

        loss_segmentation = criterion(out_mask, gt)

        # with amp.scale_loss(loss_segmentation, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss_segmentation.backward()
        optimizer.step()

        if SRM_FLAG == 1:
            bayer_mask = torch.zeros(3,3,5,5).cuda()
            bayer_mask[:,:,5//2, 5//2] = 1
            bayer_weight = model.module.encoder.bayer_conv.weight * (1-bayer_mask)
            bayer_weight = (bayer_weight / torch.sum(bayer_weight, dim=(2,3), keepdim=True)) + 1e-7
            bayer_weight -= bayer_mask
            model.module.encoder.bayer_conv.weight = nn.Parameter(bayer_weight)
            
        # ---------------------Batch Loss Update-------------------------
        segmentation_loss.update(loss_segmentation.item(), train_loader.batch_size)

        with torch.no_grad():
            out_mask = torch.sigmoid(out_mask.cpu())
            gt = gt.cpu()
            
            targets.extend(list(gt))
            outputs.extend(list(out_mask))

        # gc.collect()
        # torch.cuda.empty_cache()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    dice, _ = seg_metrics.dice_coeff(outputs, targets) 
    jaccard, _ = seg_metrics.jaccard_coeff(outputs, targets)  
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")

    train_metrics = {
        "train_loss_segmentation": segmentation_loss.avg,
        "train_dice": dice.item(),
        "train_jaccard": jaccard.item(),
        "epoch" : epoch
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    segmentation_loss = AverageMeter()
    targets = []
    outputs = []

    example_images = []
    image_names = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)
            
            out_mask = model(images, elas)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), valid_loader.batch_size)

            out_mask = torch.sigmoid(out_mask.cpu())
            gt = gt.cpu()
            
            targets.extend(list(gt))
            outputs.extend(list(out_mask))
            example_images.extend(list(images.cpu()))
            image_names.extend(batch["image_path"])

    print("~~~~~~~~~~~~~~~~~~~~~~~~~")       
    dice, best_dice = seg_metrics.dice_coeff(outputs, targets)  
    jaccard, best_iou = seg_metrics.jaccard_coeff(outputs, targets) 
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")

    examples = []
    caption = f"{epoch}Dice:{best_dice[1]}, IOU:{best_iou[1]} Path : {image_names[best_dice[0]]}"
    examples.append(wandb.Image(example_images[best_dice[0]],caption=caption))
    examples.append(wandb.Image(outputs[best_dice[0]],caption=f'{epoch}PRED'))
    examples.append(wandb.Image(targets[best_dice[0]],caption=f'{epoch}GT'))
        
    valid_metrics = {
        "valid_loss_segmentation": segmentation_loss.avg,
        "valid_dice": dice.item(),
        "valid_jaccard": jaccard.item(),
        "examples": examples,
        "epoch" : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    segmentation_loss = AverageMeter()
    targets = []
    outputs = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)

            out_mask = model(images, elas)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), test_loader.batch_size)

            out_mask = torch.sigmoid(out_mask.cpu())
            gt = gt.cpu()
            
            targets.extend(list(gt))
            outputs.extend(list(out_mask))

    dice, _ = seg_metrics.dice_coeff(outputs, targets)  
    jaccard, _ = seg_metrics.jaccard_coeff(outputs, targets) 

    test_metrics = {
        "test_loss_segmentation": segmentation_loss.avg,
        "test_dice": dice.item(),
        "test_jaccard": jaccard.item(),
    }
    wandb.log(test_metrics)
    
    #region TEST LOGGING
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
    #endregion

def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)


if __name__ == "__main__":
    patch_size = 64
    DATA_ROOT = f"Image_Manipulation_Dataset/CASIA_2.0/image_patch_{patch_size}"

    df = pd.read_csv(f"casia_{patch_size}.csv").sample(frac=1).reset_index(drop=True)
    dice = AverageMeter()
    jaccard = AverageMeter()
    loss = AverageMeter()
    for i in range(0,1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"224CASIA_{patch_size}" + config_defaults["model"],
            df=df,
            data_root=DATA_ROOT,
            patch_size=patch_size,
            VAL_FOLD=i,
            SRM_FLAG=1
        )
        dice.update(test_metrics['test_dice'])
        jaccard.update(test_metrics['test_jaccard'])
        loss.update(test_metrics['test_loss_segmentation'])
    
    print(f'DICE : {dice.avg}')
    print(f'JACCARD : {jaccard.avg}')
    print(f'LOSS : {loss.avg}')
