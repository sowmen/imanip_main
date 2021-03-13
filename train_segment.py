import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
import pickle as pkl
from sklearn import metrics
import gc
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

import albumentations
from albumentations import augmentations
import imgaug.augmenters as iaa
import albumentations.pytorch
from torchvision import transforms

torch.backends.cudnn.benchmark = True

# from apex import amp
import wandb
# import neptune
# from neptunecontrib.monitoring.metrics import *

from dataset import DATASET
import seg_metrics
from pytorch_toolbelt import losses
from utils import *

# import segmentation_models_pytorch as smp
from segmentation.timm_srm_unetpp import UnetPP
from segmentation.merged_net import SRM_Classifer 
from sim_dataset import SimDataset

OUTPUT_DIR = "weights"
device = 'cuda'
config_defaults = {
    "epochs": 100,
    "train_batch_size": 16,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.0005,
    "schedule_patience": 3,
    "schedule_factor": 0.25,
    'sampling':'nearest',
    "model": "UnetPP",
}
TEST_FOLD = 1


def train(name, df, patch_size, VAL_FOLD=0, resume=False):
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
    # model = smp.DeepLabV3('resnet34', classes=1, encoder_weights='imagenet')
    
    # encoder = EfficientNet(encoder_checkpoint='64_encoder.h5', freeze_encoder=True).get_encoder()
    # model = UnetB4_Inception(encoder, in_channels=54, num_classes=1, sampling=config.sampling, layer='end')
    encoder = SRM_Classifer(encoder_checkpoint='weights/Changed classifier+COMBO_ALL_FULLSRM+ELA_[08|03_21|22|09].h5', freeze_encoder=True)
    model = UnetPP(encoder, num_classes=1, sampling=config.sampling, layer='end')
    model.to(device)
    
    SRM_FLAG=1
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    wandb.save('segmentation/merged_net.py')
    wandb.save('segmentation/timm_srm_unetpp.py')
    wandb.save('dataset.py')
    
    
    #####################################################################################################################
    train_imgaug  = iaa.Sequential(
        [
            iaa.SomeOf((0, 5),
                [   
                    iaa.OneOf([
                        iaa.JpegCompression(compression=(10, 60)),
                        iaa.GaussianBlur((0, 1.75)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    # iaa.Sometimes(0.3, iaa.Invert(0.05, per_channel=True)), # invert color channels
                    # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    # # either change the brightness of the whole image (sometimes
                    # # per channel) or change the brightness of subareas
                    iaa.Sometimes(0.6,
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.MultiplyAndAddToBrightness(mul=(0.5, 2.5), add=(-10,10)),
                            iaa.MultiplyHueAndSaturation(),
                            # iaa.BlendAlphaFrequencyNoise(
                            #     exponent=(-4, 0),
                            #     foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                            #     background=iaa.LinearContrast((0.5, 2.0))
                            # )
                        ])
                    ),
                ], random_order=True
            )
        ], random_order=True
    )
    train_geo_aug = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.1),
            albumentations.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=35, p=0.25),
            # albumentations.OneOf([
            #     albumentations.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #     albumentations.GridDistortion(p=0.5),
            #     albumentations.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)                  
            # ], p=0.7),
        ],
        additional_targets={'ela':'image'}
    )
    ####################################################################################################################

    normalize = {
        "mean": [0.4535408213875562, 0.42862278450748387, 0.41780105499276865],
        "std": [0.2672804038612597, 0.2550410416463668, 0.29475415579144293],
    }

    transforms_normalize = albumentations.Compose(
        [
            albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
            albumentations.pytorch.transforms.ToTensorV2()
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
    train_dataset = DATASET(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        resize=256,
        transforms_normalize=transforms_normalize,
        imgaug_augment=train_imgaug,
        geo_augment=train_geo_aug
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)

    valid_dataset = DATASET(
        dataframe=df,
        mode="val",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        resize=256,
        transforms_normalize=transforms_normalize,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        resize=256,
        transforms_normalize=transforms_normalize,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)

    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )

    # bce = nn.BCEWithLogitsLoss()
    dice = losses.DiceLoss(mode='binary', log_loss=True, smooth=1e-7)
    criterion = dice #losses.JointLoss(bce, dice)

    es = EarlyStopping(patience=15, mode="min")

    # wandb.watch(model, log_freq=50, log='all')

    start_epoch = 0
    if resume:
        checkpoint = torch.load('checkpoint/224CASIA_128UnetPP_[30|10_05|21|34].pt')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("-----------> Resuming <------------")

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        # if epoch == 2:
        #     model.module.encoder.unfreeze()

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

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join('checkpoint', f"{name}_[{dt_string}].pt"))

    if os.path.exists(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5")):
        print(model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))))
        print("LOADED FOR TEST")

    test_metrics = test(model, test_loader, criterion)
    wandb.save(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))

    return test_metrics

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
        # out_mask = model(images)

        loss_segmentation = criterion(out_mask, gt)
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
            out_mask = torch.sigmoid(out_mask.detach().cpu())
            out_mask = out_mask.squeeze(1)
            gt = gt.detach().cpu()
            
            targets.extend(list(gt))
            outputs.extend(list(out_mask))

        gc.collect()
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
            # out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), valid_loader.batch_size)

            out_mask = torch.sigmoid(out_mask.detach().cpu())
            out_mask = out_mask.squeeze(1)
            gt = gt.detach().cpu()
            
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
            # out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), test_loader.batch_size)

            out_mask = torch.sigmoid(out_mask.detach().cpu())
            out_mask = out_mask.squeeze(1)
            gt = gt.detach().cpu()
            
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
    return test_metrics
    

def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)


if __name__ == "__main__":
    patch_size = "FULL"

    df = pd.read_csv(f"combo_all_{patch_size}.csv").sample(frac=1.0, random_state=123).reset_index(drop=True)
    dice = AverageMeter()
    jaccard = AverageMeter()
    loss = AverageMeter()
    for i in range(0,1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"(DICE)ChangedClass_COMBO_ALL_{patch_size}" + config_defaults["model"],
            df=df,
            patch_size=patch_size,
            VAL_FOLD=i,
            resume=False
        )
        dice.update(test_metrics['test_dice'])
        jaccard.update(test_metrics['test_jaccard'])
        loss.update(test_metrics['test_loss_segmentation'])
    
    print(f'DICE : {dice.avg}')
    print(f'JACCARD : {jaccard.avg}')
    print(f'LOSS : {loss.avg}')
