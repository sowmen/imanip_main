import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

import albumentations
import imgaug.augmenters as iaa
import albumentations.pytorch
from torchvision import transforms

torch.backends.cudnn.benchmark = True

import wandb

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
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0005,
    "schedule_patience": 3,
    "schedule_factor": 0.25,
    'sampling':'nearest',
    "model": "UnetPP",
}
TEST_FOLD = 1


def train(name, df, patch_size, VAL_FOLD=0, resume=False):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    wandb.init(project="imanip", config=config_defaults, name=run)
    config = wandb.config


    # model = smp.DeepLabV3('resnet34', classes=1, encoder_weights='imagenet')
    
    encoder = SRM_Classifer(encoder_checkpoint='weights/Changed classifier+COMBO_ALL_FULLSRM+ELA_[08|03_21|22|09].h5', freeze_encoder=True)
    model = UnetPP(encoder, num_classes=1, sampling=config.sampling, layer='end')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    wandb.save('segmentation/merged_net.py')
    wandb.save('segmentation/timm_srm_unetpp.py')
    wandb.save('dataset.py')
    
    
    train_imgaug, train_geo_aug = get_train_transforms()
    transforms_normalize = get_transforms_normalize()
    

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

    #region ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = DATASET(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
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
        transforms_normalize=transforms_normalize,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        transforms_normalize=transforms_normalize,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
    #endregion ######################################################################################



    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )
    criterion = get_lossfn()
    es = EarlyStopping(patience=20, mode="min")


    model = nn.DataParallel(model).to(device)
    
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

        # if epoch == 4:
        #     model.module.encoder.unfreeze()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
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
            model_path=os.path.join(OUTPUT_DIR, f"{run}.h5"),
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
        torch.save(checkpoint, os.path.join('checkpoint', f"{run}.pt"))


    if os.path.exists(os.path.join(OUTPUT_DIR, f"{run}.h5")):
        print(model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{run}.h5"))))
        print("LOADED FOR TEST")

    test_metrics = test(model, test_loader, criterion)
    wandb.save(os.path.join(OUTPUT_DIR, f"{run}.h5"))

    return test_metrics
    
    
    
def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    segmentation_loss = AverageMeter()
    dice = AverageMeter()
    jaccard = AverageMeter()

    scores = seg_metrics.DiceMeter()

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        gt = batch["mask"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        out_mask = model(images, elas)
        # out_mask = model(images)

        loss_segmentation = criterion(out_mask, gt)
        loss_segmentation.backward()

        optimizer.step()

        ############## SRM Step ###########
        bayer_mask = torch.zeros(3,3,5,5).cuda()
        bayer_mask[:, :, 5//2, 5//2] = 1
        bayer_weight = model.module.encoder.bayer_conv.weight * (1-bayer_mask)
        bayer_weight = (bayer_weight / torch.sum(bayer_weight, dim=(2,3), keepdim=True)) + 1e-7
        bayer_weight -= bayer_mask
        model.module.encoder.bayer_conv.weight = nn.Parameter(bayer_weight)
        ###################################
            
        # ---------------------Batch Loss Update-------------------------
        segmentation_loss.update(loss_segmentation.item(), train_loader.batch_size)

        with torch.no_grad():
            out_mask = torch.sigmoid(out_mask)
            out_mask = out_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, _, _ = seg_metrics.get_avg_batch_dice(out_mask, gt)
            dice.update(batch_dice, train_loader.batch_size)

            batch_jaccard, _, _ = seg_metrics.get_avg_batch_jaccard(out_mask, gt)
            jaccard.update(batch_jaccard, train_loader.batch_size)

            scores.update(gt, out_mask)


    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TRAIN", scores)
    train_metrics = {
        "train_loss_segmentation": segmentation_loss.avg,
        "train_dice": dice.avg,
        "train_jaccard": jaccard.avg,
        "epoch" : epoch,
        "train_dice2" : dice2,
        "train_dice_pos" : dice_pos,
        "train_dice_neg" : dice_neg,
        "train_iou2" : iou2
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    segmentation_loss = AverageMeter()
    dice = AverageMeter()
    jaccard = AverageMeter()
    scores = seg_metrics.DiceMeter()

    example_images = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].unsqueeze(1).to(device)
            
            out_mask = model(images, elas)
            # out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), valid_loader.batch_size)

            out_mask = torch.sigmoid(out_mask)
            out_mask = out_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, (mx_dice, best_dice_idx), (worst_dice, worst_dice_idx) = seg_metrics.get_avg_batch_dice(out_mask, gt)
            dice.update(batch_dice, valid_loader.batch_size)

            batch_jaccard, (mx_iou, best_iou__idx), (worst_iou, worst_idx) = seg_metrics.get_avg_batch_jaccard(out_mask, gt)
            jaccard.update(batch_jaccard, valid_loader.batch_size)

            if(np.random.rand() < 0.5 and len(example_images) < 10):
                images = images.cpu().detach()
                example_images.append({
                    "best_image" : images[best_dice_idx],
                    "best_image_name" : batch["image_path"][best_dice_idx],
                    "best_dice" : mx_dice,
                    "best_pred" : out_mask[best_dice_idx],
                    "best_gt" : gt[best_dice_idx],
                    "worst_image" : images[worst_dice_idx],
                    "worst_image_name" : batch["image_path"][worst_dice_idx],
                    "worst_dice" : worst_dice,
                    "worst_pred" : out_mask[worst_dice_idx],
                    "worst_gt" : gt[worst_dice_idx],
                })
            
            scores.update(gt, out_mask)

            gc.collect()

    examples = []
    for b in example_images:
        caption_best = f"{epoch}Best Dice:" + str(b["best_dice"]) + "Path : " + str(b["best_image_name"])
        examples.append(wandb.Image(b['best_image'],caption=caption_best))
        examples.append(wandb.Image(b['best_pred'],caption=f'{epoch}PRED'))
        examples.append(wandb.Image(b['best_gt'],caption=f'{epoch}GT'))

        caption_worst = f"{epoch}Worst Dice:" + str(b["worst_dice"]) + "Path : " + str(b["worst_image_name"])
        examples.append(wandb.Image(b['worst_image'],caption=caption_worst))
        examples.append(wandb.Image(b['worst_pred'],caption=f'{epoch}PRED'))
        examples.append(wandb.Image(b['worst_gt'],caption=f'{epoch}GT'))

    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("VALID", scores)

    valid_metrics = {
        "valid_loss_segmentation": segmentation_loss.avg,
        "valid_dice": dice.avg,
        "valid_jaccard": jaccard.avg,
        "examples": examples,
        "epoch" : epoch,
        "valid_dice2" : dice2,
        "valid_dice_pos" : dice_pos,
        "valid_dice_neg" : dice_neg,
        "valid_iou2" : iou2
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    segmentation_loss = AverageMeter()
    dice = AverageMeter()
    jaccard = AverageMeter()
    scores = seg_metrics.DiceMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].unsqueeze(1).to(device)

            out_mask = model(images, elas)
            # out_mask = model(images)

            loss_segmentation = criterion(out_mask, gt)

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), test_loader.batch_size)

            out_mask = torch.sigmoid(out_mask)
            out_mask = out_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, _, _ = seg_metrics.get_avg_batch_dice(out_mask, gt)
            dice.update(batch_dice, test_loader.batch_size)

            batch_jaccard, _, _ = seg_metrics.get_avg_batch_jaccard(out_mask, gt)
            jaccard.update(batch_jaccard, test_loader.batch_size)

            scores.update(gt, out_mask)

            gc.collect()

    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TEST", scores)
    test_metrics = {
        "test_loss_segmentation": segmentation_loss.avg,
        "test_dice": dice.avg,
        "test_jaccard": jaccard.avg,
        "test_dice2" : dice2,
        "test_dice_pos" : dice_pos,
        "test_dice_neg" : dice_neg,
        "test_iou2" : iou2
    }
    wandb.log(test_metrics)
    return test_metrics
    

def get_train_transforms():
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
    return train_imgaug, train_geo_aug

def get_transforms_normalize():
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
    return transforms_normalize

def get_lossfn():
    bce = nn.BCEWithLogitsLoss()
    dice = losses.DiceLoss(mode='binary', log_loss=True, smooth=1e-7)
    criterion = losses.JointLoss(bce, dice)
    
    return criterion


if __name__ == "__main__":
    patch_size = "64"

    df = pd.read_csv(f"dataset_csv/coverage_{patch_size}.csv").sample(frac=1.0, random_state=123).reset_index(drop=True)
    dice = AverageMeter()
    jaccard = AverageMeter()
    loss = AverageMeter()
    for i in range(0,1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"(coverage)InvCClass_COMBO_ALL_{patch_size}" + config_defaults["model"],
            df=df,
            patch_size=patch_size,
            VAL_FOLD=i,
            resume=False,
        )
        dice.update(test_metrics['test_dice'])
        jaccard.update(test_metrics['test_jaccard'])
        loss.update(test_metrics['test_loss_segmentation'])
    
    print(f'DICE : {dice.avg}')
    print(f'JACCARD : {jaccard.avg}')
    print(f'LOSS : {loss.avg}')
