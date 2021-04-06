import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import albumentations
import imgaug.augmenters as iaa
import albumentations.pytorch

torch.backends.cudnn.benchmark = True
 
from utils import *
from pretrain_dataset import DATASET
from segmentation.merged_net import SRM_Classifer
import torchmetrics


OUTPUT_DIR = "weights"
device =  'cuda'
config_defaults = {
    "epochs": 30,
    "train_batch_size": 24,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.0009,
    "weight_decay": 0.0001,
    "schedule_patience": 5,
    "schedule_factor": 0.25,
    "model": "",
}

def train(name, df, resume=False):
    now = datetime.now()
    dt_string = now.strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wandb.init(
        project="imanip", config=config_defaults, name=f"{name},{dt_string}",
    )
    config = wandb.config


    model = SRM_Classifer(num_classes=312)
    print("Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))    
    
    wandb.save('segmentation/merged_net.py')
    wandb.save('pretrain_dataset.py')

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
                    iaa.Sometimes(0.5,
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

    # -------------------------------- CREATE DATASET and DATALOADER --------------------------
    df_train, df_val, df_test = stratified_train_val_test_split(df, stratify_colname='class_idx', 
                                                                frac_train=0.96, frac_val=0.02, frac_test=0.02)

    train_dataset = DATASET(
        dataframe=df_train,
        mode="train",
        transforms_normalize=transforms_normalize,
        imgaug_augment=None,
        geo_augment=train_geo_aug
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)

    valid_dataset = DATASET(
        dataframe=df_val,
        mode="val",
        transforms_normalize=transforms_normalize,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df_test,
        mode="test",
        transforms_normalize=transforms_normalize,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)


    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )

    model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()

    es = EarlyStopping(patience=15, mode="min")

    start_epoch = 0
    if resume:
        checkpoint = torch.load('checkpoint/pretrain_[28|03_11|34|53].pt')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("-----------> Resuming <------------")

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion, epoch)

        scheduler.step(valid_metrics['valid_loss'])

        print(
            "TRAIN_ACC = %.5f, TRAIN_LOSS = %.5f" % (train_metrics['train_acc5_manual'], train_metrics['train_loss'])
        )
        print(
            "VALID_ACC = %.5f, VALID_LOSS = %.5f" % (valid_metrics['valid_acc5_manual'], valid_metrics['valid_loss'])
        )
        print("New LR", optimizer.param_groups[0]['lr'])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        es(
            valid_metrics['valid_loss'],
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
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoint', f"{name}_[{dt_string}].pt"))

    if os.path.exists(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5")):
        print(model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))))
        print("LOADED FOR TEST")
        wandb.save(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))

    test_metrics = test(model, test_loader, criterion)
    

    return test_metrics


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    manual_top1 = AverageMeter()
    manual_top5 = AverageMeter()
    torch_top1 = torchmetrics.Accuracy()
    torch_top5 = torchmetrics.Accuracy(top_k=5)
    torch_f1 = torchmetrics.F1(num_classes=312)

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        target_labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        out_logits, _ = model(images, elas)

        loss = criterion(out_logits, target_labels)
        
        loss.backward()
        optimizer.step()

        ############## SRM Step ###########
        bayer_mask = torch.zeros(3,3,5,5).cuda()
        bayer_mask[:,:,5//2, 5//2] = 1
        bayer_weight = model.module.bayer_conv.weight * (1-bayer_mask)
        bayer_weight = (bayer_weight / torch.sum(bayer_weight, dim=(2,3), keepdim=True)) + 1e-7
        bayer_weight -= bayer_mask
        model.module.bayer_conv.weight = nn.Parameter(bayer_weight)
        ###################################

        #---------------------Batch Loss Update-------------------------
        total_loss.update(loss.item(), train_loader.batch_size)
        
        # Metric
        with torch.no_grad():
            out_logits = out_logits.cpu().detach()
            target_labels = target_labels.cpu().detach()

            topk = topk_accuracy(out_logits, target_labels, topk=(1,5))
            manual_top1.update(topk[0].item(), train_loader.batch_size)
            manual_top5.update(topk[1].item(), train_loader.batch_size)

            torch_top1.update(torch.softmax(out_logits, dim=-1), target_labels)
            torch_top5.update(torch.softmax(out_logits, dim=-1), target_labels)
            torch_f1.update(torch.softmax(out_logits, dim=-1), target_labels)
    
        
    train_metrics = {
        "train_loss" : total_loss.avg,
        "train_acc1_manual": manual_top1.avg,
        "train_acc5_manual": manual_top5.avg,
        "train_acc1_torch": torch_top1.compute().item(),
        "train_acc_5_torch": torch_top5.compute().item(),
        "train_f1": torch_f1.compute().item(),
        "epoch" : epoch
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()
    
    manual_top1 = AverageMeter()
    manual_top5 = AverageMeter()
    torch_top1 = torchmetrics.Accuracy()
    torch_top5 = torchmetrics.Accuracy(top_k=5)
    torch_f1 = torchmetrics.F1(num_classes=312)

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            
            out_logits, _ = model(images, elas)

            loss = criterion(out_logits, target_labels)
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)
                    
            # Metric
            with torch.no_grad():
                out_logits = out_logits.cpu().detach()
                target_labels = target_labels.cpu().detach()

                topk = topk_accuracy(out_logits, target_labels, topk=(1,5))
                manual_top1.update(topk[0].item(), valid_loader.batch_size)
                manual_top5.update(topk[1].item(), valid_loader.batch_size)

                torch_top1.update(torch.softmax(out_logits, dim=-1), target_labels)
                torch_top5.update(torch.softmax(out_logits, dim=-1), target_labels)
                torch_f1.update(torch.softmax(out_logits, dim=-1), target_labels)


    valid_metrics = {
        "valid_loss": total_loss.avg,
        "valid_acc1_manual": manual_top1.avg,
        "valid_acc5_manual": manual_top5.avg,
        "valid_acc1_torch": torch_top1.compute().item(),
        "valid_acc_5_torch": torch_top5.compute().item(),
        "valid_f1": torch_f1.compute().item(),
        "epoch": epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    total_loss = AverageMeter()
    
    manual_top1 = AverageMeter()
    manual_top5 = AverageMeter()
    torch_top1 = torchmetrics.Accuracy()
    torch_top5 = torchmetrics.Accuracy(top_k=5)
    torch_f1 = torchmetrics.F1(num_classes=312)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            
            out_logits, _ = model(images, elas)

            loss = criterion(out_logits, target_labels)
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), test_loader.batch_size)
                    
            # Metric
            with torch.no_grad():
                out_logits = out_logits.cpu().detach()
                target_labels = target_labels.cpu().detach()

                topk = topk_accuracy(out_logits, target_labels, topk=(1,5))
                manual_top1.update(topk[0].item(), test_loader.batch_size)
                manual_top5.update(topk[1].item(), test_loader.batch_size)

                torch_top1.update(torch.softmax(out_logits, dim=-1), target_labels)
                torch_top5.update(torch.softmax(out_logits, dim=-1), target_labels)
                torch_f1.update(torch.softmax(out_logits, dim=-1), target_labels)


    test_metrics = {
        "test_loss": total_loss.avg,
        "test_acc1_manual": manual_top1.avg,
        "test_acc5_manual": manual_top5.avg,
        "test_acc1_torch": torch_top1.compute().item(),
        "test_acc_5_torch": torch_top5.compute().item(),
        "test_f1": torch_f1.compute().item(),
    }
    wandb.log(test_metrics)

    return test_metrics


if __name__ == "__main__":

    df = pd.read_csv('pretrain_images.csv').sample(frac=1.0, random_state=123).reset_index(drop=True)


    test_metrics = train(
        name=f"pretrain" + config_defaults["model"],
        df=df,
        resume=True
    )
    
    print(test_metrics)