import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import albumentations
from albumentations import augmentations
import imgaug.augmenters as iaa
import albumentations.pytorch

torch.backends.cudnn.benchmark = True

from utils import *
from dataset import DATASET
from segmentation.merged_net import SRM_Classifer


OUTPUT_DIR = "weights"
device =  'cuda'
config_defaults = {
    "epochs": 100,
    "train_batch_size": 40,
    "valid_batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "schedule_patience": 5,
    "schedule_factor": 0.25,
    "model": "",
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



    model = SRM_Classifer(num_classes=1, encoder_checkpoint='weights/pretrain_[31|03_12|16|32].h5')

    # for name_, param in model.named_parameters():
    #     if 'classifier' in name_:
    #         continue
    #     else:
    #         param.requires_grad = False

    print("Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))    
    

    wandb.save('segmentation/merged_net.py')
    wandb.save('dataset.py')


    train_imgaug, train_geo_aug = get_train_transforms()
    transforms_normalize = get_transforms_normalize()
    

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
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    valid_dataset = DATASET(
        dataframe=df,
        mode="val",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        transforms_normalize=transforms_normalize,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        patch_size=patch_size,
        transforms_normalize=transforms_normalize,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    #endregion ######################################################################################



    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=20, mode="min")


    model = nn.DataParallel(model).to(device)
    
    # wandb.watch(model, log_freq=50, log='all')

    start_epoch = 0
    if resume:
        checkpoint = torch.load('checkpoint/(using pretrain)COMBO_ALL_FULL_[09|04_12|46|35].pt')
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

        print(f"TRAIN_ACC = {train_metrics['train_acc_05']}, TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_ACC = {valid_metrics['valid_acc_05']}, VALID_LOSS = {valid_metrics['valid_loss']}")
        print("New LR", optimizer.param_groups[0]['lr'])

        
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

    total_loss = AverageMeter()
    
    predictions = []
    targets = []

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        target_labels = batch["label"].to(device)
        # dft_dwt_vector = batch["dft_dwt_vector"].to(device)
        
        optimizer.zero_grad()
        
        out_logits, _ = model(images, elas)#, dft_dwt_vector)

        loss = criterion(out_logits, target_labels.view(-1, 1).type_as(out_logits))
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
        
        targets.append((target_labels.view(-1, 1).cpu() >= 0.5) * 1.0)
        predictions.append(torch.sigmoid(out_logits).cpu().detach().numpy())

        gc.collect()

    # Epoch Logging
    with torch.no_grad():
        targets = np.vstack((targets)).ravel()
        predictions = np.vstack((predictions)).ravel()

        train_auc = metrics.roc_auc_score(targets, predictions)
        train_f1_05 = metrics.f1_score(targets, (predictions >= 0.5) * 1)
        train_acc_05 = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
        train_balanced_acc_05 = metrics.balanced_accuracy_score(targets, (predictions >= 0.5) * 1)
        
    train_metrics = {
        "train_loss" : total_loss.avg,
        "train_auc": train_auc,
        "train_f1_05": train_f1_05,
        "train_acc_05": train_acc_05,
        "train_balanced_acc_05": train_balanced_acc_05,
        "epoch" : epoch
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()
    
    predictions = []
    targets = []
    example_images = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            # dft_dwt_vector = batch["dft_dwt_vector"].to(device)

            out_logits, _ = model(images, elas)#, dft_dwt_vector)

            loss = criterion(out_logits, target_labels.view(-1, 1).type_as(out_logits))
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)
            
            batch_targets = (target_labels.view(-1, 1).cpu() >= 0.5) * 1.0
            batch_preds = torch.sigmoid(out_logits).cpu()          
            
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
        "valid_auc": valid_auc,
        "valid_f1_05": valid_f1_05,
        "valid_acc_05": valid_acc_05,
        "valid_balanced_acc_05": valid_balanced_acc_05,
        "valid_examples": example_images[-10:],
        "epoch": epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    total_loss = AverageMeter()
    
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            # dft_dwt_vector = batch["dft_dwt_vector"].to(device)

            out_logits, _ = model(images, elas)#, dft_dwt_vector)

            loss = criterion(out_logits, target_labels.view(-1, 1).type_as(out_logits))
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), test_loader.batch_size)
            
            targets.append((target_labels.view(-1, 1).cpu() >= 0.5) * 1.0)
            predictions.append(torch.sigmoid(out_logits).cpu() )

    # Logging
    targets = np.vstack((targets)).ravel()
    predictions = np.vstack((predictions)).ravel()

    test_auc = metrics.roc_auc_score(targets, predictions)
    test_f1_05 = metrics.f1_score(targets, (predictions >= 0.5) * 1)
    test_acc_05 = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
    test_balanced_acc_05 = metrics.balanced_accuracy_score(targets, (predictions >= 0.5) * 1)

    test_metrics = {
        "test_loss": total_loss.avg,
        "test_auc": test_auc,
        "test_f1_05": test_f1_05,
        "test_acc_05": test_acc_05,
        "test_balanced_acc_05": test_balanced_acc_05,
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



if __name__ == "__main__":
    patch_size = 'FULL'

    # df = pd.read_csv(f"combo_all_{patch_size}.csv").sample(frac=1.0, random_state=123).reset_index(drop=True)

    combo_all_df = pd.read_csv('combo_all_FULL.csv').sample(frac=1.0, random_state=123)
    
    # df_without_cmfd = combo_all_df[~combo_all_df['root_dir'].str.contains('CMFD')]
    
    # cmfd = combo_all_df[combo_all_df['root_dir'].str.contains('CMFD')]
    # cmfd_real = cmfd[cmfd['label'] == 0].sample(n=1000, random_state=123)
    # cmfd_fake = cmfd[cmfd['label'] == 1].sample(n=1800, random_state=123)
    # cmfd = pd.concat([cmfd_real, cmfd_fake]).sample(frac=1.0, random_state=123)
    
    nist_extend = pd.read_csv('nist_extend.csv').sample(frac=1.0, random_state=123)
    # # nist_extend_real = nist_extend[nist_extend['label'] == 0].sample(n=1000, random_state=123)
    # # nist_extend_fake = nist_extend[nist_extend['label'] == 1].sample(n=1500, random_state=123)
    # # nist_extend = pd.concat([nist_extend_real, nist_extend_fake]).sample(frac=1.0, random_state=123)
    
    coverage_extend = pd.read_csv('coverage_extend.csv').sample(frac=1.0, random_state=123)
    # # coverage_extend_real = coverage_extend[coverage_extend['label'] == 0].sample(n=800, random_state=123)
    # # coverage_extend_fake = coverage_extend[coverage_extend['label'] == 1].sample(n=800, random_state=123)
    # # coverage_extend = pd.concat([coverage_extend_real, coverage_extend_fake]).sample(frac=1.0, random_state=123)
            
    df_full = pd.concat([combo_all_df, nist_extend, coverage_extend])
    df_full.insert(0, 'image', -1)

    df_128 = pd.read_csv('combo_all_128.csv').sample(frac=1.0, random_state=123)

    df = pd.concat([df_full, df_128])
    print(df.groupby('root_dir').label.value_counts())

    acc = AverageMeter()
    f1 = AverageMeter()
    loss = AverageMeter()
    auc = AverageMeter()
    for i in range(1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"(full+128)COMBO_ALL_{patch_size}" + config_defaults["model"],
            df=df,
            patch_size=patch_size,
            VAL_FOLD=i,
            resume=False
        )
        acc.update(test_metrics['test_acc_05'])
        f1.update(test_metrics['test_f1_05'])
        loss.update(test_metrics['test_loss'])
        auc.update(test_metrics['test_auc'])
    
    print(f'FINAL ACCURACY : {acc.avg}')
    print(f'FINAL F1 : {f1.avg}')
    print(f'FINAL LOSS : {loss.avg}')
    print(f'FINAL AUC : {auc.avg}')