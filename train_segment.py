import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

import wandb

from dataset import DATASET
import seg_metrics
from pytorch_toolbelt import losses
from utils import *

from segmentation.smp_srm import SMP_SRM_UPP
from segmentation.timm_srm_unetpp import UnetPP
from segmentation.merged_net import SRM_Classifer 
from sim_dataset import SimDataset

from segmentation.merged_netv2 import Mani_FeatX
from segmentation.unetpp_v2 import MyUnetPP

OUTPUT_DIR = "/content/drive/MyDrive/Image_Manipulation_Dataset/weights"
CKPT_DIR = "/content/drive/MyDrive/Image_Manipulation_Dataset/checkpoint"
device = 'cuda'
config_defaults = {
    "epochs": 60,
    "train_batch_size": 20,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0005,
    "schedule_patience": 5,
    "schedule_factor": 0.25,
    'sampling':'nearest',
    "model": "SMP-UnetPP",
}
TEST_FOLD = 1


def train(name, df, VAL_FOLD=0, resume=None):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    print("Starting -->", run)
    
    wandb.init(project="imanip", config=config_defaults, name=run)
    config = wandb.config


    # model = smp.DeepLabV3('resnet34', classes=1, encoder_weights='imagenet')
    # model = SMP_SRM_UPP()
    
    # encoder = SRM_Classifer(encoder_checkpoint='weights/pretrain_[31|03_12|16|32].h5', freeze_encoder=True)
    # model = UnetPP(encoder, num_classes=1, sampling=config.sampling, layer='end')

    encoder = Mani_FeatX()
    model = MyUnetPP(encoder)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    wandb.save('segmentation/merged_netv2.py')
    wandb.save('segmentation/unetpp_v2.py')
    wandb.save('segmentation/layers.py')
    wandb.save('dataset.py')
    
    
    train_imgaug, train_geo_aug = get_train_transforms()
    transforms_normalize = get_transforms_normalize()
    

    #region SIMULATION
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    # ])

    # train_set = SimDataset(2000, transform = trans)
    # train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, num_workers=4)
    # val_set = SimDataset(500, transform = trans)
    # valid_loader = DataLoader(val_set, batch_size=config.valid_batch_size, shuffle=False, num_workers=4)
    #endregion

    #region ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = DATASET(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        segment=False,
        transforms_normalize=transforms_normalize,
        imgaug_augment=None,
        geo_augment=train_geo_aug,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    valid_dataset = DATASET(
        dataframe=df,
        mode="val",
        segment=False,
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        transforms_normalize=transforms_normalize,
        equal_sample=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        segment=False,
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        transforms_normalize=transforms_normalize,
        equal_sample=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    #endregion ######################################################################################



    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
        verbose=True
    )
    criterion = get_lossfn()
    es = EarlyStopping(patience=20, mode="min")


    model = nn.DataParallel(model).to(device)
    # print(model.load_state_dict(torch.load('(defacto+customloss)UnetPP_[29|04_21|04|41].h5')))
    
    # wandb.watch(model, log_freq=50, log='all')
    
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("-----------> Resuming <------------")

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        # if epoch == 6:
        #     model.module.encoder.unfreeze()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion,  epoch)
        
        scheduler.step(valid_metrics["valid_loss_segmentation"])

        print(
            f"TRAIN_LOSS = {train_metrics['train_loss_segmentation']}, \
            TRAIN_FAKE_DICE = {train_metrics['train_fake_dice']}, \
            TRAIN_REAL_FPR = {train_metrics['train_real_fpr']}"
        )
        print(
            f"VALID_LOSS = {valid_metrics['valid_loss_segmentation']}, \
            VALID_FAKE_DICE = {valid_metrics['valid_fake_dice']}, \
            VALID_REAL_FPR = {valid_metrics['valid_real_fpr']}"
        )
        print("New LR", optimizer.param_groups[0]['lr'])
        wandb.log({
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        es(valid_metrics["valid_loss_segmentation"],
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
        torch.save(checkpoint, os.path.join(CKPT_DIR, f"{run}.pt"))


    if os.path.exists(os.path.join(OUTPUT_DIR, f"{run}.h5")):
        print(model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{run}.h5"))))
        print("LOADED FOR TEST")

    test(model, test_loader, criterion)
    calculate_auc(model, test_dataset, 'TEST')
    calculate_auc(model, valid_dataset, 'VAL')

    wandb.save(os.path.join(OUTPUT_DIR, f"{run}.h5"))
    
    
    
def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    classification_loss = AverageMeter()
    mask_loss = AverageMeter()
    
    metrics = seg_metrics.MetricMeter("TRAIN")
    scores = seg_metrics.SegMeter()

    for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        gt = batch["mask"].to(device)
        target_labels = batch["label"].to(device)

        optimizer.zero_grad()
        pred_mask, label_tensor = model(images, elas)

        loss_segmentation, bce_loss, dice_loss = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))
        loss_segmentation.backward()

        optimizer.step()

        ############## SRM Step ###########
        bayer_mask = torch.zeros(3,3,5,5).cuda()
        bayer_mask[:, :, 5//2, 5//2] = 1
        bayer_weight = model.module.bayer_conv.weight * (1-bayer_mask)
        bayer_weight = (bayer_weight / torch.sum(bayer_weight, dim=(2,3), keepdim=True)) + 1e-7
        bayer_weight -= bayer_mask
        model.module.bayer_conv.weight = nn.Parameter(bayer_weight)
        ###################################
            
        # ---------------------Batch Loss Update-------------------------
        total_loss.update(loss_segmentation.item(), train_loader.batch_size)
        classification_loss.update(bce_loss.item(), train_loader.batch_size)
        mask_loss.update(dice_loss.item(), train_loader.batch_size)

        with torch.no_grad():
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            metrics.update(pred_mask, gt, batch)
            scores.update(pred_mask, gt)
            

    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TRAIN", scores)

    train_metrics = {
        "epoch" : epoch,
        "train_loss_segmentation": total_loss.avg,
        "train_classification_loss": classification_loss.avg,
        "train_mask_loss": mask_loss.avg,

        "train_fake_dice": metrics.fake_dice.avg,
        "train_fake_jaccard": metrics.fake_jaccard.avg,
        "train_fake_pixel_auc": metrics.fake_pixel_auc.avg,
        "train_fake_fpr": metrics.fake_fpr.avg,
        "train_real_fpr": metrics.real_fpr.avg,
        "train_total_fpr": metrics.total_fpr.avg,

        "train_dice2" : dice2,
        "train_dice_pos" : dice_pos,
        "train_dice_neg" : dice_neg,
        "train_iou2" : iou2,
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()
    classification_loss = AverageMeter()
    mask_loss = AverageMeter()
    
    metrics = seg_metrics.MetricMeter("VALID")
    scores = seg_metrics.SegMeter()
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Valid epoch {epoch}", dynamic_ncols=True):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)
            target_labels = batch["label"].to(device)
                    
            
            pred_mask, label_tensor = model(images, elas)

            loss_segmentation, bce_loss, dice_loss = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))

            # ---------------------Batch Loss Update-------------------------
            total_loss.update(loss_segmentation.item(), valid_loader.batch_size)
            classification_loss.update(bce_loss.item(), valid_loader.batch_size)
            mask_loss.update(dice_loss.item(), valid_loader.batch_size)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            metrics.update(pred_mask, gt, batch)
            scores.update(pred_mask, gt)


    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("VALID", scores)

    valid_metrics = {
        "epoch" : epoch,
        "valid_loss_segmentation": total_loss.avg,
        "valid_classification_loss": classification_loss.avg,
        "valid_mask_loss": mask_loss.avg,

        "valid_fake_dice": metrics.fake_dice.avg,
        "valid_fake_jaccard": metrics.fake_jaccard.avg,
        "valid_fake_pixel_auc": metrics.fake_pixel_auc.avg,
        "valid_fake_fpr": metrics.fake_fpr.avg,
        "valid_real_fpr": metrics.real_fpr.avg,
        "valid_total_fpr": metrics.total_fpr.avg,

        "valid_dice2" : dice2,
        "valid_dice_pos" : dice_pos,
        "valid_dice_neg" : dice_neg,
        "valid_iou2" : iou2,
        
        "examples": metrics.example_images,
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    total_loss = AverageMeter()
    classification_loss = AverageMeter()
    mask_loss = AverageMeter()
    
    metrics = seg_metrics.MetricMeter("TEST")
    scores = seg_metrics.SegMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader, dynamic_ncols=True):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)
            target_labels = batch["label"].to(device)
                    
            
            pred_mask, label_tensor = model(images, elas)
            # pred_mask = model(images)

            loss_segmentation, bce_loss, dice_loss = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))

            # ---------------------Batch Loss Update-------------------------
            total_loss.update(loss_segmentation.item(), test_loader.batch_size)
            classification_loss.update(bce_loss.item(), test_loader.batch_size)
            mask_loss.update(dice_loss.item(), test_loader.batch_size)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            metrics.update(pred_mask, gt, batch)
            scores.update(pred_mask, gt)


    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TEST", scores)

    test_metrics = {
        "test_loss_segmentation": total_loss.avg,
        "test_classification_loss": classification_loss.avg,
        "test_mask_loss": mask_loss.avg,

        "test_fake_dice": metrics.fake_dice.avg,
        "test_fake_jaccard": metrics.fake_jaccard.avg,
        "test_fake_pixel_auc": metrics.fake_pixel_auc.avg,
        "test_fake_fpr": metrics.fake_fpr.avg,
        "test_real_fpr": metrics.real_fpr.avg,
        "test_total_fpr": metrics.total_fpr.avg,

        "test_dice2" : dice2,
        "test_dice_pos" : dice_pos,
        "test_dice_neg" : dice_neg,
        "test_iou2" : iou2,
        
        "examples": metrics.example_images,
    }
    wandb.log(test_metrics)


from sklearn.metrics import roc_auc_score
def calculate_auc(model, dataset, step):
    model.eval()

    preds = []
    truths = []
    paths = []

    for data in tqdm(dataset, desc=f"{step} AUC: "):
        images = data["image"].unsqueeze(0).to(device)
        elas = data["ela"].unsqueeze(0).to(device)
        gt = data["mask"].unsqueeze(0)
        target_labels = data["label"]
        mask_path = data['mask_path']

        if(target_labels > 0.5 and np.count_nonzero(gt.numpy().ravel() >= 0.5) > 0):
            pred_mask, _ = model(images, elas)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            
            preds.append(pred_mask.squeeze())
            truths.append(gt.squeeze())
            paths.append(mask_path)
    

    thrs = [0.0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    example_images = []
    auc, cnt = 0, 0
    for gtr, pr, path in tqdm(zip(truths, preds, paths),total=len(truths)):
        best, bt = -1, 0
        for thr in thrs:
            tmp = roc_auc_score(gtr.numpy().ravel() >= 0.5, pr.numpy().ravel() >= thr)          
            if tmp > best: 
                best = tmp
                bt = thr
        auc += best
        cnt += 1
        example_images.append((pr, gtr, path, bt))
    dataset_auc = auc / cnt

    examples = []
    for b in example_images:
        caption = "Thr:" + str(b[-1]) + b[-2]
        examples.append(wandb.Image(b[0],caption=caption))
        examples.append(wandb.Image(b[1]))

    wandb.log({
        "examples": examples,
        f"{step}_pixel_auc" : dataset_auc
    })
    print(f"{step} AUC : ", dataset_auc)


from losses import DiceLoss, ImanipLoss
def get_lossfn():
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss(mode='binary', log_loss=True, smooth=1e-7)
    criterion = ImanipLoss(bce, dice)
    return criterion

    
if __name__ == "__main__":

    #---------------------------------- FULL --------------------------------------#
    # combo_all_df = get_dataframe('combo_all_FULL.csv', folds=None)
    casia_full = get_dataframe('dataset_csv/casia_FULL.csv', folds=None)
    # imd_full = get_dataframe('dataset_csv/imd_FULL.csv', folds=None)
    # cmfd_full = get_dataframe('dataset_csv/cmfd_FULL.csv', folds=-1)
    # nist_full = get_dataframe('dataset_csv/nist16_FULL.csv', folds=None)
    # coverage_full = get_dataframe('dataset_csv/coverage_FULL.csv', folds=None)
    
    # nist_extend = get_dataframe('nist_extend.csv', folds=12)
    # coverage_extend = get_dataframe('coverage_extend.csv', folds=12)
    # defacto_cp = get_dataframe('dataset_csv/defacto_copy_move.csv', folds=-1)
    # defacto_inpaint = get_dataframe('dataset_csv/defacto_inpainting.csv', folds=-1)
    # defacto_s1 = get_dataframe('dataset_csv/defacto_splicing1.csv', folds=-1)
    # defacto_s2 = get_dataframe('dataset_csv/defacto_splicing2.csv', folds=-1)
    # defacto_s3 = get_dataframe('dataset_csv/defacto_splicing3.csv', folds=-1)
    

    # df_full = pd.concat([casia_full, imd_full, cmfd_full, nist_full, coverage_full,\
    #                 nist_extend, coverage_extend, defacto_cp, \
    #                 defacto_inpaint, defacto_s1, defacto_s2, defacto_s3])
    df_full = casia_full
    df_full.insert(0, 'image', '')
    
    # casia128 = get_dataframe('dataset_csv/casia_128.csv', folds=-1)
    # casia128_real = casia128[casia128['label'] == 0]
    
    # df = pd.concat([df_full, casia128_real])
    df = df_full
    
    
    #---------------------------------- 128 ---------------------------------------#

    # casia128 = get_dataframe('dataset_csv/casia_128.csv', folds=41)
    # imd128 = get_dataframe('dataset_csv/imd_128.csv', folds=41)
    # cmfd128 = get_dataframe('dataset_csv/cmfd_128.csv', folds=-1)
    # coverage128 = get_dataframe('dataset_csv/coverage_128.csv', folds=12)
    # nist128 = get_dataframe('dataset_csv/nist16_128.csv', folds=15)

    # df_128 = pd.concat([casia128, imd128, coverage128, cmfd128, nist128, coverage128])
    # df = df_128
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.label.value_counts())
        print('------')
        print(df.groupby('fold').root_dir.value_counts())

    
    train(
        name=f"(CASIA_FULL+customloss)" + config_defaults["model"],
        df=df,
        VAL_FOLD=0,
        resume=None,
    )
