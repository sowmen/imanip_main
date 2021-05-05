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
    "epochs": 60,
    "train_batch_size": 12,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0005,
    "schedule_patience": 4,
    "schedule_factor": 0.25,
    'sampling':'nearest',
    "model": "UnetPP",
}
TEST_FOLD = 1


def train(name, df, VAL_FOLD=0, resume=False):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    print("Starting -->", run)
    
    wandb.init(project="imanip", config=config_defaults, name=run)
    config = wandb.config


    # model = smp.DeepLabV3('resnet34', classes=1, encoder_weights='imagenet')
    
    encoder = SRM_Classifer(encoder_checkpoint='weights/pretrain_[31|03_12|16|32].h5', freeze_encoder=True)
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
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    valid_dataset = DATASET(
        dataframe=df,
        mode="val",
        segment=False,
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        transforms_normalize=transforms_normalize,
        equal_sample=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        segment=False,
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        transforms_normalize=transforms_normalize,
        equal_sample=True
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

        if epoch == 6:
            model.module.encoder.unfreeze()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion,  epoch)
        
        scheduler.step(valid_metrics["valid_loss_segmentation"])

        print(
            f"TRAIN_LOSS = {train_metrics['train_loss_segmentation']}, \
            TRAIN_DICE = {train_metrics['train_dice']}, \
            TRAIN_JACCARD = {train_metrics['train_jaccard']}"
        )
        print(
            f"VALID_LOSS = {valid_metrics['valid_loss_segmentation']}, \
            VALID_DICE = {valid_metrics['valid_dice']}, \
            VALID_JACCARD = {valid_metrics['valid_jaccard']}"
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
    # pixel_auc = AverageMeter()

    scores = seg_metrics.SegMeter()

    for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        gt = batch["mask"].to(device)
        target_labels = batch["label"].to(device)

        optimizer.zero_grad()
        pred_mask, label_tensor = model(images, elas)
        # pred_mask = model(images)

        loss_segmentation = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))
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
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, _, _ = seg_metrics.get_avg_batch_dice(pred_mask, gt)
            dice.update(batch_dice, train_loader.batch_size)

            batch_jaccard, _, _ = seg_metrics.get_avg_batch_jaccard(pred_mask, gt)
            jaccard.update(batch_jaccard, train_loader.batch_size)

            
            # batch_pixel_auc, num = seg_metrics.batch_pixel_auc(pred_mask, gt, batch["image_path"])
            # pixel_auc.update(batch_pixel_auc, num)
            
            scores.update(pred_mask, gt)
            

    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TRAIN", scores)
    train_metrics = {
        "train_loss_segmentation": segmentation_loss.avg,
        "train_dice": dice.avg,
        "train_jaccard": jaccard.avg,
        "epoch" : epoch,
        "train_dice2" : dice2,
        "train_dice_pos" : dice_pos,
        "train_dice_neg" : dice_neg,
        "train_iou2" : iou2,
        # "train_pixel_auc" : pixel_auc.avg
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    segmentation_loss = AverageMeter()
    dice = AverageMeter()
    jaccard = AverageMeter()
    # pixel_auc = AverageMeter()
    scores = seg_metrics.SegMeter()

    example_images = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Valid epoch {epoch}", dynamic_ncols=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)
            target_labels = batch["label"].to(device)
                    
            
            pred_mask, label_tensor = model(images, elas)
            # pred_mask = model(images)

            loss_segmentation = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), valid_loader.batch_size)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, (mx_dice, best_dice_idx), (worst_dice, worst_dice_idx) = seg_metrics.get_avg_batch_dice(pred_mask, gt)
            dice.update(batch_dice, valid_loader.batch_size)

            batch_jaccard, (mx_iou, best_iou__idx), (worst_iou, worst_idx) = seg_metrics.get_avg_batch_jaccard(pred_mask, gt)
            jaccard.update(batch_jaccard, valid_loader.batch_size)

            if(np.random.rand() < 0.5 and len(example_images) < 20):
                images = images.cpu().detach()
                example_images.append({
                    "best_image" : images[best_dice_idx],
                    "best_image_name" : batch["image_path"][best_dice_idx],
                    "best_dice" : mx_dice,
                    "best_pred" : pred_mask[best_dice_idx],
                    "best_gt" : gt[best_dice_idx],
                    "worst_image" : images[worst_dice_idx],
                    "worst_image_name" : batch["image_path"][worst_dice_idx],
                    "worst_dice" : worst_dice,
                    "worst_pred" : pred_mask[worst_dice_idx],
                    "worst_gt" : gt[worst_dice_idx],
                })
            
            scores.update(pred_mask, gt)

            # batch_pixel_auc, num = seg_metrics.batch_pixel_auc(pred_mask, gt, batch["image_path"])
            # pixel_auc.update(batch_pixel_auc, valid_loader.batch_size)

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
        "valid_iou2" : iou2,
        # "valid_pixel_auc" : pixel_auc.avg
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model, test_loader, criterion):
    model.eval()

    segmentation_loss = AverageMeter()
    dice = AverageMeter()
    jaccard = AverageMeter()
    # pixel_auc = AverageMeter()
    scores = seg_metrics.SegMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader, dynamic_ncols=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            gt = batch["mask"].to(device)
            target_labels = batch["label"].to(device)
                    
            
            pred_mask, label_tensor = model(images, elas)
            # pred_mask = model(images)

            loss_segmentation = criterion(pred_mask, gt, label_tensor, target_labels.view(-1, 1))

            # ---------------------Batch Loss Update-------------------------
            segmentation_loss.update(loss_segmentation.item(), test_loader.batch_size)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach()
            gt = gt.cpu().detach()
            
            batch_dice, _, _ = seg_metrics.get_avg_batch_dice(pred_mask, gt)
            dice.update(batch_dice, test_loader.batch_size)

            batch_jaccard, _, _ = seg_metrics.get_avg_batch_jaccard(pred_mask, gt)
            jaccard.update(batch_jaccard, test_loader.batch_size)

            scores.update(pred_mask, gt)

            # batch_pixel_auc, num = seg_metrics.batch_pixel_auc(pred_mask, gt, batch["image_path"])
            # pixel_auc.update(batch_pixel_auc, test_loader.batch_size)

            gc.collect()

    dice2, dice_neg, dice_pos, iou2 = seg_metrics.epoch_score_log("TEST", scores)
    test_metrics = {
        "test_loss_segmentation": segmentation_loss.avg,
        "test_dice": dice.avg,
        "test_jaccard": jaccard.avg,
        "test_dice2" : dice2,
        "test_dice_pos" : dice_pos,
        "test_dice_neg" : dice_neg,
        "test_iou2" : iou2,
        # "test_pixel_auc" : pixel_auc.avg
    }
    wandb.log(test_metrics)
    return test_metrics



from losses import DiceLoss, ImanipLoss
def get_lossfn():
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss(mode='binary', log_loss=True, smooth=1e-7)
    criterion = ImanipLoss(bce, dice)
    return criterion

    

if __name__ == "__main__":

    #---------------------------------- FULL --------------------------------------#
    combo_all_df = get_dataframe('combo_all_FULL.csv', folds=None)
    casia_full = get_dataframe('dataset_csv/casia_FULL.csv', folds=None)
    imd_full = get_dataframe('dataset_csv/imd_FULL.csv', folds=None)
    cmfd_full = get_dataframe('dataset_csv/cmfd_FULL.csv', folds=-1)
    nist_full = get_dataframe('dataset_csv/nist16_FULL.csv', folds=None)
    coverage_full = get_dataframe('dataset_csv/coverage_FULL.csv', folds=None)
    
    nist_extend = get_dataframe('nist_extend.csv', folds=12)
    coverage_extend = get_dataframe('coverage_extend.csv', folds=12)
    defacto_cp = get_dataframe('dataset_csv/defacto_copy_move.csv', folds=-1)
    defacto_inpaint = get_dataframe('dataset_csv/defacto_inpainting.csv', folds=-1)
    defacto_s1 = get_dataframe('dataset_csv/defacto_splicing1.csv', folds=-1)
    defacto_s2 = get_dataframe('dataset_csv/defacto_splicing2.csv', folds=-1)
    defacto_s3 = get_dataframe('dataset_csv/defacto_splicing3.csv', folds=-1)
    

    df_full = pd.concat([casia_full, imd_full, cmfd_full, nist_full, coverage_full,\
                    nist_extend, coverage_extend, defacto_cp, \
                    defacto_inpaint, defacto_s1, defacto_s2, defacto_s3])
    df_full.insert(0, 'image', '')
    
    casia128 = get_dataframe('dataset_csv/casia_128.csv', folds=-1)
    casia128_real = casia128[casia128['label'] == 0]
    
    df = pd.concat([df_full, casia128_real])
    
    
    #---------------------------------- 128 ---------------------------------------#

    # casia128 = get_dataframe('dataset_csv/casia_128.csv', folds=41)
    # imd128 = get_dataframe('dataset_csv/imd_128.csv', folds=41)
    # cmfd128 = get_dataframe('dataset_csv/cmfd_128.csv', folds=-1)
    # coverage128 = get_dataframe('dataset_csv/coverage_128.csv', folds=12)
    # nist128 = get_dataframe('dataset_csv/nist16_128.csv', folds=15)

    # df_128 = pd.concat([casia128, imd128, cmfd128, nist128, coverage128])
    # df = df_128
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.label.value_counts())
        # print('------')
        # print(df_full.groupby('fold').root_dir.value_counts())

    dice = AverageMeter()
    jaccard = AverageMeter()
    loss = AverageMeter()
    for i in range(0,1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"(defacto+customloss)" + config_defaults["model"],
            df=df,
            VAL_FOLD=i,
            resume=False,
        )
        dice.update(test_metrics['test_dice'])
        jaccard.update(test_metrics['test_jaccard'])
        loss.update(test_metrics['test_loss_segmentation'])
    
    print(f'DICE : {dice.avg}')
    print(f'JACCARD : {jaccard.avg}')
    print(f'LOSS : {loss.avg}')
