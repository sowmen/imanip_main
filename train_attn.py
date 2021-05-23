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
from warmup_scheduler import GradualWarmupScheduler

torch.backends.cudnn.benchmark = True

from utils import *
from dataset import DATASET
from segmentation.merged_net import SRM_Classifer
from segmentation.smp_srm import SMP_SRM_UPP


OUTPUT_DIR = "weights"
device =  'cuda'
config_defaults = {
    "epochs": 200,
    "train_batch_size": 40,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "schedule_patience": 5,
    "schedule_factor": 0.25,
    "warmup" : 3,
    "model": "",
}

TEST_FOLD = 1

def train(name, df, VAL_FOLD=0, resume=False):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    wandb.init(project="imanip", config=config_defaults, name=run)
    config = wandb.config


    # model = SRM_Classifer(num_classes=1, encoder_checkpoint='weights/pretrain_[31|03_12|16|32].h5')
    model = SMP_SRM_UPP(classifier_only=True)

    # for name_, param in model.named_parameters():
    #     if 'classifier' in name_:
    #         continue
    #     else:
    #         param.requires_grad = False

    print("Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))    
    

    wandb.save('segmentation/smp_srm.py')
    wandb.save('dataset.py')


    train_imgaug, train_geo_aug = get_train_transforms()
    transforms_normalize = get_transforms_normalize()
    

    #region ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = DATASET(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
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
        transforms_normalize=transforms_normalize,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    test_dataset = DATASET(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
        transforms_normalize=transforms_normalize,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    #endregion ######################################################################################



    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    # after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     patience=config.schedule_patience,
    #     mode="min",
    #     factor=config.schedule_factor,
    # )
    after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_0=35, T_mult=2)
    scheduler = GradualWarmupScheduler(optimizer       = optimizer, 
                                       multiplier      = 1, 
                                       total_epoch     = config.warmup + 1, 
                                       after_scheduler = after_scheduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    # optimizer.zero_grad()
    # optimizer.step()

    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=200, mode="min")


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

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion, epoch)
        
        scheduler.step(valid_metrics['valid_loss'])

        print(f"TRAIN_ACC = {train_metrics['train_acc_05']}, TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_ACC = {valid_metrics['valid_acc_05']}, VALID_LOSS = {valid_metrics['valid_loss']}")
        # print("Optimizer LR", optimizer.param_groups[0]['lr'])
        print("Scheduler LR", scheduler.get_lr()[0])
        wandb.log({
            'schedule_lr' : optimizer.param_groups[0]['lr']
        })

        
        es(
            valid_metrics["valid_loss"],
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


def train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    
    predictions = []
    targets = []

    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader),  desc=f"Train epoch {epoch}", dynamic_ncols=True):
        images = batch["image"].to(device)
        elas = batch["ela"].to(device)
        target_labels = batch["label"].to(device)
        # dft_dwt_vector = batch["dft_dwt_vector"].to(device)
        
        # scheduler.step(epoch + 1 + batch_idx / len(train_loader)) #-> For CosineAnnealingWarmRestarts

        out_logits = model(images, elas)#, dft_dwt_vector)
        loss = criterion(out_logits, target_labels.view(-1, 1).type_as(out_logits))

        loss.backward()
        optimizer.step()     
        optimizer.zero_grad()   

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
        for batch in tqdm(valid_loader, desc=f"Valid epoch {epoch}", dynamic_ncols=True):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            # dft_dwt_vector = batch["dft_dwt_vector"].to(device)

            out_logits = model(images, elas)#, dft_dwt_vector)

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
        for batch in tqdm(test_loader, dynamic_ncols=True):
            images = batch["image"].to(device)
            elas = batch["ela"].to(device)
            target_labels = batch["label"].to(device)
            # dft_dwt_vector = batch["dft_dwt_vector"].to(device)

            out_logits = model(images, elas)#, dft_dwt_vector)

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




if __name__ == "__main__":

    # combo_all_df = get_dataframe('combo_all_FULL.csv', folds=None)
    casia_full = get_dataframe('dataset_csv/casia_FULL.csv', folds=None)
    # imd_full = get_dataframe('dataset_csv/imd_FULL.csv', folds=None)
    # cmfd_full = get_dataframe('dataset_csv/cmfd_FULL.csv', folds=-1)
    # nist_full = get_dataframe('dataset_csv/nist16_FULL.csv', folds=None)
    # coverage_full = get_dataframe('dataset_csv/coverage_FULL.csv', folds=None)
    
    # nist_extend = get_dataframe('dataset_csv/nist_extend.csv', folds=12)
    # # nist_extend_real = nist_extend[nist_extend['label'] == 0].sample(n=1000, random_state=123)
    # # nist_extend_fake = nist_extend[nist_extend['label'] == 1].sample(n=1500, random_state=123)
    # # nist_extend = pd.concat([nist_extend_real, nist_extend_fake]).sample(frac=1.0, random_state=123)
    
    # coverage_extend = get_dataframe('dataset_csv/coverage_extend.csv', folds=12)
    # # coverage_extend_real = coverage_extend[coverage_extend['label'] == 0].sample(n=800, random_state=123)
    # # coverage_extend_fake = coverage_extend[coverage_extend['label'] == 1].sample(n=800, random_state=123)
    # # coverage_extend = pd.concat([coverage_extend_real, coverage_extend_fake]).sample(frac=1.0, random_state=123)
            
    # df_full = pd.concat([casia_full, imd_full, cmfd_full, nist_full, coverage_full,\
    #                      nist_extend, coverage_extend])
    df_full = casia_full
    df_full.insert(0, 'image', '')

    # df_128 = pd.read_csv('combo_all_128.csv').sample(frac=1.0, random_state=123)
    # df = pd.concat([df_full, df_128])

    df = df_full
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.label.value_counts())
        print('------')
        print(df.groupby('fold').root_dir.value_counts())

    acc = AverageMeter()
    f1 = AverageMeter()
    loss = AverageMeter()
    auc = AverageMeter()
    for i in range(1):
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"(RL Plateau scheduler)CASIA_FULL" + config_defaults["model"],
            df=df,
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