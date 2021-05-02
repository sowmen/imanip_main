import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

import wandb

from utils import *
from classifier_dataset import Classifier_Dataset
from classifier import Classifier_Linear

OUTPUT_DIR = "weights"
device =  'cuda'
config_defaults = {
    "epochs": 50,
    "train_batch_size": 512,
    "valid_batch_size": 256,
    "optimizer": "adam",
    "learning_rate": 0.005,
    "weight_decay": 0.0005,
    "schedule_patience": 3,
    "schedule_factor": 0.15,
    "model": "CLASSIFIER-LINEAR",
}

TEST_FOLD = 1


def train(name, df, VAL_FOLD=0, resume=False):
    now = datetime.now()
    dt_string = now.strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init(project="imanip", config=config_defaults, name=f"{name},{dt_string}")
    config = wandb.config


    model = Classifier_Linear()
    print("Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # wandb.watch(model)
    wandb.save('classifier.py')
    wandb.save('classifier_dataset.py')

    # -------------------------------- CREATE DATASET and DATALOADER --------------------------
    train_dataset = Classifier_Dataset(
        dataframe=df,
        mode="train",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = Classifier_Dataset(
        dataframe=df,
        mode="val",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = Classifier_Dataset(
        dataframe=df,
        mode="test",
        val_fold=VAL_FOLD,
        test_fold=TEST_FOLD,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=8, pin_memory=True)


    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        mode="min",
        factor=config.schedule_factor,
    )

    model = nn.DataParallel(model).to(device)

    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=20, mode="min")

    start_epoch = 0
    # if resume:
    #     checkpoint = torch.load('checkpoint/')
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print("-----------> Resuming <------------")

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion, epoch)
        scheduler.step(valid_metrics["valid_loss"])

        print(
            f"TRAIN_ACC = {train_metrics['train_acc_05']}, TRAIN_LOSS = {train_metrics['train_loss']}"
        )
        print(
            f"VALID_ACC = {valid_metrics['valid_acc_05']}, VALID_LOSS = {valid_metrics['valid_loss']}"
        )
        print("New LR", optimizer.param_groups[0]['lr'])
        es(
            valid_metrics["valid_loss"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

        # checkpoint = {
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict' : optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        # }
        # torch.save(checkpoint, os.path.join('checkpoint', f"{name}_[{dt_string}].pt"))

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5")))

    test_metrics = test(model, test_loader, criterion)
    wandb.save(os.path.join(OUTPUT_DIR, f"{name}_[{dt_string}].h5"))

    return test_metrics


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    
    predictions = []
    targets = []

    for batch in tqdm(train_loader):
        tensors = batch["tensor"].to(device)
        target_labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        out_labels = model(tensors)

        loss = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
        loss.backward()

        optimizer.step()

        #---------------------Batch Loss Update-------------------------)
        total_loss.update(loss.item(), train_loader.batch_size)
        
        total_loss.update(loss.item(), train_loader.batch_size)
        
        targets.append((target_labels.view(-1, 1).cpu() >= 0.5) * 1.0)
        predictions.append(torch.sigmoid(out_labels).cpu().detach().numpy())

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

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            tensors = batch["tensor"].to(device)
            target_labels = batch["label"].to(device)

            out_labels = model(tensors)

            loss = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)
            
            batch_targets = (target_labels.view(-1, 1).cpu() >= 0.5) * 1.0
            batch_preds = torch.sigmoid(out_labels).cpu()          
            
            targets.append(batch_targets)
            predictions.append(batch_preds)

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
        "epoch" : epoch
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
            tensors = batch["tensor"].to(device)
            target_labels = batch["label"].to(device)

            out_labels = model(tensors)

            loss = criterion(out_labels, target_labels.view(-1, 1).type_as(out_labels))
            
            #---------------------Batch Loss Update-------------------------
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
        "test_auc": test_auc,
        "test_f1_05": test_f1_05,
        "test_acc_05": test_acc_05,
        "test_balanced_acc_05": test_balanced_acc_05,
    }
    wandb.log(test_metrics)
    return test_metrics


if __name__ == "__main__":

    df = pd.read_csv('all_features.csv').sample(frac=1.0, random_state=123).reset_index(drop=True)
    acc = AverageMeter()
    f1 = AverageMeter()
    loss = AverageMeter()
    auc = AverageMeter()
    for i in [0]:
        print(f'>>>>>>>>>>>>>> CV {i} <<<<<<<<<<<<<<<')
        test_metrics = train(
            name=f"Classifier" + config_defaults["model"],
            df=df,
            VAL_FOLD=i,
            resume=False
        )
        acc.update(test_metrics['test_acc_05'])
        f1.update(test_metrics['test_f1_05'])
        loss.update(test_metrics['test_loss'])
        auc.update(test_metrics['test_auc'])
    
    print(f'ACCURACY : {acc.avg}')
    print(f'F1 : {f1.avg}')
    print(f'LOSS : {loss.avg}')
    print(f'AUC : {auc.avg}')

