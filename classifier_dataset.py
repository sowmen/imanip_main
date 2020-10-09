import os
import random
import numpy as np
import pandas as pd
import cv2
import math

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import albumentations
from albumentations import augmentations

from segmentation.timm_efficientnet import EfficientNet
from image_ensemble import ensemble

class Classifier_Dataset(Dataset):
    def __init__(self, dataframe, mode, val_fold, test_fold, root_dir, 
                 label_smoothing=0.1, equal_sample=False
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.root_dir = root_dir
        self.label_smoothing = label_smoothing
        self.equal_sample = equal_sample


        if self.mode == "train":
            rows = self.dataframe[~self.dataframe["fold"].isin([self.val_fold, self.test_fold])]
        elif self.mode == "val":
            rows = self.dataframe[self.dataframe["fold"] == self.val_fold]
        else:
            rows = self.dataframe[self.dataframe["fold"] == self.test_fold]

        if self.equal_sample:
            rows = self._equalize(rows)

        print(
            "real:{}, fakes:{}, mode = {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        self.data = rows.values
        np.random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_patch, mask_patch, label, fold, tensor_name = self.data[index]
 
        if self.label_smoothing:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        tensor = torch.load(os.path.join(self.root_dir, tensor_name))
        B,C,H,W = tensor.shape
        tensor = tensor.view(-1,H,W)
        # tensor = torch.nn.functional.adaptive_avg_pool2d(tensor, 1).squeeze()
        
        return {"image": tensor, "label": label}

    def _equalize(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Equalizes count of fake and real samples
        """
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_fake = fakes["image"].count()
        num_real = real["image"].count()
        if self.mode == "train":
            if int(num_fake * 1.5) <= num_real:
                real = real.sample(n=int(num_fake * 1.5), replace=False)
            else:
                real = real.sample(n=num_fake, replace=False)
        return pd.concat([real, fakes])
