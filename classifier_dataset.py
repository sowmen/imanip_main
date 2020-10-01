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
    def __init__(self, dataframe, mode, val_fold, test_fold, root_dir, patch_size, 
                 transforms=None, label_smoothing=0.1, equal_sample=False, attention=False
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.label_smoothing = label_smoothing
        self.equal_sample = equal_sample
        self.normalize = {
            "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
            "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
        }


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
        
        # self.encoder = EfficientNet(freeze_encoder=True).get_encoder().to('cuda')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_patch, mask_patch, label, fold, tensor = self.data[index]
 
        if self.label_smoothing:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        # image_path = os.path.join(self.root_dir, image_patch)

        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # if image is None:
        #     print(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if self.transforms:
        #     data = self.transforms(image=image)
        #     image = data["image"]

        # tensor = ensemble(self.encoder, image)
        tensor = torch.load(os.path.join(self.root_dir, 'full_tensors', tensor))
        tensor = torch.nn.functional.adaptive_avg_pool2d(tensor, 1).squeeze()
        
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
