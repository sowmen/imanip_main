import os
import random
import numpy as np
import pandas as pd
import cv2
import math
import copy
import gc

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import albumentations
from albumentations import augmentations
from albumentations.augmentations import functional


class DATASET(Dataset):
    def __init__(self, dataframe, mode, val_fold, test_fold, patch_size, imgaug_augment=None,
                 transforms=None, equal_sample=False, segment=False
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.patch_size = patch_size
        self.imgaug_augment = imgaug_augment
        self.transforms = transforms
        self.label_smoothing = 0.1
        self.equal_sample = equal_sample
        self.segment = segment # Returns only fake rows for segmentation
        self.root_folder = "Image_Manipulation_Dataset"

        # self.attn_mask_transforms = albumentations.Compose([
        #     augmentations.transforms.Resize(
        #         32, 32, interpolation=cv2.INTER_LANCZOS4, always_apply=True, p=1
        #     ),
        #     albumentations.Normalize(mean=self.normalize['mean'], std=self.normalize['std'], p=1, always_apply=True),
        #     albumentations.pytorch.ToTensor()
        # ])

        if self.mode == "train":
            rows = self.dataframe[~self.dataframe["fold"].isin([self.val_fold, self.test_fold])]
        elif self.mode == "val":
            rows = self.dataframe[self.dataframe["fold"] == self.val_fold]
        else:
            rows = self.dataframe[self.dataframe["fold"] == self.test_fold]

        #---- For checking. Get all rows -------#
        # rows = self.dataframe 

        if self.equal_sample:
            rows = self._equalize(rows)
        if self.segment:
            rows = self._segment(rows)

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

        if self.patch_size == 'FULL':
            image_patch, mask_patch, label, _, ela, root_dir = self.data[index]
        else:
            image_name, image_patch, mask_patch, label, _, ela, root_dir = self.data[index]

        if self.label_smoothing:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        if self.patch_size == 'FULL':
            image_path = os.path.join(self.root_folder, root_dir, image_patch)
            ela_path = os.path.join(self.root_folder, root_dir, ela)
        else:
            image_path = os.path.join(self.root_folder, root_dir, image_name, image_patch)
            ela_path = os.path.join(self.root_folder, root_dir, image_name, ela)

        if(not os.path.exists(ela_path)):
            print(f"ELA Not Found : {ela_path}")
        if(not os.path.exists(image_path)):
            print(f"Image Not Found : {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ela_image = cv2.imread(ela_path, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ela_image = cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB)

        if not isinstance(mask_patch, str) and np.isnan(mask_patch):
            mask_image = np.zeros((image.shape[0], image.shape[1])).astype('uint8')
        else:
            if self.patch_size == 'FULL':
                mask_path = os.path.join(self.root_folder, root_dir, mask_patch)
            else:
                mask_path = os.path.join(self.root_folder, root_dir, image_name, mask_patch)

            if(not os.path.exists(mask_path)):
                print(f"Mask Not Found : {mask_path}")
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # ----- For NIST16 Invert Mask Here ----- #

            ##########################################
            
        # attn_mask_image = copy.deepcopy(mask_image)

        if self.imgaug_augment:
            try :
                # temp_image = copy.deepcopy(image)
                image = self.imgaug_augment.augment_image(image)
            except Exception as e:
                print(image_path, e) 
                # image = temp_image
                # del(temp_image)
                # gc.collect()
        
        # print("--- bfr ---")
        # print(image.shape, image.dtype)
        # print(ela_image.shape, ela_image.dtype)
        # print(mask_image.shape, mask_image.dtype)
        # print("-------")
        if self.transforms:
            data = self.transforms(image=image, mask=mask_image, ela=ela_image)
            image = data["image"]
            mask_image = data["mask"]
            ela_image = data["ela"]#.permute(2,0,1)
        # attn_mask_image = self.attn_mask_transforms(image=attn_mask_image)["image"]

        # print("--- afr ---")
        # print(image.shape, image.type())
        # print(ela_image.shape, ela_image.type())
        # print(mask_image.shape, mask_image.type())
        # print("-------")

        # image = img_to_tensor(image, self.normalize)
        # mask_image = img_to_tensor(mask_image).unsqueeze(0)
        # attn_mask_image = img_to_tensor(attn_mask_image).unsqueeze(0)
        # print("LOADED DATA")
        return {
            "image": image,
            "image_path" : image_path, 
            "label": label, 
            "mask": mask_image,
            "ela" : ela_image 
            # "attn_mask": attn_mask_image
        }

    def _equalize(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Equalizes count of fake and real samples
        """
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]

        num_fake = fakes["label"].count()
        num_real = real["label"].count()

        if num_fake < num_real:
            real = real.sample(n=num_fake, replace=False)
        else:
            fakes = fakes.sample(n=num_real, replace=False)
        return pd.concat([real, fakes])

    def _segment(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Returns only fake rows for segmentation
        """
        return rows[rows["label"] == 1]
