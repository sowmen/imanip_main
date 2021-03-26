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
from albumentations import augmentations
from torchvision import transforms
from utils import get_ela


class DATASET(Dataset):
    def __init__(self, dataframe, mode, imgaug_augment=None,
                 transforms_normalize=None, geo_augment=None
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.imgaug_augment = imgaug_augment
        self.geo_augment = geo_augment
        self.transforms_normalize = transforms_normalize
        self.label_smoothing = 0.1
        self.root_folder = "Image_Manipulation_Dataset"

        
        if self.mode == "train":
            rows = self.dataframe[~self.dataframe["fold"].isin([self.val_fold, self.test_fold])]
        elif self.mode == "val":
            rows = self.dataframe[self.dataframe["fold"] == self.val_fold]
        else:
            rows = self.dataframe[self.dataframe["fold"] == self.test_fold]


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
            if('NIST' in root_dir):
                mask_image = 255 - mask_image
            ##########################################
            
        # attn_mask_image = copy.deepcopy(mask_image)

        if self.imgaug_augment:
            try :
                image = self.imgaug_augment.augment_image(image)
            except Exception as e:
                print(image_path, e) 
        
        if self.geo_augment:
            data = self.geo_augment(image=image, mask=mask_image, ela=ela_image)
            image = data["image"]
            mask_image = data["mask"]
            ela_image = data["ela"]
        

        image = augmentations.geometric.functional.resize(image, self.resize, self.resize, cv2.INTER_AREA)
        mask_image = augmentations.geometric.functional.resize(mask_image, self.resize, self.resize, cv2.INTER_AREA)
        ela_image = augmentations.geometric.functional.resize(ela_image, self.resize, self.resize, cv2.INTER_AREA)


        ###--- Generate DFT DWT Vector -----------------
        # dft_dwt_vector = generate_dft_dwt_vector(image)
        # dft_dwt_vector = torch.from_numpy(dft_dwt_vector).float()

        ##########------Normalize-----##########
        # image_normalize = {
        #     "mean": [0.4535408213875562, 0.42862278450748387, 0.41780105499276865],
        #     "std": [0.2672804038612597, 0.2550410416463668, 0.29475415579144293],
        # }
        # transNormalize = transforms.Normalize(mean=image_normalize['mean'], std=image_normalize['std'])
        # transTensor = transforms.ToTensor()

        # tensor_image = transTensor(image)
        # tensor_ela = transTensor(ela_image)
        # tensor_mask = transTensor(mask_image)

        # tensor_image = transNormalize(tensor_image)
        # tensor_ela = transforms.functional.normalize(tensor_ela, mean=[0.0640, 0.05255, 0.0766], std=[0.0871, 0.0722, 0.1013])
        ########################################

        if self.transforms_normalize:
            data = self.transforms_normalize(image=image, mask=mask_image, ela=ela_image)
            image = data["image"]
            mask_image = data["mask"] / 255.0
            ela_image = data["ela"]
        # attn_mask_image = self.attn_mask_transforms(image=attn_mask_image)["image"]


        return {
            "image": image,
            "image_path" : image_path, 
            "label": label, 
            "mask": mask_image,
            "ela" : ela_image ,
            # "dft_dwt_vector" : dft_dwt_vector
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
