import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math

from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import albumentations
from albumentations import augmentations


class CASIA(Dataset):
    def __init__(
        self,
        dataframe,
        mode,
        val_fold,
        test_fold,
        root_dir,
        patch_size,
        transforms=None,
        label_smoothing=None,
        equal_sample=False,
        normalize={
            "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
            "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
        },
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
        self.normalize = normalize

        self.mask_transforms = albumentations.Compose(
            [
                augmentations.transforms.Resize(
                    28, 28, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1
                )
            ]
        )

        if self.mode == "train":
            rows = self.dataframe[
                ~self.dataframe["fold"].isin([self.val_fold, self.test_fold])
            ]
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

        if self.patch_size == 256:
            image_patch, mask_patch, label, fold = self.data[index]
        else:
            image_name, image_patch, mask_patch, label, fold = self.data[index]

        if self.label_smoothing:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        if self.patch_size == 256:
            image_path = os.path.join(self.root_dir, image_patch)
        else:
            image_path = os.path.join(self.root_dir, image_name, image_patch)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not isinstance(mask_patch, str) and np.isnan(mask_patch):
            mask_image = np.zeros((image.shape[0], image.shape[1]))
        else:
            if self.patch_size == 256:
                mask_path = os.path.join(self.root_dir, mask_patch)
            else:
                mask_path = os.path.join(self.root_dir, image_name, mask_patch)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            data = self.transforms(image=image, mask=mask_image)
            image = data["image"]
            mask_image = data["mask"]
        mask_image = self.mask_transforms(image=mask_image)["image"]

        image = img_to_tensor(image, self.normalize)
        mask_image = img_to_tensor(mask_image).unsqueeze(0)

        return {"image": image, "label": label, "mask": mask_image}

    def _equalize(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Equalizes count of fake and real samples
        """
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_fake = fakes["image"].count()
        num_real = real["image"].count()
        if self.mode == "train":
            if int(num_fake * 1.5) >= num_real:
                real = real.sample(n=int(num_fake * 1.5), replace=False)
            else:
                real = real.sample(n=num_fake, replace=False)
        return pd.concat([real, fakes])
