import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class Classifier_Dataset(Dataset):
    def __init__(self, dataframe, mode, val_fold, test_fold, 
                 label_smoothing=0.1, equal_sample=False
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
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

        self.data = []

        for row in tqdm(rows.values):
            _, label, _, _, _, feature = row

            feature_array = torch.load(feature)
            self.data.append((feature_array, label))

        np.random.shuffle(self.data)

        print(
            "\n\nreal:{}, fakes:{}, mode = {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index: int):
        feat_tensor, label = self.data[index]
 
        if self.label_smoothing:
            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        feat_tensor = feat_tensor.view(-1)
        
        return {"tensor": feat_tensor, "label": label}


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