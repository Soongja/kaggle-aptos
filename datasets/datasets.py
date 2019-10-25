import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        frame = pd.read_csv(self.config.CSV)
        self.frame = frame.loc[frame['split'] == self.split].reset_index(drop=True)
        if self.config.DEBUG:
            self.frame = self.frame[:100]
        print(self.split, 'set:', self.frame.shape[0])

        self.labels = self.frame['diagnosis'].values

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        ext = '.jpeg' if '_' in self.frame["id_code"][idx] else '.png'
        image = cv2.imread(os.path.join(self.config.DATA_DIR, self.frame["id_code"][idx] + ext), 1)
        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        label = torch.tensor([label]).float()
        # label = torch.zeros([1], dtype=torch.int32)
        # label = label.fill_(label_val)
        # label = label.float()

        return image, label
