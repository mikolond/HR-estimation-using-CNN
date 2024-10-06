import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self,path):
        self.n = n
        self.f = f
        self.fs = fs
        self.length = length

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        c = 2 * np.pi * self.f / self.fs
        amplitude = 50
        x = np.array([[[100 + np.sin(i * c) * amplitude, 100 + np.sin(i * c) * amplitude, 0]] * 640] * 480, dtype=np.uint8)
        x = torch.tensor(x, dtype=torch.float32)
        return x