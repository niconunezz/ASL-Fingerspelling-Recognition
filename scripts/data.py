import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np


class CustomDataset(Dataset):
    def __init__(self):
        self.files = {k:v for k, v in enumerate(os.listdir("data/extracted"))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f = np.load(f"data/extracted/{self.files[idx]}")
        return f['arr_0'], f['arr_1']