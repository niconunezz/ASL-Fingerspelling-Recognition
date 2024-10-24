import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, validation = False) -> None:
        path = "data/extracted"
        self.files = os.listdir(path)
        if validation:
            path = "data/validation"
            self.files = os.listdir(path)
        print(f"Loading {len(self.files)} files")
        self.data = []
        self.labels = []

        counter = 0
        for file in self.files:
            f = np.load(f"{path}/{file}")
            x = torch.from_numpy(f['arr_0'])
            x = self.fill_nans(x)
            
            try:
                y = torch.tensor(f['arr_1'])
            except TypeError:
                counter += 1
                continue
            self.data.extend(torch.unbind(x, dim=0))
            self.labels.extend(torch.unbind(y,dim=0))
        
        print(f"Skipped {counter} files")

    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.labels[idx]
