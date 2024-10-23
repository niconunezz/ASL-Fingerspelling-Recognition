import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self) -> None:
        #! must take out the [:5] for the final version
        self.files = os.listdir("data/extracted")[1:5]
        self.data = []
        self.labels = []

        for file in self.files:
            f = np.load(f"data/extracted/{file}")
            x = torch.from_numpy(f['arr_0'])
            x = self.fill_nans(x)
            
            try:
                y = torch.tensor(f['arr_1'])
            except TypeError:
                continue
            self.data.extend(torch.unbind(x, dim=0))
            self.labels.extend(torch.unbind(y,dim=0))

    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.labels[idx]
