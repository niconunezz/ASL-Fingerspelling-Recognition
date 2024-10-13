import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self) -> None:
        #! must take out the [:5] for the final version
        self.files = os.listdir("data/extracted")[:5]
        self.data = []
        self.labels = []

        for file in self.files:
            f = np.load(f"data/extracted/{file}")
            x = torch.from_numpy(f['arr_0'])
            print(x.shape)
            print(f['arr_1'])
            y = torch.from_numpy(f['arr_1'])
            self.data.extend(torch.unbind(x, dim=0))
            self.labels.extend(torch.unbind(y,dim=0))

       
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.labels[idx]
