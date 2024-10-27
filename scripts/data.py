import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def forward(self, x):
        x = self.normalize(x)
        x = self.fill_nans(x)
    
    def normalize(self, x):
        nonan = x[~torch.isnan(x)]
        x = (x - nonan.mean(0).reshape(1,1,-1))
        x = x/nonan.std(0, unbiased = False).reshape(1,1,-1)
        return x
        
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
        


def padd_or_interpolate(x, max_len, vocab_size: int = 502):
        if x.shape[0] > max_len:
            return F.interpolate(x, max_len)
        if x.shape[0] < max_len:
            return np.pad(x, ((0, max_len - x.shape[0])), mode='constant', constant_values = vocab_size)
            
        


class CustomDataset(Dataset):
    def __init__(self, validation = False) -> None:

        path = "data/extracted"
        self.df = pd.read_csv("data/train.csv")
        self.processor = Preprocessing()    

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        
        row = self.df.iloc[idx]
        file_id, sequence_id, _ = row[['file_id', 'sequence_id', 'sentence']]
        data = self.load_seq(file_id, sequence_id)
        x, y = data['arr_0'], data['arr_1']
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        x = self.processor(x)
        



        return x, y


    def load_seq(self, file, seq):
        path = f"data/extracted/{file}/{seq}.npz"
        return np.load(path)
