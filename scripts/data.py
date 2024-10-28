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
        return x
    
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
            matrix = F.interpolate(x.permute(1,2,0), max_len).permute(2,0,1)
            assert matrix.shape[0] == max_len
            return matrix
        
        if x.shape[0] < max_len:
            pad = torch.full((max_len - x.shape[0], x.shape[1], x.shape[2]), vocab_size)
            matrix = torch.cat((x, pad), dim=0)
            assert matrix.shape[0] == max_len

            return matrix
            


def padd_sequence(x, max_len, vocab_size: int = 502):
    return np.pad(x, ((0, max_len - x.shape[0])), mode='constant', constant_values = vocab_size)
            
        


class CustomDataset(Dataset):
    def __init__(self, cfg ,mode = "train") -> None:
        
        self.config = cfg
        self.path = "data/tensors"
        self.df = pd.read_csv("data/train.csv")
        self.processor = Preprocessing()
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        
        row = self.df.iloc[idx]
        file_id, sequence_id, _ = row[['file_id', 'sequence_id', 'phrase']]
        data = self.load_seq(file_id, sequence_id)
        x, y = data['arr_0'], data['arr_1']
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        x = self.processor(x)
        
        if self.mode == "train":
            x = padd_or_interpolate(x, self.config.block_size)
            y = padd_sequence(y, self.config.max_seq_len)


        return x, y


    def load_seq(self, file, seq):
        path = f"{self.path}/{file}/{seq}.npy.npz"
        return np.load(path)
