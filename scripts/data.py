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
        x = self.fill_nans(x)
        x = self.normalize(x)
        return x
    
    def normalize(self, x):
        nonan = x[~torch.isnan(x)]
        x = (x - nonan.mean(0))
        x = x/nonan.std(0, unbiased = False)
        return x
        
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
        


def padd_or_interpolate(x, max_len, vocab_size: int = 502):
        if x.shape[0] > max_len:
            matrix = F.interpolate(x.permute(1,2,0), max_len).permute(2,0,1)
            assert matrix.shape[0] == max_len
            mask = torch.ones_like(matrix[:, 0, 0])
            return matrix, mask
        
        else:
            diff = max_len - x.shape[0]
            pad = torch.full((diff, x.shape[1], x.shape[2]), vocab_size)
            matrix = torch.cat((x, pad), dim=0)
            assert matrix.shape[0] == max_len

            mask = torch.ones_like(x[:, 0, 0])
            
            mask = torch.cat([mask, pad[:, 0, 0]* 0])
            return matrix, mask

            


def padd_sequence(x, max_len, vocab_size: int = 502):
    return np.pad(x, ((0, max_len - x.shape[0])), mode='constant', constant_values = vocab_size)
            
        


class CustomDataset(Dataset):
    def __init__(self, cfg , mode = "train") -> None:
        
        
        self.config = cfg
        self.path = "data/tensors"

        self.df = df = pd.read_csv("data/train.csv")
        if cfg.max_ex:
            self.df = df = df.iloc[:cfg.max_ex]
        if mode == "train":
            self.df = df = df.iloc[:len(df)*4//5]
        if mode == "val":
            self.df = df = df.iloc[len(df)*4//5:]
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
        
        if self.mode == "train" or self.mode == "val":
            x, mask = padd_or_interpolate(x, self.config.block_size)
            y = padd_sequence(y, self.config.max_seq_len)


        return {"data": x, "mask": mask, "target": y}


    def load_seq(self, file, seq):
        path = f"{self.path}/{file}/{seq}.npy.npz"
        return np.load(path)
