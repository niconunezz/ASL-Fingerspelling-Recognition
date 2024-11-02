import os
import json
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
        
        


def padd_or_interpolate(x, max_len, pad_token: int = 61):
        if x.shape[0] > max_len:
            matrix = F.interpolate(x.permute(1,2,0), max_len).permute(2,0,1)
            mask = torch.ones_like(matrix[:, 0, 0])
            assert matrix.shape[0] == max_len
            return matrix, mask
        
        else:
            diff = max_len - x.shape[0]
            pad = torch.full((diff, x.shape[1], x.shape[2]), pad_token)
            matrix = torch.cat((x, pad), dim=0)
            assert matrix.shape[0] == max_len

            mask = torch.ones_like(x[:, 0, 0])
            
            mask = torch.cat([mask, pad[:, 0, 0]* 0])
            return matrix, mask

            


def padd_sequence(x, max_len, pad_token: int = 61):
    return np.pad(x, ((0, max_len - x.shape[0])), mode='constant', constant_values = pad_token)
            




class CustomDataset(Dataset):
    def __init__(self, cfg , mode = "train") -> None:
        
        
        self.config = cfg
        self.path = "data/tensors"
        self.tokenizer = self.setup_tokenizer()

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
        file_id, sequence_id, phrase = row[['file_id', 'sequence_id', 'phrase']]
        data = self.load_seq(file_id, sequence_id)
        x = data['arr_0']
        x = torch.from_numpy(x)
        x = self.processor(x)
        
        y = np.array([self.tokenizer[char] for char in list(phrase)])
        
        if self.mode == "train" or self.mode == "val":
            x, mask = padd_or_interpolate(x, self.config.block_size)
            y = padd_sequence(y, self.config.max_seq_len)


        return {"data": x, "mask": mask, "target": y}


    def load_seq(self, file, seq):
        path = f"{self.path}/{file}/{seq}.npy.npz"
        return np.load(path)


    def setup_tokenizer(self):
        self.tokenizer = json.load(open("data/supplemental_landmarks/character_to_prediction_index.json"))
        self.tokenizer["<sos>"] = 59
        self.tokenizer["<eos>"] = 60
        self.tokenizer["<pad>"] = 61
        return self.tokenizer 
