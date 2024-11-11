import os
import json
import time
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
            mask = torch.ones((matrix.shape[0]))
            assert matrix.shape[0] == max_len
            return matrix, mask
        
        else:
            diff = max_len - x.shape[0]
            pad = torch.full((diff, x.shape[1], x.shape[2]), pad_token)
            matrix = torch.cat((x, pad), dim=0)
            assert matrix.shape[0] == max_len

            mask = torch.ones((x.shape[0]))
            
            mask = torch.cat([mask, pad[:, 0, 0]* 0])
            return matrix, mask

            


def padd_sequence(x, max_len, pad_token: int = 61):
    return np.pad(x, ((0, max_len - x.shape[0])), mode='constant', constant_values = pad_token)
            




class CustomDataset(Dataset):
    def __init__(self, cfg , folder ="1", mode = "train", aug = None, verbose: bool = False) -> None:
        
        self.verbose = verbose
        self.config = cfg
        self.path = f"data/tensors2/{folder}"
        self.tokenizer = self.setup_tokenizer()
        self.aug = aug

        self.df = df = pd.read_csv("data/train.csv").query(f"fold == {folder}").reset_index(drop = True)
        if cfg.max_ex:
            self.df = df = df.iloc[:cfg.max_ex]
        if mode == "train":
            self.df = df = df.iloc[:int(len(df)*0.9)]
        if mode == "val":
            self.df = df = df.iloc[int(len(df)*0.9):]
        self.processor = Preprocessing()
        self.mode = mode


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        
        t0 = time.time()
        row = self.df.iloc[idx]
        sequence_id, phrase = row[['sequence_id', 'phrase']]
        t1 = time.time()
        if self.verbose:
            print(f"Indexing took {(t1-t0)*1000:.2f} ms")
        
        t0 = time.time()
        x = self.load_seq(sequence_id)
        t1 = time.time()
        if self.verbose:
            print(f"Loading data took {(t1-t0)*1000:.2f} ms")

        t0 = time.time()
        
        x = torch.from_numpy(x)
        t1 = time.time()
        if self.verbose:
            print(f"Converting to tensor took {(t1-t0)*1000:.2f} ms")
        
        t0 = time.time()
        x = self.processor(x)
        t1 = time.time()
        if self.verbose:
            print(f"Preprocessing took {(t1-t0)*1000:.2f} ms")
        
        t0 = time.time()
        y = np.array([self.tokenizer[char] for char in list(phrase)])
        t1 = time.time()
        if self.verbose:
            print(f"Tokenizing took {(t1-t0)*1000:.2f} ms")

        
        if self.aug:
            self.augment(x.numpy())

        t0 = time.time()

        x, mask = padd_or_interpolate(x, self.config.block_size)
        y = padd_sequence(y, self.config.max_seq_len)
        t1 = time.time()
        if self.verbose:
            print(f"Padding took {(t1-t0)*1000:.2f} ms")


        return {"data": x, "mask": mask, "target": y}


    def load_seq(self, seq):
        path = f"{self.path}/{seq}.npy"
        return np.load(path)
    
    def augment(self, x):
        x_aug = self.aug(image = x)['image']
        return x_aug


    def setup_tokenizer(self):
        self.tokenizer = json.load(open("data/supplemental_landmarks/character_to_prediction_index.json"))
        self.tokenizer["<sos>"] = 59
        self.tokenizer["<eos>"] = 60
        self.tokenizer["<pad>"] = 61
        return self.tokenizer 
