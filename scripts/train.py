from model.model import Net
from data import CustomDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass


data = CustomDataset()
print("Data loaded")

@dataclass
class config:
    n_dim: int = 208
    n_heads: int = 8
    block_size: int = 16

dataloader = DataLoader(data, batch_size=32, shuffle=False)
cfg = config()

model = Net(cfg)
    
for (x, y) in (dataloader):
    
    model(x)
    break