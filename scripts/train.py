from model import Net
from data import CustomDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


data = CustomDataset()
print("Data loaded")
 
dataloader = DataLoader(data, batch_size=32, shuffle=False)

model = Net(32)

for (x, y) in (dataloader):
    print(x.shape)
    model(x)
    break