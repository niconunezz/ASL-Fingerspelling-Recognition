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
    block_size: int = 130
    encoder_layers: int = 1
    vocab_size: int = 502
    n_layer: int = 1
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(data, batch_size=32, shuffle=False)
cfg = config()
print(f"Device: {cfg.device}")


model = Net(cfg)
model.to(cfg.device)

print(f"Model has {sum(p.numel() for p in model.parameters())/1e6}M parameters")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Data loader examples: {len(dataloader)}")

def validate(model):
    data = CustomDataset(validation=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    model.eval()
    for x, y in dataloader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits, loss = model(x, y)
        print(f"Validation loss: {loss.item()}")
        break



for epoch in range(1):
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, loss: {loss.item()}")
    

validate(model)