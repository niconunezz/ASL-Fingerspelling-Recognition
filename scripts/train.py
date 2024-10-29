import sys
import time
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
from model.model import Net
import torch.optim as optim
from data import CustomDataset
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class config:
    n_dim: int = 208
    n_heads: int = 2
    block_size: int = 128
    max_seq_len: int = 31
    encoder_layers: int = 2
    vocab_size: int = 502
    n_layer: int = 3
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_files: int = 3
    epochs: int = 2

cfg = config()
print(f"Device: {cfg.device}")

data = CustomDataset(cfg)
print("Data loaded")
print(f"Data has {len(data)} examples")

dataloader = DataLoader(data, batch_size=32, shuffle=False)
model = Net(cfg)
model.to(cfg.device)

print(f"Model has {sum(p.numel() for p in model.parameters())/1e6} M parameters")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Data loader examples: {len(dataloader)}")

def validate(model, config):
    data = CustomDataset(validation=True)
    dataloader = DataLoader(data, batch_size= config.val_files, shuffle=False)
    vocab = pickle.load(open("data/extractor.pkl", "rb"))
    model.eval()
    for el in dataloader:
        x, mask, y = el
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits, loss = model(x, mask, y)
        print(f"Validation loss: {loss.item()}")

        print(f"Sentence: {''.join([vocab[token.item()].decode('utf-8') for token in y[1]])}")

        print(f"Logits: {logits.shape}")
        print("".join([vocab[i.item()].decode('utf-8', errors = 'replace') for i in logits[0].argmax(dim=-1)]))
        logits = logits[0].argmax(dim=-1)

        break
        

for epoch in range(config.epochs):
    for i, batch in (enumerate(dataloader)):
        t0 = time.time()
        x, mask, y = batch['data'], batch['mask'], batch['target']
        x, y = x.to(cfg.device), y.to(cfg.device)
        mask = mask.to(cfg.device)

        logits, loss = model(x, mask, y)
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 1.0:
                print(f"Layer: {name}, grad norm: {param.grad.norm()}")
        sys.exit(0)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        if i % 10 == 0:
            print(f"Epoch {epoch}| iteration {i}| loss: {loss.item()}| norm: {norm}| time: {(t1-t0)*1000:.2f} ms")
            


       
    

validate(model, cfg)