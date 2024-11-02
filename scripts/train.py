import sys
import time
import math
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from model.model import Net
import torch.optim as optim
from data import CustomDataset
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class config:
    n_dim: int = 208
    n_heads: int = 2
    block_size: int = 16
    max_seq_len: int = 31
    encoder_layers: int = 5
    vocab_size: int = 62
    n_layer: int = 6
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_files: int = 512
    epochs: int = 5
    max_ex: int = None

cfg = config()
print(f"Device: {cfg.device}")



data = CustomDataset(cfg)
print("Data loaded")
print(f"Data has {len(data)} examples")

dataloader = DataLoader(data, batch_size=512, shuffle=False)
model = Net(cfg)
model.to(cfg.device)

print(f"Model has {(sum(p.numel() for p in model.parameters())/1e6):.2f} M parameters")

mx_lr = 1e-3
mn_lr = 1e-5
warmup_epoch = 1
max_epochs = cfg.epochs 

def get_lr(it):
    if it < warmup_epoch:
        return mx_lr * (it+1) / warmup_epoch
    if it > max_epochs:
        return mn_lr
    
    return (mx_lr-mn_lr) * ((1 + math.cos(math.pi * (it - warmup_epoch) / (max_epochs - warmup_epoch))) / 2) + mn_lr


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Data loader examples: {len(dataloader)}")

        

for epoch in range(config.epochs):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for i, batch in (enumerate(dataloader)):
        t0 = time.time()
        x, mask, y = batch['data'], batch['mask'], batch['target']
        x, y = x.to(cfg.device), y.to(cfg.device)
        mask = mask.to(cfg.device)

        logits, loss = model(x, mask, y)
        optimizer.zero_grad()
        loss.backward()
       
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        if i % 10 == 0:
            print(f"Epoch {epoch}| iteration {i}| loss: {loss.item()}| norm: {norm}| time: {(t1-t0)*1000:.2f} ms")
        

def validate(model, config):
    data = CustomDataset(cfg = config, mode="val")
    dataloader = DataLoader(data, batch_size= config.val_files, shuffle=False)
    print(f"Validation data has {len(data)} examples")
    print(f"Validation data loader has {len(dataloader)} examples")
    
    vocab = json.load(open("data/supplemental_landmarks/prediction_index_to_character.json"))
    vocab['59'] = '<sos>'
    vocab['60'] = '<eos>'
    vocab['61'] = '<pad>'
    model.eval()
    
    for batch in dataloader:
        x, mask, y = batch['data'], batch['mask'], batch['target']
        
        x, y = x.to(cfg.device), y.to(cfg.device)
        mask = mask.to(cfg.device)
        
        logits, loss = model(x, mask, y)
        print(f"Validation loss: {loss.item()}")

        print(f"Logits: {logits.shape}")
        probs = F.softmax(logits, dim = -1)
        for i in range(5):
            print(f"Sentence: {''.join([vocab[str(token.item())] for token in y[i]])}")

            
            print("".join([vocab[str(i.item())] for i in torch.multinomial(probs[0], 1)]))
            print('\n')
        

        break


       
    

validate(model, cfg)