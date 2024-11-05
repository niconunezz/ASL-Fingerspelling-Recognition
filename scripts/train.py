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
from transformers.models.speech_to_text import Speech2TextConfig


@dataclass
class config:
    n_dim: int = 208
    n_heads: int = 2
    block_size: int = 384
    max_seq_len: int = 31
    encoder_layers: int = 4
    vocab_size: int = 62
    n_layer: int = 8
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_files: int = 512
    epochs: int = 10
    max_ex: int = None

    config = Speech2TextConfig.from_pretrained("facebook/s2t-small-librispeech-asr")
    config.encoder_layers = 0
    config.decoder_layers = 2
    config.d_model = n_dim
    config.max_target_positions = 1024 #?
    config.num_hidden_layers = 1
    config.vocab_size = 62
    config.bos_token_id = 59
    config.eos_token_id = 60
    config.decoder_start_token_id = 59
    config.pad_token_id = 61
    config.num_conv_layers = 0
    config.conv_kernel_sizes = []
    config.max_length = n_dim
    config.input_feat_per_channel = n_dim
    config.num_beams = 1
    config.attention_dropout = 0.2
    # config.dropout = 0.2
    config.decoder_ffn_dim = 512
    config.init_std = 0.02


cfg = config()
print(f"Device: {cfg.device}")

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True

data = CustomDataset(cfg, verbose = False)
print("Data loaded")
print(f"Data has {len(data)} examples")

dataloader = DataLoader(data, batch_size=128, shuffle=False)
model = Net(cfg, cfg.config)
model.to(cfg.device)

print(f"Model has {(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6):.2f} M parameters")

mx_lr = 4.5e-3
mn_lr = 1e-5
warmup_epoch = 0
max_epochs = len(dataloader)

def get_lr(it):
    if it < warmup_epoch:
        return mx_lr * (it+1) / warmup_epoch
    if it > max_epochs:
        return mn_lr
    
    return (mx_lr-mn_lr) * ((1 + math.cos(math.pi * (it - warmup_epoch) / (max_epochs - warmup_epoch))) / 2) + mn_lr


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Data loader examples: {len(dataloader)}")


verbose = False
t0 = 0
for epoch in range(config.epochs):
    
    for i, batch in (enumerate(dataloader)):
        t0 = time.time()
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        

        x, mask, y = batch['data'], batch['mask'], batch['target']
        x, y = x.to(cfg.device), y.to(cfg.device)
        mask = mask.to(cfg.device)

        
        logits = model(x, mask, y, verbose = False)
            
        B, T, C = logits.shape
        targets = y.view(B*T)    
        loss = F.cross_entropy(logits.view(B*T, C), targets, ignore_index=61)

        t2 = time.time()
        optimizer.zero_grad()
        loss.backward()
        t3 = time.time()

        if verbose:
            print(f"Backward took {(t3-t2)*1000:.2f} ms")

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Epoch {epoch}| iteration {i}| loss: {loss.item():.3f}| lr: {lr:.4f} | norm: {norm:.4f}| time: {(t1-t0)*1000:.2f} ms")
            
        

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
    losses = []
    for i,batch in enumerate(dataloader):
        x, mask, y = batch['data'], batch['mask'], batch['target']
        
        x, y = x.to(cfg.device), y.to(cfg.device)
        mask = mask.to(cfg.device)
        
        logits = model(x, mask, y)

        B, T, C = logits.shape
        targets = y.view(B*T)
        loss = F.cross_entropy(logits.view(B*T, C), targets, ignore_index=61)

        losses.append(loss.item())

        
        probs = F.softmax(logits, dim = -1)
        if i == len(dataloader) - 1:
            
            for i in range(5):
                print(f"Sentence: {''.join([vocab[str(token.item())] for token in y[i]])}")

                
                print("".join([vocab[str(i.item())] for i in torch.multinomial(probs[0], 1)]))
                print('\n')
            
    print(f"Validation loss: {sum(losses)/len(losses)}")

    file = json.load(open("scripts/test/loss.json", "r"))

    max_loss = file['loss']
    new_loss = sum(losses)/len(losses)
    if  new_loss < max_loss:
        file['loss'] = new_loss
        json.dump(file, open("scripts/test/loss.json", "w"))
            

if __name__ == "__main__":
    validate(model, cfg)