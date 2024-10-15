import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freqs(dim, block_size ,theta: float = 10000):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    m = torch.arange(0, block_size)
    freqs = torch.outer(m, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

    