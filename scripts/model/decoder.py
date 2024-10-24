import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple



class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.n_dim // config.n_heads
        self.kqv = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        self.proj = nn.Linear(config.n_dim, config.n_dim)
        self.att_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size, config.block_size)))
    
    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        # linearly transform the inputs to have queries, keys, and values
        qkv = self.kqv(x)
        

        qkv = qkv.chunk(3, dim=-1)
        
        assert C%self.n_heads == 0, "Embedding dimension must be divisible by number of heads"


        q, k, v = map(lambda t: t.view(B, T, self.n_heads, C//self.n_heads), qkv)

        q,k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.res_dropout(self.proj(out))
        return out
        


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_dim, 4 * config.n_dim),
            nn.ReLU(),
            nn.Linear(4 * config.n_dim, config.n_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    m = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(m, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # print(f" freqs_cis before reshaPE: {freqs_cis.shape}")
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # print(f"freqs_cis: {freqs_cis.shape}")
    # print(f"xq_: {xq_.shape}")

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # config.n_dim: embedding dimension, config.n_heads: the number of heads we'd like
        super().__init__()
        head_size = config.n_dim // config.n_heads
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_dim)
        self.ln2 = nn.LayerNorm(config.n_dim)

    def forward(self, x, freqs_cis):
        x = x + self.sa(self.ln1(x), freqs_cis)
        x = x + self.ffwd(self.ln2(x))
        return x
class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_dim)
        # self.position_embedding_table = nn.Embedding(block_size, config.n_dim)

        self.layers = nn.ModuleList()
        for _ in range(config.n_layer):
            self.layers.append(Block(config))
        self.ln_f = nn.LayerNorm(config.n_dim) # final layer norm
        self.lm_head = nn.Linear(config.n_dim * config.block_size, 31*config.vocab_size)
        self.freqs_cis = precompute_freqs_cis(config.n_dim//config.n_heads , config.block_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T, C = x.shape

        self.freqs_cis = self.freqs_cis.to(x.device)

        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # x = tok_emb # (B,T,C)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        
        x = self.ln_f(x) # (B,T,C)
        x = x.view(B, T*C)
        logits = self.lm_head(x) # (B,31*vocab_size)
        logits = logits.view(B, 31, -1)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx