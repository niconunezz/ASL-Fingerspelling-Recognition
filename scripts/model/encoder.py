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




def apply_rotary_emb(xq: torch.Tensor,
                     xk: torch.Tensor,
                     freqs: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    
    xq_= torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_= torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x, freqs, mask):
        B, T, C = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim), qkv)

        q, k = apply_rotary_emb(q, k, freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # att = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        # out = att @ v
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, is_causal=False)


        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        
        return out

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)


class PointwiseConv(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True
            ) -> None:
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DepthWiseConv(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True
                 ) -> None:
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size=kernel_size, 
            groups=in_channels, 
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        outputs, gate = x.chunk(2, dim=self.dim)
        return  outputs * gate.sigmoid()

class ConvolutionModule(nn.Module):
    # see https://www.youtube.com/watch?v=vVaRhZXovbw

    def __init__(self, in_channels, kernel_size, expansion_factor, dropout_p = 0.1):
        super(ConvolutionModule, self).__init__()
       
        self.pointwise = PointwiseConv(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.depthwise = DepthWiseConv(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pointwise2 = PointwiseConv(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p= dropout_p)
    
    def forward(self, x_or, mask):
        B, C, T = x_or.shape
        mask = mask.unsqueeze(1) # B, 1, T
        
        if mask.size(2) > 0: # time > 0
            x = x_or.masked_fill(~mask.bool(), 0.0)
        
        x = self.pointwise(x_or)
        x = self.act1(x)
        x = self.depthwise(x)

        x_bn = x.view(B, T, C).reshape(B*T, C)
        mask_bn = mask.view(-1).bool()
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        
        x = x_bn.view(B, C, T)

        x = self.act2(x)
        x = self.pointwise2(x)
        x = self.dropout(x)

        if mask.size(2) > 0:
            x = x.masked_fill(~mask.bool(), 0.0)
        
        return x


def create_scale(dim):
    scale = nn.Parameter(torch.ones(1 ,1, dim))
    bias = nn.Parameter(torch.zeros(1,1, dim))
    return scale, bias

class SqueezeformerBlock(nn.Module):
    def __init__(self, config):
        super(SqueezeformerBlock, self).__init__()
        dim = config.n_dim
        n_heads = config.n_heads

        self.scale_mhsa, self.bias_mhsa = create_scale(dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = create_scale(dim)
        self.scale_conv, self.bias_conv = create_scale(dim)
        self.scale_ff_conv, self.bias_ff_conv = create_scale(dim)

        self.freqs = precompute_freqs(dim//n_heads, config.block_size)
        self.freqs = self.freqs.to(config.device)
        """
        #TODO
        In addition, we cached the rotary embeddings once and fed them into each layer with
        the input data so they are not duplicated in each layer. This resulted in 20% less parameters in the model. 
        
        """

        self.attn = MultiHeadAttention(dim, n_heads)
        self.ln_att = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, expansion_factor= 4)
        self.ln_ff = nn.LayerNorm(dim)
        self.conv = ConvolutionModule(dim, kernel_size=3, expansion_factor=2)
        self.ln_conv = nn.LayerNorm(dim)
        self.ff_2 = FeedForward(dim, expansion_factor= 4)
        self.ln_ff_2 = nn.LayerNorm(dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    

    def forward(self, x, mask):
        """
            x: B, T, C
            mask: B, T
        """
        B, T, C = x.shape
        mask_pad = mask.bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0,2,1) * mask_pad)
        mask_flat = mask.view(-1).bool() # B*T
        

        res = x
        x = x * self.scale_mhsa + self.bias_mhsa

        x = res + self.attn(x, self.freqs, mask_pad.unsqueeze(1))

        # skip 1
        x_skip = x.view(B*T, C) # B*T, C
        x = x_skip[mask_flat].unsqueeze(0) # 1, B*T, C

        x = self.ln_att(x)
        res = x

        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = res + self.ff(x)

        x = self.ln_ff(x)
        # unskip 1
        x_skip[mask_flat] = x.squeeze(0).to(x_skip.dtype)
        x = x_skip.view(B, T, C)

        res = x
        x = x * self.scale_conv + self.bias_conv
        x = x.permute(0, 2, 1)
        x = res + self.conv(x, mask).permute(0, 2, 1)

        # skip 2
        x_skip = x.view(B*T, C) # B*T, C
        x = x_skip[mask_flat].unsqueeze(0) # 1, B*T, C

        x = self.ln_conv(x)

        res = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = res + self.ff_2(x)
        x = self.ln_ff_2(x)


        # unskip 2
        x_skip[mask_flat] = x.squeeze(0).to(x_skip.dtype)
        x = x_skip.view(B, T, C)

        return x

    