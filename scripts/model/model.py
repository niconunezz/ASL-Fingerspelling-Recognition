
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers.norm_act import BatchNormAct2d
from model.encoder import SqueezeformerBlock
from model.decoder import Decoder

class FeatureExtraction(nn.Module):
    def __init__(self, out_dim ,channels: int = 3, height: int = 130, width: int = 130):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels = 32 * math.ceil(height / 2)
        self.conv_stem = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(1,2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, act_layer = nn.SiLU, drop_layer=None)
        self.stem_linear = nn.Linear(in_channels, out_dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        

    def forward(self, data, mask, verbose = False):
        import time
        t0 = time.time()
        xc = data.permute(0, 3, 1, 2) # B, C, H, W
        t1 = time.time()
        if verbose:
            print(f"Permuting data took {(t1-t0)*1000:.2f} ms")

        t0 = time.time()
        xc = self.conv_stem(xc) # B, 32, H, W//2
        t1 = time.time()
        if verbose:
            print(f"Conv stem took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        xc = self.bn_conv(xc) # B, 32, H, W//2
        
        t1 = time.time()
        if verbose:
            print(f"Batch norm took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        xc = xc.reshape(*data.shape[:2], -1) # B, H, 32 * W//2
        t1 = time.time()
        if verbose:
            print(f"Reshaping took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        xc = self.stem_linear(xc) # B, out_dim
        t1 = time.time()
        if verbose:
            print(f"Stem linear took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        B, T, C = xc.shape # 512, 16, 52
        
        xc = xc.view(-1, C) # B*T, C
        t1 = time.time()
        if verbose:
            print(f"Reshaping for BN took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
       
        x_bn = xc[(mask.view(-1) == 1)].unsqueeze(0) # 1, B*T, C
        x_bn = self.stem_bn(x_bn.permute(0, 2, 1)).permute(0, 2, 1) # B*T, C
        xc[mask.view(-1) == 1] = x_bn[0]
        t1 = time.time()
        if verbose:
            print(f"Batch norm for non-padded tokens took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        xc = xc.view(B, T, C)
        xc = xc.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        t1 = time.time()
        if verbose:
            print(f"Final reshaping and masking took {(t1-t0)*1000:.2f} Mseconds")

        import sys
        # sys.exit(0)
        
        return xc


class Net(nn.Module):
    def __init__(self, config, decoder_cfg):
        super(Net, self).__init__()
        self.n_heads = config.n_heads
        self.feature_extraction = FeatureExtraction(out_dim = 208)
        self.face_fe = FeatureExtraction(out_dim = 52, height=75)
        self.pose_fe = FeatureExtraction(out_dim = 52, height=15)
        self.lhand = FeatureExtraction(out_dim = 52, height=20)
        self.rhand = FeatureExtraction(out_dim = 52, height=20)
        
        self.encoder = nn.ModuleList([SqueezeformerBlock(config) for _ in range(config.encoder_layers)])
        self.decoder = Decoder(decoder_cfg)

    def forward(self, data, mask, targets, verbose = False):

        import time 

        t0 = time.time()
        right_hand = data[:, :, :20, :]
        left_hand = data[:, :, 20:40, :]
        face = data[:, :, 40:115, :]
        pose = data[:, :, 115:, :]
        t1 = time.time()
        if verbose:
            print(f"Splitting data took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        all_together = self.feature_extraction(data, mask)
        t1 = time.time()
        if verbose:
            print(f"Feature extraction took {(t1-t0)*1000:.2f} Mseconds")

        t0 = time.time()
        face = self.face_fe(face, mask)
        pose = self.pose_fe(pose, mask)
        left_hand = self.lhand(left_hand, mask)
        right_hand = self.rhand(right_hand, mask)
        t1 = time.time()
        if verbose:
            print(f"Feature extraction of all took {(t1-t0)*1000} Mseconds")

        ccat = torch.cat([face, pose, left_hand, right_hand], dim=2)

        xc = all_together + ccat
        t0 = time.time()
        for layer in self.encoder:
            xc = layer(xc, mask)
        t1 = time.time()
        if verbose:
            print(f"Encoder took {(t1-t0)*1000} Mseconds")
        t0 = time.time()

        xc = self.decoder(xc, labels = targets, encoder_attention_mask= mask.long())
        t1 = time.time()
        if verbose:
            print(f"Decoder took {(t1-t0)*1000} Mseconds")
        return xc
