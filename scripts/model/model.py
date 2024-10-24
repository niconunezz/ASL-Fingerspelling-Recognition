
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
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        self.stem_linear = nn.Linear(in_channels, out_dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)

    def forward(self, data):
        xc = data.permute(0, 3, 1, 2) # B, C, H, W
        xc = self.conv_stem(xc) # B, 32, H, W//2
        xc = self.bn_conv(xc) # B, 32, H, W//2
        xc = xc.reshape(*data.shape[:2], -1) # B, H, 32 * W//2
        xc = self.stem_linear(xc) # B, out_dim

        B, T, C = xc.shape
        xc = xc.view(1, -1, C)
        xc = self.stem_bn(xc.permute(0, 2, 1)).permute(0, 2, 1) # B*T, C
        xc = xc.view(B, T, C)
        return xc


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.n_heads = n_heads = config.n_heads
        self.feature_extraction = FeatureExtraction(out_dim = 208)
        self.face_fe = FeatureExtraction(out_dim = 52, height=75)
        self.pose_fe = FeatureExtraction(out_dim = 52, height=15)
        self.lhand = FeatureExtraction(out_dim = 52, height=20)
        self.rhand = FeatureExtraction(out_dim = 52, height=20)
        
        self.encoder = nn.Sequential(*[SqueezeformerBlock(config) for _ in range(config.encoder_layers)])
        self.decoder = Decoder(config)

    def forward(self, data, targets):
        right_hand = data[:, :, :20, :]
        left_hand = data[:, :, 20:40, :]
        face = data[:, :, 40:115, :]
        pose = data[:, :, 115:, :]


        all_together = self.feature_extraction(data)
        face = self.face_fe(face)
        pose = self.pose_fe(pose)
        left_hand = self.lhand(left_hand)
        right_hand = self.rhand(right_hand)

        ccat = torch.cat([face, pose, left_hand, right_hand], dim=2)
        xc = all_together + ccat
        xc = self.encoder(xc)
        xc = self.decoder(xc, targets)

        return xc
