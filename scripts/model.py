import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers.norm_act import BatchNormAct2d



class FeatureExtraction(nn.Module):
    def __init__(self, out_dim ,channels: int = 3, height: int = 130, width: int = 130):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels = 32 * math.ceil(width / 2)
        self.conv_stem = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(1,2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        self.stem_linear = nn.Linear(height * in_channels, out_dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)

    def forward(self, data):
        xc = data.permute(0, 3, 1, 2) # B, C, H, W
        xc = self.conv_stem(xc) # B, 32, H, W//2
        xc = self.bn_conv(xc) # B, 32, H, W//2
        xc = xc.flatten(1) # B, 32 * H * W//2
        xc = self.stem_linear(xc) # B, out_dim
        xc = self.stem_bn(xc) # B, out_dim

        return xc


class Net(nn.Module):
    def __init__(self, out_dim):
        super(Net, self).__init__()
        self.feature_extraction = FeatureExtraction(out_dim)
        self.face_fe = FeatureExtraction(out_dim, height=75)
        

    def forward(self, data):
        right_hand = data[:, :, :20, :]
        left_hand = data[:, :, 20:40, :]
        face = data[:, :, 40:115, :]
        pose = data[:, :, 115:, :]


        all_together = self.feature_extraction(data)
        face = self.face_fe(face)
        pose = self.feature_extraction(pose)
        left_hand = self.feature_extraction(left_hand)
        right_hand = self.feature_extraction(right_hand)

        xc = torch.cat([face, pose, left_hand, right_hand], dim=1)

        return all_together
