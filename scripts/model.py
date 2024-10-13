import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers.norm_act import BatchNormAct2d



class FeatureExtraction(nn.Module):
    def __init__(self, out_dim ,channels: int = 3, n_landmarks: int = 130):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels = 32 * math.ceil(n_landmarks / 2)
        self.stem_linear = nn.Linear(in_channels, out_dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(1,2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)

    def forward(self, data):
        xc = data.permute(0, 3, 1, 2)
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.flatten(1)
        xc = self.stem_linear(xc)
        xc = self.stem_bn(xc)

        return xc


class Net(nn.Module):
    def __init__(self, out_dim):
        super(Net, self).__init__()
        self.feature_extraction = FeatureExtraction(out_dim)
        

    def forward(self, data):
        xc = self.feature_extraction(data)
        return xc
