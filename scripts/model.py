import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from timm.layers.norm_act import BatchNormAct2d


# from @dieter
class FeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_dim):
        super().__init__()   

        self.conv_stem = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        self.in_channels = in_channels
        self.stem_linear = nn.Linear(32*256*128,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
    
    def forward(self, x):
        # x.shape = (B, in_channels, 256, 256)
        x = self.conv_stem(x)
        # print(f"conv_stem: {x.shape}")
        x = self.bn_conv(x)
        # print(f"bn_conv: {x.shape}")
        x = x.flatten(1)
        # print(f"flatten: {x.shape}")
        x = self.stem_linear(x)
        # print(f"stem_linear: {x.shape}")
        x = self.stem_bn(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sagt1 = FeatureExtractor(15, 512)
        self.axt2 = FeatureExtractor(20, 512)
        self.sagt2 = FeatureExtractor(15, 512)
        self.fc1 = nn.Linear(512*3, 512)
    
    def forward(self, sagt1, axt2, sagt2):
        x1 = self.sagt1(sagt1)
        x2 = self.axt2(axt2)
        x3 = self.sagt2(sagt2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fc1(x)
        return x