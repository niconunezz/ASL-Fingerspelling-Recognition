import random
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F
from albumentations import Compose
import torch
import numpy as np
import math
import typing


class Resample(BasicTransform):
    def __init__(self, sample_rate = (0.8, 1.2), p: float = 0.5):
        super(Resample, self).__init__(always_apply=False, p=p)

        lower, upper = sample_rate

        self.lower = lower
        self.upper = upper
    
    def apply(self, data, sample_rate = 1., **params):
        data = torch.from_numpy(data)
        length = data.shape[0]
        new_length = max(int(length * sample_rate), 1)
        data = F.interpolate(data.permute(1, 2, 0), new_length).permute(2, 0, 1)
        return data

    @property
    def targets(self):
        return {"image": self.apply}
    

class TemporalCrop(BasicTransform):
    def __init__(
        self,
        length=384,
        always_apply=False,
        p=0.5,
    ):
        super(TemporalCrop, self).__init__(always_apply, p=0.5)
        self.length = length
    
    def apply(self, data, length=350, offset_01=0.5, **params):
        orig_len = data.shape[0]
        max_l = np.clip(orig_len - length, 1, orig_len) # num beetween 1 and orig_len
        start = int(max_l*offset_01) # cut in the middle
        end = start + length
        return data[start:end] # crop the data


    @property
    def targets(self):
        return {"image": self.apply}
    

class TimeShift(BasicTransform):
    def __init__(self, p: float = 0.5):
        super(TimeShift, self).__init__(always_apply=False, p=p)
    
    def apply(self, data, shift = 5, **params):        
        B,T,C = data.shape
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        zeros = torch.zeros((shift, T, C))
        data = torch.cat([zeros, data.clone()], dim=0)

        return data

    @property
    def targets(self):
        return {"image": self.apply}