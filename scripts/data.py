import pandas as pd
import os
from tqdm import tqdm
import json
import albumentations as A
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    x = ((x * 255).astype("uint8")) 
    return x
        

def load_dicom_stack(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    
    array = np.stack([cv2.resize(d.pixel_array,(256,256),interpolation=cv2.INTER_AREA).astype("float32") for d in dicoms])
    array = array[idx]
    return {"array": convert_to_8bit(array), "positions": ipp, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float")}


class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.maxlen = {"Sagittal T1": 15, "Axial T2": 20, "Sagittal T2/STIR": 15}
    
    def check_size(self, x, desctype):
        if x.shape[0] > self.maxlen[desctype]:
            x = x[:self.maxlen[desctype]]
        elif x.shape[0] < self.maxlen[desctype]:
            #! should we zero pad or repeat the last image?
            if (self.maxlen[desctype] - x.shape[0]) > x.shape[0]:
                x = np.concatenate((x, np.zeros((self.maxlen[desctype] - x.shape[0], x.shape[1], x.shape[2]))))
            else:
                x = np.concatenate((x, x[:(self.maxlen[desctype] - x.shape[0])]))
        return x

    def forward(self, x, desctype):
        # x is a batch of images, a series
       
        x = self.check_size(x, desctype)
        return x


def generate_data():
    states = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    df=pd.read_csv("train.csv")
    metaobj = json.load(open("data/metaobj.json", "r"))
    
    preprocessor = Preprocessing()
    xdata = {"Sagittal T1": [], "Axial T2": [], "Sagittal T2/STIR": []}
    ys = []
    for idx, patient in (enumerate(metaobj)):
        print(f"Processing {idx+1}/{len(metaobj)}")
        try:
            y = torch.tensor([states[i] for i in df[df["study_id"] == int(patient)].values[0][1:]])
        except:
            continue
        ys.append(F.one_hot(y, 3))
        
        
        
        cache = []
        for i,serie in enumerate(metaobj[patient]["series_ids"]):
            imgs = load_dicom_stack(f"train_images/{patient}/{serie}", "sagittal")['array']
            desctype = metaobj[patient]["series_description"][i]
            if desctype in cache:
                continue
            procesed_data = preprocessor(imgs, desctype = desctype)
            cache.append(desctype)
            if desctype == "Sagittal T1" or desctype == "Sagittal T2/STIR":
                assert procesed_data.shape[0] == 15
            else:
                assert procesed_data.shape[0] == 20, f"is actually {procesed_data.shape}"

            xdata[metaobj[patient]["series_description"][i]].append(procesed_data)

            #! some patiens have more than 3 series for now we will only use the first 3
        
        
    sagt1 = np.stack(xdata["Sagittal T1"])
    axial = np.stack(xdata["Axial T2"])
    sagt2 = np.stack(xdata["Sagittal T2/STIR"])
    ys = np.stack(ys)

    print(f"shape of Sagittal T1: {sagt1.shape}")
    print(f"shape of Axial T2: {axial.shape}")
    print(f"shape of Sagittal T2/STIR: {sagt2.shape}")
    print(f"shape of ys: {ys.shape}")

    assert len(xdata["Sagittal T1"]) == len(xdata["Axial T2"]) == len(xdata["Sagittal T2/STIR"]) == len(ys)
    np.savez_compressed("data/sagitalt1.npz", sagt1)
    np.savez_compressed("data/axialt2.npz", axial)
    np.savez_compressed("data/sagitalt2.npz", sagt2)
    np.savez_compressed("data/ydata.npz", ys)

class CustomDataset(Dataset):
    def __init__(self):
        self.sagt1 = torch.tensor(np.load("data/sagitalt1.npz")['arr_0'])
        self.sagt2 = torch.tensor(np.load("data/sagitalt2.npz")['arr_0'])
        self.axt2 = torch.tensor(np.load("data/axialt2.npz")['arr_0'])
        self.y = torch.tensor(np.load("data/ydata.npz")['arr_0'])
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5)])
        
    def __len__(self):
        return len(self.sagt1)
    
    def __getitem__(self, idx):
        # Apply transformations to the images
        return self.sagt1[idx], self.sagt2[idx], self.axt2[idx], self.y[idx]