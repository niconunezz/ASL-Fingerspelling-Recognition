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
import math
import torch
import torch.nn as nn



def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x) 
    return ((x * 255).astype("uint8"))

def resize(img):
        size = (256, 256)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def load_dicom_stack(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    # print(np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float"))
    # print("index: ", idx)
    array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    return {"array": np.asarray([resize(img) for img in convert_to_8bit(array)]), "positions": ipp, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float")}


             
def n_of_images(desctype):
    df = pd.read_csv("train_series_descriptions.csv")
    studies = df[df["series_description"] == f"{desctype}"]["study_id"].values
    descriptions = df[df["series_description"] == f"{desctype}"]["series_id"].values
    n=0
    elements = [(s,d) for s,d in zip(studies, descriptions)]
    for study, description in tqdm(elements):
        path = f"train_images/{study}/{description}"    
        n += len(glob.glob(os.path.join(path, "*.dcm")))
    
    print(f"Number of images in {desctype} is {n}")
    print(f"Mean of images in {desctype} is {n/len(elements)}")
    print(f"Median of images in {desctype} is {np.median([len(glob.glob(os.path.join(f'train_images/{s}/{d}', '*.dcm'))) for s,d in elements])}")




class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.maxlen = {"Sagittal T1": 15, "Axial T2": 20, "Sagittal T2/STIR": 15}
    
    def check_size(self, x, desctype):
        if x.shape[0] > self.maxlen[desctype]:
            x = x[:self.maxlen[desctype]]
        if x.shape[0] < self.maxlen[desctype]:
            x = torch.cat([x, x[:(self.maxlen[desctype] - x.shape[0])]])
        return x

    def forward(self, x, desctype):
        # x is a batch of images, a series
        x = self.check_size(x, desctype)

        return x
    

def main():
    imgs = load_dicom_stack("train_images/4003253/2448190387", "axial")['array']
    preprocessor = Preprocessing()
    imgs = preprocessor(imgs, "Axial T2")
    print(imgs.shape)


if __name__ == "__main__":
    main()