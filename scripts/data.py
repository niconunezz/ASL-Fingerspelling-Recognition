import pandas as pd
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np

train_df = pd.read_csv("/kaggle/input/asl-fingerspelling/train.csv")

class CustomDataset(Dataset):
    def __init__(self, df, config, mode='train'):
        
        self.df = df.copy()
        self.config = config
        self.mode = mode
    
        with open(config.data_path + "selected_columns.json", "r") as f:
            columns = json.load(f)