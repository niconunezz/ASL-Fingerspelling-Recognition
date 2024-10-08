import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm

class Extractor():

    def __init__(self):
        self.train_df = pd.read_csv("data/train.csv")
        # self.unique_files = self.train_df.file_id.unique
        self.unique_files = [1019715464, 1021040628]
        self.mapper = {file : self.train_df.loc[self.train_df['file_id'] == file].sequence_id.unique() for file in self.unique_files}
    
        self.drop_rows(self.mapper)

    def drop_rows(self, mapper):
        for file in self.unique_files:
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            to_drop = f.groupby('sequence_id').filter(lambda x: len(x) <= 130).index
            f.drop(to_drop, inplace=True)
            f.to_parquet(f"data/train_landmarks/{file}.parquet")

    def extract(self):
        for file in self.unique_files:
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            # to_drop = f.groupby('sequence_id').filter(lambda x: len(x) <= 130).index
            # f.drop(to_drop, inplace=True)

            for sequence in (self.mapper[file]):
                curr_sqnce = f.loc[f.index == sequence]

                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [21, 21, 76, 12]

                print(curr_sqnce.shape)

                something = [curr_sqnce.loc[:130, f'{dim}_{col}_0' : f"{dim}_{col}_{r}"].to_numpy()
                for dim in ['x','y', 'z']
                for col, r in zip(kpoints, ranges)]

                for s in something:
                    print(s.shape)


                sys.exit()
                # data = {}
                # for col, r in zip(kpoints, ranges):
                #     data[col] = [[[curr_sqnce[f"{dim}_{col}_{i}"].iloc[frame]
                #                 for dim in ['x', 'y', 'z']]
                #                 for i in range(r)]
                #                 for frame in range(130)]

                
        

    def get_mapper(self):
        return self.mapper


ex = Extractor()
t1  = time.time()
ex.extract()
t2 = time.time()
print(t2-t1)

        



