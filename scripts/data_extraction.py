import pandas as pd
import numpy as np

import sys
import time

class Extractor():

    def __init__(self):
        self.train_df = pd.read_csv("data/train.csv")
        # self.unique_files = self.train_df.file_id.unique
        self.unique_files = [1019715464, 1021040628]
        self.mapper = {file : self.train_df.loc[self.train_df['file_id'] == file].sequence_id.unique() for file in self.unique_files}
    
    
    def extract(self):
        for file in self.unique_files:
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            for sequence in self.mapper[file]:
                curr_sqnce = f.loc[f.index == sequence]
                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [21, 21, 76, 12]
                

                # print(curr_sqnce["x_right_hand_0"].iloc[12])

                
                # for frame in range(0,5):
                #     for i in range(21):
                #         try:
                #             print(curr_sqnce[f"x_right_hand_{i}"].iloc[frame])
                #         except IndexError:
                #             print(f" Error en frame {frame} y punto {i}")
                            # sys.exit()

                print(curr_sqnce[f"x_right_hand_0"].iloc[3])
                sys.exit()
                # data = {}
                # for col, r in zip(kpoints, ranges):
                #     data[col] = [[[curr_sqnce[f"{dim}_{col}_{i}"].iloc[5]
                #                     for dim in ['x', 'y', 'z']]
                #                     for i in range(r)]
                #                     for frame in range(0,5)]
                #     break
        
    
    
                



                




    def get_mapper(self):
        return self.mapper


ex = Extractor()
t1  = time.time()
ex.extract()
t2 = time.time()
print(t2-t1)

        



