import pandas as pd
import numpy as np


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
                frames = f.loc[f.index == sequence]

                for frame in range(len(frames.index)):
                    height = []
                    row = frames.loc[frames.frame == frame]
                    
                    right_hand = [[row[f"x_right_hand_{i}"], row[f"y_right_hand_{i}"], row[f"z_right_hand_{i}"]] for i in range(21)] 
                    left_hand = [[row[f"x_left_hand_{i}"], row[f"y_left_hand_{i}"], row[f"z_left_hand_{i}"]] for i in range(21)]
                    face = [[row[f"x_face_{i}"], row[f"y_face_{i}"], row[f"z_face_{i}"]] for i in range(76)]
                    pose = [[row[f"x_pose_{i}"], row[f"y_pose_{i}"], row[f"z_pose_{i}"]] for i in range(12)]
                    
                    height.extend(right_hand); height.extend(left_hand); height.extend(face); height.extend(pose)
                    



                    print(np.array(height).shape)


                    break
                break
                    
                
            break



    def get_mapper(self):
        return self.mapper


ex = Extractor()
ex.extract()
        



