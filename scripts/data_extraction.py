import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from tokenizer import RegexTokenizer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class Extractor():
    def __init__(self, merges = False):
        self.train_df = train_df = pd.read_csv("data/merged.csv")
        self.sequence_to_phrase = {sequence: phrase for sequence, phrase in zip(train_df.sequence_id, train_df.phrase)}
        self.unique_files = unique_files = train_df.file_id.unique()
        self.file_to_sequences = {file : train_df.loc[train_df['file_id'] == file].sequence_id for file in unique_files}
        self.base_out_dir = Path("data/tensors3")

    def process_seq(self, file, sequence, f, debug = False):
        seq_start = time.time()
                
              
        process_start = time.time()
        curr_sqnce = f.loc[f.index == sequence]
        ranges = [21, 21, 76, 12]

        start = 0
        arr = []
        for r2 in ranges:
            c = curr_sqnce.iloc[:, start: start+r2 *3]
            v = c.to_numpy()
            arr.append(v.reshape(-1, r2, 3))
                    
            start += r2*3
                
            array = np.concatenate(arr, axis=1)

        if debug:
            print(f"processing {(time.time() - process_start)*100}")
        assert array.shape[1] == 130 and array.shape[2] == 3, f"Shape mismatch: {array.shape}"

        save_start = time.time()
        output_dir = self.base_out_dir / f"{file}"
        output_dir.mkdir(parents=True, exist_ok=True)

               
        out_path = output_dir / f"{sequence}.npz"
        np.savez(out_path, np.array(array))

        if debug:
            print(f"saving {(time.time() - save_start)*100:.2f} ms")
                
        if debug:
            print(f"Sequence processed in {(time.time() - seq_start)*100:.2f} seconds")



    def extract(self, debug = False):
        total_start = time.time()

        # Define columns to read from parquet
        cols = []
        for kpoint, r in zip(['right_hand', 'left_hand', 'face', 'pose'],[21, 21, 76, 12]):
            for dim in ['x','y','z']:
                for i in range(r):
                    cols.append(f'{dim}_{kpoint}_{i}')

        for file in tqdm(self.unique_files):
            file_start = time.time()
            
            # Timer for parquet reading
            parquet_start = time.time()
            
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet", columns = cols)
            if debug:
                print(f"parquet read {(time.time() - parquet_start)*100} ms")
            
    
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(self.process_seq, file, sequence, f, debug) for sequence in (self.file_to_sequences[file])]
                
                

            if debug:
                print(f"File {file} processed in {(time.time() - file_start)*100:.2f} seconds")

        
        total_time = time.time() - total_start
        if debug:
            print(f"Total time: {total_time} seconds")
        
        

if __name__ == "__main__":
    extractor = Extractor(merges=True)
    extractor.extract()
    print("done")