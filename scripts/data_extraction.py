import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class Extractor():
    def __init__(self):
        self.train_df = train_df = pd.read_csv("data/merged.csv")
        self.sequence_to_phrase = {sequence: phrase for sequence, phrase in zip(train_df.sequence_id, train_df.phrase)}
        self.unique_files = unique_files = train_df.file_id.unique()
        self.file_to_sequences = {file : train_df.loc[train_df['file_id'] == file].sequence_id for file in unique_files}
        self.base_out_dir = Path("data/tensors")

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

        cols = []
        for kpoint, r in zip(['right_hand', 'left_hand', 'face', 'pose'],[21, 21, 76, 12]):
            for dim in ['x','y','z']:
                for i in range(r):
                    cols.append(f'{dim}_{kpoint}_{i}')

        for file in (self.unique_files):
            file_start = time.time()
            
            # Timer for parquet reading
            parquet_start = time.time()
            try:
                f = pd.read_parquet(f"data/train_landmarks/{file}.parquet", columns = cols)
            except Exception as e:
                continue
            if debug:
                print(f"parquet read {(time.time() - parquet_start)*100} ms")
            
    
            self.process_seq(file, self.file_to_sequences[file].iloc[0], f, debug)
                
                
            n_seq =len(self.file_to_sequences[file])
            tt = time.time() - file_start
            print(f"processed in {tt:.2f} seconds| {n_seq/tt:.2f} seq/s")

        
        total_time = time.time() - total_start
        if debug:
            print(f"Total time: {total_time} seconds")
        
        

if __name__ == "__main__":
    extractor = Extractor()
    extractor.extract(debug=False)
    print("done")