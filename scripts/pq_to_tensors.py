import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path




class Extractor():
    def __init__(self):
        self.train_df = train_df = pd.read_csv("data/newtrain.csv", dtype_backend='pyarrow')
        self.sequence_to_phrase = dict(zip(train_df.sequence_id, train_df.phrase))
        self.unique_files = unique_files = train_df.file_id.unique()
        self.file_to_sequences = file_to_sequences = {file : list(map(self.ensure_5_dig ,train_df.loc[train_df['file_id'] == file].sequence_id.values.tolist())) for file in unique_files}
        
        self.base_out_dir = Path("data/newtensors")
    
    def ensure_5_dig(self, x):
        x = str(x)
        n = 5-len(x)
        x = "0"*n + x
        return x

    def process_seq(self, sequence, f, debug = False):
        seq_start = time.time()
                
        process_start = time.time()
        curr_sqnce = f.loc[f.index == sequence]

        start = 0
        arr = []
        for n in range(1, 22):
            c = curr_sqnce.iloc[:, start: start+(3)]
            v = c.to_numpy()
            arr.append(v.reshape(-1, 1, 3))
                    
            start = n*3
                
        array = np.concatenate(arr, axis=1)
       
        
        if debug:
            print(f"processing {(time.time() - process_start)*100} ms")
        assert array.shape[1] == 21 and array.shape[2] == 3, f"Shape mismatch: {array.shape}"

        save_start = time.time()
        output_dir = self.base_out_dir
               
        np.save(output_dir / f"{sequence}.npy", np.array(array))

        if debug:
            print(f"saving {(time.time() - save_start)*100:.2f} ms")
                
        if debug:
            print(f"Sequence processed in {(time.time() - seq_start)*100:.2f} ms")


    def process_file(self, file, cols, debug = False):
            
        file_start = time.time()
        parquet_start = time.time()
        try:
            f = pd.read_parquet(f"data/files/{file}.parquet", columns = cols, engine='pyarrow')
        except Exception as e:
            pass
        if debug:
            print(f"parquet read {(time.time() - parquet_start)*100:.2f} ms")
        
        for sequence in self.file_to_sequences[file]:
            self.process_seq(sequence, f, debug)
            
                
        n_seq =len(self.file_to_sequences[file])
        tt = time.time() - file_start
        print(f"processed in {tt:.2f} seconds| {n_seq/tt:.2f} seq/s")

    def extract(self, debug = False):
        total_start = time.time()

        cols = [f"{hand}_{i}_{dim}" for hand in ["right_hand", "left_hand"] for i in range(21) for dim in ['x', 'y', 'z']]
        
        cols.append("frame")
                

        for file in self.unique_files:
            self.process_file(file, cols, debug)
        
        
        total_time = time.time() - total_start
        if debug:
            print(f"Total time: {total_time} seconds")
        
        

if __name__ == "__main__":
    extractor = Extractor()
    extractor.extract(debug=False)
    print("done")