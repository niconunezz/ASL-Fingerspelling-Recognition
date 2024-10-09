import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
from tokenizer import RegexTokenizer
from typing import Tuple

class Extractor():

    def __init__(self):
        self.train_df = pd.read_csv("data/train.csv")
        # self.unique_files = self.train_df.file_id.unique
        self.unique_files = [1019715464, 1021040628]

        self.mapper = {file : self.train_df.loc[self.train_df['file_id'] == file].sequence_id.unique() for file in self.unique_files}

        #TODO must obviously refactor haha
        self.sequence_to_phrase = {}
        for file in self.unique_files:
            for sequence in self.mapper[file]:
                phrase = self.train_df.loc[(self.train_df['sequence_id'] == sequence)].phrase.values[0]
                self.sequence_to_phrase[sequence] = phrase
                
            
        self.tok = RegexTokenizer()
        self.merges, self.vocab = self.init_tokenizer()

       
    
    def init_tokenizer(self) -> tuple[dict, dict]:
        t = pd.read_csv("data/train.csv")
        text = ' '.join(t.loc[:, 'phrase'].values)
        tokens, merges = self.tok.train(text, 275, True)

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (el0, el1), v in merges.items():
            vocab[v] = vocab[el0] + vocab[el1]
        
        return merges, vocab
    

    
    

    def extract(self):
        for file in self.unique_files:
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            
            examples = []
            for sequence in (self.mapper[file]):
                
                #TODO: think about the regez pattern of the tokenizer,
                #TODO a lot of phone numbers are being tokenized as a single token
                
                print(self.sequence_to_phrase[sequence])
                print(self.tok.encode(self.sequence_to_phrase[sequence], self.merges))
                sys.exit()



                curr_sqnce = f.loc[f.index == sequence]
                if curr_sqnce.shape[0] == 0 or curr_sqnce.shape[0] < 130:
                    continue

                
                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [20, 20, 75, 11]

                array = [np.stack([curr_sqnce.iloc[:130].loc[:, f'{dim}_{kpoint}_0' : f"{dim}_{kpoint}_{r}"].to_numpy()
                                      for dim in ['x','y','z']], axis=2)
                                      for kpoint, r in zip(kpoints, ranges)]
                
                assert array[0].shape == (130, 21, 3)
                examples.append(array)
            

            
            # np.savez_compressed(f"data/extracted/{file}.npz", np.array(examples))
                
                
                


ex = Extractor()
t1  = time.time()
ex.extract()
t2 = time.time()
print(t2-t1)

        



