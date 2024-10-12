import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
import json
import pickle
from tokenizer import RegexTokenizer
from typing import Tuple

class Extractor():

    def __init__(self):
        self.train_df = pd.read_csv("data/train.csv")
        self.unique_files = self.train_df.file_id.unique()
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

        phrases = t.loc[:, 'phrase'].values
        maxm = 0
        
        text = ' '.join(t.loc[:, 'phrase'].values)
        tokens, merges = self.tok.train(text, 500, True)

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (el0, el1), v in merges.items():
            vocab[v] = vocab[el0] + vocab[el1]
        
        with open("data/extractor.pkl", "wb") as f:
            pickle.dump(vocab, f)
        return merges, vocab
    
    def add_padding(self, array: list, max_len: int) -> np.ndarray:
        if not isinstance(array, list):
            array = [array]
        array = np.array(array)
        
        
        if  array.shape[0] == max_len:
            return array[:max_len]
        elif array.shape[0] > max_len:
            sys.exit("Array is bigger than max_len")
        else:
            return np.pad(array, ((0, max_len - array.shape[0])), mode='constant', constant_values=276)
        

    def extract(self):
        for file in self.unique_files:
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            
            examples = []
            labels = []
            for sequence in tqdm(self.mapper[file]):
                
                #TODO: think about the regex pattern of the tokenizer, a lot of phone numbers are being tokenized as a single token

                curr_sqnce = f.loc[f.index == sequence]
                if curr_sqnce.shape[0] == 0 or curr_sqnce.shape[0] < 130:
                    continue

                
                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [20, 20, 75, 11]

                array = np.concatenate([np.stack([curr_sqnce.iloc[:130].loc[:, f'{dim}_{kpoint}_0' : f"{dim}_{kpoint}_{r}"].to_numpy()
                                      for dim in ['x','y','z']], axis=2)
                                      for kpoint, r in zip(kpoints, ranges)], axis=1)
                
                assert array.shape == (130, 130, 3)

                examples.append(array)

                tokenized_phrase = self.tok.encode(self.sequence_to_phrase[sequence], self.merges)
                padded = self.add_padding(tokenized_phrase, 31)

                assert padded.shape == (31,), f"padded shape: {padded.shape}"
                labels.append(padded)

                
            np.savez_compressed(f"data/extracted/{file}.npz", np.array(examples), np.array(labels))
    
  
                
                


ex = Extractor()
t1  = time.time()
ex.extract()
t2 = time.time()
print(t2-t1)



        



