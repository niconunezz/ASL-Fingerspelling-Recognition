import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer import RegexTokenizer

class Extractor():

    def __init__(self, merges = False):
        self.train_df = train_df = pd.read_csv("data/merged.csv")

        self.sequence_to_phrase = {sequence: phrase for sequence, phrase in zip(train_df.sequence_id, train_df.phrase)}
        
        self.unique_files = unique_files = train_df.file_id.unique()
        self.file_to_sequences = {file : train_df.loc[train_df['file_id'] == file].sequence_id for file in unique_files}

        self.tok = RegexTokenizer()
        if not merges:
            self.merges, self.vocab = self.init_tokenizer(vocab_size=500)
            self.vocab_size = len(self.vocab) + 1 # +1 for padding token
        else:
            self.merges= pickle.load(open("data/extractor_merges.pkl", "rb"))
            self.vocab = pickle.load(open("data/extractor.pkl", "rb"))
            self.vocab_size = len(self.vocab) + 1 # +1 for padding token
        print("Tokenizer initialized")
        
    def init_tokenizer(
            self, 
            vocab_size: int
            ) -> tuple[dict, dict]:
        
        phrases = self.sequence_to_phrase.values()
        text = ' '.join(phrases)
        tokens, merges = self.tok.train(text, vocab_size, verbose=False)

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (el0, el1), v in merges.items():
            vocab[v] = vocab[el0] + vocab[el1]
        

        with open("data/extractor.pkl", "wb") as f:
            pickle.dump(vocab, f)
        
        with open("data/extractor_merges.pkl", "wb") as f:
            pickle.dump(merges, f)

        return merges, vocab
    
    

    def extract(self):
        for file in tqdm(self.unique_files):
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            
            for sequence in (self.file_to_sequences[file]):
                
                curr_sqnce = f.loc[f.index == sequence]
                
                
                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [20, 20, 75, 11]

                array = np.concatenate([np.stack([curr_sqnce.loc[:, f'{dim}_{kpoint}_0' : f"{dim}_{kpoint}_{r}"].to_numpy()
                                        for dim in ['x','y','z']], axis=2)
                                        for kpoint, r in zip(kpoints, ranges)], axis=1)
                
                assert array.shape[1] == 130 and array.shape[2] == 3, f"Shape mismatch: {array.shape}"
                
                #TODO: Must add start and end token
                label = np.array(self.tok.encode(self.sequence_to_phrase[sequence], self.merges))

                try:
                    label = label.astype(int)
                except Exception as e:
                    print(f" Exception: {e}")
                    print(f"Some error occured with sequence: {sequence}")
                    print(f"original phrase: {self.sequence_to_phrase[sequence]}")
                    print(f"tokenized phrase: {label}")

                os.makedirs(f"data/tensors/{file}", exist_ok=True)
                np.savez_compressed(f"data/tensors/{file}/{sequence}", np.array(array), np.array(label))
        
    

if __name__ == "__main__":
    extractor = Extractor(merges=True)
    extractor.extract()
    print("done")