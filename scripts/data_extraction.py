import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer import RegexTokenizer

class Extractor():

    def __init__(self):
        self.train_df = train_df = pd.read_csv("data/merged.csv")

        self.sequence_to_phrase = {sequence: phrase for sequence, phrase in zip(train_df.sequence_id, train_df.phrase)}
        
        self.unique_files = unique_files = train_df.file_id.unique()
        self.file_to_sequences = {file : train_df.loc[train_df['file_id'] == file].sequence_id for file in unique_files}


        self.tok = RegexTokenizer()
        self.merges, self.vocab = self.init_tokenizer(vocab_size=500)
        print("Tokenizer initialized")
        self.vocab_size = len(self.vocab) + 1 # +1 for padding token
        

    def init_tokenizer(self, vocab_size: int) -> tuple[dict, dict]:
        
        phrases = self.sequence_to_phrase.values()
        text = ' '.join(phrases)
        tokens, merges = self.tok.train(text, vocab_size, verbose=False)

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
        
        if array.shape[0] >= max_len:
            return array[:max_len]
        
        else:
            return np.pad(array, ((0, max_len - array.shape[0])), mode='constant', constant_values = self.vocab_size)

    def extract(self):
        for file in tqdm(self.unique_files):
            f = pd.read_parquet(f"data/train_landmarks/{file}.parquet")
            
            examples = []
            labels = []
            for sequence in (self.file_to_sequences[file]):
                
                curr_sqnce = f.loc[f.index == sequence]
                if curr_sqnce.shape[0] == 0 or curr_sqnce.shape[0] < 130:
                    continue
                
                kpoints = ['right_hand', 'left_hand', 'face', 'pose']
                ranges = [20, 20, 75, 11]

                array = np.concatenate([np.stack([curr_sqnce.iloc[:130].loc[:, f'{dim}_{kpoint}_0' : f"{dim}_{kpoint}_{r}"].to_numpy()
                                        for dim in ['x','y','z']], axis=2)
                                        for kpoint, r in zip(kpoints, ranges)], axis=1)
                assert array.shape == (130, 130, 3), f"something wrong, array shape: {array.shape}"     

                tokenized_phrase = self.tok.encode(self.sequence_to_phrase[sequence], self.merges)
                if not isinstance(tokenized_phrase[0], int):
                    continue

                examples.append(array)

                
                padded = self.add_padding(tokenized_phrase, max_len=31)
                assert padded.shape == (31,), f"something wrong, padded shape: {padded.shape}"
                labels.append(padded)

                

                
            np.savez_compressed(f"data/extracted/{file}.npz", np.array(examples), np.array(labels))
        
    

if __name__ == "__main__":
    extractor = Extractor()
    extractor.extract()
    print("done")