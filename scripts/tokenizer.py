import regex as re
import pandas as pd
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    def __init__(self):
        pass

    
    def stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    
        return counts
    
    def merge(self, ids, pair, sub):
        n_ids = []
        for el in ids:
            i = 0
            curr = []
            while i < len(el):

                if (i < len(el)-1) and (el[i] == pair[0]) and (el[i+1] == pair[1]):
                    curr.append(sub)
                    i += 2
                    
                
                else:
                    curr.append(el[i])
                    i += 1
                
            n_ids.append(curr)
            
        return n_ids
    
    def sm_dicts(self, dic1, dic2):
        
        for k,v in dic2.items():
            if k in dic1:
                dic1[k] += v
            else:
                dic1[k] = v
        
        return dic1
    
        
    def train(self, text, vocab_size, verbose = False):
        chunks = re.findall(GPT4_SPLIT_PATTERN, text)
        tokens = [chk.encode("utf-8") for chk in chunks]
        
        ids = [list(map(int, tok)) for tok in tokens]
        n_merges = vocab_size - 256

        merges = {}
        for i in range(n_merges):
            ntoken = 256 + i

            counts = {}
            for chunk in tokens:
                self.sm_dicts(counts, self.stats(chunk))

            mcp = max(counts, key = counts.get)


            if verbose:
                print(f" {mcp} occurred {counts[mcp]} times")
                print(f" Changing {mcp} --> {ntoken} ")
                if mcp[0]<256 and mcp[1]<256:
                    print(f" This is ({chr(mcp[0])}|{chr(mcp[1])}) .")
            
            merges[mcp] = ntoken
            tokens = self.merge(tokens, mcp, ntoken)
            
        return tokens, merges
    
    def decode(self, ids, vocab):
 
        tokens = b"".join([vocab[i] for i in ids])

        return tokens.decode("utf-8", errors = "replace")
    
    def merge_for_encoding(self, ids, pair, sub):
        n_ids = []
        i = 0
        while i < len(ids):

            if (i < len(ids)-1) and (ids[i] == pair[0]) and (ids[i+1] == pair[1]):
                n_ids.append(sub)
                i += 2
                
            else:
                n_ids.append(ids[i])
                i += 1
        
        return n_ids
    
    def encode(self, text, merges):

        ids = text.encode("utf-8")

        while len(ids) > 1:

            counts = self.stats(ids)
            pair = min(counts, key = lambda k: merges.get(k, float("inf")))

            if pair not in merges:
                break
                
            ids = self.merge_for_encoding(ids, pair, merges[pair])
        
        return ids
    


