import json
import pandas as pd
import os

wlasl = json.load(open("data/WLASL_v0.3.json", "r"))

data = {'file_id':[],'sequence_id':[],'phrase':[]}


vid_per_file = 1000

videos = os.listdir("data/videos")
indexes = [i for i in range(0, len(videos),vid_per_file)]

seq_to_file = dict()
for idx, i in enumerate(range(0, 12000, vid_per_file)):
    print(f"Processing {i} to {i+vid_per_file}")
    sel = videos[i:i+vid_per_file]
    for s in sel:
        seq_to_file[s.split('.')[0]] = idx

    
# print(seq_to_file.values())


# print('692'videos[11000:12000])

for entry in (wlasl):
    for inst in entry['instances']:
        try:
            data['file_id'].append(seq_to_file[inst['video_id']])
            data['phrase'].append(entry['gloss'])
            data['sequence_id'].append(inst['video_id'])
        except KeyError:
            print("Error with ", inst['video_id'])
            continue
data = pd.DataFrame(data)
data.to_csv("data/sequences.csv", index = False)
        

