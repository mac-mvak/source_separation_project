import os
import json
import tqdm

index_path = '/home/mac-mvak/code_disk/hw2/data/datasets/librispeech/train-clean-360_index.json'
cur_path = os.getcwd()


with open(index_path, 'r') as f:
    files = json.load(f)

for i in range(len(files)):
    now_path = files[i]['path']
    splitted = now_path.split('/', 5)
    files[i]['path'] = cur_path + '/' + splitted[-1]

with open(index_path, 'w') as f:
    json.dump(files, f, indent=2)


