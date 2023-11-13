import json 
import os


a = 'val_index.json'
ans = []
i = 0
for w in os.listdir('/home/mac-mvak/code_disk/hw2_copy/aaa/snr0-20LUFSsmall/mix'):
    ans_dict = {}
    ans_dict['snr'] = 0
    ans_dict['target_id'] = 0
    ans_dict['noise_id'] = 0
    ans_dict['path_mix'] = '/home/mac-mvak/code_disk/hw2/data/datasets/mixture/val/' + w
    ans_dict['path_target'] = '/home/mac-mvak/code_disk/hw2/data/datasets/mixture/val/' + w[:-9] + 'target.wav'
    ans_dict['path_ref'] = '/home/mac-mvak/code_disk/hw2/data/datasets/mixture/val/' + w[:-9] + 'ref.wav'
    ans.append(ans_dict)

with open(a, 'w') as f:
    json.dump(ans, f)

print(ans_dict)






