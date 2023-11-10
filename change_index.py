import json

a = "/home/ubuntu/hw2_copy"
l = len("/home/mac-mvak/code_disk/hw2")
b = "/home/mac-mvak/code_disk/hw2/data/datasets/mixture/train/1018_101_005738_0-mixed.wav"
print(b[l:])

json_path = "/home/ubuntu/hw2_copy/data/datasets/mixture/val_index.json"
ans = []
with open(json_path) as f:
    ww = json.load(f)

for u in ww:
    u['path_mix'] = a + u['path_mix'][l:]
    u['path_target'] = a + u['path_target'][l:]
    u['path_ref'] = a + u['path_ref'][l:]
    ans.append(u)

print(ans[0])
with open(json_path, 'w') as f:
    json.dump(ans, f, indent=2)