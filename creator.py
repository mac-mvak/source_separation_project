import json 
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import shutil


def changer(test_path):
    os.makedirs('data/datasets/mixture/val', exist_ok=True)
    a = 'data/datasets/mixture/val_index.json'
    ans = []
    mix_path = test_path / 'mix'
    target_path = test_path / 'targets'
    ref_path = test_path / 'refs'
    i = 0
    for w in tqdm(os.listdir(mix_path)):
        mix_name = w
        target_name = w[:-9] + 'target.wav'
        ref_name = w[:-9] + 'ref.wav'
        ans_dict = {}
        ans_dict['snr'] = 0
        ans_dict['target_id'] = 0
        ans_dict['noise_id'] = 0
        ans_dict['path_mix'] = 'data/datasets/mixture/val/' + mix_name
        ans_dict['path_target'] = 'data/datasets/mixture/val/' + target_name
        ans_dict['path_ref'] = 'data/datasets/mixture/val/' + ref_name
        ans.append(ans_dict)
        shutil.copy(mix_path/mix_name, ans_dict['path_mix'])
        shutil.copy(ref_path/ref_name, ans_dict['path_ref'])
        shutil.copy(target_path/target_name, ans_dict['path_target'])

    with open(a, 'w') as f:
        json.dump(ans, f, indent=2)

    print(ans_dict)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-t",
        "--test-data-folder",
        default="snr0-20LUFSsmall",
        type=str,
        help="Path to dataset",
    )
    args = args.parse_args()
    test_path = Path(args.test_data_folder)
    changer(test_path)









