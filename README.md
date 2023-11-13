# ASR project barebones

## Installation guide

Copy this repo.

```shell
pip install -r ./requirements.txt
gdown https://drive.google.com/uc?id=1sRipq4sY8YHdVhJqQUyp8OSnqP3pv_l6 -O final_data/model.pth
python3 mixture.py
```

Then run scripts using `train.py` and `test.py`. `test.py` will print metrics in `output_metrics.json`.

Now we will define scripts.

1. `config_train.json` -- train model.
2. `config_test.json` -- test model.

If you need to move custom validation into validation folder use `creator.py` which will parse three libraries into one.

## Wandb Report

[Link to report](https://wandb.ai/svak/ss_project/reports/Source-Separation-project--Vmlldzo1OTU4NjE4)

