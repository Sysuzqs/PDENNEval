## train:
1. Make sure config file:
    1. `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`
2. Run
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ${config file path}
```

## test
1. Modify config file:
    1. Set `if_training` as `False`;
    2. Set `model_path` as checkpoint path to be evaluated.
2. Run
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ${config file}
```