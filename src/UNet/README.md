## Config files

The `config` directory contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml` where all args for training and testing are saved. The explanations of some U-Net specific args are as follows:
* `training_type`: string, set 'autoregressive' for autoregressive trainging using autoregressive loss or 'single' for single step training using single step loss.
* `pushforward`: bool, set 'True' for pushforward training. And `training_type` also must be set to true at the same time.
* `initial_step`: int, the number of input time steps. (default: 10)
* `unroll_step`: int, the number of time steps to backpropagate in the pushforward training. (default: 20)
* `in_channels`: int, the number of input channels that equals to the number of variables to be solved. For example, there are 3 variables to be solved for 1D Compressible NS equation: density, pressure and velocity.
* `out_channels`: int, the number of output channels that equals to the `in_channels`.
* `init_features`: int, the number of channels in the 1st upsample block of U-Net.

We believe that the meanings of other args are explicit. For reproducibility, we give our training hyperparameters for solving different PDEs.

| PDE Name                    | spatial resolution / downsample rate | temporal resolution / downsample rate | lr    | epochs | batch size | weight decay | initial step | unroll step |
| :-------------------------- | :---------------------- | :----------------------- | :---- | :----- | :--------- | :----------- | :----------- | :---------- |
| 1D Advection                | 1024/4                  | 201/5                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Diffusion-Reaction       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Burgers                  | 1024/4                  | 201/5                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Diffusion-Sorption       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Allen Cahn               | 1024/4                  | 101/1                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Cahn Hilliard            | 1024/4                  | 101/1                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 1D Compressible NS          | 1024/4                  | 101/1                    | 1.e-3 | 500    | 64         | 1.e-4        | 10           | 20          |
| 2D Burgers                  | 128/1                   | 101/1                    | 1.e-3 | 500    | 8          | 1.e-4        | 10           | 20          |
| 2D Compressible NS          | 128/2                   | 21/1                     | 1.e-3 | 500    | 32         | 1.e-4        | 10           | 20          |
| 2D DarcyFlow                | 128/1                   | -                        | 1.e-3 | 500    | 64         | 1.e-4        | 1            | 1           |
| 2D Shallow Water            | 128/1                   | 101/1                    | 1.e-3 | 500    | 8          | 1.e-4        | 10           | 20          |
| 2D Allen Cahn               | 128/1                   | 101/1                    | 1.e-3 | 500    | 8          | 1.e-4        | 10           | 20          |
| 2D Black-Scholes-Barenblatt | 128/1                   | 101/1                    | 1.e-3 | 500    | 8          | 1.e-4        | 10           | 20          |
| 3D Compressible NS          | 128/2                   | 21/1                     | 1.e-3 | 500    | 2          | 1.e-4        | 10           | 20          |
| 3D Eular                    | 128/2                   | 21/1                     | 1.e-3 | 500    | 2          | 1.e-4        | 10           | 20          |
| 3D Maxwell                  | 32                      | 8                        | 1.e-3 | 500    | 2          | 1.e-4        | 2            | -           |

## Train

1. Check the following args in the config file:
    1. The path of `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`;
2. Set hyperparameters for training, such as `lr`, `batch size`, etc. You can use the default values we provide;
3. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config file name}
# Example: CUDA_VISIBLE_DEVICES=0 python train.py ./configs/config_1D_Advection.yaml
```

## Resume training

1. Modify config file:
    1. Make sure `if_training` is `True`;
    2. Set `continue_training` to `True`;
    3. Set `model_path` to the checkpoint path where traing restart;
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config file name}
```

## Test

1. Modify config file:
    1. Set `if_training` to `False`;
    2. Set `model_path` to the checkpoint path where the model to be evaluated is saved.
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config file name}
```

## Estimate lipschitz upper bound

1. Set `model_path` to the checkpoint path where the model to be evaluated is saved.
2. Compute Singular value decomposition (SVD) for model. Run command:
    ```bash
    python get_model_sv.py ./configs/${config file name} ${directory path to save results}
    ```
3. Estimate lipschitz upper bound using SeqLip algorithm. Run command:
    ```bash
    python get_model_sv.py ./configs/${config file name} ${directory path to save results} --n_sv 1
    ```