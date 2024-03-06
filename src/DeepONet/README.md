# DeepOnet

## Config file

The `config` directory contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml`. The explanations of some DeepOnet specific args are as follows:

* training args:
    * `training_type`: string, 'single' specific the non-autoregressive processing of DeepOnet.
    * `initial_step`: int, the number of input time steps. (default: 1)

* model args:
    * `input_size`: int, specify the length of the spatial grid per dimension, which is to calculate the final input shape of the MLP corresponding to the branch net.  A two-dimension grid of 256*256 should set `input_size`=256.   
    * `in_channels`: int, the number of input channels that equals to the number of variables to be solved. For example, there are 3 variables to be solved for 1D Compressible NS equation: density, pressure and velocity.
    * `out_channels`: int, the number of output channels.
    * `query_dim`: int, the length of a query position for the input of the MLP corresponding to the trunk net.
    
    Considering a 1D example with `input_size`=256,`in_channels`=1, the MLP of brunch net should take a tensor of shape $(N_b,256)$ as the input, where $N_b$ is the batch size and 256 is `input_size`^{`spatial_dimansion`}$\times$`in_channels`$\times$`initial_step`.
  
Training hyperparameters we used are as follows:

| PDE Name                    | spatial resolution / downsample rate (pde) | temporal resolution / downsample rate | lr    | epochs | batch size | weight decay |
| :-------------------------- | :---------------------- | :----------------------- | :---- | :----- | :--------- | :----------- | 
| 1D Advection                | 1024/4                  | 201/5                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Diffusion-Reaction       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Burgers                  | 1024/4                  | 201/5                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Diffusion-Sorption       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Allen Cahn               | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Cahn Hilliard            | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 1D Compressible NS          | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        |
| 2D Burgers                  | 128/1                   | 101/1                    | 1.e-3 | 200    | 50          | 1.e-4        |
| 2D Compressible NS          | 128/1                   | 21/1                     | 1.e-3 | 200    | 50          | 1.e-4        |
| 2D DarcyFlow                | 128/1                   | -                        | 1.e-3 | 500    | 50         | 1.e-4        |
| 2D Shallow Water            | 128/1                   | 101/1                    | 1.e-3 | 200    | 50          | 1.e-4        |
| 2D Allen Cahn               | 128/1                   | 101/1                    | 1.e-3 | 200    | 50          | 1.e-4        |
| 2D Black-Scholes-Barenblatt | 128/1                   | 101/1                    | 1.e-3 | 200    | 50          | 1.e-4        |


## Loss function

DeepOnet solves PDEs in non-autoregressive manner where model $f_{\theta}$ predicts the solution of all time steps $\{\hat{u}^{0}\dots \hat{u}^{T}\}$ based on the solution of initial (`initial_step`=1) time steps $\hat{u}^{0}$. 

The loss function has the form:

$$
\mathcal{L}=\frac{1}{N_b}\sum_{i=1}^{N_b}l(u_{\text{pred}}, u)
$$ 

where $N_b$ is the batchsize, $u$ and $u_{\text{pred}}$ are ground truth solution and predeicted solution respectively, $l$ is MSE in our implementation.

## Train

1. Check the following args in the config file:
    1. The path of `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`;
2. Set hyperparameters for training, such as `lr`, `batch size`, etc. You can use the default values we provide;
3. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/${config file name}
# Example: CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/config_1D_Advection.yaml
```

## Resume training

1. Modify config file:
    1. Make sure `if_training` is `True`;
    2. Set `continue_training` to `True`;
    3. Set `model_path` to the checkpoint path where training restart;
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/${config file name}
```

## Test

1. Modify config file:
    1. Set `if_training` to `False`;
    2. Set `model_path` to the checkpoint path where the model to be evaluated is saved.
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config file name}
```