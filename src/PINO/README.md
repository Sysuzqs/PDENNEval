# PINO

## Config file

The `config` directory contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml`. The explanations of some PINO specific args are as follows:

* training args:
    * `training_type`: string, 'single' specific the non-autoregressive processing of PINO.
    * `initial_step`: int, the number of input time steps. (default: 1)

* model args:
    * `modes`: list[int], where the length of the list specifics the number of Fourier blocks, and the values specific the Fourier modes in FNO architecture. Notice that there are multiple modes to be set, i.e. `modes1`,`modes2`,` modes3` for 3D architecture, like 3D spatial model or 2D spatial+1D temporal model, where the last dim is for temporal.
    * `width`: int, number of channels for the Fourier layer.
    * `in_channels`: int, the number of input channels that equals to the number of variables to be solved. For example, there are 3 variables to be solved for 1D Compressible NS equation: density, pressure and velocity.
    * `out_channels`: int, the number of output channels
* datasets:
    * `reduced_resolution`: int, the resolution downsampling rate on spatial for data-driving part in PINO. For example, if the data provides spatial resolution of 1024 and set `reduced_resolution`=16, then use resolution of 1024/16= 64 for data-driven part.
    * `reduced_resolution_t`: int, the resolution downsampling rate on temporal for data-driven part in PINO.
    * `reduced_resolution_pde`: int, the resolution downsampling rate on spatial for physics-driven part in PINO.
    * `reduced_resolution_pde_t`: int, the resolution downsampling rate on temporal for physics-driven part in PINO.
    
    Note that here we provide 4x physics/data resolution rates on spatial and 1x on temporal for training PINO, so by default `reduced_resolution` is 4 times `reduced_resolution_pde`.   
* loss weights:
    * ic_loss: float, default 2.0, the weight of initial conditions loss for physics-driven part.
    * f_loss: float, default 1.0, the weight of equations loss for physics-driven part.
    * xy_loss: float, default 10.0, the weight of data-driven loss.
Training hyperparameters we used are as follows:

| PDE Name                    | spatial resolution / downsample rate (pde) | temporal resolution / downsample rate | lr    | epochs | batch size | weight decay | width | modes |
| :-------------------------- | :---------------------- | :----------------------- | :---- | :----- | :--------- | :----------- | :----------- | :---------- |
| 1D Advection                | 1024/4                  | 201/5                    | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 1D Diffusion-Reaction       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 1D Burgers                  | 1024/4                  | 201/5                    | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 1D Diffusion-Sorption       | 1024/4                  | 101/1                    | 1.e-3 | 500    | 20         | 1.e-4        | 32          | 12          |
| 1D Allen Cahn               | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 1D Cahn Hilliard            | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 1D Compressible NS          | 1024/4                  | 101/1                    | 1.e-3 | 500    | 50         | 1.e-4        | 64          | 12          |
| 2D Burgers                  | 128/1                   | 101/1                    | 1.e-3 | 200    | 2          | 1.e-4        | 64          | 12          |
| 2D Compressible NS          | 128/1                   | 21/1                     | 1.e-3 | 200    | 8          | 1.e-4        | 64          | 12          |
| 2D DarcyFlow                | 128/1                   | -                        | 1.e-3 | 500    | 50         | 1.e-4        | 32          | 12          |
| 2D Shallow Water            | 128/1                   | 101/1                    | 1.e-3 | 200    | 2          | 1.e-4        | 64          | 12          |
| 2D Allen Cahn               | 128/1                   | 101/1                    | 1.e-3 | 200    | 2          | 1.e-4        | 64          | 12          |
| 2D Black-Scholes-Barenblatt | 128/1                   | 101/1                    | 1.e-3 | 200    | 2          | 1.e-4        | 64          | 12          |


## Loss function

PINO solves PDEs in non-autoregressive manner where model $f_{\theta}$ predicts the solution of all time steps $\{\hat{u}^{0}\dots \hat{u}^{T}\}$ based on the solution of initial (`initial_step`=1) time steps $\hat{u}^{0}$. 

The loss function has the form:

$$
\mathcal{L}=\frac{1}{N_b}\sum_{i=1}^{N_b}(W_{1}*l(u_{\text{pred}}, u) + W_{2}*l(\mathcal{F}(u_{\text{pred}}),\mathcal{F}(u))+ W_{3}*l(u^{0}_{\text{pred}},u^{0}))
$$ 
where $N_b$ is the batchsize, $u$ and $u_{\text{pred}}$ are the ground truth solution and the predicted solution respectively, $\mathcal{F}$ represents the PDE operators and $l$ is MSE in our implementation. $W_{1}$,$W_{2}$,$W_{3}$ are the weight for data, physics and initial condition, respectively.

## Train

1. Check the following args in the config file:
    1. The path of `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`;
2. Set hyperparameters for training, such as `lr`, `batch size`, etc. You can use the default values we provide;
3. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/4x/${config file name}
# Example: CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/4x/config_1D_Advection.yaml
```

## Resume training

1. Modify config file:
    1. Make sure `if_training` is `True`;
    2. Set `continue_training` to `True`;
    3. Set `model_path` to the checkpoint path where training restart;
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/4x/${config file name}
```

## Test

1. Modify config file:
    1. Set `if_training` to `False`;
    2. Set `model_path` to the checkpoint path where the model to be evaluated is saved.
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/4x/${config file name}
```