# PDENNEval

The official code repository for PDENNEval.

## Requirements

### Installation

Create a conda environment and install dependencies (ours):
* python 3.8
* CUDA 11.6
* pytorch 1.13.1
* torch-geometric (for MPNN)
* DeepXDE 1.10.0 (for PINNS)

```bash
# create environment
conda create -n PDENNEval python=3.8 
conda activate PDENNEval

# install pytorch
conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# For PINNs
pip install deepxde # install DeepXDE
conda env config vars set DDE_BACKEND=pytorch # set backend as pytorch

# For MPNN
pip install torch_geometric # install torch geometric
conda install pytorch-cluster -c pyg # install torch cluster

# Other dependencies
pip install h5py # to read dataset file in HDF5 format
pip install tensorboard matplotlib tqdm # visualization
```

### Datasets

The data used in our evaluation are from two sources: PDEBench and self-generated.

#### PDEBench data

Download data file from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986. The data files for each equation are as follows:

| PDE | file name | file size | 
| :--- | :--- | :---: |
| 1D Advection | 1D_Advection_Sols_beta0.1.hdf5 | 7.7G |
| 1D Diffusion-Reaction | ReacDiff_Nu0.5_Rho1.0.hdf5 | 3.9G | 
| 1D Burgers| 1D_Burgers_Sols_Nu0.001.hdf5 | 7.7G |
| 1D Diffusion-Sorption | 1D_diff-sorp_NA_NA.h5 | 4.0G |
| 1D Compressible NS | 1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5 | 12G | 
| 2D Compressible NS | 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 | 52G | 
| 2D Darcy Flow | 2D_DarcyFlow_beta1.0_Train.hdf5 | 1.3G |
| 2D Shallow Water | 2D_rdb_NA_NA.h5 | 6.2G |
| 3D Compressible NS | 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08 _periodic_Train.hdf5 | 83G |

#### self-generated data

comming soon

## Get started

In this work, we provide code reference of 12 advanced NN methods, including 6 function learning-based NN methods: [DRM](https://arxiv.org/abs/1710.00211), [PINN](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), [WAN](https://arxiv.org/abs/1907.08272), [DFLM](https://arxiv.org/abs/2001.06145), [RFM](https://arxiv.org/abs/2207.13380), [DFVM](https://arxiv.org/abs/2305.06863v2), and 6 operator learning-based NN methods: [U-Net](https://arxiv.org/abs/1505.04597), [MPNN](https://arxiv.org/abs/2202.03376), [FNO](https://arxiv.org/abs/2010.08895), [DeepONet](https://arxiv.org/abs/1910.03193), [PINO](https://arxiv.org/abs/2111.03794), [U-NO](https://arxiv.org/abs/2204.11127). The source code can be found in [models](https://github.com/zhouzy36/PDENNEval/tree/main/models). The relevant code files for the most of methods (DeepONet, DFVM, FNO, MPNN, PINNs, PINO, UNet, UNO) are saved in a subfolder named after the method name. And it contains at least the following contents:
* `config`(folder): contains yaml configuration files for solving different PDEs using this method. All arguments for model training and testing are saved in this file.
* `{model name}.py`: contains implementation of evaluated model.
* `train.py`: contains training and testing scripts.
* `utils.py`: contains code related to data reading and some tool functions, such as data format conversion, random seed initialization, timer and etc.

`metrics.py` contains implementation of metric functions shared by most of NN methods.

### Train 

1. Check configuration file:
    1. `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ${config file path}
```

### Resume training

1. Modify configuration file:
    1. Make sure `if_training` is `True`;
    2. Set `continue_training` to `True`;
    3. Set `model_path` to the checkpoint path where traing restart;
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ${config file path}
```

### Test

1. Modify configuration file:
    1. Set `if_training` to `False`;
    2. Set `model_path` to the checkpoint path where the model to be evaluated is saved.
2. Run command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ${config file}
```

## Contributors

Changye He, [Haolong Fan](https://github.com/fhl2000), Hongji Li, [Jianhuan Cen](https://github.com/12138xs), Liao Chen

## Citation