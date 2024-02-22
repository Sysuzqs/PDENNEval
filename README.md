# PDENNEval

## Introduction

This is the official code repository for PDENNEval. PDENNEval conducted a comprehensive and systematic evaluation of 12 NN methods for PDEs, including 6 function learning-based NN methods: [DRM](https://arxiv.org/abs/1710.00211), [PINN](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), [WAN](https://arxiv.org/abs/1907.08272), [DFLM](https://arxiv.org/abs/2001.06145), [RFM](https://arxiv.org/abs/2207.13380), [DFVM](https://arxiv.org/abs/2305.06863v2), and 6 operator learning-based NN methods: [U-Net](https://arxiv.org/abs/1505.04597), [MPNN](https://arxiv.org/abs/2202.03376), [FNO](https://arxiv.org/abs/2010.08895), [DeepONet](https://arxiv.org/abs/1910.03193), [PINO](https://arxiv.org/abs/2111.03794), [U-NO](https://arxiv.org/abs/2204.11127). In this repository, we provide code reference for all evaluated methods. If this repository is helpful to your research, please cite our paper.

## Requirement

Our implementation is based on PyTorch. Before starting, make sure you have configured environment.

### Installation

Create a conda environment and install dependencies (ours):
* Python 3.8
* CUDA 11.6
* PyTorch 1.13.1
* PyTorch Geometric (for MPNN)
* DeepXDE 1.10.0 (for PINNs)

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

The data used in our evaluation are from two sources: [PDEBench](https://arxiv.org/abs/2210.07182) and self-generated.

#### PDEBench Data

PDEBench provides large datasets covering wide range PDEs. You can download these datasets from [DaRUS data repository](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986). The data files used in our work are as follows:

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

If you use PDEBench datasets in your reseach, please cite their papers:

[PDEBench: An Extensive Benchmark for Scientific Machine Learning - NeurIPS'2022](https://arxiv.org/abs/2210.07182)

```
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}
```

[PDEBench Datasets - NeurIPS'2022](https://doi.org/10.18419/darus-2986)

```
@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}
```
</details>

#### Self-generated Data

comming soon

## Getting Started

## Contributors

Changye He, [Haolong Fan](https://github.com/fhl2000), Hongji Li, [Jianhuan Cen](https://github.com/12138xs), Liao Chen

## Citation

Our work is based on many previous work. If you use the corresponding codes, please cite their papers. In details:

DeepONet:

MPNN:

FNO:

U-NO:

DeepXDE:

SeqLip: