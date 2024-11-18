# PDENNEval

This is the official code repository for [PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/573). The appendix can be found in [here](https://github.com/Sysuzqs/PDENNEval/blob/main/PDENNEval_appendix.pdf).

## Introduction

PDENNEval conducts a comprehensive and systematic evaluation of 12 NN methods for PDEs, including 6 function learning-based NN methods: [DRM](https://arxiv.org/abs/1710.00211), [PINN](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), [WAN](https://arxiv.org/abs/1907.08272), [DFLM](https://arxiv.org/abs/2001.06145), [RFM](https://arxiv.org/abs/2207.13380), [DFVM](https://arxiv.org/abs/2305.06863v2), and 6 operator learning-based NN methods: [U-Net](https://arxiv.org/abs/1505.04597), [MPNN](https://arxiv.org/abs/2202.03376), [FNO](https://arxiv.org/abs/2010.08895), [DeepONet](https://arxiv.org/abs/1910.03193), [PINO](https://arxiv.org/abs/2111.03794), [U-NO](https://arxiv.org/abs/2204.11127). In this repository, we provide code reference for all evaluated methods. If this repository is helpful to your research, please cite our paper.

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

| PDE | File Name | File Size | 
| :--- | :---: | :---: |
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

<details>
<summary>
    <a href="https://arxiv.org/abs/2210.07182">PDEBench: An Extensive Benchmark for Scientific Machine Learning - NeurIPS'2022 </a>
</summary>
<br/>

```
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}
```

</details>


<details>
<summary>
    <a href="https://doi.org/10.18419/darus-2986">PDEBench Datasets - NeurIPS'2022 </a>
</summary>
<br/>

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

| PDE | File Size | Download Link | 
| :--- | :--- | :---: |
| 1D Allen-Cahn Equation | 3.9G | [AI4SC Website](http://aisccc.cn/database/data-details?id=52&type=resource), [Google Drive](https://drive.google.com/file/d/10Ee3EqcZyAE0s1Q_rO5XbrRwYZDYKf-o/view?usp=drive_link) |
| 1D Cahn-Hilliard Equation | 3.9G | [AI4SC Website](http://aisccc.cn/database/data-details?id=48&type=resource), [Google Drive](https://drive.google.com/file/d/10D25VbDAnYtEOxSO18pw_o_BteBn5MnB/view?usp=drive_link) | 
| 2D Allen-Cahn Equation | 6.2G | [AI4SC Website](http://aisccc.cn/database/data-details?id=56&type=resource), [Google Drive](https://drive.google.com/file/d/11AA7RAts9ErTaY7Qk4W3TWFu6OKp1RJx/view?usp=drive_link) |
| 2D Black-Scholes-Barenblatt Equation | 6.2G | [AI4SC Website](http://aisccc.cn/database/data-details?id=53&type=resource), [Google Drive](https://drive.google.com/file/d/11WgIOSYR6UKk16G_NK82k0Y_QM7azQlH/view?usp=drive_link) |
| 2D Burgers Equation | 12.3G | AI4SC Website, [Google Drive](https://drive.google.com/file/d/11ICqL_oK52nCW5u3r31WUINtbPnRHGeo/view?usp=drive_link) | 
| 3D Euler Equation | 83G | [AI4SC Website](http://aisccc.cn/database/data-details?id=54&type=resource), [Google Drive](https://drive.google.com/file/d/11aRPB5RdoDOH8nef3J96RpJANV-WhNv1/view?usp=drive_link) | 
| 3D Maxwell Equation | 5.9G | [AI4SC Website](http://aisccc.cn/database/data-details?id=55&type=resource), [Google Drive](https://drive.google.com/file/d/11b3p8zqEu1vawtZkI1ThCSilYAz1Hyya/view?usp=drive_link) | 

## Getting Started

### Train and Test

Our implementation is saved in the `src` directory. The relevant code files for each methods are saved in a subdirectorys named after the method name. If you want to evaluate a certain method, please enter the corresponding subdirectory. A detailed guidance is provided to help you running training and testing.

### Estimate Lipschitz Upper Bound

We use [SeqLip](https://arxiv.org/abs/1805.10965) algorithm to estimate the Lipschitz upper bound of trained neural networks. Specifically, we provide estimation scripts for UNet, DeepONet, and all methods that only use MLP. You can find these scripts in the folder corresponding to each method.

## Contributors

[Changye He](https://github.com/Hechy23), [Haolong Fan](https://github.com/fhl2000), [Hongji Li](https://github.com/Lowbcgz), [Jianhuan Cen](https://github.com/12138xs), [Liao Chen](https://github.com/liaochenl), [Ping Wei](http://github.com/weip7), [Ziyang Zhou](https://github.com/zhouzy36)

## Citation

Our work is based on many previous work. If you use the corresponding codes, please cite their papers. In details:

<details>
<summary>
DeepONet
</summary>
<br/>

```
@article{lu2021learning,
  title={Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators},
  author={Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  journal={Nature machine intelligence},
  volume={3},
  number={3},
  pages={218--229},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```
</details>

<details>
<summary>
MPNN
</summary>
<br/>

```
@article{brandstetter2022message,
  title={Message passing neural PDE solvers},
  author={Brandstetter, Johannes and Worrall, Daniel and Welling, Max},
  journal={arXiv preprint arXiv:2202.03376},
  year={2022}
}
```
</details>

<details>
<summary>
FNO
</summary>
<br/>

```
@article{li2020fourier,
  title={Fourier neural operator for parametric partial differential equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2010.08895},
  year={2020}
}
```
</details>

<details>
<summary>
U-NO
</summary>
<br/>

```
@article{rahman2022u,
  title={U-no: U-shaped neural operators},
  author={Rahman, Md Ashiqur and Ross, Zachary E and Azizzadenesheli, Kamyar},
  journal={arXiv preprint arXiv:2204.11127},
  year={2022}
}
```
</details>

<details>
<summary>
DeepXDE
</summary>
<br/>

```
@article{lu2021deepxde,
  title={DeepXDE: A deep learning library for solving differential equations},
  author={Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  journal={SIAM review},
  volume={63},
  number={1},
  pages={208--228},
  year={2021},
  publisher={SIAM}
}
```
</details>

<details>
<summary>
SeqLip
</summary>
<br/>

```
@article{virmaux2018lipschitz,
  title={Lipschitz regularity of deep neural networks: analysis and efficient estimation},
  author={Virmaux, Aladin and Scaman, Kevin},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```
</details>
