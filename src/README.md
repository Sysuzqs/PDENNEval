## Overview

We offer training and testing code for 12 neural network (NN) methods, with each method's code saved in a subdirectory named after the method name. And each subdirectory typically has the following content:
```bash
.
├── configs
│   ├── config_1D_Advection.yaml
│   ├── ...
├── README.md
├── train.py
├── ${model name}.py
└── utils.py
```
* `config`(directory): contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml` where all args for training and testing are saved.
* `README.md`: provide guidance to run training and testing.
* `{model name}.py`: contains implementation of evaluated model.
* `train.py`: contains training and testing scripts.
* `utils.py`: contains code related to data reading and some tool functions, such as data format conversion, random seed initialization, timer and etc.

## The short explanation for config file content

