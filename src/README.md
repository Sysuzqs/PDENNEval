## Overview

We offer training and testing code for 12 neural network (NN) methods, with each method's code saved in a subdirectory named after the method name. Each subdirectory typically has the following content:
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
* `config` (directory): contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml` where all args for training and testing are included.
* `README.md`: provide guidance to run training and testing.
* `{model name}.py`: contains implementation of evaluated model.
* `train.py`: contains training and testing scripts.
* `utils.py`: contains code related to dataset and some tool functions, such as data format conversion, random seed initialization, timer and etc.

## The explanation for common used config args

* `if_training`: bool, set `True` for training, `False` for testing.
* `continue_training`: bool, set `True` to recover training from checkpoint.
* `model_path`: string, set to checkpoint path when test model at checkpoint or recover training from checkpoint.
* `output_dir`: string, the directory path to save checkpoint.
* `save_period`: int, the frequency of model validation, and the unit is epoch. (default: 20)
* `epochs`: int, the total number of training epochs.

Dataset related args:
* `single_file`: bool, set `False` for 1D Diffusion-Sorption, 2D Burgers, 2D SWE and 3D Maxwell problems. And set `True` for others.
* `file_name`: string, the file name of dataset.
* `saved_folder`: string, the directory path of dataset file.
* `reduced_resolution`: int, spatial downsampling rate.
* `reduced_resolution_t`: int, temporal downsampling rate.
* `reduced_batch`: int, sample downsampling rate. (default: 1)
* `test_ratio`: float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. (default 0.1)

Dataloader related args: parameters for dataloader in PyTorch, such as `batch_size`, `num_workers`, `pin_memory`, etc.

Optimizer related args: include optimzer `name` (default: Adam) supported by PyTorch and its parameters such as `lr` and `weight_decay`.

Learning scheduler related args: include scheduler `name` supported by PyTorch and its parameters. For example, U-Net use "StepLR" scheduler with parameters `step_size` and `gamma`.