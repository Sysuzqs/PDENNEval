## enviroments
```
python=3.8
pytorch=1.13.1
cuda=11.6
deepxde=1.9.3
```

## the configuration files are in ./config/
parameters in the config files:

* model_name: the name of the model (PINN)
* scenario: the type of PDE
* data_path: the directory where the data set is stored
* filename: the file name of the data set
* model_update: interval for printing loss information
* learning_rate: the learning rate used during training. We use 1e-3 for all the scenarios of PINN.
* aux_params: parameters used in the PDE.
* val_batch_idx: a int which is the index of the part of the data set we used. In some scenario we use "seed", for the indices in the data set are strings.
* seed: As described above.

## train and test: 
```bash
CUDA_VISIBLE_DEVICES=0 python run.py ${configfile}
```
where ${configfile} is the name of the config file, config_pinn_2dAllen-Cahn.yaml for example.