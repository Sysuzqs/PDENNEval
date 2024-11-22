# PINN

## Poisson equation

`GenerateData.py` provides functions for generating sample points within a cube or sphere using various methods, including uniform mesh, Monte Carlo, and quasi-Monte Carlo (Sobol sequences) sampling techniques.

### Part 1: high dimension problems

```python
python PINN_Poisson-Ph.py --dimension 3 --seed 0
# --dimension: dimension of the problem (default: 100)
# --seed: random seed (default: 0)
```

### Part 2: Singularity problem

```python
python PINN_Poisson-Ps.py --dimension 2 --seed 0
```

### Part 3: L-shape problem

```python
python PINN_Poisson-Pl.py --dimension 2 --seed 0
```


### Results after training

All metrics are recorded in `{DIMENSION}DIM-PINN-{BETA}weight-{NUM_ITERATION}itr-{LEARN_RATE}lr.csv`.

- step: The current iteration number during the training process.
- L2error: The L2 norm error, measuring the difference between predicted and true values.
- MaxError: The maximum error observed between predicted and true values.
- loss: The total loss value, combining interior and boundary loss components.
- elapsed_time: The total time elapsed since the beginning of the training.
- epoch_time: The time taken to complete one training step.
- inference_time: The time taken to perform inference on the test data.


## Other PDEs

### enviroments
```
python=3.8
pytorch=1.13.1
cuda=11.6
deepxde=1.9.3
```

### the configuration files are in ./config/
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

### train and test: 
```bash
CUDA_VISIBLE_DEVICES=0 python run.py ${configfile}
```
where ${configfile} is the name of the config file, config_pinn_2dAllen-Cahn.yaml for example.