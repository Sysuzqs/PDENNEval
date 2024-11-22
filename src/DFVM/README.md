# DFVM

## Poisson equation

`GenerateData.py` provides functions for generating sample points within a cube or sphere using various methods, including uniform mesh, Monte Carlo, and quasi-Monte Carlo (Sobol sequences) sampling techniques.

### Part 1: high dimension problems

```python
python DFVM_Poisson-Ph.py --dimension 3 --seed 0
# --dimension: dimension of the problem (default: 100)
# --seed: random seed (default: 0)
```

### Part 2: Singularity problem

```python
python DFVM_Poisson-Ps.py --dimension 2 --seed 0
```

### Part 3: L-shape problem

```python
python DFVM_Poisson-Pl.py --dimension 2 --seed 0
```

### Results after training

All metrics are recorded in `{DIMENSION}DIM-DFVM-{BETA}weight-{NUM_ITERATION}itr-{EPSILON}R-{BDSIZE}bd-{LEARN_RATE}lr.csv`.

- step: The current iteration number during the training process.
- L2error: The L2 norm error, measuring the difference between predicted and true values.
- MaxError: The maximum error observed between predicted and true values.
- loss: The total loss value, combining interior and boundary loss components.
- elapsed_time: The total time elapsed since the beginning of the training.
- epoch_time: The time taken to complete one training step.
- inference_time: The time taken to perform inference on the test data.
