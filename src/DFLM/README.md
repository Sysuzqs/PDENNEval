# DFLM

## Poisson equation

### Part 1: high dimension problems

```python
python DFLM_Poisson-Ph.py --dimension 3 --seed 0
# --dimension: dimension of the problem (default: 100)
# --seed: random seed (default: 0)
```

### Part 2: Singularity problem

```python
python DFLM_Poisson-Ps.py --dimension 2 --seed 0
```

### Part 3: L-shape problem

```python
python DFLM_Poisson-Pl.py --dimension 2 --seed 0
```

### Results after training

All metrics are recorded in `{dimension}DIM-DFLM-global-[{day}{hour}{mind}].csv`.

- step: The current iteration number during the training process.
- loss: The total loss value, combining interior and boundary loss components.
- L2: The L2 norm error, measuring the difference between predicted and true values.
- epoch_time: The time taken to complete one training step.
- inference_time: The time taken to perform inference on the test data.
