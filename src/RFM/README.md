# RFM

## Installation
```
* Python 3.8
* CUDA 11.6
* PyTorch 1.13.1
* DeepXDE
* scipy
* scikit-learn
```

## Poisson Problems

### Training Scripts

#### Part 1: High Dimension Problems

Reproduce the RFM results in Table 5 in our paper:

```python
# Argument '--dimension' specifics the dimension of the problem to solve
python RFM_Poisson-PH.py --dimension 3
python RFM_Poisson-PH.py --dimension 5
python RFM_Poisson-PH.py --dimension 10
python RFM_Poisson-PH.py --dimension 20
python RFM_Poisson-PH.py --dimension 40
python RFM_Poisson-PH.py --dimension 80
python RFM_Poisson-PH.py --dimension 120
```

#### Part 2: Singularity Problem

Reproduce the RFM results in Table 6 in our paper:

```python
python RFM_Poisson-PS.py
```

#### Part 3: L-shape Problem

Reproduce the RFM results in Table 6 in our paper:

```python
python RFM_Poisson-PL.py
```