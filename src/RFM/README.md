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

#### High Dimension Cases

Reproduce the RFM results in Table 5 of our paper:

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

Improved version:

```python
python RFM_Poisson-PH_improved.py --dimension 3
python RFM_Poisson-PH_improved.py --dimension 5
python RFM_Poisson-PH_improved.py --dimension 10
python RFM_Poisson-PH_improved.py --dimension 20
python RFM_Poisson-PH_improved.py --dimension 40
python RFM_Poisson-PH_improved.py --dimension 80
python RFM_Poisson-PH_improved.py --dimension 120
```

#### Singularity Case

Reproduce the RFM results in Table 6 of our paper:

```python
python RFM_Poisson-PS.py
```

#### L-shape Domain Case

Reproduce the RFM results in Table 6 of our paper:

```python
python RFM_Poisson-PL.py
```

### Hyperparameter Setting

The random seed is set to `2024` for all problems.

For the high dimensional cases, the number of basis functions is `1000`, the scale of network initialization is `1.0`, the number of internal points is `3000*d`, and the number of boundary points is `400*d`, where `d` is the dimension of the equation.

For the singular case, the number of basis functions is `1000`, and the scale of network initialization is `0.5`.

For the L-shape domain case, the number of basis functions is `800`, and the scale of network initialization is `0.1`.

### Results

#### Statement

After reviewing the implementation for solving high-dimensional cases, we identified an error in the original code. We have corrected the error and were fortunate to invite the authors of [RFM](https://arxiv.org/abs/2207.13380) to assist us in verifying the corrected code. We re-ran all the experiments and presented the results before and after the correction below. We apologize for our mistakes. If you have any questions about our work, feel free to contact us!

In addition, the authors of [RFM](https://arxiv.org/abs/2207.13380) pointed out that when solving high-dimensional problems, it is necessary to choose a smaller initialization range that matches the dimensionality of the problem and the number of basis functions. A direct and effective approach is to use Xavier initialization, which adjusts the weights and biases according to network size. We also provide the improved implementation `RFM_Poisson-PH_improved.py` that achieves better performance by simply changing the initialization method of the neural network weights. The improved results are presented below. Thank them for their valuable suggestions!

#### High Dimension Cases

##### L2RE

|  d   | Paper Results | Reproduction Results | Improved Results |
| :--: | :-----------: | :------------------: | :--------------: |
|  3   |    0.0074     |      9.4010E-07      |    1.4247E-08    |
|  5   |    0.0479     |        0.0323        |    1.0439E-05    |
|  10  |    0.3079     |        0.2237        |      0.0546      |
|  20  |    0.5133     |        0.3371        |      0.1864      |
|  40  |    0.5108     |        0.3770        |      0.1361      |
|  80  |    0.5678     |        0.4450        |      0.1153      |
| 120  |    0.6654     |        0.5254        |      0.1058      |

##### Max Error

|  d   | Paper Results | Reproduction Results | Improved Results |
| :--: | :-----------: | :------------------: | :--------------: |
|  3   |    0.0270     |      7.5621E-07      |    1.1441E-08    |
|  5   |    0.0962     |        0.0376        |    1.8226E-05    |
|  10  |    0.5767     |        0.3077        |      0.0769      |
|  20  |    0.5144     |        0.2694        |      0.2371      |
|  40  |    0.3169     |        0.1960        |      0.0974      |
|  80  |    0.1862     |        0.1521        |      0.0507      |
| 120  |    0.1681     |        0.1252        |      0.0419      |

#### Singularity Case

##### L2RE

| Paper Results | Reproduction Results |
| :-----------: | :------------------: |
|    1.0298     |        1.1880        |

##### Max Error

| Paper Results | Reproduction Results |
| :-----------: | :------------------: |
|    0.2853     |        0.3362        |

#### L-shape Domain Case

##### L2RE

| Paper Results | Reproduction Results |
| :-----------: | :------------------: |
|    0.0189     |        0.0193        |

##### Max Error

| Paper Results | Reproduction Results |
| :-----------: | :------------------: |
|    0.0188     |        0.0191        |

## Other PDEs