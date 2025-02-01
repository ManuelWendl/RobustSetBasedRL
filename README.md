# SBML Set-Based Machine Learning

This package provides a code base for set-based neural network training on regression and classification tasks as well as set-based reinforcement learning. 
The advantage of set-based neural network training is to obtain certifiably robust models, for bounded input noise.

The package is based on the following papers:

- [End-to-End set-based neural networks](https://arxiv.org/abs/2401.14961)
- [Training Verifiably Robust Agents using Set-Based Reinforcement Learning](https://arxiv.org/abs/2408.09112)

## Structure

The package is structured as follows:

- `SBML` contains the main code base
    - `SBML/ZonoTorch` contains the set-based zonotope extension for PyTorch
    - `SBML/SBRL` contains the set-based reinforcement learning algorithms

- `examples` contains example scripts for regression, classification and reinforcement learning tasks

