# SBML Set-Based Machine Learning

This package provides the Python code for set-based neural network training on regression and classification tasks, as well as set-based reinforcement learning. 
The advantage of set-based neural network training is to obtain certifiably robust models for bounded input noise.

The package is based on the following papers:

- [1] [End-to-End set-based neural networks](https://arxiv.org/abs/2401.14961), Koller et. al, 2024
- [2] [Training Verifiably Robust Agents using Set-Based Reinforcement Learning](https://arxiv.org/abs/2408.09112), Wendl et. al, 2024

With this repository, the results of [2] can be recreated for the MuJoCo Hopper benchmark. 
By using set-based reinforcement learning, the robustness against several attack types can be improved. 

These are the example videos from [2] for the uniform-random attack,and the MAD (maximum action difference) attack.

<img src="examples/Hopper/videosRand.gif" alt="SetBasedRL" style="height: 400px; margin-right: 10px;"> <img src="examples/Hopper/videosMad.gif" alt="SetBasedRL" style="height: 400px; margin-right: 10px;">

## Structure

The code is structured as a Python package:

- `SBML` contains the main code base
    - `SBML/ZonoTorch` contains the set-based zonotope extension for PyTorch
    - `SBML/SBRL` contains the set-based reinforcement learning algorithms

- `examples` contains example scripts for regression, classification, and reinforcement learning tasks

## Installation

This repository uses **Python 3.8.** and [mujoco210](https://github.com/google-deepmind/mujoco). 

All requirements are given in 
```
requirements.txt
```
and can be installed with
```bash
pip install -r requirements.txt
```
Examples can be run in the Jupyter Notebooks in
```
./examples
```
