"""
PyTorch Zonotope:
=================

This package extends pytorch to zonotopes and introduces set-based neural network training. 

Modules:
---------
- train: contains train function for set-based nn training
- layers: implements the different set-based nn layers
- set: contains the zonotope class as set description
- losses: implements a set-based regression and classification loss
"""

from SBML.PyTorchZonotope import train
from SBML.PyTorchZonotope import layers
from SBML.PyTorchZonotope import set
from SBML.PyTorchZonotope import losses


__all__ = [
    "train",
    "layers",
    "set",
    "losses"
]