"""
ZonoTorch:
==========

This package extends pytorch to zonotopes and introduces set-based neural network training. 

Modules:
---------
- train: train function for set-based nn training
- layers: implements the different set-based nn layers
- set: contains the zonotope class as set description
- losses: implements a set-based regression and classification loss
"""

from SBML.ZonoTorch.train import train
from SBML.ZonoTorch import layers
from SBML.ZonoTorch import set
from SBML.ZonoTorch import losses


__all__ = [
    "train",
    "layers",
    "set",
    "losses"
]