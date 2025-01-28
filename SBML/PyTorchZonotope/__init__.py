"""
PyTorch Zonotope:
=================

This package extends pytorch to zonotopes and introduces set-based neural network training. 

Modules:
---------
- core: All basic Zonotope implementations are contained in core
- functions: Implements functions for neural network training
"""

from SBML.PyTorchZonotope import core
from SBML.PyTorchZonotope import functions

__all__ = [
    "core",
    "functions"
]