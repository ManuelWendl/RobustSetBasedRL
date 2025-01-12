"""
PyTorch Zonotope Extension:
===========================

This package extends pytorch to zonotopes and introduces set-based neural network training. 

Modules:
---------
- core: All basic Zonotope implementations are contained in core
- functions: Implements functions for neural network training
"""

from SBRL.PyTorchZonotopeExtension import core
from SBRL.PyTorchZonotopeExtension import functions

__all__ = [
    "core",
    "functions"
]