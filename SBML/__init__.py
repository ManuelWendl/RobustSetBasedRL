"""
SBML Set-Based Machine Learnin:
===============================

This package uses a zonotope extension of pytorch to implement set-based machine learning. 

Modules:
---------
- ZonoTorch: contains the pytorch zonotope extension and implements set-based regressoin and classification with NNs.
- SBRL: contains the set-based reinforement learning implementation based on the pytorch zonotope extension.
"""

from SBML import ZonoTorch
from SBML import SBRL

__all__ = [
    "ZonoTorch"
    "SBRL"
]