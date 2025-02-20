"""
SBRL Set-Based Reinforcement Learning:
=================

This package uses a zonotope extension of pytorch to implement set-based machine learning. 

"""

from .buffer import Buffer
from .senv import SetEnvironmnent, GymEnvironment
from .algorithms.ddpg import DDPG

__all__ = ['DDPG', 'Buffer', 'SetEnvironmnent', 'GymEnvironment']