"""
Core attack algorithms and loss functions

Contains:
- GI-SMN attack implementation
- Loss functions (gradient matching, regularization)
- Optimization strategies
"""

from .gi_smn_attack import GISMNAttack
from .loss_functions import (
    gradient_matching_loss,
    total_variation_loss,
    l2_regularization,
    group_consistency_loss
)

__all__ = [
    'GISMNAttack',
    'gradient_matching_loss',
    'total_variation_loss',
    'l2_regularization',
    'group_consistency_loss',
]
