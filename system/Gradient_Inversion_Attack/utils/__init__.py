"""
Utility functions for GIA

Contains:
- Gradient extraction and processing
- Data loading helpers
- Configuration management
"""

from .gradient_utils import extract_gradients, compute_gradient
from .config import GIAConfig

__all__ = [
    'extract_gradients',
    'compute_gradient',
    'GIAConfig',
]
