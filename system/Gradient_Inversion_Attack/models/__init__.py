"""
Models module for Gradient Inversion Attack

Contains:
- StyleGAN-XL wrapper for image generation
- Target model definitions and interfaces
"""

from .stylegan_wrapper import StyleGANWrapper

__all__ = ['StyleGANWrapper']
