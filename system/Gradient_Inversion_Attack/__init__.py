"""
Gradient Inversion Attack (GIA) Module

This module implements GI-SMN (Gradient Inversion attack based on Style Migration Network),
a state-of-the-art gradient inversion attack method proposed by Jin Qian et al. (2024).

Key Features:
- StyleGAN-XL based image reconstruction from gradients
- Two-stage optimization with regularization
- Standard image quality metrics (PSNR, SSIM, LPIPS)
- Defense mechanism testing (DP noise, gradient clipping)
- Visualization tools for reconstruction quality

Reference:
    Jin Qian, Kaimin Wei, Yongdong Wu, et al.
    "GI-SMN: Gradient Inversion Attack against Federated Learning without Prior Knowledge"
    arXiv:2405.03516, 2024
"""

__version__ = "0.1.0"
__author__ = "Fed_ADP Project"

from .gia_evaluator import FederatedGIAEvaluator

__all__ = [
    'FederatedGIAEvaluator',
]
