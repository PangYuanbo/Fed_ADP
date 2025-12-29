"""
Image Quality Metrics for GI-SMN Attack Evaluation

Implements standard metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Author: Fed_ADP Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
import warnings


class ImageQualityMetrics:
    """
    Compute image quality metrics for reconstruction evaluation

    Supports:
    - PSNR: Peak Signal-to-Noise Ratio (higher is better)
    - SSIM: Structural Similarity Index (higher is better)
    - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)

    Args:
        device (str): Device for computation
        data_range (float): Maximum pixel value (1.0 for normalized images)
    """

    def __init__(self, device: str = 'cuda', data_range: float = 1.0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_range = data_range

        # Try to import LPIPS
        self.lpips_fn = self._init_lpips()

    def _init_lpips(self):
        """Initialize LPIPS metric"""
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            lpips_fn.eval()
            return lpips_fn
        except ImportError:
            warnings.warn(
                "lpips package not found. LPIPS metric will not be available.\n"
                "Install with: pip install lpips"
            )
            return None

    def calculate_psnr(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        data_range: Optional[float] = None
    ) -> float:
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio)

        PSNR = 10 * log10((max_pixel_value)² / MSE)

        Args:
            reconstructed (torch.Tensor): Reconstructed images [B, C, H, W]
            original (torch.Tensor): Original images [B, C, H, W]
            data_range (float, optional): Max pixel value (default: self.data_range)

        Returns:
            psnr (float): PSNR value in dB (higher is better)
                         > 30 dB: high quality
                         20-30 dB: medium quality
                         < 20 dB: low quality
        """
        if data_range is None:
            data_range = self.data_range

        # Ensure tensors are on the same device
        reconstructed = reconstructed.to(self.device)
        original = original.to(self.device)

        # Compute MSE
        mse = F.mse_loss(reconstructed, original)

        # Compute PSNR
        if mse == 0:
            return float('inf')

        psnr = 10 * torch.log10((data_range ** 2) / mse)
        return psnr.item()

    def calculate_ssim(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        window_size: int = 11,
        data_range: Optional[float] = None
    ) -> float:
        """
        Calculate SSIM (Structural Similarity Index)

        SSIM measures structural similarity between images.

        Args:
            reconstructed (torch.Tensor): Reconstructed images [B, C, H, W]
            original (torch.Tensor): Original images [B, C, H, W]
            window_size (int): Window size for SSIM computation (default: 11)
            data_range (float, optional): Max pixel value

        Returns:
            ssim (float): SSIM value in [0, 1] (higher is better)
                         > 0.9: high similarity
                         0.7-0.9: medium similarity
                         < 0.7: low similarity
        """
        if data_range is None:
            data_range = self.data_range

        # Ensure tensors are on the same device
        reconstructed = reconstructed.to(self.device)
        original = original.to(self.device)

        # Use piq library if available, otherwise use simple implementation
        try:
            import piq
            ssim_value = piq.ssim(
                reconstructed,
                original,
                data_range=data_range,
                kernel_size=window_size
            )
            return ssim_value.item()
        except ImportError:
            # Fallback: simple SSIM implementation
            return self._simple_ssim(reconstructed, original, window_size, data_range)

    def _simple_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        data_range: float = 1.0
    ) -> float:
        """
        Simple SSIM implementation (fallback)

        Args:
            img1, img2: Images [B, C, H, W]
            window_size: Window size
            data_range: Max pixel value

        Returns:
            ssim: SSIM value
        """
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # Create Gaussian window
        channel = img1.size(1)
        window = self._create_window(window_size, channel).to(img1.device)

        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def calculate_lpips(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        normalize: bool = True
    ) -> Optional[float]:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)

        LPIPS uses deep features to measure perceptual similarity.

        Args:
            reconstructed (torch.Tensor): Reconstructed images [B, C, H, W]
            original (torch.Tensor): Original images [B, C, H, W]
            normalize (bool): Normalize images to [-1, 1] (default: True)

        Returns:
            lpips (float or None): LPIPS value in [0, 1] (lower is better)
                                  < 0.1: high similarity
                                  0.1-0.3: medium similarity
                                  > 0.3: low similarity
                                  Returns None if lpips not available
        """
        if self.lpips_fn is None:
            warnings.warn("LPIPS not available. Skipping metric.")
            return None

        # Ensure tensors are on the same device
        reconstructed = reconstructed.to(self.device)
        original = original.to(self.device)

        # Normalize to [-1, 1] if needed
        if normalize and reconstructed.min() >= 0:
            reconstructed = reconstructed * 2 - 1
            original = original * 2 - 1

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_fn(reconstructed, original)

        return lpips_value.mean().item()

    def calculate_all(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        data_range: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics

        Args:
            reconstructed (torch.Tensor): Reconstructed images [B, C, H, W]
            original (torch.Tensor): Original images [B, C, H, W]
            data_range (float, optional): Max pixel value

        Returns:
            metrics (dict): Dictionary containing all metrics
        """
        metrics = {}

        # PSNR
        try:
            metrics['psnr'] = self.calculate_psnr(reconstructed, original, data_range)
        except Exception as e:
            warnings.warn(f"Failed to calculate PSNR: {e}")
            metrics['psnr'] = None

        # SSIM
        try:
            metrics['ssim'] = self.calculate_ssim(reconstructed, original, data_range=data_range)
        except Exception as e:
            warnings.warn(f"Failed to calculate SSIM: {e}")
            metrics['ssim'] = None

        # LPIPS
        try:
            metrics['lpips'] = self.calculate_lpips(reconstructed, original)
        except Exception as e:
            warnings.warn(f"Failed to calculate LPIPS: {e}")
            metrics['lpips'] = None

        # Assess quality
        metrics['quality_assessment'] = self.assess_quality(metrics)

        return metrics

    @staticmethod
    def assess_quality(metrics: Dict[str, Optional[float]]) -> str:
        """
        Assess overall reconstruction quality

        Args:
            metrics (dict): Dictionary of metric values

        Returns:
            quality (str): 'high', 'medium', 'low', or 'failed'
        """
        psnr = metrics.get('psnr')
        ssim = metrics.get('ssim')
        lpips = metrics.get('lpips')

        # Count metrics indicating high quality
        high_quality_count = 0
        total_metrics = 0

        if psnr is not None:
            total_metrics += 1
            if psnr > 30:
                high_quality_count += 1

        if ssim is not None:
            total_metrics += 1
            if ssim > 0.9:
                high_quality_count += 1

        if lpips is not None:
            total_metrics += 1
            if lpips < 0.1:
                high_quality_count += 1

        if total_metrics == 0:
            return 'unknown'

        # Determine quality
        ratio = high_quality_count / total_metrics

        if ratio >= 0.67:  # At least 2/3 metrics indicate high quality
            return 'high'
        elif psnr and psnr > 25 or (ssim and ssim > 0.8):
            return 'medium'
        elif psnr and psnr > 20 or (ssim and ssim > 0.7):
            return 'low'
        else:
            return 'failed'


# Example usage
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create metric calculator
    metrics_calculator = ImageQualityMetrics(device=device, data_range=1.0)

    # Generate test images
    original = torch.rand(4, 3, 32, 32, device=device)
    reconstructed = original + torch.randn_like(original) * 0.1  # Add some noise

    # Calculate metrics
    print("Testing image quality metrics:")
    print("="*50)

    # Individual metrics
    psnr = metrics_calculator.calculate_psnr(reconstructed, original)
    print(f"PSNR: {psnr:.2f} dB")

    ssim = metrics_calculator.calculate_ssim(reconstructed, original)
    print(f"SSIM: {ssim:.4f}")

    lpips = metrics_calculator.calculate_lpips(reconstructed, original)
    if lpips is not None:
        print(f"LPIPS: {lpips:.4f}")
    else:
        print("LPIPS: Not available")

    # All metrics at once
    print("\nCalculating all metrics:")
    all_metrics = metrics_calculator.calculate_all(reconstructed, original)
    print(all_metrics)
