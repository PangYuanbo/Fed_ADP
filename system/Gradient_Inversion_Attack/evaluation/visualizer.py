"""
Visualization Tools for GI-SMN Attack Results

Creates visual comparisons and plots for attack evaluation.

Author: Fed_ADP Project
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


class GIAVisualizer:
    """
    Visualization utilities for GI-SMN attack results

    Args:
        save_dir (str): Directory to save visualizations
    """

    def __init__(self, save_dir: str = 'gia_results/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_reconstruction_comparison(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        title: str = "Reconstruction Comparison"
    ):
        """
        Plot side-by-side comparison of original vs reconstructed images

        Args:
            original (torch.Tensor): Original images [B, C, H, W]
            reconstructed (torch.Tensor): Reconstructed images [B, C, H, W]
            metrics (dict, optional): Quality metrics to display
            save_path (str, optional): Path to save figure
            title (str): Figure title
        """
        batch_size = min(original.shape[0], 8)  # Show max 8 images

        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size*2, 4))

        # Ensure axes is always 2D array
        if batch_size == 1:
            axes = axes.reshape(2, 1)

        for i in range(batch_size):
            # Original image
            orig_img = self._tensor_to_numpy(original[i])
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # Reconstructed image
            recon_img = self._tensor_to_numpy(reconstructed[i])
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

        # Add metrics if provided
        if metrics:
            metrics_text = f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB\n" \
                          f"SSIM: {metrics.get('ssim', 'N/A'):.4f}\n" \
                          f"Quality: {metrics.get('quality_assessment', 'N/A')}"
            fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Saved comparison to {save_path}")

        plt.close()

    def plot_loss_history(
        self,
        loss_history: List[Dict],
        save_path: Optional[str] = None,
        title: str = "GI-SMN Attack Loss History"
    ):
        """
        Plot loss evolution during attack

        Args:
            loss_history (list): List of loss dictionaries
            save_path (str, optional): Path to save figure
            title (str): Figure title
        """
        iterations = [entry['iteration'] for entry in loss_history]
        total_loss = [entry['total'] for entry in loss_history]
        grad_loss = [entry['gradient'] for entry in loss_history]
        tv_loss = [entry['tv'] for entry in loss_history]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Total loss
        axes[0].plot(iterations, total_loss, label='Total Loss')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Component losses
        axes[1].plot(iterations, grad_loss, label='Gradient Matching')
        axes[1].plot(iterations, tv_loss, label='TV Regularization')
        axes[1].set_ylabel('Component Loss')
        axes[1].set_xlabel('Iteration')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Saved loss history to {save_path}")

        plt.close()

    def plot_defense_impact(
        self,
        defense_results: Dict,
        defense_type: str = 'DP Noise',
        save_path: Optional[str] = None
    ):
        """
        Plot impact of defense mechanism on reconstruction quality

        Args:
            defense_results (dict): Results from DefenseTester
            defense_type (str): Type of defense ('DP Noise', 'Clipping', etc.)
            save_path (str, optional): Path to save figure
        """
        params = []
        psnr_values = []
        ssim_values = []

        for key, value in defense_results.items():
            if key == 'summary':
                continue
            params.append(key)
            if 'metrics' in value:
                psnr_values.append(value['metrics'].get('psnr', 0))
                ssim_values.append(value['metrics'].get('ssim', 0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # PSNR vs defense parameter
        axes[0].plot(params, psnr_values, marker='o')
        axes[0].set_xlabel(f'{defense_type} Parameter')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title(f'PSNR vs {defense_type}')
        axes[0].grid(True, alpha=0.3)
        if defense_type == 'DP Noise':
            axes[0].set_xscale('log')

        # SSIM vs defense parameter
        axes[1].plot(params, ssim_values, marker='o', color='orange')
        axes[1].set_xlabel(f'{defense_type} Parameter')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title(f'SSIM vs {defense_type}')
        axes[1].grid(True, alpha=0.3)
        if defense_type == 'DP Noise':
            axes[1].set_xscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Saved defense impact to {save_path}")

        plt.close()

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization"""
        # Detach and move to CPU
        img = tensor.detach().cpu()

        # Convert from [C, H, W] to [H, W, C]
        if img.dim() == 3:
            img = img.permute(1, 2, 0)

        # Normalize to [0, 1]
        img = img.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        return np.clip(img, 0, 1)


if __name__ == '__main__':
    print("Visualizer implementation complete.")
