"""
Defense Mechanism Tester for GI-SMN Attack

Tests the effectiveness of various defense mechanisms:
- Differential Privacy (DP) noise
- Gradient clipping
- Gradient pruning/sparsification

Author: Fed_ADP Project
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy
from ..core.gi_smn_attack import GISMNAttack
from .metrics import ImageQualityMetrics


class DefenseTester:
    """
    Test GI-SMN attack against various defense mechanisms

    Args:
        attack (GISMNAttack): Configured GI-SMN attack instance
        metrics (ImageQualityMetrics): Metrics calculator
    """

    def __init__(
        self,
        attack: GISMNAttack,
        metrics: Optional[ImageQualityMetrics] = None
    ):
        self.attack = attack
        self.metrics = metrics or ImageQualityMetrics(device=attack.device)

    def test_dp_noise(
        self,
        true_gradient: Dict[str, torch.Tensor],
        target_model: nn.Module,
        ground_truth_images: torch.Tensor,
        noise_levels: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        batch_size: int = 1
    ) -> Dict:
        """
        Test attack against Differential Privacy noise

        Args:
            true_gradient (dict): Clean gradient
            target_model (nn.Module): Target model
            ground_truth_images (torch.Tensor): Original images for comparison
            noise_levels (list): List of noise standard deviations
            batch_size (int): Batch size

        Returns:
            results (dict): Test results for each noise level
        """
        print(f"[Defense Test] Testing DP noise defense...")
        results = {}

        for noise_level in noise_levels:
            print(f"  Testing noise level: {noise_level}")

            # Add DP noise to gradient
            noisy_gradient = self._add_dp_noise(true_gradient, noise_level)

            # Run attack
            attack_result = self.attack.reconstruct_from_gradient(
                noisy_gradient,
                target_model,
                batch_size=batch_size,
                save_intermediate=False
            )

            # Evaluate quality
            quality_metrics = self.metrics.calculate_all(
                attack_result['reconstructed_images'],
                ground_truth_images.cpu()
            )

            results[noise_level] = {
                'noise_level': noise_level,
                'metrics': quality_metrics,
                'final_loss': attack_result['loss_history'][-1]['total']
            }

        # Compute degradation summary
        results['summary'] = self._summarize_defense_impact(results)

        return results

    def test_gradient_clipping(
        self,
        true_gradient: Dict[str, torch.Tensor],
        target_model: nn.Module,
        ground_truth_images: torch.Tensor,
        clip_values: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
        batch_size: int = 1
    ) -> Dict:
        """
        Test attack against gradient clipping defense

        Args:
            true_gradient (dict): Clean gradient
            target_model (nn.Module): Target model
            ground_truth_images (torch.Tensor): Original images
            clip_values (list): List of clipping thresholds
            batch_size (int): Batch size

        Returns:
            results (dict): Test results for each clip value
        """
        print(f"[Defense Test] Testing gradient clipping defense...")
        results = {}

        for clip_value in clip_values:
            print(f"  Testing clip value: {clip_value}")

            # Apply gradient clipping
            clipped_gradient = self._clip_gradient(true_gradient, clip_value)

            # Run attack
            attack_result = self.attack.reconstruct_from_gradient(
                clipped_gradient,
                target_model,
                batch_size=batch_size,
                save_intermediate=False
            )

            # Evaluate quality
            quality_metrics = self.metrics.calculate_all(
                attack_result['reconstructed_images'],
                ground_truth_images.cpu()
            )

            results[clip_value] = {
                'clip_value': clip_value,
                'metrics': quality_metrics,
                'final_loss': attack_result['loss_history'][-1]['total']
            }

        results['summary'] = self._summarize_defense_impact(results)

        return results

    def test_gradient_pruning(
        self,
        true_gradient: Dict[str, torch.Tensor],
        target_model: nn.Module,
        ground_truth_images: torch.Tensor,
        keep_ratios: List[float] = [1.0, 0.5, 0.1, 0.01, 0.001],
        batch_size: int = 1
    ) -> Dict:
        """
        Test attack against gradient pruning defense

        Args:
            true_gradient (dict): Clean gradient
            target_model (nn.Module): Target model
            ground_truth_images (torch.Tensor): Original images
            keep_ratios (list): List of keep ratios (1.0 = 100%)
            batch_size (int): Batch size

        Returns:
            results (dict): Test results for each keep ratio
        """
        print(f"[Defense Test] Testing gradient pruning defense...")
        results = {}

        for keep_ratio in keep_ratios:
            print(f"  Testing keep ratio: {keep_ratio*100:.1f}%")

            # Apply gradient pruning
            pruned_gradient = self._prune_gradient(true_gradient, keep_ratio)

            # Run attack
            attack_result = self.attack.reconstruct_from_gradient(
                pruned_gradient,
                target_model,
                batch_size=batch_size,
                save_intermediate=False
            )

            # Evaluate quality
            quality_metrics = self.metrics.calculate_all(
                attack_result['reconstructed_images'],
                ground_truth_images.cpu()
            )

            results[keep_ratio] = {
                'keep_ratio': keep_ratio,
                'metrics': quality_metrics,
                'final_loss': attack_result['loss_history'][-1]['total']
            }

        results['summary'] = self._summarize_defense_impact(results)

        return results

    def _add_dp_noise(
        self,
        gradient: Dict[str, torch.Tensor],
        noise_std: float
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to gradient (DP)"""
        noisy_gradient = {}
        for name, grad in gradient.items():
            noise = torch.randn_like(grad) * noise_std
            noisy_gradient[name] = grad + noise
        return noisy_gradient

    def _clip_gradient(
        self,
        gradient: Dict[str, torch.Tensor],
        clip_value: float
    ) -> Dict[str, torch.Tensor]:
        """Clip gradient by L2 norm"""
        clipped_gradient = {}
        for name, grad in gradient.items():
            norm = torch.norm(grad)
            if norm > clip_value:
                clipped_gradient[name] = grad / norm * clip_value
            else:
                clipped_gradient[name] = grad.clone()
        return clipped_gradient

    def _prune_gradient(
        self,
        gradient: Dict[str, torch.Tensor],
        keep_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Prune gradient to keep only top-k% values"""
        pruned_gradient = {}
        for name, grad in gradient.items():
            flat_grad = grad.view(-1)
            k = max(1, int(len(flat_grad) * keep_ratio))

            topk_values, topk_indices = torch.topk(flat_grad.abs(), k)
            pruned_flat = torch.zeros_like(flat_grad)
            pruned_flat[topk_indices] = flat_grad[topk_indices]

            pruned_gradient[name] = pruned_flat.view(grad.shape)
        return pruned_gradient

    def _summarize_defense_impact(self, results: Dict) -> Dict:
        """Summarize impact of defense mechanism"""
        summary = {
            'num_tests': len(results) - 1 if 'summary' in results else len(results),
            'baseline_psnr': None,
            'worst_psnr': None,
            'avg_degradation': None
        }

        psnr_values = []
        for key, value in results.items():
            if key != 'summary' and 'metrics' in value:
                psnr = value['metrics'].get('psnr')
                if psnr is not None:
                    psnr_values.append(psnr)

        if psnr_values:
            summary['baseline_psnr'] = max(psnr_values)
            summary['worst_psnr'] = min(psnr_values)
            summary['avg_psnr'] = sum(psnr_values) / len(psnr_values)
            summary['avg_degradation'] = summary['baseline_psnr'] - summary['avg_psnr']

        return summary


if __name__ == '__main__':
    print("Defense Tester implementation complete.")
    print("Usage: See gia_evaluator.py for integration example.")
