"""
Loss Functions for GI-SMN Attack

Implements the two-stage loss strategy:
1. Stage 1 (0 < t < 4T/9): Pure gradient matching
2. Stage 2 (4T/9 < t < T): Gradient matching + regularization

Author: Fed_ADP Project
Reference: Jin Qian et al., "GI-SMN", arXiv:2405.03516, 2024
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def gradient_matching_loss(
    dummy_gradient: Dict[str, torch.Tensor],
    true_gradient: Dict[str, torch.Tensor],
    norm: str = 'fro'
) -> torch.Tensor:
    """
    Compute gradient matching loss (Frobenius norm)

    This is the core loss function that minimizes the distance between
    the gradient computed from dummy images and the intercepted true gradient.

    L_grad = ||∇w* - ∇w||²_F

    Args:
        dummy_gradient (dict): Gradient from dummy images
        true_gradient (dict): Intercepted true gradient
        norm (str): Norm type ('fro' for Frobenius, 'l2', 'l1')

    Returns:
        loss (torch.Tensor): Gradient matching loss (scalar)
    """
    total_loss = 0.0
    num_layers = 0

    for name in true_gradient.keys():
        if name not in dummy_gradient:
            continue

        grad_true = true_gradient[name]
        grad_dummy = dummy_gradient[name]

        # Compute norm-based loss
        if norm == 'fro':
            # Frobenius norm (default in GI-SMN)
            loss = torch.norm(grad_dummy - grad_true, p='fro') ** 2
        elif norm == 'l2':
            loss = torch.norm(grad_dummy - grad_true, p=2) ** 2
        elif norm == 'l1':
            loss = torch.norm(grad_dummy - grad_true, p=1)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        total_loss += loss
        num_layers += 1

    # Average over layers
    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    """
    Total Variation (TV) regularization

    Encourages smoothness by penalizing pixel-wise differences.

    R_TV = Σ|x[i,j] - x[i+1,j]| + |x[i,j] - x[i,j+1]|

    Args:
        images (torch.Tensor): Images of shape [B, C, H, W]

    Returns:
        tv_loss (torch.Tensor): TV loss (scalar)
    """
    batch_size = images.shape[0]

    # Horizontal differences
    diff_h = torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :])
    # Vertical differences
    diff_v = torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:])

    # Sum and average
    tv_loss = (diff_h.sum() + diff_v.sum()) / batch_size

    return tv_loss


def l2_regularization(images: torch.Tensor) -> torch.Tensor:
    """
    L2 regularization on image pixels

    Constrains the magnitude of individual images.

    R_L2 = ||x*||²_2

    Args:
        images (torch.Tensor): Images of shape [B, C, H, W]

    Returns:
        l2_loss (torch.Tensor): L2 loss (scalar)
    """
    batch_size = images.shape[0]
    l2_loss = torch.norm(images, p=2) ** 2 / batch_size
    return l2_loss


def group_consistency_loss(images: torch.Tensor) -> torch.Tensor:
    """
    Group consistency regularization

    Penalizes deviation from the group mean (for batch attacks).

    R_group = Σ||x*_i - mean(x*)||²_2

    Args:
        images (torch.Tensor): Images of shape [B, C, H, W]

    Returns:
        group_loss (torch.Tensor): Group consistency loss (scalar)
    """
    if images.shape[0] <= 1:
        # No group for single image
        return torch.tensor(0.0, device=images.device)

    # Compute mean image
    mean_image = images.mean(dim=0, keepdim=True)  # [1, C, H, W]

    # Compute deviation from mean
    deviations = images - mean_image  # [B, C, H, W]
    group_loss = torch.norm(deviations, p=2) ** 2 / images.shape[0]

    return group_loss


def cosine_similarity_loss(
    dummy_gradient: Dict[str, torch.Tensor],
    true_gradient: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Cosine similarity loss between gradients

    Alternative to Frobenius norm, measures angle between gradient vectors.

    L_cos = 1 - cosine_similarity(∇w*, ∇w)

    Args:
        dummy_gradient (dict): Gradient from dummy images
        true_gradient (dict): Intercepted true gradient

    Returns:
        cos_loss (torch.Tensor): Cosine similarity loss (scalar)
    """
    total_similarity = 0.0
    num_layers = 0

    for name in true_gradient.keys():
        if name not in dummy_gradient:
            continue

        grad_true = true_gradient[name].view(-1)
        grad_dummy = dummy_gradient[name].view(-1)

        # Cosine similarity
        similarity = F.cosine_similarity(
            grad_dummy.unsqueeze(0),
            grad_true.unsqueeze(0),
            dim=1
        )

        total_similarity += similarity
        num_layers += 1

    # Average similarity
    if num_layers > 0:
        avg_similarity = total_similarity / num_layers
    else:
        avg_similarity = torch.tensor(0.0)

    # Convert to loss (minimize dissimilarity)
    cos_loss = 1.0 - avg_similarity

    return cos_loss


class GISMNLoss:
    """
    Combined loss function for GI-SMN attack with two-stage strategy

    Stage 1 (0 < t < 4T/9): Pure gradient matching
    Stage 2 (4T/9 < t < T): Gradient matching + regularization

    Args:
        total_iterations (int): Total number of optimization iterations
        lambda_tv (float): Weight for total variation loss
        lambda_l2 (float): Weight for L2 regularization
        lambda_group (float): Weight for group consistency loss
        stage1_ratio (float): Ratio of stage 1 (default: 4/9 ≈ 0.444)
        gradient_norm (str): Norm type for gradient matching
    """

    def __init__(
        self,
        total_iterations: int = 10000,
        lambda_tv: float = 0.001,
        lambda_l2: float = 0.0001,
        lambda_group: float = 0.01,
        stage1_ratio: float = 4/9,
        gradient_norm: str = 'fro'
    ):
        self.total_iterations = total_iterations
        self.lambda_tv = lambda_tv
        self.lambda_l2 = lambda_l2
        self.lambda_group = lambda_group
        self.stage1_end = int(total_iterations * stage1_ratio)
        self.gradient_norm = gradient_norm

        self.current_iteration = 0

    def __call__(
        self,
        dummy_gradient: Dict[str, torch.Tensor],
        true_gradient: Dict[str, torch.Tensor],
        dummy_images: torch.Tensor,
        iteration: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss based on current iteration stage

        Args:
            dummy_gradient (dict): Gradient from dummy images
            true_gradient (dict): Intercepted true gradient
            dummy_images (torch.Tensor): Current dummy images [B, C, H, W]
            iteration (int, optional): Current iteration number

        Returns:
            loss_dict (dict): Dictionary containing:
                - 'total': Total loss
                - 'gradient': Gradient matching loss
                - 'tv': TV loss (0 in stage 1)
                - 'l2': L2 loss (0 in stage 1)
                - 'group': Group consistency loss (0 in stage 1)
        """
        if iteration is not None:
            self.current_iteration = iteration

        # Gradient matching loss (always present)
        grad_loss = gradient_matching_loss(
            dummy_gradient,
            true_gradient,
            norm=self.gradient_norm
        )

        loss_dict = {
            'gradient': grad_loss,
            'tv': torch.tensor(0.0, device=dummy_images.device),
            'l2': torch.tensor(0.0, device=dummy_images.device),
            'group': torch.tensor(0.0, device=dummy_images.device),
        }

        # Stage 1: Pure gradient matching
        if self.current_iteration < self.stage1_end:
            total_loss = grad_loss

        # Stage 2: Add regularization
        else:
            tv_loss = total_variation_loss(dummy_images)
            l2_loss = l2_regularization(dummy_images)
            group_loss = group_consistency_loss(dummy_images)

            total_loss = (
                grad_loss +
                self.lambda_tv * tv_loss +
                self.lambda_l2 * l2_loss +
                self.lambda_group * group_loss
            )

            loss_dict['tv'] = tv_loss
            loss_dict['l2'] = l2_loss
            loss_dict['group'] = group_loss

        loss_dict['total'] = total_loss
        self.current_iteration += 1

        return loss_dict

    def reset(self):
        """Reset iteration counter"""
        self.current_iteration = 0


# Example usage
if __name__ == '__main__':
    # Test loss functions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dummy gradients
    dummy_grad = {
        'conv1.weight': torch.randn(64, 3, 3, 3, device=device),
        'fc.weight': torch.randn(10, 128, device=device),
    }
    true_grad = {
        'conv1.weight': torch.randn(64, 3, 3, 3, device=device),
        'fc.weight': torch.randn(10, 128, device=device),
    }

    # Dummy images
    images = torch.randn(4, 3, 32, 32, device=device)

    # Test individual losses
    print("Testing individual loss functions:")
    print(f"Gradient matching loss: {gradient_matching_loss(dummy_grad, true_grad):.4f}")
    print(f"Total variation loss: {total_variation_loss(images):.4f}")
    print(f"L2 regularization: {l2_regularization(images):.4f}")
    print(f"Group consistency loss: {group_consistency_loss(images):.4f}")

    # Test combined loss
    print("\nTesting combined GI-SMN loss:")
    loss_fn = GISMNLoss(total_iterations=1000, stage1_ratio=0.444)

    # Stage 1 (iteration 100)
    loss_dict = loss_fn(dummy_grad, true_grad, images, iteration=100)
    print(f"Stage 1 (iter 100): Total={loss_dict['total']:.4f}, "
          f"Grad={loss_dict['gradient']:.4f}, TV={loss_dict['tv']:.4f}")

    # Stage 2 (iteration 600)
    loss_dict = loss_fn(dummy_grad, true_grad, images, iteration=600)
    print(f"Stage 2 (iter 600): Total={loss_dict['total']:.4f}, "
          f"Grad={loss_dict['gradient']:.4f}, TV={loss_dict['tv']:.4f}")
