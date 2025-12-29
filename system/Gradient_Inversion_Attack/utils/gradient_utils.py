"""
Gradient Extraction and Processing Utilities

Provides functions to extract and process gradients from PyTorch models,
optimized for integration with existing Fed_ADP codebase.

Author: Fed_ADP Project
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def extract_gradients(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: Optional[nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract gradients from a model given input images and labels

    Args:
        model (nn.Module): Target model
        images (torch.Tensor): Input images [B, C, H, W]
        labels (torch.Tensor): Target labels [B]
        criterion (nn.Module, optional): Loss function (default: CrossEntropyLoss)

    Returns:
        gradients (dict): Dictionary of gradients {param_name: gradient_tensor}
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Zero out existing gradients
    model.zero_grad()

    # Forward pass
    outputs = model(images)

    # Compute loss
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Extract gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach()

    return gradients


def compute_gradient(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    criterion: Optional[nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients (alias for extract_gradients for compatibility)

    Args:
        model (nn.Module): Target model
        data (torch.Tensor): Input data
        labels (torch.Tensor): Labels
        criterion (nn.Module, optional): Loss function

    Returns:
        gradients (dict): Gradient dictionary
    """
    return extract_gradients(model, data, labels, criterion)


def gradient_cosine_similarity(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute cosine similarity between two gradients

    Args:
        grad1, grad2: Gradient dictionaries

    Returns:
        similarity (float): Cosine similarity in [-1, 1]
    """
    # Flatten all gradients
    flat1 = torch.cat([g.view(-1) for g in grad1.values()])
    flat2 = torch.cat([g.view(-1) for g in grad2.values()])

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0),
        flat2.unsqueeze(0)
    )

    return similarity.item()


def gradient_l2_distance(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute L2 distance between two gradients

    Args:
        grad1, grad2: Gradient dictionaries

    Returns:
        distance (float): L2 distance
    """
    total_dist = 0.0

    for name in grad1.keys():
        if name in grad2:
            dist = torch.norm(grad1[name] - grad2[name])
            total_dist += dist.item() ** 2

    return total_dist ** 0.5


def apply_dp_noise(
    gradient: Dict[str, torch.Tensor],
    noise_scale: float,
    clip_norm: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply differential privacy noise to gradient

    Args:
        gradient (dict): Original gradient
        noise_scale (float): Standard deviation of Gaussian noise
        clip_norm (float, optional): Clip gradient to this L2 norm before adding noise

    Returns:
        noisy_gradient (dict): Gradient with DP noise
    """
    noisy_gradient = {}

    for name, grad in gradient.items():
        # Clip if requested
        if clip_norm is not None:
            norm = torch.norm(grad)
            if norm > clip_norm:
                grad = grad / norm * clip_norm

        # Add Gaussian noise
        noise = torch.randn_like(grad) * noise_scale
        noisy_gradient[name] = grad + noise

    return noisy_gradient


def gradient_magnitude_stats(
    gradient: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute statistics about gradient magnitudes

    Args:
        gradient (dict): Gradient dictionary

    Returns:
        stats (dict): Statistics including mean, std, min, max norms
    """
    norms = []
    for name, grad in gradient.items():
        norms.append(torch.norm(grad).item())

    stats = {
        'mean_norm': sum(norms) / len(norms),
        'max_norm': max(norms),
        'min_norm': min(norms),
        'std_norm': torch.tensor(norms).std().item() if len(norms) > 1 else 0.0,
        'num_params': len(norms)
    }

    return stats


if __name__ == '__main__':
    print("Gradient utilities implementation complete.")
