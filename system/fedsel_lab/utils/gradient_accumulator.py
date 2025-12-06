"""
Gradient Accumulator for FedSel

Implements momentum-based gradient accumulation to stabilize Top-k dimension selection.

Formula: r_t = β * r_{t-1} + g_t

Where:
- r_t: Accumulated gradient at round t
- g_t: Current gradient at round t
- β: Decay factor (typically 0.9)

This accumulation helps to:
1. Smooth out noisy gradient estimates
2. Stabilize Top-k selection across rounds
3. Reduce variance in private dimension selection
"""

import torch
from typing import Dict


class GradientAccumulator:
    """
    Accumulates gradients across training rounds using exponential moving average.

    Attributes:
        beta (float): Decay factor for exponential moving average (0 < β < 1)
        accumulated_grads (Dict[str, torch.Tensor]): Dictionary storing accumulated gradients
    """

    def __init__(self, model, beta=0.9):
        """
        Initialize the gradient accumulator.

        Args:
            model: PyTorch model (used to initialize gradient shapes)
            beta (float): Decay factor for accumulation (default: 0.9)
                         Typical range: [0.7, 0.95]
                         - Higher β: More history, smoother gradients
                         - Lower β: More responsive to recent changes
        """
        if not (0 < beta < 1):
            raise ValueError(f"Beta must be in (0, 1), got {beta}")

        self.beta = beta
        self.accumulated_grads = {}

        # Initialize accumulated gradients to zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.accumulated_grads[name] = torch.zeros_like(param.data)

    def accumulate(self, current_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update accumulated gradients with current gradients.

        Formula: r_t = β * r_{t-1} + g_t

        Args:
            current_grads: Dictionary mapping parameter names to current gradient tensors

        Returns:
            Dictionary of updated accumulated gradients
        """
        for name, grad in current_grads.items():
            if name in self.accumulated_grads:
                # Core accumulation formula
                self.accumulated_grads[name] = (
                    self.beta * self.accumulated_grads[name] + grad
                )
            else:
                # First occurrence: directly assign (no history)
                self.accumulated_grads[name] = grad.clone()

        return self.accumulated_grads

    def get_accumulated(self) -> Dict[str, torch.Tensor]:
        """
        Get current accumulated gradients.

        Returns:
            Dictionary of accumulated gradient tensors
        """
        return self.accumulated_grads

    def reset(self):
        """Reset all accumulated gradients to zero."""
        for name in self.accumulated_grads:
            self.accumulated_grads[name].zero_()

    def reset_param(self, param_name: str):
        """
        Reset accumulated gradient for a specific parameter.

        Args:
            param_name: Name of the parameter to reset
        """
        if param_name in self.accumulated_grads:
            self.accumulated_grads[param_name].zero_()

    def get_beta(self) -> float:
        """Get the current beta value."""
        return self.beta

    def set_beta(self, new_beta: float):
        """
        Update the beta value.

        Args:
            new_beta: New decay factor (must be in (0, 1))
        """
        if not (0 < new_beta < 1):
            raise ValueError(f"Beta must be in (0, 1), got {new_beta}")
        self.beta = new_beta
