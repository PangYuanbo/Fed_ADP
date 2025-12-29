"""
Exponential Mechanism implementation for FedSel (Algorithm 2).

Selects dimensions proportionally to their rank-based utility scores.
"""

import torch


class ExponentialMechanism:
    """Private Top-k selector using the exponential mechanism."""

    def __init__(self, epsilon: float, delta: float = 0.0, k: int = 100):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.epsilon = epsilon
        self.delta = delta
        self.k = k

    def _compute_rank_probabilities(self, gradient_vector: torch.Tensor) -> torch.Tensor:
        abs_grad = gradient_vector.abs()
        d = abs_grad.shape[0]
        if d == 1:
            return torch.ones(1, device=gradient_vector.device)

        # Obtain ranks (1..d) where larger gradients have higher ranks
        sorted_indices = torch.argsort(abs_grad)  # ascending
        ranks = torch.zeros_like(abs_grad, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(1, d + 1, device=gradient_vector.device, dtype=torch.float32)

        # Utility proportional to rank / (d-1) as described in Algorithm 2
        scale = (ranks - 1) / max(d - 1, 1)
        logits = self.epsilon * scale
        probs = torch.softmax(logits, dim=0)
        return probs

    def select_topk(self, gradient_vector: torch.Tensor, k: int = None) -> torch.Tensor:
        if k is None:
            k = self.k
        d = gradient_vector.shape[0]
        if k > d:
            raise ValueError(f"k ({k}) cannot exceed dimension ({d})")

        probs = self._compute_rank_probabilities(gradient_vector)
        # torch.multinomial expects probs sum to 1; ensure numerical stability
        probs = probs / probs.sum()
        selected = torch.multinomial(probs, num_samples=k, replacement=False)
        return selected
