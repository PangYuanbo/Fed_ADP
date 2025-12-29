"""
Perturbed Sampling (PS) mechanism for FedSel (Algorithm 4).

Randomly samples indices from Top-k or non-Top-k pools with
privacy-preserving probabilities.
"""

import math
import torch


class PerturbedSampling:
    """Private Top-k selector using perturbed sampling."""

    def __init__(self, epsilon: float, delta: float = 0.0, k: int = 100):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.epsilon = epsilon
        self.delta = delta
        self.k = k

    def select_topk(self, gradient_vector: torch.Tensor, k: int = None) -> torch.Tensor:
        if k is None:
            k = self.k

        d = gradient_vector.shape[0]
        if k > d:
            raise ValueError(f"k ({k}) cannot exceed dimension ({d})")

        # Determine deterministic Top-k set (by absolute gradient)
        abs_grad = gradient_vector.abs()
        _, base_topk = torch.topk(abs_grad, k)
        base_topk_list = base_topk.tolist()
        base_topk_set = set(base_topk_list)
        other_pool = [idx for idx in range(d) if idx not in base_topk_set]

        if not other_pool:
            return base_topk

        exp_eps = math.exp(self.epsilon)
        p = exp_eps * len(base_topk_list) / ((d - len(base_topk_list)) + exp_eps * len(base_topk_list))

        selected = []
        top_pool = base_topk_list.copy()
        rng_device = gradient_vector.device

        while len(selected) < k and (top_pool or other_pool):
            choose_top = torch.rand(1, device=rng_device).item() < p
            pool = None
            if choose_top and top_pool:
                pool = top_pool
            elif other_pool:
                pool = other_pool
            elif top_pool:
                pool = top_pool

            if not pool:
                break

            idx = torch.randint(len(pool), (1,)).item()
            choice = pool.pop(idx)
            selected.append(choice)

        if len(selected) < k:
            # Fill remaining slots without replacement
            remaining = top_pool + other_pool
            if remaining:
                perm = torch.randperm(len(remaining))[:k - len(selected)]
                selected.extend([remaining[i] for i in perm.tolist()])

        return torch.tensor(selected[:k], device=gradient_vector.device, dtype=torch.long)
