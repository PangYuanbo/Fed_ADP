"""
Privacy Budget Allocator for FedSel

Automatically allocates total privacy budget ε into:
- ε1: Budget for dimension selection (Stage 1)
- ε2: Budget for value perturbation (Stage 2)

The allocation strategy depends on:
- Dimension d
- Top-k value
- Selection mechanism type (PE recommended)

Goal: Minimize total error while satisfying ε1 + ε2 ≤ ε_total
"""

import math
from typing import Tuple


class BudgetAllocator:
    """
    Automatic privacy budget allocation for two-stage LDP framework.

    Attributes:
        epsilon_total (float): Total privacy budget available
        delta (float): Privacy parameter for (ε, δ)-DP
    """

    def __init__(self, epsilon_total: float, delta: float = 1e-6):
        """
        Initialize budget allocator.

        Args:
            epsilon_total: Total privacy budget
            delta: Privacy parameter (default: 1e-6)
        """
        if epsilon_total <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon_total}")
        if not (0 < delta < 1):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")

        self.epsilon_total = epsilon_total
        self.delta = delta

    def allocate(self, d: int, k: int, mechanism: str = 'PE') -> Tuple[float, float]:
        """
        Allocate budget between selection and perturbation stages.

        Args:
            d: Dimension of parameter vector
            k: Top-k dimensions to select
            mechanism: Selection mechanism ('PE', 'EXP', or 'PS')

        Returns:
            (epsilon_1, epsilon_2): Budget for selection and perturbation

        Raises:
            ValueError: If k > d or invalid mechanism
        """
        if k > d:
            raise ValueError(f"k ({k}) cannot exceed d ({d})")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")

        # Compute allocation ratio based on mechanism
        if mechanism == 'PE':
            ratio = self._compute_pe_ratio(d, k)
        elif mechanism == 'EXP':
            ratio = self._compute_exp_ratio(d, k)
        elif mechanism == 'PS':
            ratio = self._compute_ps_ratio(d, k)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}. Use 'PE', 'EXP', or 'PS'")

        # Allocate budget
        epsilon_1 = ratio * self.epsilon_total
        epsilon_2 = (1 - ratio) * self.epsilon_total

        return epsilon_1, epsilon_2

    def _compute_pe_ratio(self, d: int, k: int) -> float:
        """
        Compute budget ratio for Perturbed Encoding (PE) mechanism.

        PE uses k-hot encoding with Laplacian noise (sensitivity = 2k).
        Allocation strategy: Balanced, slightly favoring perturbation.

        Args:
            d: Dimension
            k: Top-k

        Returns:
            Ratio of budget for selection (ε1 / ε_total)
        """
        # Base ratio: 0.4 (40% for selection, 60% for perturbation)
        base_ratio = 0.4

        # Adjust based on sparsity
        sparsity = k / d
        if sparsity < 0.1:
            # Very sparse: selection needs less budget
            ratio = base_ratio - 0.1
        elif sparsity > 0.5:
            # Dense: selection needs more budget
            ratio = base_ratio + 0.1
        else:
            # Moderate sparsity: use base ratio
            ratio = base_ratio

        # Clamp to reasonable range [0.2, 0.6]
        return max(0.2, min(0.6, ratio))

    def _compute_exp_ratio(self, d: int, k: int) -> float:
        """
        Compute budget ratio for Exponential Mechanism (EXP).

        EXP iterates Top-1 selection k times, requiring more budget.
        Privacy cost scales with k due to composition.

        Args:
            d: Dimension
            k: Top-k

        Returns:
            Ratio of budget for selection
        """
        # Base ratio: 0.5, increases with k
        base_ratio = 0.5
        k_factor = min(0.2, 0.05 * math.log(k + 1))

        ratio = base_ratio + k_factor

        # Clamp to [0.3, 0.7]
        return max(0.3, min(0.7, ratio))

    def _compute_ps_ratio(self, d: int, k: int) -> float:
        """
        Compute budget ratio for Perturbed Sampling (PS).

        PS is most efficient: perturbs probabilities (sensitivity = 1/d).
        Requires least budget for selection.

        Args:
            d: Dimension
            k: Top-k

        Returns:
            Ratio of budget for selection
        """
        # PS is efficient: allocate less to selection
        base_ratio = 0.3

        # Slight adjustment for very large k
        if k > 200:
            base_ratio += 0.05

        # Clamp to [0.2, 0.4]
        return max(0.2, min(0.4, base_ratio))

    def compute_noise_scale(
        self,
        epsilon_2: float,
        sensitivity: float,
        delta: float = None
    ) -> float:
        """
        Compute Gaussian noise standard deviation for value perturbation.

        Uses Gaussian mechanism for (ε, δ)-DP:
        σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε

        Args:
            epsilon_2: Privacy budget for perturbation stage
            sensitivity: Sensitivity of the query (typically clip_value)
            delta: Privacy parameter (default: self.delta)

        Returns:
            Standard deviation for Gaussian noise
        """
        if delta is None:
            delta = self.delta

        if epsilon_2 <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon_2}")

        # Gaussian mechanism formula
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon_2

        return sigma

    def compute_laplace_scale(self, epsilon_1: float, sensitivity: float) -> float:
        """
        Compute Laplacian noise scale for selection stage.

        Laplace mechanism for ε-DP:
        scale = sensitivity / ε

        Args:
            epsilon_1: Privacy budget for selection
            sensitivity: Sensitivity of selection (e.g., 2k for PE)

        Returns:
            Scale parameter for Laplace distribution
        """
        if epsilon_1 <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon_1}")

        scale = sensitivity / epsilon_1
        return scale

    def get_total_budget(self) -> float:
        """Get the total privacy budget."""
        return self.epsilon_total

    def get_delta(self) -> float:
        """Get the delta parameter."""
        return self.delta

    def verify_allocation(self, epsilon_1: float, epsilon_2: float, tolerance: float = 1e-6) -> bool:
        """
        Verify that allocated budgets sum to total budget.

        Args:
            epsilon_1: Budget for selection
            epsilon_2: Budget for perturbation
            tolerance: Numerical tolerance

        Returns:
            True if ε1 + ε2 ≈ ε_total
        """
        return abs((epsilon_1 + epsilon_2) - self.epsilon_total) < tolerance
