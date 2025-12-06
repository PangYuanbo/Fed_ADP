"""
Perturbed Encoding (PE) Mechanism for FedSel

Three-stage workflow for private Top-k dimension selection:
1. Encode: Convert gradient vector to k-hot binary encoding
2. Perturb: Add Laplacian noise to encoding
3. Decode: Extract Top-k indices from noisy encoding

Privacy Guarantee: (epsilon, delta)-LDP

Recommended for: 10 < k < 200 (balanced privacy-utility)

Reference: FedSel Paper, Algorithm 3
"""

import torch
from typing import Optional


class PerturbedEncoding:
    """
    Perturbed Encoding mechanism for private Top-k dimension selection.

    Attributes:
        epsilon (float): Privacy budget for selection
        delta (float): Privacy parameter
        k (int): Number of dimensions to select
    """

    def __init__(self, epsilon: float, delta: float = 1e-6, k: int = 100):
        """
        Initialize PE mechanism.

        Args:
            epsilon: Privacy budget for dimension selection
            delta: Privacy parameter (default: 1e-6)
            k: Top-k dimensions to select

        Raises:
            ValueError: If parameters are out of valid range
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.epsilon = epsilon
        self.delta = delta
        self.k = k

    def encode(self, gradient_vector: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Stage 1: Encode gradient vector as k-hot binary vector.

        Creates binary encoding where Top-k positions are 1, others are 0.

        Args:
            gradient_vector: 1D tensor of gradients [d]
            k: Number of top dimensions (default: self.k)

        Returns:
            k-hot binary encoding [d] where Top-k positions = 1
        """
        if k is None:
            k = self.k

        d = gradient_vector.shape[0]
        if k > d:
            raise ValueError(f"k ({k}) cannot exceed dimension ({d})")

        # Create zero encoding
        encoding = torch.zeros(d, device=gradient_vector.device, dtype=torch.float32)

        # Find Top-k indices based on absolute gradient values
        abs_grad = gradient_vector.abs()
        topk_values, topk_indices = torch.topk(abs_grad, k)

        # Set Top-k positions to 1
        encoding[topk_indices] = 1.0

        return encoding

    def add_noise(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Stage 2: Add Laplacian noise to encoding.

        Sensitivity analysis:
        - k-hot encoding changes by at most 2k when switching one gradient value
        - Therefore, sensitivity Δf = 2k
        - Laplace scale = sensitivity / epsilon = 2k / epsilon

        Args:
            encoding: Binary k-hot encoding [d]

        Returns:
            Noisy encoding [d]
        """
        d = encoding.shape[0]

        # Compute sensitivity: Δf = 2k (worst case: flip k positions)
        sensitivity = 2 * self.k

        # Laplace scale parameter: b = Δf / ε
        scale = sensitivity / self.epsilon

        # Generate Laplacian noise
        # Laplace distribution: f(x) = (1/2b) * exp(-|x|/b)
        noise = torch.distributions.laplace.Laplace(0, scale).sample((d,))
        noise = noise.to(encoding.device)

        # Add noise to encoding
        noisy_encoding = encoding + noise

        return noisy_encoding

    def decode(self, noisy_encoding: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Stage 3: Decode Top-k indices from noisy encoding.

        Selects k dimensions with highest noisy encoding values.

        Args:
            noisy_encoding: Noisy binary encoding [d]
            k: Number of dimensions to extract (default: self.k)

        Returns:
            Tensor of Top-k indices [k]
        """
        if k is None:
            k = self.k

        # Select indices with k highest values
        topk_values, topk_indices = torch.topk(noisy_encoding, k)

        return topk_indices

    def select_topk(self, gradient_vector: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Complete PE pipeline: Encode → Perturb → Decode.

        Args:
            gradient_vector: 1D gradient tensor [d]
            k: Top-k dimensions to select (default: self.k)

        Returns:
            Tensor of selected indices [k]

        Example:
            >>> pe = PerturbedEncoding(epsilon=1.0, k=10)
            >>> gradients = torch.randn(100)
            >>> selected_indices = pe.select_topk(gradients)
            >>> print(selected_indices.shape)  # torch.Size([10])
        """
        if k is None:
            k = self.k

        # Stage 1: Encode
        encoding = self.encode(gradient_vector, k)

        # Stage 2: Add noise
        noisy_encoding = self.add_noise(encoding)

        # Stage 3: Decode
        selected_indices = self.decode(noisy_encoding, k)

        return selected_indices

    def get_epsilon(self) -> float:
        """Get the privacy budget."""
        return self.epsilon

    def get_delta(self) -> float:
        """Get the privacy parameter."""
        return self.delta

    def get_k(self) -> int:
        """Get the Top-k value."""
        return self.k

    def compute_sensitivity(self, k: Optional[int] = None) -> float:
        """
        Compute sensitivity of k-hot encoding.

        Sensitivity = maximum change in encoding when one gradient changes.
        For k-hot encoding: Δf = 2k

        Args:
            k: Top-k value (default: self.k)

        Returns:
            Sensitivity value
        """
        if k is None:
            k = self.k
        return 2 * k

    def compute_expected_noise_magnitude(self, k: Optional[int] = None) -> float:
        """
        Compute expected magnitude of Laplacian noise.

        For Laplace(0, b): E[|noise|] = b = sensitivity / epsilon

        Args:
            k: Top-k value (default: self.k)

        Returns:
            Expected noise magnitude
        """
        if k is None:
            k = self.k

        sensitivity = self.compute_sensitivity(k)
        scale = sensitivity / self.epsilon

        return scale  # E[|Laplace(0, b)|] = b
