"""
Unit Tests for LDP Dimension Selection Mechanisms

Tests the Perturbed Encoding (PE) mechanism:
- Encoding stage (k-hot binary vector)
- Perturbation stage (Laplacian noise)
- Decoding stage (Top-k extraction)
- Complete pipeline
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.mechanisms.perturbed_encoding import PerturbedEncoding


class TestPerturbedEncoding(unittest.TestCase):
    """Test cases for Perturbed Encoding mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.delta = 1e-6
        self.k = 10
        self.d = 100

    def test_initialization(self):
        """Test PE initialization with valid parameters."""
        pe = PerturbedEncoding(epsilon=self.epsilon, delta=self.delta, k=self.k)

        self.assertEqual(pe.get_epsilon(), self.epsilon)
        self.assertEqual(pe.get_delta(), self.delta)
        self.assertEqual(pe.get_k(), self.k)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        # Invalid epsilon
        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=0, delta=self.delta, k=self.k)

        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=-1.0, delta=self.delta, k=self.k)

        # Invalid delta
        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=self.epsilon, delta=0, k=self.k)

        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=self.epsilon, delta=1.5, k=self.k)

        # Invalid k
        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=self.epsilon, delta=self.delta, k=0)

        with self.assertRaises(ValueError):
            PerturbedEncoding(epsilon=self.epsilon, delta=self.delta, k=-5)

    def test_encode_creates_khot_vector(self):
        """Test that encode() creates valid k-hot binary vector."""
        pe = PerturbedEncoding(epsilon=self.epsilon, k=self.k)

        gradients = torch.randn(self.d)
        encoding = pe.encode(gradients, k=self.k)

        # Check shape
        self.assertEqual(encoding.shape[0], self.d)

        # Check exactly k ones
        num_ones = (encoding == 1.0).sum().item()
        self.assertEqual(num_ones, self.k)

        # Check exactly d-k zeros
        num_zeros = (encoding == 0.0).sum().item()
        self.assertEqual(num_zeros, self.d - self.k)

    def test_encode_selects_topk_correctly(self):
        """Test that encode() selects actual Top-k gradient indices."""
        pe = PerturbedEncoding(epsilon=self.epsilon, k=5)

        # Create gradient with known top-5 indices
        gradients = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.0])
        # Top-5 by absolute value: indices [1, 3, 5, 7, 8] (values 0.9, 0.8, 0.7, 0.6, 0.5)

        encoding = pe.encode(gradients, k=5)

        # Find which indices are selected (encoding == 1)
        selected_indices = torch.where(encoding == 1.0)[0]

        # Check that top-5 are selected
        expected_indices = torch.tensor([1, 3, 5, 7, 8])
        self.assertTrue(torch.equal(selected_indices.sort()[0], expected_indices.sort()[0]))

    def test_add_noise_changes_encoding(self):
        """Test that add_noise() modifies the encoding."""
        pe = PerturbedEncoding(epsilon=self.epsilon, k=self.k)

        gradients = torch.randn(self.d)
        encoding = pe.encode(gradients, k=self.k)
        noisy_encoding = pe.add_noise(encoding)

        # Noisy encoding should be different
        self.assertFalse(torch.allclose(encoding, noisy_encoding))

        # Noisy encoding should have same shape
        self.assertEqual(noisy_encoding.shape, encoding.shape)

    def test_add_noise_scale(self):
        """Test that noise scale depends on epsilon and k."""
        pe_high_eps = PerturbedEncoding(epsilon=2.0, k=10)
        pe_low_eps = PerturbedEncoding(epsilon=0.5, k=10)

        gradients = torch.randn(self.d)
        encoding = torch.zeros(self.d)

        # Add noise multiple times and measure variance
        noise_magnitudes_high = []
        noise_magnitudes_low = []

        for _ in range(100):
            noisy_high = pe_high_eps.add_noise(encoding.clone())
            noisy_low = pe_low_eps.add_noise(encoding.clone())

            noise_magnitudes_high.append(noisy_high.abs().mean().item())
            noise_magnitudes_low.append(noisy_low.abs().mean().item())

        avg_noise_high = sum(noise_magnitudes_high) / len(noise_magnitudes_high)
        avg_noise_low = sum(noise_magnitudes_low) / len(noise_magnitudes_low)

        # Higher epsilon should have less noise
        self.assertLess(avg_noise_high, avg_noise_low)

    def test_decode_returns_k_indices(self):
        """Test that decode() returns exactly k indices."""
        pe = PerturbedEncoding(epsilon=self.epsilon, k=self.k)

        noisy_encoding = torch.randn(self.d)
        selected_indices = pe.decode(noisy_encoding, k=self.k)

        # Check correct number of indices
        self.assertEqual(selected_indices.shape[0], self.k)

        # Check indices are unique
        unique_indices = torch.unique(selected_indices)
        self.assertEqual(unique_indices.shape[0], self.k)

        # Check indices are in valid range
        self.assertTrue(torch.all(selected_indices >= 0))
        self.assertTrue(torch.all(selected_indices < self.d))

    def test_complete_pipeline(self):
        """Test the complete select_topk() pipeline."""
        pe = PerturbedEncoding(epsilon=1.0, k=10)

        gradients = torch.randn(100)
        selected_indices = pe.select_topk(gradients, k=10)

        # Check output shape
        self.assertEqual(selected_indices.shape[0], 10)

        # Check indices are valid
        self.assertTrue(torch.all(selected_indices >= 0))
        self.assertTrue(torch.all(selected_indices < 100))

        # Check indices are unique
        self.assertEqual(len(torch.unique(selected_indices)), 10)

    def test_privacy_randomness(self):
        """Test that PE mechanism is randomized (privacy property)."""
        pe = PerturbedEncoding(epsilon=1.0, k=10)

        gradients = torch.randn(100)

        # Run mechanism multiple times
        results = []
        for _ in range(10):
            selected = pe.select_topk(gradients, k=10)
            results.append(selected)

        # Check that results are not always identical (due to randomness)
        all_identical = all(torch.equal(results[0], r) for r in results[1:])
        self.assertFalse(all_identical, "PE should produce different results due to noise")

    def test_sensitivity_computation(self):
        """Test sensitivity computation for k-hot encoding."""
        pe = PerturbedEncoding(epsilon=1.0, k=10)

        # Sensitivity should be 2k
        sensitivity = pe.compute_sensitivity(k=10)
        self.assertEqual(sensitivity, 20)

        sensitivity_20 = pe.compute_sensitivity(k=20)
        self.assertEqual(sensitivity_20, 40)

    def test_expected_noise_magnitude(self):
        """Test expected noise magnitude calculation."""
        pe = PerturbedEncoding(epsilon=1.0, k=10)

        # Expected noise = sensitivity / epsilon = 2k / eps = 20 / 1.0 = 20
        expected_noise = pe.compute_expected_noise_magnitude(k=10)
        self.assertAlmostEqual(expected_noise, 20.0)

        # Higher epsilon → less noise
        pe_high_eps = PerturbedEncoding(epsilon=2.0, k=10)
        expected_noise_high = pe_high_eps.compute_expected_noise_magnitude(k=10)
        self.assertAlmostEqual(expected_noise_high, 10.0)

    def test_k_exceeds_d(self):
        """Test that k > d raises ValueError."""
        pe = PerturbedEncoding(epsilon=1.0, k=10)

        gradients = torch.randn(5)  # d=5 < k=10

        with self.assertRaises(ValueError):
            pe.encode(gradients, k=10)

    def test_different_k_values(self):
        """Test PE with various k values."""
        for k in [5, 10, 50, 100]:
            pe = PerturbedEncoding(epsilon=1.0, k=k)
            gradients = torch.randn(200)

            selected = pe.select_topk(gradients, k=k)

            self.assertEqual(selected.shape[0], k)
            self.assertEqual(len(torch.unique(selected)), k)


if __name__ == '__main__':
    unittest.main()
