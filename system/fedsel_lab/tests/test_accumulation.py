"""
Unit Tests for Gradient Accumulator

Tests the gradient accumulation formula: r_t = β * r_{t-1} + g_t
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.utils.gradient_accumulator import GradientAccumulator


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)


class TestGradientAccumulator(unittest.TestCase):
    """Test cases for GradientAccumulator."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.beta = 0.9

    def test_initialization(self):
        """Test that accumulator initializes with correct beta and zero gradients."""
        accumulator = GradientAccumulator(self.model, beta=self.beta)

        # Check beta
        self.assertEqual(accumulator.get_beta(), self.beta)

        # Check all accumulated gradients are zero
        for name, grad in accumulator.get_accumulated().items():
            self.assertTrue(torch.allclose(grad, torch.zeros_like(grad)))

    def test_invalid_beta(self):
        """Test that invalid beta values raise ValueError."""
        with self.assertRaises(ValueError):
            GradientAccumulator(self.model, beta=0.0)

        with self.assertRaises(ValueError):
            GradientAccumulator(self.model, beta=1.0)

        with self.assertRaises(ValueError):
            GradientAccumulator(self.model, beta=-0.5)

        with self.assertRaises(ValueError):
            GradientAccumulator(self.model, beta=1.5)

    def test_accumulation_formula(self):
        """Test the core accumulation formula: r_t = β * r_{t-1} + g_t"""
        accumulator = GradientAccumulator(self.model, beta=self.beta)

        # Create mock gradients
        current_grads = {}
        for name, param in self.model.named_parameters():
            current_grads[name] = torch.ones_like(param.data)

        # First accumulation: r_1 = β * 0 + g_1 = g_1
        result1 = accumulator.accumulate(current_grads)
        for name, grad in result1.items():
            expected = torch.ones_like(grad)
            self.assertTrue(torch.allclose(grad, expected),
                          f"First accumulation failed for {name}")

        # Second accumulation: r_2 = β * r_1 + g_2 = β * 1 + 1 = β + 1
        result2 = accumulator.accumulate(current_grads)
        for name, grad in result2.items():
            expected = torch.ones_like(grad) * (self.beta + 1)
            self.assertTrue(torch.allclose(grad, expected),
                          f"Second accumulation failed for {name}")

        # Third accumulation: r_3 = β * r_2 + g_3 = β * (β + 1) + 1 = β² + β + 1
        result3 = accumulator.accumulate(current_grads)
        for name, grad in result3.items():
            expected = torch.ones_like(grad) * (self.beta**2 + self.beta + 1)
            self.assertTrue(torch.allclose(grad, expected),
                          f"Third accumulation failed for {name}")

    def test_varying_gradients(self):
        """Test accumulation with varying gradient values."""
        accumulator = GradientAccumulator(self.model, beta=0.5)  # β = 0.5 for simplicity

        # Round 1: gradients = 2.0
        grads_1 = {}
        for name, param in self.model.named_parameters():
            grads_1[name] = torch.ones_like(param.data) * 2.0

        result1 = accumulator.accumulate(grads_1)
        for name, grad in result1.items():
            expected = torch.ones_like(grad) * 2.0  # r_1 = 0.5 * 0 + 2 = 2
            self.assertTrue(torch.allclose(grad, expected))

        # Round 2: gradients = 4.0
        grads_2 = {}
        for name, param in self.model.named_parameters():
            grads_2[name] = torch.ones_like(param.data) * 4.0

        result2 = accumulator.accumulate(grads_2)
        for name, grad in result2.items():
            expected = torch.ones_like(grad) * 5.0  # r_2 = 0.5 * 2 + 4 = 5
            self.assertTrue(torch.allclose(grad, expected))

    def test_reset(self):
        """Test that reset() zeros all accumulated gradients."""
        accumulator = GradientAccumulator(self.model, beta=self.beta)

        # Accumulate some gradients
        current_grads = {}
        for name, param in self.model.named_parameters():
            current_grads[name] = torch.randn_like(param.data)

        accumulator.accumulate(current_grads)

        # Verify non-zero
        for name, grad in accumulator.get_accumulated().items():
            self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))

        # Reset
        accumulator.reset()

        # Verify all zero
        for name, grad in accumulator.get_accumulated().items():
            self.assertTrue(torch.allclose(grad, torch.zeros_like(grad)))

    def test_reset_param(self):
        """Test that reset_param() zeros a specific parameter."""
        accumulator = GradientAccumulator(self.model, beta=self.beta)

        # Accumulate gradients
        current_grads = {}
        for name, param in self.model.named_parameters():
            current_grads[name] = torch.ones_like(param.data)

        accumulator.accumulate(current_grads)

        # Reset only fc1.weight
        accumulator.reset_param('fc1.weight')

        # Check fc1.weight is zero
        self.assertTrue(torch.allclose(
            accumulator.get_accumulated()['fc1.weight'],
            torch.zeros_like(accumulator.get_accumulated()['fc1.weight'])
        ))

        # Check others are non-zero
        for name in ['fc1.bias', 'fc2.weight', 'fc2.bias']:
            self.assertFalse(torch.allclose(
                accumulator.get_accumulated()[name],
                torch.zeros_like(accumulator.get_accumulated()[name])
            ))

    def test_set_beta(self):
        """Test that set_beta() updates beta correctly."""
        accumulator = GradientAccumulator(self.model, beta=0.9)

        # Change beta
        accumulator.set_beta(0.8)
        self.assertEqual(accumulator.get_beta(), 0.8)

        # Test invalid beta
        with self.assertRaises(ValueError):
            accumulator.set_beta(1.5)

    def test_different_beta_values(self):
        """Test accumulation with different beta values."""
        # High beta (0.95): more history
        acc_high = GradientAccumulator(self.model, beta=0.95)

        # Low beta (0.7): more responsive
        acc_low = GradientAccumulator(self.model, beta=0.7)

        # Same gradients
        grads = {}
        for name, param in self.model.named_parameters():
            grads[name] = torch.ones_like(param.data)

        # Accumulate twice
        for _ in range(2):
            result_high = acc_high.accumulate(grads)
            result_low = acc_low.accumulate(grads)

        # High beta should have larger accumulated values
        for name in result_high.keys():
            self.assertTrue(
                result_high[name].mean() > result_low[name].mean(),
                f"High beta should accumulate more for {name}"
            )


if __name__ == '__main__':
    unittest.main()
