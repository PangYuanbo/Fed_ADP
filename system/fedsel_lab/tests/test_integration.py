"""
Integration Tests for FedSel Clients

Tests the complete client implementation:
- Client initialization
- Two-stage LDP application
- Gradient accumulation integration
- Sparse gradient generation
- Model parameter updates
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.clients.client_fedsel_pe import ClientFedSelPE
from flcore.trainmodel.models import LocalModel


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestFedSelIntegration(unittest.TestCase):
    """Integration tests for FedSel clients."""

    def setUp(self):
        """Set up test fixtures."""
        # Create args object (use simple class instead of Mock to avoid format issues)
        class Args:
            pass

        self.args = Args()
        self.args.device = 'cpu'
        self.args.dataset = 'test-dataset'
        self.args.alpha = 1.0
        self.args.batch_size = 4
        self.args.local_learning_rate = 0.01
        self.args.local_steps = 1
        self.args.num_classes = 10
        self.args.difference_privacy = True
        self.args.lamda = 0.0
        self.args.global_rounds = 10

        # DP parameters
        self.args.epsilon = 1.0
        self.args.delta = 1e-6
        self.args.clip_value = 0.005

        # FedSel parameters
        self.args.fedsel_beta = 0.9
        self.args.fedsel_k = 50
        self.args.fedsel_mechanism = 'PE'

        # Create model
        feature_extractor = SimpleCNN()
        head = nn.Linear(10, 10)
        self.args.model = LocalModel(feature_extractor, head)

        # Set layer noise mode (clientCP expects this)
        self.args.layer_noise_mode = 'global'
        self.args.global_noise = False
        self.args.threshold_high = 0.6
        self.args.threshold_low = 0.4

        # Selective upload parameters (clientCP checks these)
        self.args.enable_selective_upload = False
        self.args.gradient_keep_ratio = 0.3
        self.args.noise_multiplier = 1.0

    def test_client_initialization(self):
        """Test that FedSel-PE client initializes correctly."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Check basic attributes
        self.assertEqual(client.id, 0)
        self.assertEqual(client.train_samples, 100)
        self.assertEqual(client.test_samples, 20)
        self.assertEqual(client.fedsel_mechanism, 'PE')
        self.assertEqual(client.fedsel_k, 50)
        self.assertEqual(client.fedsel_beta, 0.9)

        # Check components exist
        self.assertIsNotNone(client.grad_accumulator)
        self.assertIsNotNone(client.budget_allocator)

    def test_select_topk_dimensions(self):
        """Test that _select_topk_dimensions returns correct number of indices."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Create test gradient
        gradient = torch.randn(1000)
        k = 50
        epsilon_1 = 0.4

        # Select Top-k
        selected_indices = client._select_topk_dimensions(gradient, k, epsilon_1)

        # Check output
        self.assertEqual(selected_indices.shape[0], k)
        self.assertTrue(torch.all(selected_indices >= 0))
        self.assertTrue(torch.all(selected_indices < 1000))
        self.assertEqual(len(torch.unique(selected_indices)), k)

    def test_perturb_selected_values(self):
        """Test that _perturb_selected_values creates sparse gradient."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Create test gradient
        gradient = torch.randn(1000)
        selected_indices = torch.tensor([10, 20, 30, 40, 50])
        epsilon_2 = 0.6

        # Perturb
        perturbed = client._perturb_selected_values(gradient, selected_indices, epsilon_2)

        # Check sparsity
        self.assertEqual(perturbed.shape[0], 1000)
        nonzero_mask = perturbed != 0
        nonzero_count = nonzero_mask.sum().item()

        # Should have exactly 5 non-zero values
        self.assertEqual(nonzero_count, 5)

        # Check that non-zero indices match selected indices
        nonzero_indices = torch.where(nonzero_mask)[0]
        self.assertTrue(torch.equal(nonzero_indices.sort()[0], selected_indices.sort()[0]))

    def test_apply_fedsel_ldp(self):
        """Test the complete two-stage LDP application."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Create mock parameter differences
        param_diff = {
            'feature_extractor.conv1.weight': torch.randn(16, 3, 3, 3),
            'feature_extractor.conv1.bias': torch.randn(16),
        }

        # Apply FedSel LDP
        perturbed_diff = client._apply_fedsel_ldp(param_diff, round=1)

        # Check output
        self.assertIn('feature_extractor.conv1.weight', perturbed_diff)
        self.assertIn('feature_extractor.conv1.bias', perturbed_diff)

        # Check shapes preserved
        self.assertEqual(
            perturbed_diff['feature_extractor.conv1.weight'].shape,
            param_diff['feature_extractor.conv1.weight'].shape
        )

        # Check sparsity (most values should be zero)
        weight_perturbed = perturbed_diff['feature_extractor.conv1.weight']
        bias_perturbed = perturbed_diff['feature_extractor.conv1.bias']

        total_nonzero = (weight_perturbed != 0).sum().item() + (bias_perturbed != 0).sum().item()
        self.assertLessEqual(total_nonzero, client.fedsel_k)
        self.assertGreater(total_nonzero, 0)

    def test_gradient_accumulation_integration(self):
        """Test that gradient accumulation works across multiple rounds."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Simulate multiple rounds with different gradients
        param_diff = {
            'feature_extractor.conv1.weight': torch.randn(16, 3, 3, 3),  # Random values
        }

        # Round 1
        perturbed_1 = client._apply_fedsel_ldp(param_diff, round=1)

        # Check that accumulator has non-zero gradients
        accumulated = client.grad_accumulator.get_accumulated()
        self.assertIn('feature_extractor.conv1.weight', accumulated)
        self.assertFalse(torch.allclose(
            accumulated['feature_extractor.conv1.weight'],
            torch.zeros_like(accumulated['feature_extractor.conv1.weight'])
        ))

        # Save first accumulated state
        accumulated_copy = accumulated['feature_extractor.conv1.weight'].clone()

        # Round 2 with different gradients
        param_diff_2 = {
            'feature_extractor.conv1.weight': torch.randn(16, 3, 3, 3),  # Different random values
        }
        perturbed_2 = client._apply_fedsel_ldp(param_diff_2, round=2)

        # Accumulated gradients should have changed
        accumulated_2 = client.grad_accumulator.get_accumulated()
        self.assertFalse(torch.equal(
            accumulated_copy,
            accumulated_2['feature_extractor.conv1.weight']
        ))

    def test_budget_allocation_integration(self):
        """Test that budget allocation is applied correctly."""
        client = ClientFedSelPE(
            args=self.args,
            id=0,
            train_samples=100,
            test_samples=20
        )

        # Check budget allocator parameters
        self.assertEqual(client.budget_allocator.get_total_budget(), client.epsilon)
        self.assertEqual(client.budget_allocator.get_delta(), client.delta)

        # Test allocation for typical parameter
        d = 1000
        k = 100  # 10% sparsity - should get base PE ratio
        eps1, eps2 = client.budget_allocator.allocate(d, k, mechanism='PE')

        # Check allocation sums to total
        self.assertAlmostEqual(eps1 + eps2, client.epsilon, places=6)

        # Check expected allocation for PE with 10% sparsity (40% / 60%)
        self.assertAlmostEqual(eps1, 0.4 * client.epsilon, places=2)
        self.assertAlmostEqual(eps2, 0.6 * client.epsilon, places=2)


if __name__ == '__main__':
    unittest.main()
