"""
FedSel Base Client Implementation

Implements the two-stage LDP framework for federated learning:
1. Stage 1: Private dimension selection using LDP mechanisms
2. Stage 2: Value perturbation using Gaussian noise

Key Features:
- Gradient accumulation for stability (r_t = β*r_{t-1} + g_t)
- Automatic privacy budget allocation (ε1/ε2)
- Sparse gradient upload (only Top-k dimensions)
- Compatible with existing FedCP framework

Subclasses must implement:
- _select_topk_dimensions(): Specific dimension selection mechanism
"""

import sys
import os
import torch
import copy

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from flcore.clients.clientcp import clientCP
from fedsel_lab.utils.gradient_accumulator import GradientAccumulator
from fedsel_lab.mechanisms.budget_allocation import BudgetAllocator


class ClientFedSelBase(clientCP):
    """
    Base client for FedSel framework with two-stage LDP.

    Inherits all standard FL functionality from clientCP and adds:
    - Gradient accumulation
    - Privacy budget allocation
    - Two-stage LDP (selection + perturbation)
    - Sparse gradient updates
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        Initialize FedSel client.

        Args:
            args: Training arguments
            id: Client ID
            train_samples: Number of training samples
            test_samples: Number of test samples
            **kwargs: Additional arguments
        """
        # Call parent constructor
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # FedSel-specific parameters
        self.fedsel_beta = getattr(args, 'fedsel_beta', 0.9)
        self.fedsel_k = getattr(args, 'fedsel_k', 100)
        self.fedsel_mechanism = getattr(args, 'fedsel_mechanism', 'PE').upper()
        self.fedsel_mu = getattr(args, 'fedsel_mu', None)
        if self.fedsel_mu is not None and not (0 < self.fedsel_mu < 1):
            raise ValueError(f"fedsel_mu must be in (0, 1), got {self.fedsel_mu}")

        # Initialize gradient accumulator
        self.grad_accumulator = GradientAccumulator(self.model, beta=self.fedsel_beta)

        # Initialize budget allocator
        self.budget_allocator = BudgetAllocator(
            epsilon_total=self.epsilon if self.dp else 1.0,
            delta=self.delta,
            allocation_ratio=self.fedsel_mu
        )

        # Logging
        if self.id == 0:  # Only log for first client
            print(f"[FedSel] Client {self.id} initialized:")
            print(f"  Mechanism: {self.fedsel_mechanism}")
            print(f"  Top-k: {self.fedsel_k}")
            print(f"  Beta (accumulation): {self.fedsel_beta}")
            if self.fedsel_mu is not None:
                print(f"  Manual µ (budget split): {self.fedsel_mu:.3f}")
            print(f"  DP enabled: {self.dp}")
            if self.dp:
                print(f"  Epsilon: {self.epsilon}, Delta: {self.delta}")

    def train_cs_model(self, round, args):
        """
        Train model with FedSel two-stage LDP.

        Overrides clientCP.train_cs_model() to implement:
        1. Standard local training
        2. Gradient computation
        3. Gradient accumulation
        4. Stage 1: Private Top-k dimension selection
        5. Stage 2: Value perturbation with DP noise
        6. Sparse model update

        Args:
            round: Current training round
            args: Training arguments
        """
        if not self.dp:
            # DP not enabled, use parent's method
            super().train_cs_model(round, args)
            return

        # Load data
        trainloader = self.load_train_data()
        testloader = self.load_test_data()

        # Pre-training evaluation
        self.get_module_gradient_norm(dataloader=testloader, filename="test_before")

        self.model.train()

        # Save initial parameters
        self.inital_pra_dp = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Standard local training
        for _ in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Compute parameter differences (gradients)
        param_diff = {}
        modules = {'feature_extractor': self.model.feature_extractor}

        for module_name, module in modules.items():
            for name, param in module.named_parameters():
                full_name = f"{module_name}.{name}".lstrip('.')
                diff = (param - self.inital_pra_dp[full_name]).detach()
                param_diff[full_name] = diff

        # Apply FedSel two-stage LDP
        perturbed_diff = self._apply_fedsel_ldp(param_diff, round)

        # Update model with perturbed differences
        for name, param in self.model.named_parameters():
            if name in perturbed_diff:
                param.data = self.inital_pra_dp[name] + perturbed_diff[name]
            else:
                # Head parameters: use original (no perturbation)
                param.data = self.inital_pra_dp[name]

        # Post-training evaluation
        self.get_module_gradient_norm(dataloader=trainloader, filename="train_after")

    def _apply_fedsel_ldp(self, param_diff, round):
        """
        Apply FedSel two-stage LDP to parameter differences.

        Args:
            param_diff: Dictionary of parameter differences
            round: Current training round

        Returns:
            Dictionary of perturbed (sparse) parameter differences
        """
        perturbed_diff = {}

        # Accumulate gradients
        accumulated_grads = self.grad_accumulator.accumulate(param_diff)

        # Flatten all parameters into a single vector so one privacy budget
        # is consumed per round (aligns with paper's global Top-k selection).
        flat_diffs = []
        flat_accums = []
        names = []
        shapes = []

        for name, diff in param_diff.items():
            names.append(name)
            shapes.append(diff.shape)
            flat_diffs.append(diff.view(-1))
            flat_accums.append(accumulated_grads[name].view(-1))

        if not flat_diffs:
            return perturbed_diff

        full_diff = torch.cat(flat_diffs)
        full_accum = torch.cat(flat_accums)

        total_d = full_diff.numel()
        k = min(self.fedsel_k, total_d)

        # Allocate privacy budget once for the full vector
        epsilon_1, epsilon_2 = self.budget_allocator.allocate(
            total_d, k, mechanism=self.fedsel_mechanism
        )

        # Stage 1: Select Top-k dimensions (implemented by subclasses)
        selected_indices = self._select_topk_dimensions(
            full_accum, k, epsilon_1
        )

        # Stage 2: Perturb selected values
        perturbed_flat = self._perturb_selected_values(
            full_diff, selected_indices, epsilon_2
        )

        # Reconstruct per-parameter tensors
        start = 0
        for name, shape, original_flat in zip(names, shapes, flat_diffs):
            size = original_flat.numel()
            slice_flat = perturbed_flat[start:start + size]
            perturbed_diff[name] = slice_flat.view(shape)
            start += size

        # Logging (every 10 rounds for client 0)
        if round % 10 == 0 and self.id == 0:
            sparsity = k / total_d
            nonzero = (perturbed_flat != 0).sum().item()
            print(f"[FedSel Client {self.id} Round {round}]:")
            print(f"  total_d={total_d}, k={k}, sparsity={sparsity:.3f}")
            print(f"  eps1={epsilon_1:.3f}, eps2={epsilon_2:.3f}")
            print(f"  Non-zero dims: {nonzero}")

        return perturbed_diff

    def _select_topk_dimensions(self, gradient_vector, k, epsilon_1):
        """
        Select Top-k dimensions privately (ABSTRACT METHOD).

        Subclasses must implement this method with specific mechanisms
        (EXP, PE, or PS).

        Args:
            gradient_vector: 1D tensor of gradients [d]
            k: Number of dimensions to select
            epsilon_1: Privacy budget for selection

        Returns:
            Tensor of selected indices [k]

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclass must implement _select_topk_dimensions() method"
        )

    def _perturb_selected_values(self, gradient_vector, selected_indices, epsilon_2):
        """
        Perturb selected gradient values with Gaussian noise (Stage 2).

        Uses Gaussian mechanism for (ε, δ)-DP:
        1. Extract values at selected indices
        2. Clip to sensitivity bound
        3. Add Gaussian noise
        4. Create sparse vector (only selected indices non-zero)

        Args:
            gradient_vector: 1D gradient tensor [d]
            selected_indices: Indices of selected dimensions [k]
            epsilon_2: Privacy budget for perturbation

        Returns:
            Sparse perturbed gradient [d] with only k non-zero values
        """
        d = gradient_vector.shape[0]
        perturbed = torch.zeros_like(gradient_vector)

        # Extract selected values
        selected_values = gradient_vector[selected_indices]

        # L2 norm clipping
        norm = torch.norm(selected_values)
        clip_value = self.clip_value

        if norm > clip_value:
            selected_values = selected_values / norm * clip_value

        # Compute Gaussian noise scale
        noise_scale = self.budget_allocator.compute_noise_scale(
            epsilon_2,
            sensitivity=clip_value,
            delta=self.delta
        )

        # Add Gaussian noise
        noise = torch.normal(
            mean=0,
            std=noise_scale,
            size=selected_values.shape
        ).to(gradient_vector.device)

        noisy_values = selected_values + noise

        # Create sparse vector (only selected indices have values)
        perturbed[selected_indices] = noisy_values

        return perturbed
