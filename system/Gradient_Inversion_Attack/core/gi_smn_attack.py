"""
GI-SMN (Gradient Inversion attack based on Style Migration Network) Core Implementation

Implements the main attack algorithm that reconstructs private data from gradients
using StyleGAN-XL's latent space optimization.

Author: Fed_ADP Project
Reference: Jin Qian et al., "GI-SMN: Gradient Inversion Attack against Federated
           Learning without Prior Knowledge", arXiv:2405.03516, 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import copy

from .loss_functions import GISMNLoss
from ..models.stylegan_wrapper import StyleGANWrapper


class GISMNAttack:
    """
    GI-SMN Attack: Reconstruct images from gradients using StyleGAN-XL

    Key features:
    - Optimization in 64-dim latent space (vs 3072-dim pixel space)
    - Two-stage loss strategy
    - No prior knowledge required
    - Robust against weak defenses

    Args:
        generator (StyleGANWrapper): StyleGAN-XL generator wrapper
        device (str): Device for computation
        num_iterations (int): Total optimization iterations (default: 10000)
        learning_rate (float): Learning rate for latent optimization (default: 0.01)
        lambda_tv (float): TV regularization weight (default: 0.001)
        lambda_l2 (float): L2 regularization weight (default: 0.0001)
        lambda_group (float): Group consistency weight (default: 0.01)
        stage1_ratio (float): Stage 1 duration ratio (default: 4/9)
        latent_clamp (float): Clamp latent codes to [-c, c] (default: 2.0)
        verbose (bool): Print progress (default: True)
    """

    def __init__(
        self,
        generator: StyleGANWrapper,
        device: str = 'cuda',
        num_iterations: int = 10000,
        learning_rate: float = 0.01,
        lambda_tv: float = 0.001,
        lambda_l2: float = 0.0001,
        lambda_group: float = 0.01,
        stage1_ratio: float = 4/9,
        latent_clamp: float = 2.0,
        verbose: bool = True
    ):
        self.generator = generator
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.latent_clamp = latent_clamp
        self.verbose = verbose

        # Initialize loss function
        self.loss_fn = GISMNLoss(
            total_iterations=num_iterations,
            lambda_tv=lambda_tv,
            lambda_l2=lambda_l2,
            lambda_group=lambda_group,
            stage1_ratio=stage1_ratio
        )

    def reconstruct_from_gradient(
        self,
        true_gradient: Dict[str, torch.Tensor],
        target_model: nn.Module,
        batch_size: int = 1,
        ground_truth_labels: Optional[torch.Tensor] = None,
        init_latent: Optional[torch.Tensor] = None,
        save_intermediate: bool = False
    ) -> Dict:
        """
        Reconstruct images from intercepted gradients

        Args:
            true_gradient (dict): Intercepted gradient from victim
            target_model (nn.Module): Victim model architecture (without weights)
            batch_size (int): Number of images in the batch
            ground_truth_labels (torch.Tensor, optional): True labels (if known)
            init_latent (torch.Tensor, optional): Initial latent code (random if None)
            save_intermediate (bool): Save intermediate results

        Returns:
            result (dict): Dictionary containing:
                - 'reconstructed_images': Final reconstructed images [B, 3, H, W]
                - 'optimal_latent': Optimal latent code [B, latent_dim]
                - 'loss_history': Loss values over iterations
                - 'intermediate_images': List of images (if save_intermediate=True)
        """
        # Initialize latent code
        if init_latent is None:
            z = torch.randn(
                batch_size,
                self.generator.latent_dim,
                device=self.device,
                requires_grad=True
            )
        else:
            z = init_latent.clone().detach().to(self.device)
            z.requires_grad = True

        # Setup optimizer
        optimizer = optim.Adam([z], lr=self.learning_rate)

        # Move target model to device
        target_model = target_model.to(self.device)
        target_model.eval()

        # Move true gradient to device
        true_gradient = {k: v.to(self.device) for k, v in true_gradient.items()}

        # Storage for results
        loss_history = []
        intermediate_images = [] if save_intermediate else None

        # Optimization loop
        iterator = tqdm(range(self.num_iterations), desc="GI-SMN Attack") if self.verbose else range(self.num_iterations)

        for iteration in iterator:
            optimizer.zero_grad()

            # Generate dummy images from latent code
            dummy_images = self.generator.generate_from_latent(z)

            # Compute dummy gradient
            dummy_gradient = self._compute_gradient(
                target_model,
                dummy_images,
                ground_truth_labels
            )

            # Compute loss
            loss_dict = self.loss_fn(
                dummy_gradient,
                true_gradient,
                dummy_images,
                iteration=iteration
            )

            total_loss = loss_dict['total']

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Clamp latent code
            with torch.no_grad():
                z.clamp_(-self.latent_clamp, self.latent_clamp)

            # Record history
            loss_history.append({
                'iteration': iteration,
                'total': total_loss.item(),
                'gradient': loss_dict['gradient'].item(),
                'tv': loss_dict['tv'].item(),
                'l2': loss_dict['l2'].item(),
                'group': loss_dict['group'].item(),
            })

            # Save intermediate images
            if save_intermediate and (iteration % (self.num_iterations // 10) == 0 or iteration == self.num_iterations - 1):
                with torch.no_grad():
                    intermediate_images.append({
                        'iteration': iteration,
                        'images': dummy_images.cpu().clone()
                    })

            # Update progress bar
            if self.verbose and iteration % 100 == 0:
                iterator.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'grad': f"{loss_dict['gradient'].item():.4f}",
                    'stage': 1 if iteration < self.loss_fn.stage1_end else 2
                })

        # Generate final reconstructed images
        with torch.no_grad():
            reconstructed_images = self.generator.generate_from_latent(z)

        result = {
            'reconstructed_images': reconstructed_images.cpu(),
            'optimal_latent': z.detach().cpu(),
            'loss_history': loss_history,
        }

        if save_intermediate:
            result['intermediate_images'] = intermediate_images

        return result

    def _compute_gradient(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient of model on given images

        Args:
            model (nn.Module): Target model
            images (torch.Tensor): Input images [B, C, H, W]
            labels (torch.Tensor, optional): Labels [B]

        Returns:
            gradients (dict): Dictionary of gradients {param_name: gradient}
        """
        # Zero out existing gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)

        # If labels unknown, use argmax of outputs (label inference)
        if labels is None:
            labels = outputs.argmax(dim=1)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        # Backward pass with create_graph=True to maintain gradient connection
        loss.backward(create_graph=True)

        # Extract gradients (do NOT clone to maintain gradient graph)
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad

        return gradients

    def batch_reconstruct(
        self,
        gradient_list: List[Dict[str, torch.Tensor]],
        target_model: nn.Module,
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict]:
        """
        Reconstruct multiple batches of images

        Args:
            gradient_list (list): List of gradients to reconstruct from
            target_model (nn.Module): Victim model architecture
            batch_size (int): Batch size for each gradient
            **kwargs: Additional arguments for reconstruct_from_gradient()

        Returns:
            results (list): List of reconstruction results
        """
        results = []

        for idx, gradient in enumerate(gradient_list):
            if self.verbose:
                print(f"\n[GI-SMN] Reconstructing batch {idx+1}/{len(gradient_list)}")

            result = self.reconstruct_from_gradient(
                gradient,
                target_model,
                batch_size=batch_size,
                **kwargs
            )

            results.append(result)

        return results


class DefendedGISMNAttack(GISMNAttack):
    """
    GI-SMN Attack against defended gradients

    Extends GISMNAttack to handle:
    - Differential privacy (DP) noise
    - Gradient clipping/pruning
    - Gradient compression

    Args:
        Same as GISMNAttack, plus:
        defense_type (str): Type of defense ('dp_noise', 'clipping', 'pruning', 'none')
        defense_params (dict): Defense-specific parameters
    """

    def __init__(
        self,
        generator: StyleGANWrapper,
        defense_type: str = 'none',
        defense_params: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(generator, **kwargs)
        self.defense_type = defense_type
        self.defense_params = defense_params or {}

    def _apply_defense(
        self,
        gradient: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply defense mechanism to gradient (for simulation)

        Args:
            gradient (dict): Original gradient

        Returns:
            defended_gradient (dict): Gradient after defense
        """
        if self.defense_type == 'none':
            return gradient

        defended_gradient = {}

        for name, grad in gradient.items():
            if self.defense_type == 'dp_noise':
                # Add Gaussian noise
                noise_scale = self.defense_params.get('noise_scale', 0.001)
                noise = torch.randn_like(grad) * noise_scale
                defended_gradient[name] = grad + noise

            elif self.defense_type == 'clipping':
                # Gradient clipping
                clip_value = self.defense_params.get('clip_value', 1.0)
                norm = torch.norm(grad)
                if norm > clip_value:
                    defended_gradient[name] = grad / norm * clip_value
                else:
                    defended_gradient[name] = grad

            elif self.defense_type == 'pruning':
                # Gradient pruning (keep top-k%)
                keep_ratio = self.defense_params.get('keep_ratio', 0.1)
                flat_grad = grad.view(-1)
                k = int(len(flat_grad) * keep_ratio)
                topk_values, topk_indices = torch.topk(flat_grad.abs(), k)

                pruned_grad = torch.zeros_like(flat_grad)
                pruned_grad[topk_indices] = flat_grad[topk_indices]
                defended_gradient[name] = pruned_grad.view(grad.shape)

            else:
                raise ValueError(f"Unknown defense type: {self.defense_type}")

        return defended_gradient


# Example usage
if __name__ == '__main__':
    from ..models.stylegan_wrapper import StyleGANWrapper

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load generator (placeholder for testing)
    generator = StyleGANWrapper(
        model_path='pretrained_models/stylegan_xl_cifar10.pkl',
        device=device,
        latent_dim=64,
        img_resolution=32
    )

    # Create a simple target model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 32 * 32, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    target_model = SimpleModel()

    # Simulate true gradient (from victim)
    true_images = torch.randn(1, 3, 32, 32, device=device)
    true_labels = torch.tensor([5], device=device)

    target_model.zero_grad()
    outputs = target_model(true_images)
    loss = nn.CrossEntropyLoss()(outputs, true_labels)
    loss.backward()

    true_gradient = {name: param.grad.clone() for name, param in target_model.named_parameters()}

    # Run GI-SMN attack
    attack = GISMNAttack(
        generator=generator,
        device=device,
        num_iterations=1000,  # Reduced for testing
        verbose=True
    )

    result = attack.reconstruct_from_gradient(
        true_gradient=true_gradient,
        target_model=target_model,
        batch_size=1,
        ground_truth_labels=true_labels,
        save_intermediate=True
    )

    print(f"\nReconstruction complete!")
    print(f"Reconstructed images shape: {result['reconstructed_images'].shape}")
    print(f"Final loss: {result['loss_history'][-1]['total']:.4f}")
