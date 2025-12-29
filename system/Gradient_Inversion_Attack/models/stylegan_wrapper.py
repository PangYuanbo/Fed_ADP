"""
StyleGAN-XL Wrapper for GI-SMN Attack

This module provides a wrapper around StyleGAN-XL generator for image reconstruction
from latent codes. Supports loading pretrained models and batch generation.

Author: Fed_ADP Project
Reference: https://github.com/autonomousvision/stylegan-xl
"""

import torch
import torch.nn as nn
import os
import sys
import pickle
import warnings
from typing import Optional, Tuple
from pathlib import Path


class StyleGANWrapper:
    """
    Wrapper class for StyleGAN-XL generator

    Provides a simplified interface for:
    - Loading pretrained models (CIFAR-10, ImageNet, etc.)
    - Generating images from latent codes
    - Batch processing and GPU acceleration

    Args:
        model_path (str): Path to pretrained StyleGAN-XL model (.pkl file)
        device (str): Device for computation ('cuda' or 'cpu')
        latent_dim (int): Dimension of latent space (default: 64 for CIFAR-10)
        img_resolution (int): Output image resolution (default: 32)
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        latent_dim: int = 64,
        img_resolution: int = 32
    ):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.img_resolution = img_resolution

        # Load generator
        self.generator = self._load_generator()
        self.generator.eval()

        print(f"[StyleGAN] Loaded generator from {model_path}")
        print(f"[StyleGAN] Latent dim: {latent_dim}, Resolution: {img_resolution}x{img_resolution}")

    def _load_generator(self) -> nn.Module:
        """
        Load StyleGAN-XL generator from pretrained model

        Returns:
            generator (nn.Module): Loaded generator model
        """
        if not os.path.exists(self.model_path):
            warnings.warn(
                f"Model file not found: {self.model_path}\n"
                f"Please download from: https://github.com/autonomousvision/stylegan-xl\n"
                f"Or run: bash system/Gradient_Inversion_Attack/setup.sh\n"
                f"Falling back to placeholder generator for testing."
            )
            return self._create_placeholder_generator()

        # Add StyleGAN-XL repository to path BEFORE any loading attempts
        stylegan_repo = Path(__file__).parent.parent / 'stylegan_xl_repo'
        if stylegan_repo.exists() and str(stylegan_repo) not in sys.path:
            sys.path.insert(0, str(stylegan_repo))
            print(f"[StyleGAN] Added StyleGAN-XL repo to path: {stylegan_repo}")

        try:
            # Try loading StyleGAN-XL model using legacy.load_network_pkl
            try:
                import dnnlib
                import legacy

                print(f"[StyleGAN] Loading model using legacy.load_network_pkl...")
                with dnnlib.util.open_url(self.model_path) as f:
                    model_data = legacy.load_network_pkl(f)

                # Extract generator (usually stored as 'G_ema')
                if 'G_ema' in model_data:
                    generator = model_data['G_ema']
                elif 'G' in model_data:
                    generator = model_data['G']
                else:
                    raise KeyError("Cannot find generator in model file (expected 'G' or 'G_ema')")

                print(f"[StyleGAN] Successfully loaded StyleGAN-XL generator")

            except (ImportError, ModuleNotFoundError) as e:
                # Fallback to direct pickle load
                print(f"[StyleGAN] legacy module not available, trying direct pickle load...")
                # Force all storages to target device
                if torch.cuda.is_available():
                    map_location = self.device
                else:
                    # Use lambda to force ALL storages to CPU, even CUDA ones
                    map_location = lambda storage, loc: storage.cpu()
                print(f"[StyleGAN] Using map_location to force device mapping")
                with open(self.model_path, 'rb') as f:
                    # Use torch.load with map_location and weights_only=False
                    # Note: weights_only=False is needed for StyleGAN-XL models
                    model_data = torch.load(f, map_location=map_location, weights_only=False)

                # Extract generator (usually stored as 'G' or 'G_ema')
                if isinstance(model_data, dict):
                    if 'G_ema' in model_data:
                        generator = model_data['G_ema']
                    elif 'G' in model_data:
                        generator = model_data['G']
                    else:
                        raise KeyError("Cannot find generator in model file (expected 'G' or 'G_ema')")
                else:
                    generator = model_data

            generator = generator.to(self.device)
            generator.eval()
            return generator

        except Exception as e:
            warnings.warn(
                f"Failed to load StyleGAN-XL model: {e}\n"
                f"Falling back to simple generator placeholder"
            )
            # Fallback: simple generator for testing
            return self._create_placeholder_generator()

    def _create_placeholder_generator(self) -> nn.Module:
        """
        Create a placeholder generator for testing when StyleGAN-XL is unavailable

        This is a simple MLP-based generator that can be used for initial testing.
        WARNING: This will NOT produce realistic images!
        """
        class SimpleGenerator(nn.Module):
            def __init__(self, latent_dim, img_resolution):
                super().__init__()
                output_dim = 3 * img_resolution * img_resolution
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, output_dim),
                    nn.Tanh()
                )
                self.img_resolution = img_resolution

            def forward(self, z, c=None, **kwargs):
                batch_size = z.shape[0]
                flat_img = self.net(z)
                img = flat_img.view(batch_size, 3, self.img_resolution, self.img_resolution)
                return img

        warnings.warn("Using placeholder generator - results will not be realistic!")
        return SimpleGenerator(self.latent_dim, self.img_resolution).to(self.device)

    def generate_from_latent(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        noise_mode: str = 'const'
    ) -> torch.Tensor:
        """
        Generate images from latent codes

        Args:
            z (torch.Tensor): Latent codes of shape [batch_size, latent_dim]
            truncation_psi (float): Truncation parameter for style mixing (default: 1.0)
            noise_mode (str): Noise injection mode ('const', 'random', 'none')

        Returns:
            images (torch.Tensor): Generated images of shape [batch_size, 3, H, W]
                                   Values in range [-1, 1]
        """
        assert z.shape[1] == self.latent_dim, \
            f"Expected latent dim {self.latent_dim}, got {z.shape[1]}"

        z = z.to(self.device)

        # Keep gradients for GI-SMN attack optimization
        try:
            # Try StyleGAN-XL interface
            images = self.generator(
                z,
                c=None,  # No class conditioning for CIFAR-10
                truncation_psi=truncation_psi,
                noise_mode=noise_mode
            )
        except TypeError:
            # Fallback for simpler generators
            images = self.generator(z)

        return images

    def sample_latent(
        self,
        batch_size: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample random latent codes from Gaussian distribution

        Args:
            batch_size (int): Number of samples
            seed (int, optional): Random seed for reproducibility

        Returns:
            z (torch.Tensor): Sampled latent codes of shape [batch_size, latent_dim]
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return z

    def generate_random(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate random images by sampling latent codes

        Args:
            batch_size (int): Number of images to generate
            seed (int, optional): Random seed
            **kwargs: Additional arguments for generate_from_latent()

        Returns:
            images (torch.Tensor): Generated images
        """
        z = self.sample_latent(batch_size, seed)
        return self.generate_from_latent(z, **kwargs)

    def optimize_latent(
        self,
        target_gradient: dict,
        target_model: nn.Module,
        num_iterations: int = 10000,
        learning_rate: float = 0.01,
        loss_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize latent code to match target gradient (core of GI-SMN)

        This is a placeholder method. The actual implementation is in
        core/gi_smn_attack.py for better modularity.

        Args:
            target_gradient (dict): Target gradient from victim model
            target_model (nn.Module): Victim model architecture
            num_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimizer
            loss_fn (callable, optional): Custom loss function

        Returns:
            optimal_z (torch.Tensor): Optimized latent code
            reconstructed_image (torch.Tensor): Final reconstructed image
        """
        raise NotImplementedError(
            "This method is a placeholder. Use GISMNAttack from core.gi_smn_attack instead."
        )

    @staticmethod
    def download_pretrained_model(
        model_name: str = 'cifar10',
        save_dir: str = 'pretrained_models'
    ) -> str:
        """
        Download pretrained StyleGAN-XL model

        Args:
            model_name (str): Model identifier ('cifar10', 'imagenet256', etc.)
            save_dir (str): Directory to save the model

        Returns:
            model_path (str): Path to downloaded model
        """
        import urllib.request

        os.makedirs(save_dir, exist_ok=True)

        # Model URLs (from StyleGAN-XL official repository)
        model_urls = {
            'cifar10': 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl',
            'imagenet256': 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl',
        }

        if model_name not in model_urls:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_urls.keys())}")

        url = model_urls[model_name]
        model_path = os.path.join(save_dir, f"stylegan_xl_{model_name}.pkl")

        if os.path.exists(model_path):
            print(f"[StyleGAN] Model already exists: {model_path}")
            return model_path

        print(f"[StyleGAN] Downloading {model_name} from {url}...")
        print(f"[StyleGAN] This may take several minutes...")

        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[StyleGAN] Download complete: {model_path}")
            return model_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")


# Example usage
if __name__ == '__main__':
    # Test the wrapper
    model_path = 'pretrained_models/stylegan_xl_cifar10.pkl'

    # Create wrapper
    generator = StyleGANWrapper(
        model_path=model_path,
        device='cuda',
        latent_dim=64,
        img_resolution=32
    )

    # Generate random images
    images = generator.generate_random(batch_size=4, seed=42)
    print(f"Generated images shape: {images.shape}")
    print(f"Value range: [{images.min():.2f}, {images.max():.2f}]")
