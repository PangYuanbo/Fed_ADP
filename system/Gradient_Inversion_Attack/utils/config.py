"""
Configuration Management for GI-SMN Attack

Provides configuration classes and utilities for attack parameters.

Author: Fed_ADP Project
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class GIAConfig:
    """
    Configuration for GI-SMN Attack

    Attack parameters:
        num_iterations: Total optimization iterations
        learning_rate: Learning rate for latent optimization
        latent_dim: Dimension of latent space
        latent_clamp: Clamp latent codes to [-c, c]
        stage1_ratio: Stage 1 duration ratio (default: 4/9)

    Loss weights:
        lambda_tv: Total variation regularization weight
        lambda_l2: L2 regularization weight
        lambda_group: Group consistency weight

    Evaluation:
        num_samples_per_client: Number of samples to reconstruct per client
        save_intermediate: Save intermediate reconstruction results
        save_visuals: Generate visualization plots

    Defense testing:
        test_defense: Whether to test defense mechanisms
        dp_noise_levels: List of DP noise levels to test
        clip_values: List of gradient clip values to test
    """

    # Attack parameters
    num_iterations: int = 10000
    learning_rate: float = 0.01
    latent_dim: int = 64
    latent_clamp: float = 2.0
    stage1_ratio: float = 4/9

    # Loss weights
    lambda_tv: float = 0.001
    lambda_l2: float = 0.0001
    lambda_group: float = 0.01

    # Evaluation
    num_samples_per_client: int = 5
    save_intermediate: bool = False
    save_visuals: bool = True

    # Defense testing
    test_defense: bool = False
    dp_noise_levels: list = None
    clip_values: list = None

    def __post_init__(self):
        """Initialize default values for lists"""
        if self.dp_noise_levels is None:
            self.dp_noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        if self.clip_values is None:
            self.clip_values = [0.001, 0.005, 0.01, 0.05, 0.1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GIAConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    def save_yaml(self, filepath: str):
        """Save config to YAML file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"[Config] Saved configuration to {filepath}")

    @classmethod
    def load_yaml(cls, filepath: str) -> 'GIAConfig':
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"[Config] Loaded configuration from {filepath}")
        return cls.from_dict(config_dict)

    def update(self, **kwargs):
        """Update config with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: Unknown parameter '{key}'")

    def __str__(self) -> str:
        """String representation"""
        lines = ["GI-SMN Attack Configuration:"]
        lines.append("  Attack Parameters:")
        lines.append(f"    - Iterations: {self.num_iterations}")
        lines.append(f"    - Learning rate: {self.learning_rate}")
        lines.append(f"    - Latent dim: {self.latent_dim}")
        lines.append(f"    - Stage 1 ratio: {self.stage1_ratio:.3f}")
        lines.append("  Loss Weights:")
        lines.append(f"    - λ_TV: {self.lambda_tv}")
        lines.append(f"    - λ_L2: {self.lambda_l2}")
        lines.append(f"    - λ_group: {self.lambda_group}")
        lines.append("  Evaluation:")
        lines.append(f"    - Samples per client: {self.num_samples_per_client}")
        lines.append(f"    - Test defense: {self.test_defense}")
        return "\n".join(lines)


def get_default_config() -> GIAConfig:
    """Get default configuration"""
    return GIAConfig()


def create_config_from_args(args) -> GIAConfig:
    """
    Create GIA config from command-line arguments

    Args:
        args: Argument namespace from argparse

    Returns:
        config (GIAConfig): GIA configuration
    """
    config = GIAConfig()

    # Update from args if available
    if hasattr(args, 'gia_iterations'):
        config.num_iterations = args.gia_iterations
    if hasattr(args, 'gia_lr'):
        config.learning_rate = args.gia_lr
    if hasattr(args, 'gia_num_samples'):
        config.num_samples_per_client = args.gia_num_samples
    if hasattr(args, 'gia_save_visuals'):
        config.save_visuals = args.gia_save_visuals
    if hasattr(args, 'gia_test_defense'):
        config.test_defense = args.gia_test_defense

    return config


if __name__ == '__main__':
    # Test configuration
    config = GIAConfig()
    print(config)

    # Save and load
    config.save_yaml('test_config.yaml')
    loaded_config = GIAConfig.load_yaml('test_config.yaml')
    print("\nLoaded config:")
    print(loaded_config)
