"""
Standalone GI-SMN Attack Evaluation Script

Run GIA evaluation independently without full FL training.
Useful for testing and analyzing specific models.

Author: Fed_ADP Project
Usage:
    python run_gia_standalone.py --model_path results/model.pt --test_defense
"""

import torch
import argparse
import os
import sys

# Add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Gradient_Inversion_Attack.gia_evaluator import FederatedGIAEvaluator
from Gradient_Inversion_Attack.utils.config import GIAConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Standalone GI-SMN Attack Evaluation')

    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--stylegan_path', type=str,
                        default='Gradient_Inversion_Attack/pretrained_models/stylegan_xl_cifar10.pkl',
                        help='Path to StyleGAN-XL model')
    parser.add_argument('--dataset', type=str, default='cifar-10-normal',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory')

    # Attack parameters
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to attack')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')

    # Evaluation options
    parser.add_argument('--test_defense', action='store_true',
                        help='Test defense mechanisms')
    parser.add_argument('--save_visuals', action='store_true', default=True,
                        help='Save visualization results')
    parser.add_argument('--results_dir', type=str, default='gia_results_standalone',
                        help='Results directory')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for computation')

    args = parser.parse_args()
    return args


def load_model_and_data(args):
    """Load model and sample data"""
    print(f"[Standalone] Loading model from {args.model_path}")

    # TODO: Implement model loading based on your model architecture
    # This is a placeholder - adapt to your specific model structure
    from flcore.trainmodel.models import *

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Create model (you may need to adjust this based on your architecture)
    if 'cifar' in args.dataset.lower():
        model = CifarNet()  # Replace with your actual model class
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[Standalone] Model loaded successfully")

    # Load sample data (you may need to implement this based on your data structure)
    # For now, we'll create dummy data
    batch_size = min(args.num_samples, 4)
    sample_images = torch.randn(batch_size, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (batch_size,))

    print(f"[Standalone] Loaded {batch_size} sample images")

    return model, sample_images, sample_labels


def main():
    args = parse_args()

    print("="*60)
    print("GI-SMN Standalone Attack Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"StyleGAN: {args.stylegan_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Iterations: {args.iterations}")
    print(f"Test defense: {args.test_defense}")
    print("="*60 + "\n")

    # Create config
    config = GIAConfig(
        num_iterations=args.iterations,
        learning_rate=args.lr,
        num_samples_per_client=args.num_samples,
        save_visuals=args.save_visuals,
        test_defense=args.test_defense
    )

    # Initialize evaluator
    evaluator = FederatedGIAEvaluator(
        stylegan_model_path=args.stylegan_path,
        device=args.device,
        config=config,
        results_dir=args.results_dir
    )

    # Load model and data
    model, sample_images, sample_labels = load_model_and_data(args)
    model = model.to(args.device)
    sample_images = sample_images.to(args.device)
    sample_labels = sample_labels.to(args.device)

    # Extract gradient from sample
    from Gradient_Inversion_Attack.utils.gradient_utils import extract_gradients

    print("\n[Standalone] Extracting gradient from sample batch...")
    true_gradient = extract_gradients(model, sample_images, sample_labels)

    # Run attack
    print("\n[Standalone] Running GI-SMN attack...")
    attack_result = evaluator.attack.reconstruct_from_gradient(
        true_gradient=true_gradient,
        target_model=model,
        batch_size=sample_images.shape[0],
        ground_truth_labels=sample_labels,
        save_intermediate=False
    )

    # Evaluate quality
    print("\n[Standalone] Evaluating reconstruction quality...")
    quality_metrics = evaluator.metrics.calculate_all(
        attack_result['reconstructed_images'],
        sample_images.cpu()
    )

    print("\n" + "="*60)
    print("Attack Results:")
    print("="*60)
    print(f"PSNR: {quality_metrics.get('psnr', 'N/A'):.2f} dB")
    print(f"SSIM: {quality_metrics.get('ssim', 'N/A'):.4f}")
    if quality_metrics.get('lpips') is not None:
        print(f"LPIPS: {quality_metrics.get('lpips', 'N/A'):.4f}")
    print(f"Quality: {quality_metrics.get('quality_assessment', 'unknown')}")
    print(f"Final loss: {attack_result['loss_history'][-1]['total']:.4f}")
    print("="*60 + "\n")

    # Save visualization
    if args.save_visuals:
        vis_path = os.path.join(args.results_dir, 'reconstruction_comparison.png')
        evaluator.visualizer.plot_reconstruction_comparison(
            original=sample_images.cpu(),
            reconstructed=attack_result['reconstructed_images'],
            metrics=quality_metrics,
            save_path=vis_path,
            title="GI-SMN Attack Result (Standalone)"
        )

        # Save loss history
        loss_path = os.path.join(args.results_dir, 'loss_history.png')
        evaluator.visualizer.plot_loss_history(
            loss_history=attack_result['loss_history'],
            save_path=loss_path
        )

    # Test defenses if requested
    if args.test_defense and evaluator.defense_tester is not None:
        print("\n[Standalone] Testing defense mechanisms...")

        # DP Noise
        print("\nTesting DP noise defense...")
        dp_results = evaluator.defense_tester.test_dp_noise(
            true_gradient, model, sample_images,
            noise_levels=config.dp_noise_levels
        )

        # Gradient Clipping
        print("\nTesting gradient clipping defense...")
        clip_results = evaluator.defense_tester.test_gradient_clipping(
            true_gradient, model, sample_images,
            clip_values=config.clip_values
        )

        # Visualize
        evaluator.visualizer.plot_defense_impact(
            dp_results,
            defense_type='DP Noise',
            save_path=os.path.join(args.results_dir, 'defense_dp_noise.png')
        )

        evaluator.visualizer.plot_defense_impact(
            clip_results,
            defense_type='Gradient Clipping',
            save_path=os.path.join(args.results_dir, 'defense_clipping.png')
        )

        print("\nDefense test results:")
        print(f"  DP Noise summary: {dp_results.get('summary', {})}")
        print(f"  Clipping summary: {clip_results.get('summary', {})}")

    print(f"\n[Standalone] Results saved to {args.results_dir}")
    print("[Standalone] Evaluation complete!\n")


if __name__ == '__main__':
    main()
