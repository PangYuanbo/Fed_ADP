"""
Federated Learning GIA Evaluator

Integrates GI-SMN attack into the Fed_ADP federated learning framework.
Evaluates clients' vulnerability to gradient inversion attacks.

Author: Fed_ADP Project
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import copy

from .models.stylegan_wrapper import StyleGANWrapper
from .core.gi_smn_attack import GISMNAttack
from .evaluation.metrics import ImageQualityMetrics
from .evaluation.defense_tester import DefenseTester
from .evaluation.visualizer import GIAVisualizer
from .utils.config import GIAConfig
from .utils.gradient_utils import extract_gradients


class FederatedGIAEvaluator:
    """
    GI-SMN Attack Evaluator for Federated Learning

    Integrates with Fed_ADP to evaluate gradient inversion vulnerability
    of federated learning clients.

    Args:
        stylegan_model_path (str): Path to pretrained StyleGAN-XL model
        device (str): Device for computation ('cuda' or 'cpu')
        config (GIAConfig, optional): Attack configuration
        results_dir (str): Directory to save results
    """

    def __init__(
        self,
        stylegan_model_path: str,
        device: str = 'cuda',
        config: Optional[GIAConfig] = None,
        results_dir: str = 'gia_results'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or GIAConfig()
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        print(f"[GIA Evaluator] Initializing...")
        print(f"[GIA Evaluator] Device: {self.device}")
        print(f"[GIA Evaluator] Results dir: {results_dir}")

        # Initialize components
        self.generator = self._init_generator(stylegan_model_path)
        self.attack = self._init_attack()
        self.metrics = ImageQualityMetrics(device=str(self.device))
        self.visualizer = GIAVisualizer(save_dir=os.path.join(results_dir, 'visualizations'))

        # Optional: Defense tester
        self.defense_tester = None
        if self.config.test_defense:
            self.defense_tester = DefenseTester(self.attack, self.metrics)

        print(f"[GIA Evaluator] Initialization complete")

    def _init_generator(self, model_path: str) -> StyleGANWrapper:
        """Initialize StyleGAN-XL generator"""
        print(f"[GIA Evaluator] Loading StyleGAN-XL from {model_path}")

        generator = StyleGANWrapper(
            model_path=model_path,
            device=str(self.device),
            latent_dim=self.config.latent_dim,
            img_resolution=32  # CIFAR-10 resolution
        )

        return generator

    def _init_attack(self) -> GISMNAttack:
        """Initialize GI-SMN attack"""
        attack = GISMNAttack(
            generator=self.generator,
            device=str(self.device),
            num_iterations=self.config.num_iterations,
            learning_rate=self.config.learning_rate,
            lambda_tv=self.config.lambda_tv,
            lambda_l2=self.config.lambda_l2,
            lambda_group=self.config.lambda_group,
            stage1_ratio=self.config.stage1_ratio,
            latent_clamp=self.config.latent_clamp,
            verbose=True
        )

        return attack

    def evaluate_client(
        self,
        client,
        num_samples: int = None,
        round_num: Optional[int] = None,
        save_visuals: bool = None
    ) -> Dict:
        """
        Evaluate a single client's vulnerability to GI-SMN attack

        Args:
            client: Client object from Fed_ADP
            num_samples (int, optional): Number of samples to attack
            round_num (int, optional): Current training round
            save_visuals (bool, optional): Whether to save visualizations

        Returns:
            result (dict): Evaluation results including metrics and reconstructions
        """
        if num_samples is None:
            num_samples = self.config.num_samples_per_client
        if save_visuals is None:
            save_visuals = self.config.save_visuals

        print(f"\n[GIA Evaluator] Evaluating client {client.id}")
        print(f"[GIA Evaluator] Samples: {num_samples}, Round: {round_num}")

        # Get client's model
        client_model = copy.deepcopy(client.model)
        client_model.eval()

        # Get training data from client
        try:
            train_loader = client.load_train_data()
        except:
            print(f"[GIA Evaluator] Warning: Could not load client train data")
            return {'status': 'error', 'message': 'Failed to load train data'}

        # Collect samples to attack
        sample_results = []
        samples_collected = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if samples_collected >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract gradient from this batch
            true_gradient = extract_gradients(client_model, images, labels)

            # Run GI-SMN attack
            attack_result = self.attack.reconstruct_from_gradient(
                true_gradient=true_gradient,
                target_model=client_model,
                batch_size=images.shape[0],
                ground_truth_labels=labels,
                save_intermediate=self.config.save_intermediate
            )

            # Evaluate reconstruction quality
            quality_metrics = self.metrics.calculate_all(
                attack_result['reconstructed_images'],
                images.cpu()
            )

            # Save visualization if requested
            if save_visuals:
                vis_path = os.path.join(
                    self.results_dir,
                    f"client_{client.id}",
                    f"sample_{samples_collected}_round_{round_num}.png"
                )
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)

                self.visualizer.plot_reconstruction_comparison(
                    original=images.cpu(),
                    reconstructed=attack_result['reconstructed_images'],
                    metrics=quality_metrics,
                    save_path=vis_path,
                    title=f"Client {client.id} - Sample {samples_collected}"
                )

            # Store result
            sample_results.append({
                'sample_idx': samples_collected,
                'batch_size': images.shape[0],
                'metrics': quality_metrics,
                'final_loss': attack_result['loss_history'][-1]['total']
            })

            samples_collected += 1

        # Compute average metrics
        avg_metrics = self._average_metrics(sample_results)

        result = {
            'client_id': client.id,
            'round': round_num,
            'num_samples': samples_collected,
            'average_metrics': avg_metrics,
            'sample_results': sample_results,
            'status': 'success'
        }

        return result

    def evaluate_all_clients(
        self,
        clients: List,
        round_num: int,
        dataset_name: str = 'unknown',
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate all clients in a federated learning round

        Args:
            clients (list): List of client objects
            round_num (int): Current training round
            dataset_name (str): Name of dataset
            save_results (bool): Whether to save results to disk

        Returns:
            all_results (dict): Combined results for all clients
        """
        print(f"\n{'='*60}")
        print(f"[GIA Evaluator] Evaluating Round {round_num}")
        print(f"[GIA Evaluator] Dataset: {dataset_name}")
        print(f"[GIA Evaluator] Clients: {len(clients)}")
        print(f"{'='*60}\n")

        client_results = {}

        for client in clients:
            try:
                result = self.evaluate_client(
                    client=client,
                    round_num=round_num,
                    num_samples=self.config.num_samples_per_client,
                    save_visuals=self.config.save_visuals
                )
                client_results[client.id] = result

            except Exception as e:
                print(f"[GIA Evaluator] Error evaluating client {client.id}: {e}")
                client_results[client.id] = {
                    'status': 'error',
                    'message': str(e)
                }

        # Compute summary statistics
        summary = self._compute_summary(client_results)

        all_results = {
            'round': round_num,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'dataset': dataset_name,
            'config': self.config.to_dict(),
            'client_results': client_results,
            'summary': summary,
        }

        # Save results
        if save_results:
            self._save_results(all_results, round_num)

        # Print summary
        self._print_summary(summary)

        return all_results

    def test_defense_mechanisms(
        self,
        client,
        num_samples: int = 3
    ) -> Dict:
        """
        Test defense mechanisms against GI-SMN attack

        Args:
            client: Client object
            num_samples (int): Number of samples to test

        Returns:
            defense_results (dict): Results for all defense tests
        """
        if self.defense_tester is None:
            print("[GIA Evaluator] Defense testing not enabled")
            return {}

        print(f"\n[GIA Evaluator] Testing defense mechanisms on client {client.id}")

        # Get a sample batch
        train_loader = client.load_train_data()
        images, labels = next(iter(train_loader))
        images = images[:num_samples].to(self.device)
        labels = labels[:num_samples].to(self.device)

        # Extract clean gradient
        client_model = copy.deepcopy(client.model)
        true_gradient = extract_gradients(client_model, images, labels)

        # Test different defenses
        defense_results = {}

        # 1. DP Noise
        dp_results = self.defense_tester.test_dp_noise(
            true_gradient, client_model, images,
            noise_levels=self.config.dp_noise_levels
        )
        defense_results['dp_noise'] = dp_results

        # 2. Gradient Clipping
        clip_results = self.defense_tester.test_gradient_clipping(
            true_gradient, client_model, images,
            clip_values=self.config.clip_values
        )
        defense_results['gradient_clipping'] = clip_results

        # Visualize defense impact
        for defense_type, results in defense_results.items():
            vis_path = os.path.join(
                self.results_dir,
                'defense_tests',
                f'{defense_type}_impact.png'
            )
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)

            self.visualizer.plot_defense_impact(
                results,
                defense_type=defense_type.replace('_', ' ').title(),
                save_path=vis_path
            )

        return defense_results

    def _average_metrics(self, sample_results: List[Dict]) -> Dict:
        """Compute average metrics across samples"""
        if not sample_results:
            return {}

        metrics_list = [r['metrics'] for r in sample_results]

        avg_metrics = {}
        for key in ['psnr', 'ssim', 'lpips']:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        # Quality assessment (majority vote)
        qualities = [m.get('quality_assessment', 'unknown') for m in metrics_list]
        avg_metrics['quality_assessment'] = max(set(qualities), key=qualities.count)

        return avg_metrics

    def _compute_summary(self, client_results: Dict) -> Dict:
        """Compute summary statistics across all clients"""
        successful_clients = [r for r in client_results.values() if r.get('status') == 'success']

        if not successful_clients:
            return {'num_clients': len(client_results), 'successful_clients': 0}

        # Collect average PSNR values
        psnr_values = [
            r['average_metrics'].get('psnr')
            for r in successful_clients
            if r.get('average_metrics', {}).get('psnr') is not None
        ]

        summary = {
            'num_clients': len(client_results),
            'successful_clients': len(successful_clients),
            'avg_psnr': sum(psnr_values) / len(psnr_values) if psnr_values else None,
            'max_psnr': max(psnr_values) if psnr_values else None,
            'min_psnr': min(psnr_values) if psnr_values else None,
        }

        # Count quality levels
        quality_levels = [r['average_metrics'].get('quality_assessment', 'unknown') for r in successful_clients]
        summary['high_quality_reconstructions'] = quality_levels.count('high')
        summary['medium_quality_reconstructions'] = quality_levels.count('medium')
        summary['low_quality_reconstructions'] = quality_levels.count('low')

        return summary

    def _save_results(self, results: Dict, round_num: int):
        """Save results to JSON file"""
        filename = f"gia_results_round_{round_num}_{results['timestamp']}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[GIA Evaluator] Results saved to {filepath}")

    def _print_summary(self, summary: Dict):
        """Print summary to console"""
        print(f"\n{'='*60}")
        print(f"[GIA Evaluator] Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total clients: {summary.get('num_clients', 0)}")
        print(f"Successful evaluations: {summary.get('successful_clients', 0)}")
        if summary.get('avg_psnr'):
            print(f"Average PSNR: {summary['avg_psnr']:.2f} dB")
            print(f"PSNR range: [{summary['min_psnr']:.2f}, {summary['max_psnr']:.2f}] dB")
        print(f"High quality: {summary.get('high_quality_reconstructions', 0)} clients")
        print(f"Medium quality: {summary.get('medium_quality_reconstructions', 0)} clients")
        print(f"Low quality: {summary.get('low_quality_reconstructions', 0)} clients")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    print("FederatedGIAEvaluator implementation complete.")
    print("Usage: See run_gia_standalone.py for usage examples.")
