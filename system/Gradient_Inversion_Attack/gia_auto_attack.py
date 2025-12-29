"""
GIA Auto Attack and Visualization Tool
自动扫描预训练模型并执行梯度反演攻击

Usage:
    python Gradient_Inversion_Attack/gia_auto_attack.py --pretrain_dir pretrain/cifar-10-normal/1.00
    python Gradient_Inversion_Attack/gia_auto_attack.py --scan_all  # 扫描所有模型
"""

import os
import sys
from pathlib import Path

# 添加项目路径（必须在其他导入之前）
current_file = Path(__file__).resolve()
system_dir = current_file.parent.parent  # system directory

# 添加system目录到sys.path（绝对路径）
system_dir_str = str(system_dir)
if system_dir_str not in sys.path:
    sys.path.insert(0, system_dir_str)

# 调试信息（如果需要的话可以取消注释）
# print(f"[GIA] system_dir: {system_dir_str}")
# print(f"[GIA] sys.path[0]: {sys.path[0]}")

import torch
import torch.nn as nn
import argparse
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

# 导入GIA模块
from Gradient_Inversion_Attack.models.stylegan_wrapper import StyleGANWrapper
from Gradient_Inversion_Attack.core.gi_smn_attack import GISMNAttack
from Gradient_Inversion_Attack.evaluation.metrics import ImageQualityMetrics
from Gradient_Inversion_Attack.evaluation.visualizer import GIAVisualizer
from Gradient_Inversion_Attack.utils.gradient_utils import extract_gradients
from Gradient_Inversion_Attack.utils.config import GIAConfig

# 导入系统工具
from utils.data_utils import read_client_data

# 导入模型定义
from flcore.trainmodel.models import *


class GIAAutoAttacker:
    """
    自动化GIA攻击器
    扫描预训练模型并执行梯度反演攻击
    """

    def __init__(
        self,
        stylegan_model_path: str,
        device: str = 'cuda',
        gia_config: Optional[GIAConfig] = None,
        results_dir: str = 'gia_auto_results'
    ):
        """
        初始化自动攻击器

        Args:
            stylegan_model_path: StyleGAN-XL模型路径
            device: 计算设备
            gia_config: GIA配置
            results_dir: 结果保存目录
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        print(f"[GIA Auto Attacker] Initializing...")
        print(f"[GIA Auto Attacker] Device: {self.device}")

        # 加载StyleGAN-XL生成器
        self.generator = StyleGANWrapper(
            model_path=stylegan_model_path,
            device=str(self.device),
            latent_dim=64,
            img_resolution=32
        )

        # 创建GIA攻击器
        if gia_config is None:
            gia_config = GIAConfig(
                num_iterations=10000,
                learning_rate=0.01,
                num_samples_per_client=5
            )
        self.config = gia_config

        self.attacker = GISMNAttack(
            generator=self.generator,
            device=str(self.device),
            num_iterations=gia_config.num_iterations,
            learning_rate=gia_config.learning_rate,
            lambda_tv=gia_config.lambda_tv,
            lambda_l2=gia_config.lambda_l2,
            lambda_group=gia_config.lambda_group,
            stage1_ratio=gia_config.stage1_ratio
        )

        # 创建评估和可视化工具
        self.metrics = ImageQualityMetrics(device=str(self.device))
        self.visualizer = GIAVisualizer(save_dir=results_dir)

        print(f"[GIA Auto Attacker] Initialization complete!")

    def scan_pretrain_models(self, base_dir: str = 'pretrain') -> List[Dict]:
        """
        扫描pretrain目录中的所有模型

        Args:
            base_dir: 预训练模型基础目录

        Returns:
            List[Dict]: 模型信息列表
        """
        print(f"\n[Scan] Scanning models in {base_dir}...")

        model_files = glob.glob(f"{base_dir}/**/*.pt", recursive=True)
        model_info_list = []

        for model_path in model_files:
            # 解析路径: pretrain/{dataset}/{alpha}/results_client{id}_{rounds}.pt
            path_parts = Path(model_path).parts
            try:
                dataset = path_parts[-3]
                alpha = float(path_parts[-2])
                filename = path_parts[-1]

                # 解析文件名
                if filename.startswith('results_client'):
                    parts = filename.replace('results_client', '').replace('.pt', '').split('_')
                    client_id = int(parts[0])
                    rounds = int(parts[1]) if len(parts) > 1 else 0

                    model_info = {
                        'model_path': model_path,
                        'dataset': dataset,
                        'alpha': alpha,
                        'client_id': client_id,
                        'rounds': rounds,
                        'is_global': (client_id >= 1000)  # 全局模型通常用大编号
                    }
                    model_info_list.append(model_info)

            except Exception as e:
                print(f"[Scan] Skipping invalid path {model_path}: {e}")
                continue

        print(f"[Scan] Found {len(model_info_list)} models")
        return model_info_list

    def load_model(self, model_path: str, dataset: str) -> torch.nn.Module:
        """
        加载预训练模型

        Args:
            model_path: 模型文件路径
            dataset: 数据集名称

        Returns:
            加载的模型
        """
        # 根据数据集创建模型架构
        if 'cifar' in dataset.lower():
            # CIFAR-10 使用 LocalModel 结构 (FedAvgCNN + head)
            feature_extractor = FedAvgCNN(in_features=3, num_classes=10, dim=1600, dim1=512)
            # 移除 FedAvgCNN 的最后一层 fc
            feature_extractor.fc = nn.Identity()
            head = nn.Linear(512, 10)
            model = LocalModel(feature_extractor, head)
        elif 'mnist' in dataset.lower():
            # MNIST 使用 FedAvgCNN
            feature_extractor = FedAvgCNN(in_features=1, num_classes=10, dim=1024, dim1=512)
            feature_extractor.fc = nn.Identity()
            head = nn.Linear(512, 10)
            model = LocalModel(feature_extractor, head)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        return model

    def attack_single_model(
        self,
        model_info: Dict,
        num_samples: int = 5,
        save_visuals: bool = True
    ) -> Dict:
        """
        对单个模型执行GIA攻击

        Args:
            model_info: 模型信息字典
            num_samples: 每个模型重建的样本数
            save_visuals: 是否保存可视化

        Returns:
            攻击结果字典
        """
        print(f"\n{'='*80}")
        print(f"[Attack] Model: {model_info['model_path']}")
        print(f"[Attack] Dataset: {model_info['dataset']}, Alpha: {model_info['alpha']}")
        print(f"[Attack] Client ID: {model_info['client_id']}, Rounds: {model_info['rounds']}")
        print(f"{'='*80}")

        try:
            # 加载模型
            model = self.load_model(model_info['model_path'], model_info['dataset'])
            print(f"[Attack] Model loaded successfully")

            # 加载对应客户端的训练数据
            train_data = read_client_data(
                dataset=model_info['dataset'],
                idx=model_info['client_id'],
                is_train=True,
                alpha=model_info['alpha']
            )
            print(f"[Attack] Loaded {len(train_data)} training samples")

            # 限制样本数
            num_samples = min(num_samples, len(train_data))

            # 执行攻击并收集结果
            attack_results = []
            for sample_idx in range(num_samples):
                print(f"\n[Attack] Sample {sample_idx + 1}/{num_samples}")

                # 获取真实数据
                real_image, real_label = train_data[sample_idx]
                real_image = real_image.unsqueeze(0).to(self.device)
                real_label = torch.tensor([real_label], dtype=torch.long).to(self.device)

                # 提取真实梯度
                true_gradient = extract_gradients(
                    model=model,
                    images=real_image,
                    labels=real_label,
                    criterion=torch.nn.CrossEntropyLoss()
                )

                # 执行GI-SMN攻击
                result = self.attacker.reconstruct_from_gradient(
                    true_gradient=true_gradient,
                    target_model=model,
                    batch_size=1,
                    ground_truth_labels=real_label,
                    save_intermediate=False
                )

                # 计算图像质量指标
                reconstructed = result['reconstructed_images'].cpu()
                original = real_image.cpu()

                quality = self.metrics.calculate_all(reconstructed, original)

                print(f"  PSNR: {quality['psnr']:.2f} dB")
                print(f"  SSIM: {quality['ssim']:.4f}")
                if quality.get('lpips') is not None:
                    print(f"  LPIPS: {quality['lpips']:.4f}")
                if 'quality_level' in quality:
                    print(f"  Quality: {quality['quality_level']}")

                # 保存可视化
                if save_visuals:
                    vis_dir = os.path.join(
                        self.results_dir,
                        f"{model_info['dataset']}_alpha{model_info['alpha']:.2f}",
                        f"client_{model_info['client_id']}_round_{model_info['rounds']}"
                    )
                    os.makedirs(vis_dir, exist_ok=True)

                    vis_path = os.path.join(vis_dir, f"sample_{sample_idx}.png")
                    self.visualizer.plot_reconstruction_comparison(
                        original=original,
                        reconstructed=reconstructed,
                        metrics=quality,
                        save_path=vis_path
                    )

                    # 保存损失曲线
                    loss_path = os.path.join(vis_dir, f"sample_{sample_idx}_loss.png")
                    self.visualizer.plot_loss_history(
                        loss_history=result['loss_history'],
                        save_path=loss_path
                    )

                # 根据PSNR确定质量等级
                if quality['psnr'] > 25:
                    q_level = 'high'
                elif quality['psnr'] > 15:
                    q_level = 'medium'
                else:
                    q_level = 'low'

                attack_results.append({
                    'sample_idx': sample_idx,
                    'psnr': quality['psnr'],
                    'ssim': quality['ssim'],
                    'lpips': quality.get('lpips'),
                    'quality_level': q_level,
                    'final_loss': result['loss_history']['total'][-1]
                })

            # 计算统计数据
            avg_psnr = np.mean([r['psnr'] for r in attack_results])
            avg_ssim = np.mean([r['ssim'] for r in attack_results])
            high_quality = sum(1 for r in attack_results if r.get('quality_level') == 'high')
            medium_quality = sum(1 for r in attack_results if r.get('quality_level') == 'medium')
            low_quality = sum(1 for r in attack_results if r.get('quality_level') == 'low')

            summary = {
                'model_info': model_info,
                'num_samples': num_samples,
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim),
                'high_quality_count': high_quality,
                'medium_quality_count': medium_quality,
                'low_quality_count': low_quality,
                'samples': attack_results,
                'status': 'success',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            print(f"\n[Attack Summary]")
            print(f"  Avg PSNR: {avg_psnr:.2f} dB")
            print(f"  Avg SSIM: {avg_ssim:.4f}")
            print(f"  Quality: High={high_quality}, Medium={medium_quality}, Low={low_quality}")

            return summary

        except Exception as e:
            print(f"[Attack] Error: {e}")
            import traceback
            traceback.print_exc()

            return {
                'model_info': model_info,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def attack_all_models(
        self,
        model_list: List[Dict],
        num_samples: int = 5,
        save_visuals: bool = True
    ) -> List[Dict]:
        """
        对所有模型执行GIA攻击

        Args:
            model_list: 模型信息列表
            num_samples: 每个模型重建的样本数
            save_visuals: 是否保存可视化

        Returns:
            所有攻击结果列表
        """
        all_results = []

        for idx, model_info in enumerate(model_list):
            print(f"\n{'#'*80}")
            print(f"# Processing model {idx + 1}/{len(model_list)}")
            print(f"{'#'*80}")

            result = self.attack_single_model(
                model_info=model_info,
                num_samples=num_samples,
                save_visuals=save_visuals
            )

            all_results.append(result)

            # 保存中间结果
            self.save_results(all_results, prefix="intermediate")

        return all_results

    def save_results(self, results: List[Dict], prefix: str = "final") -> str:
        """
        保存攻击结果到JSON文件

        Args:
            results: 攻击结果列表
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_gia_results_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[Save] Results saved to: {save_path}")
        return save_path

    def generate_summary_report(self, results: List[Dict]) -> str:
        """
        生成总结报告

        Args:
            results: 攻击结果列表

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = []

        report.append("# GIA Auto Attack Summary Report")
        report.append(f"Generated: {timestamp}")
        report.append(f"Total Models Evaluated: {len(results)}")
        report.append("")

        # 统计成功/失败
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        report.append(f"## Overall Statistics")
        report.append(f"- Successful attacks: {len(successful)}")
        report.append(f"- Failed attacks: {len(failed)}")
        report.append("")

        if successful:
            # 按数据集分组统计
            datasets = {}
            for r in successful:
                ds = r['model_info']['dataset']
                if ds not in datasets:
                    datasets[ds] = []
                datasets[ds].append(r)

            report.append(f"## Results by Dataset")
            for dataset, ds_results in datasets.items():
                avg_psnr = np.mean([r['avg_psnr'] for r in ds_results])
                avg_ssim = np.mean([r['avg_ssim'] for r in ds_results])
                total_high = sum(r['high_quality_count'] for r in ds_results)
                total_medium = sum(r['medium_quality_count'] for r in ds_results)
                total_low = sum(r['low_quality_count'] for r in ds_results)

                report.append(f"\n### {dataset}")
                report.append(f"- Models evaluated: {len(ds_results)}")
                report.append(f"- Average PSNR: {avg_psnr:.2f} dB")
                report.append(f"- Average SSIM: {avg_ssim:.4f}")
                report.append(f"- Quality distribution: High={total_high}, Medium={total_medium}, Low={total_low}")

        # 详细结果表格
        report.append("\n## Detailed Results")
        report.append("| Model | Dataset | Alpha | Client | Rounds | Samples | Avg PSNR | Avg SSIM | High/Med/Low | Status |")
        report.append("|-------|---------|-------|--------|--------|---------|----------|----------|--------------|--------|")

        for r in results:
            info = r['model_info']
            model_name = Path(info['model_path']).name
            if r['status'] == 'success':
                report.append(
                    f"| {model_name} | {info['dataset']} | {info['alpha']:.2f} | "
                    f"{info['client_id']} | {info['rounds']} | {r['num_samples']} | "
                    f"{r['avg_psnr']:.2f} | {r['avg_ssim']:.4f} | "
                    f"{r['high_quality_count']}/{r['medium_quality_count']}/{r['low_quality_count']} | ✅ |"
                )
            else:
                report.append(
                    f"| {model_name} | {info['dataset']} | {info['alpha']:.2f} | "
                    f"{info['client_id']} | {info['rounds']} | - | - | - | - | ❌ |"
                )

        # 保存报告
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gia_summary_report_{timestamp_str}.md"
        save_path = os.path.join(self.results_dir, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"\n[Report] Summary report saved to: {save_path}")
        return save_path


def main():
    parser = argparse.ArgumentParser(description='GIA Auto Attack Tool')

    # 模型扫描参数
    parser.add_argument('--pretrain_dir', type=str, default=None,
                        help='Specific pretrain directory to attack (e.g., pretrain/cifar-10-normal/1.00)')
    parser.add_argument('--scan_all', action='store_true',
                        help='Scan and attack all models in pretrain directory')
    parser.add_argument('--base_dir', type=str, default='pretrain',
                        help='Base directory for scanning models')

    # GIA攻击参数
    parser.add_argument('--stylegan_path', type=str,
                        default='Gradient_Inversion_Attack/pretrained_models/stylegan_xl_cifar10.pkl',
                        help='Path to StyleGAN-XL model')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to reconstruct per model')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation (cuda/cpu)')

    # 输出参数
    parser.add_argument('--results_dir', type=str, default='gia_auto_results',
                        help='Directory to save results')
    parser.add_argument('--save_visuals', action='store_true', default=True,
                        help='Save visualization plots')

    args = parser.parse_args()

    # 创建GIA配置
    gia_config = GIAConfig(
        num_iterations=args.iterations,
        learning_rate=args.lr,
        num_samples_per_client=args.num_samples
    )

    # 创建自动攻击器
    attacker = GIAAutoAttacker(
        stylegan_model_path=args.stylegan_path,
        device=args.device,
        gia_config=gia_config,
        results_dir=args.results_dir
    )

    # 扫描模型
    if args.scan_all:
        print(f"\n[Mode] Scanning all models in {args.base_dir}")
        model_list = attacker.scan_pretrain_models(base_dir=args.base_dir)
    elif args.pretrain_dir:
        print(f"\n[Mode] Scanning models in {args.pretrain_dir}")
        model_list = attacker.scan_pretrain_models(base_dir=args.pretrain_dir)
    else:
        print("[Error] Please specify --pretrain_dir or --scan_all")
        return

    if not model_list:
        print("[Error] No models found!")
        return

    print(f"\n[Attack] Will attack {len(model_list)} models")
    print(f"[Attack] Samples per model: {args.num_samples}")
    print(f"[Attack] Iterations: {args.iterations}")

    # 执行攻击
    results = attacker.attack_all_models(
        model_list=model_list,
        num_samples=args.num_samples,
        save_visuals=args.save_visuals
    )

    # 保存最终结果
    attacker.save_results(results, prefix="final")

    # 生成总结报告
    attacker.generate_summary_report(results)

    print(f"\n{'='*80}")
    print(f"[Complete] GIA Auto Attack Finished!")
    print(f"[Complete] Results saved to: {args.results_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
