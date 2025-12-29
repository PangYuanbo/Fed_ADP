"""
Quick GIA Attack Script
快速GIA攻击脚本 - 简化版本

一键启动对pretrain目录中所有模型的攻击

Usage:
    python Gradient_Inversion_Attack/quick_gia_attack.py
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for Chinese characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保正确的导入路径
current_file = Path(__file__).resolve()
system_dir = current_file.parent.parent  # system directory
if str(system_dir) not in sys.path:
    sys.path.insert(0, str(system_dir))

from Gradient_Inversion_Attack.gia_auto_attack import GIAAutoAttacker
from Gradient_Inversion_Attack.utils.config import GIAConfig


def main():
    print("="*80)
    print("GIA Quick Attack Tool")
    print("快速梯度反演攻击工具")
    print("="*80)

    # 配置参数
    STYLEGAN_PATH = "Gradient_Inversion_Attack/pretrained_models/stylegan_xl_cifar10.pkl"
    PRETRAIN_BASE = "pretrain"
    RESULTS_DIR = "gia_auto_results"
    NUM_SAMPLES = 5  # 每个模型重建5个样本
    ITERATIONS = 5000  # 快速模式：5000次迭代（完整版10000次）

    print(f"\n[Config]")
    print(f"  StyleGAN Model: {STYLEGAN_PATH}")
    print(f"  Pretrain Directory: {PRETRAIN_BASE}")
    print(f"  Samples per Model: {NUM_SAMPLES}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Results Directory: {RESULTS_DIR}")

    # 检查pretrain目录
    if not os.path.exists(PRETRAIN_BASE):
        print(f"\n[Error] Pretrain directory not found: {PRETRAIN_BASE}")
        print(f"[Info] Please train some models first using:")
        print(f"       python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200")
        return

    # 创建GIA配置
    gia_config = GIAConfig(
        num_iterations=ITERATIONS,
        learning_rate=0.01,
        num_samples_per_client=NUM_SAMPLES
    )

    # 创建自动攻击器
    print(f"\n[Init] Initializing GIA Auto Attacker...")
    attacker = GIAAutoAttacker(
        stylegan_model_path=STYLEGAN_PATH,
        device='cuda',
        gia_config=gia_config,
        results_dir=RESULTS_DIR
    )

    # 扫描所有模型
    print(f"\n[Scan] Scanning for pretrained models...")
    model_list = attacker.scan_pretrain_models(base_dir=PRETRAIN_BASE)

    if not model_list:
        print(f"[Error] No models found in {PRETRAIN_BASE}!")
        print(f"[Info] Directory structure should be: {PRETRAIN_BASE}/{{dataset}}/{{alpha}}/results_client*.pt")
        return

    # 显示找到的模型
    print(f"\n[Found] {len(model_list)} models:")
    datasets = {}
    for model in model_list:
        key = f"{model['dataset']}_alpha{model['alpha']}"
        if key not in datasets:
            datasets[key] = []
        datasets[key].append(model)

    for key, models in datasets.items():
        print(f"  - {key}: {len(models)} models")

    # 确认执行
    print(f"\n[Warning] This will attack {len(model_list)} models with {NUM_SAMPLES} samples each.")
    print(f"[Warning] Estimated time: {len(model_list) * NUM_SAMPLES * ITERATIONS * 0.0001:.1f} minutes")

    response = input("\n[Confirm] Continue? (y/n): ")
    if response.lower() != 'y':
        print("[Cancelled] Attack cancelled by user.")
        return

    # 执行攻击
    print(f"\n[Attack] Starting GIA attacks...")
    results = attacker.attack_all_models(
        model_list=model_list,
        num_samples=NUM_SAMPLES,
        save_visuals=True
    )

    # 保存结果
    result_path = attacker.save_results(results, prefix="quick_attack")

    # 生成报告
    report_path = attacker.generate_summary_report(results)

    # 显示总结
    print(f"\n{'='*80}")
    print(f"[Complete] GIA Quick Attack Finished!")
    print(f"{'='*80}")

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n[Summary]")
    print(f"  Total models: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        import numpy as np
        avg_psnr = np.mean([r['avg_psnr'] for r in successful])
        avg_ssim = np.mean([r['avg_ssim'] for r in successful])
        total_high = sum(r['high_quality_count'] for r in successful)

        print(f"\n[Quality Metrics]")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        print(f"  High quality reconstructions: {total_high}")

    print(f"\n[Output]")
    print(f"  Results: {result_path}")
    print(f"  Report: {report_path}")
    print(f"  Visualizations: {RESULTS_DIR}/")

    print(f"\n[Next] Check the results in '{RESULTS_DIR}' directory!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
