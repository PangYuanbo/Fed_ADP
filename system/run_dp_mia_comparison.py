"""
完整的 DP vs 无 DP 对抗 MIA 攻击效果对比实验

实验流程：
1. 训练 Shadow 模型（用于训练 MIA 攻击模型）
2. 训练无 DP 的目标模型
3. 训练有 DP 的目标模型
4. 训练 MIA 攻击模型
5. 对比两种模型的 MIA 防御效果
"""

import os
import sys
import torch
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='DP vs 无 DP 对抗 MIA 攻击效果对比')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'shadow', 'train', 'attack', 'evaluate'],
                        help='运行阶段：all(全部), shadow(训练shadow模型), train(训练目标模型), attack(训练攻击模型), evaluate(评估)')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--global_rounds', type=int, default=50, help='训练轮数')
    parser.add_argument('--epsilon', type=float, default=1.0, help='DP 隐私预算')
    parser.add_argument('--delta', type=float, default=1e-6, help='DP delta 参数')
    parser.add_argument('--clip_value', type=float, default=0.005, help='梯度裁剪值')
    parser.add_argument('--dataset', type=str, default='cifar-10-normal', help='数据集名称')
    parser.add_argument('--alpha', type=float, default=1.0, help='数据分布参数')
    parser.add_argument('--device_id', type=str, default='0', help='GPU设备ID')

    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs('shadow_model/1', exist_ok=True)
    os.makedirs('normal_model/1', exist_ok=True)
    os.makedirs('dp_model/1', exist_ok=True)
    os.makedirs('mia_results_comparison', exist_ok=True)

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  DP vs 无 DP 对抗 MIA 攻击效果对比实验                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║  客户端数量: {args.num_clients:<42} ║
    ║  训练轮数:   {args.global_rounds:<42} ║
    ║  DP epsilon: {args.epsilon:<42} ║
    ║  数据集:     {args.dataset:<42} ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Stage 1: 训练 Shadow 模型（用于训练 MIA 攻击模型）
    if args.stage in ['all', 'shadow']:
        print("\n[Stage 1/4] 训练 Shadow 模型...")
        shadow_cmd = [
            sys.executable, 'main.py',
            '-t', '1',
            '-nc', str(args.num_clients),
            '-nb', '10',
            '-data', args.dataset,
            '-m', 'cnn',
            '-algo', 'FedCP',
            '-did', args.device_id,
            '-gr', str(args.global_rounds),
            '-lr', '0.005',
            '-lam', '5'
        ]
        run_command(shadow_cmd, "训练 Shadow 模型（用于训练 MIA 攻击器）")

    # Stage 2: 训练无 DP 的目标模型
    if args.stage in ['all', 'train']:
        print("\n[Stage 2/4] 训练无 DP 的目标模型...")
        normal_cmd = [
            sys.executable, 'main.py',
            '-t', '1',
            '-nc', str(args.num_clients),
            '-nb', '10',
            '-data', args.dataset,
            '-m', 'cnn',
            '-algo', 'FedCP',
            '-did', args.device_id,
            '-gr', str(args.global_rounds),
            '-lr', '0.005',
            '-lam', '5'
        ]
        run_command(normal_cmd, "训练无 DP 的目标模型")

        # Stage 3: 训练有 DP 的目标模型
        print("\n[Stage 3/4] 训练有 DP 的目标模型...")
        dp_cmd = [
            sys.executable, 'main.py',
            '-t', '1',
            '-nc', str(args.num_clients),
            '-nb', '10',
            '-data', args.dataset,
            '-m', 'cnn',
            '-algo', 'FedCP',
            '-did', args.device_id,
            '-gr', str(args.global_rounds),
            '-lr', '0.005',
            '-lam', '5',
            '-dp',  # 启用 DP
            '--epsilon', str(args.epsilon),
            '--delta', str(args.delta),
            '--clip_value', str(args.clip_value),
            '--layer_noise_mode', 'global'
        ]
        run_command(dp_cmd, "训练有 DP 的目标模型")

    # Stage 4: 运行 MIA 攻击评估
    if args.stage in ['all', 'attack', 'evaluate']:
        print("\n[Stage 4/4] 运行 MIA 攻击评估...")
        print("\n请按照以下步骤手动执行 MIA 攻击评估：")
        print("\n1. 训练 MIA 攻击模型：")
        print(f"   cd Membership_Inference_Attack")
        print(f"   # 编辑 run_mia_pipeline.py，设置正确的模型路径")
        print(f"   # 取消注释第 40-54 行（训练攻击模型）")
        print(f"   python run_mia_pipeline.py")

        print("\n2. 评估无 DP 模型：")
        print(f"   # 在 run_mia_pipeline.py 中设置 target_client_files 为 normal_model 路径")
        print(f"   python run_mia_pipeline.py")

        print("\n3. 评估有 DP 模型：")
        print(f"   # 在 run_mia_pipeline.py 中设置 target_client_files 为 dp_model 路径")
        print(f"   python run_mia_pipeline.py")

        print("\n4. 对比结果：")
        print(f"   python plot_mia_comparison.py")

    print("\n" + "="*60)
    print("实验完成！请检查以下目录：")
    print(f"  - shadow_model/1/     : Shadow 模型")
    print(f"  - normal_model/1/     : 无 DP 模型")
    print(f"  - dp_model/1/         : 有 DP 模型")
    print(f"  - mia_results_*       : MIA 攻击结果")
    print("="*60)

if __name__ == "__main__":
    main()