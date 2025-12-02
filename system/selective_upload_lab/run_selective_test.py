# ============================
# run_selective_test.py
# 独立运行脚本：测试选择性上传+噪声添加方案
# ============================

import sys
import os
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import torch.nn as nn

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 添加当前目录到路径（用于导入本LAB的模块）
sys.path.insert(0, os.path.dirname(__file__))

from server_selective import ServerSelective
from flcore.trainmodel.models import FedAvgCNN, LocalModel, fastText

# 尝试导入MemReporter（可选）
try:
    from utils.mem_utils import MemReporter
    MEM_REPORTER_AVAILABLE = True
except ImportError:
    MEM_REPORTER_AVAILABLE = False
    class MemReporter:
        def report(self):
            pass

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News (if needed)
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    """
    主运行函数

    Args:
        args: 命令行参数
    """
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n{'='*60}")
        print(f"Running time: {i}th")
        print(f"Selective Upload LAB - Testing from Round {args.selective_upload_round}")
        print(f"{'='*60}")
        print("Creating server and clients ...")
        start = time.time()

        # Generate model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset[:5] == "cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # Setup for FedCP-style model (LocalModel with separate head)
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = LocalModel(args.model, args.head)

        # Create ServerSelective (uses ClientSelective internally)
        server = ServerSelective(args, i)

        print("\n[Selective Upload LAB] Configuration:")
        print(f"  - Selective Upload Enabled: {args.enable_selective_upload}")
        print(f"  - Differential Privacy: {args.difference_privacy}")
        if args.enable_selective_upload:
            print(f"  - Selective Upload Round: {args.selective_upload_round}")
            print(f"  - Gradient Keep Ratio: {args.gradient_keep_ratio}")
            print(f"  - Noise Multiplier: {args.noise_multiplier}")
        print(f"  - MIA Evaluation: {args.enable_mia} (Global Model only)")
        print(f"  - Dataset: {args.dataset}")
        print(f"  - Global Rounds: {args.global_rounds}")
        print(f"  - Alpha (heterogeneity): {args.alpha}")

        # Train
        server.train(args)

        time_list.append(time.time() - start)
        reporter.report()

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("\nAll done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser(description='Selective Upload LAB - Test selective upload with noise')

    # ========== General Parameters ==========
    parser.add_argument('-dp', '--difference_privacy', action='store_true',
                        help="Enable differential privacy")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar-10-normal")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCP")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", action='store_true',
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-lam', "--lamda", type=float, default=0.0)

    # ========== MIA Parameters ==========
    parser.add_argument('-mia', "--enable_mia", action='store_true',
                        help="Enable MIA evaluation during training")

    # ========== DP Parameters ==========
    parser.add_argument('--threshold_high', type=float, default=0.6,
                        help="High threshold percentile for noise mask (default: 0.6)")
    parser.add_argument('--threshold_low', type=float, default=0.4,
                        help="Low threshold percentile for noise mask (default: 0.4)")
    parser.add_argument('--clip_value', type=float, default=0.005,
                        help="Gradient clipping value for DP (default: 0.005)")
    parser.add_argument('--epsilon', type=float, default=0.8,
                        help="Privacy budget epsilon for DP (default: 0.8)")
    parser.add_argument('--delta', type=float, default=1e-5,
                        help="Privacy parameter delta for DP (default: 1e-5)")
    parser.add_argument('--global_noise', action='store_true',
                        help="Apply noise to all parameters globally")

    # Layer-wise noise selection
    parser.add_argument('--layer_noise_mode', type=str, default='threshold',
                        choices=['threshold', 'global', 'feature_only', 'classifier_only',
                                 'front_half', 'back_half', 'conv_only', 'fc_only'],
                        help="Layer selection mode for noise addition")
    parser.add_argument('--layer_noise_ratio', type=float, default=1.0,
                        help="Ratio of layers to apply noise")

    # ========== Selective Upload Parameters (NEW) ==========
    parser.add_argument('--enable_selective_upload', action='store_true',
                        help="Enable selective upload mechanism (default: False)")
    parser.add_argument('--selective_upload_round', type=int, default=50,
                        help="Round to start selective upload (default: 50)")
    parser.add_argument('--gradient_keep_ratio', type=float, default=0.3,
                        help="Ratio of large gradient parameters to keep locally (default: 0.3)")
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help="Noise multiplier for DP noise (default: 1.0, larger=more noise)")

    # ========== Hessian Computation Parameters ==========
    parser.add_argument('--hessian_compute_interval', type=int, default=5,
                        help="Compute Hessian (second-order gradient) every N rounds (default: 5)")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*60)
    print("SELECTIVE UPLOAD LAB - Configuration")
    print("="*60)
    print(f"Selective Upload Enabled: {args.enable_selective_upload}")
    print(f"Differential Privacy: {args.difference_privacy}")
    if args.enable_selective_upload:
        print(f"Selective Upload Round: {args.selective_upload_round}")
        print(f"Gradient Keep Ratio: {args.gradient_keep_ratio}")
        print(f"Noise Multiplier: {args.noise_multiplier}")
    print(f"MIA Evaluation: {args.enable_mia} (Global Model only)")
    print(f"Dataset: {args.dataset}")
    print(f"Global Rounds: {args.global_rounds}")
    print(f"Alpha: {args.alpha}")
    print("="*60 + "\n")

    # Setup CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"

    # Run experiment
    run(args)

    print(f"\nTotal time: {time.time() - total_start:.2f}s")
