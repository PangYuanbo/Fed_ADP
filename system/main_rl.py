# ============================
# main_rl.py
# 支持RL自适应差分隐私的主程序
# 基于原始main.py修改，集成RL-DP系统
# ============================

import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

from flcore.servers.servercp_rl import FedCP_RL
from flcore.trainmodel.models import *
from utils.mem_utils import MemReporter
from utils.rl_config import create_rl_args_parser, RLDPPresets, get_rl_dp_config

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    """
    运行支持RL-DP的联邦学习训练

    Args:
        args: 命令行参数
    """
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    # 打印配置信息
    print(f"\n{'='*60}")
    print("RL-Adaptive Differential Privacy Federated Learning")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Clients: {args.num_clients}")
    print(f"Global Rounds: {args.global_rounds}")
    print(f"Differential Privacy: {args.difference_privacy}")
    print(f"RL-DP System: {getattr(args, 'enable_rl_dp', True)}")
    print(f"MIA Evaluation: {args.enable_mia}")
    print(f"Alpha (Data Distribution): {args.alpha}")
    print(f"{'='*60}\n")

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
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
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # 初始化RL-DP服务器
        if args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)

            # 使用RL增强的服务器
            server = FedCP_RL(args, i)
            print("[Main] Using RL-enhanced FedCP")
        else:
            raise NotImplementedError

        server.train(args)

        time_list.append(time.time()-start)
        reporter.report()

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")


def create_preset_configs():
    """
    创建不同的预设配置供快速测试使用

    Returns:
        dict: 预设配置字典
    """
    return {
        "conservative": RLDPPresets.conservative_exploration(),
        "aggressive": RLDPPresets.aggressive_exploration(),
        "balanced": RLDPPresets.balanced(),
        "privacy_focused": RLDPPresets.privacy_focused(),
        "accuracy_focused": RLDPPresets.accuracy_focused()
    }


def apply_preset_config(args, preset_name):
    """
    应用预设配置

    Args:
        args: 参数对象
        preset_name: 预设名称
    """
    presets = create_preset_configs()
    if preset_name in presets:
        config = presets[preset_name]
        config_dict = config.get_config_dict()

        print(f"[Main] Applying '{preset_name}' preset configuration...")
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"  {key}: {value}")
    else:
        print(f"[Main] Warning: Unknown preset '{preset_name}'. Available: {list(presets.keys())}")


if __name__ == "__main__":
    total_start = time.time()

    # 使用RL增强的参数解析器
    parser = create_rl_args_parser()

    # 添加预设配置参数
    parser.add_argument('--preset', type=str, default=None,
                        choices=['conservative', 'aggressive', 'balanced', 'privacy_focused', 'accuracy_focused'],
                        help="Use a preset configuration for RL-DP system")

    args = parser.parse_args()

    # 应用预设配置（如果指定）
    if args.preset:
        apply_preset_config(args, args.preset)

    # 基础信息打印
    print("Differential Privacy:", args.difference_privacy)
    print("MIA evaluation enabled:", args.enable_mia)
    print("RL-DP enabled:", getattr(args, 'enable_rl_dp', True))
    print("DP noise multiplier:", args.difference_privacy_number)

    # 设备配置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"

    # 运行训练
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user")
    except Exception as e:
        print(f"\n[Main] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - total_start
        print(f"\n[Main] Total execution time: {total_time:.2f} seconds")