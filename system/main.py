import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

from flcore.servers.servercp import FedCP
from flcore.trainmodel.models import *

from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len=200

hidden_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

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

        if args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            
            # 选择服务器类型：标准FedCP、RL-DP或精细化RL-DP
            if getattr(args, 'enable_rl_dp', False) and getattr(args, 'enable_fine_rl', False):
                from flcore.servers.servercp_fine_rl import FedCPFineRL
                server = FedCPFineRL(args, i)
                print(f"[Main] Using FedCP with Fine-grained RL-DP")
                print(f"  Granularity: {args.fine_granularity}")
                print(f"  Privacy budget: {args.privacy_budget}")
                print(f"  Block size: {args.block_size}")
            elif getattr(args, 'enable_rl_dp', False):
                from flcore.servers.servercp_rl import FedCPRL
                server = FedCPRL(args, i)
                print(f"[Main] Using FedCP with standard RL-DP (privacy_budget={args.privacy_budget})")
            else:
                server = FedCP(args, i)
                print("[Main] Using standard FedCP")
        else:
            raise NotImplementedError
            
        server.train(args)
        
        # torch.cuda.empty_cache()

        time_list.append(time.time()-start)

        reporter.report()

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dp', '--difference_privacy', type=bool, default=False)
    parser.add_argument('-dpn', '--difference_privacy_number', type=float, default=5)
    parser.add_argument('-dpl', '--difference_privacy_layer', type=str, default="model.head")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedGP")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-al', "--alpha", type=float, default=1)
    parser.add_argument('-lam', "--lamda", type=float, default=0.0)
    parser.add_argument('-mia', "--enable_mia", type=bool, default=False,
                        help="Enable MIA evaluation during training")
    
    # RL-DP (Reinforcement Learning Differential Privacy) 参数
    parser.add_argument('-rl_dp', "--enable_rl_dp", type=bool, default=False,
                        help="Enable reinforcement learning-based differential privacy")
    parser.add_argument('-pb', "--privacy_budget", type=float, default=10.0,
                        help="Total privacy budget for RL-DP")
    parser.add_argument('-rl_lr', type=float, default=3e-4,
                        help="Learning rate for RL agent")
    parser.add_argument('-rl_train_interval', type=int, default=10,
                        help="RL agent training interval (rounds)")
    parser.add_argument('-rl_model_path', type=str, default="rl_models/rl_dp_agent.pth",
                        help="Path to save/load RL model")
    
    # Fine-grained RL-DP 精细化控制参数
    parser.add_argument('-fine_rl', "--enable_fine_rl", type=bool, default=False,
                        help="Enable fine-grained weight-level RL-DP control")
    parser.add_argument('-granularity', "--fine_granularity", type=str, default="block",
                        choices=['layer', 'block', 'weight'],
                        help="Granularity level: layer, block, or weight")
    parser.add_argument('-block_size', type=str, default="8,8",
                        help="Block size for block-level control (format: 'h,w')")
    parser.add_argument('-importance_method', type=str, default="combined",
                        choices=['gradient', 'fisher', 'activation', 'weight_magnitude', 'combined'],
                        help="Weight importance evaluation method")
    parser.add_argument('-visualize_noise', type=bool, default=False,
                        help="Enable noise pattern visualization")

    args = parser.parse_args()
    
    # 处理block_size参数
    if hasattr(args, 'block_size') and isinstance(args.block_size, str):
        try:
            args.block_size = tuple(map(int, args.block_size.split(',')))
        except:
            args.block_size = (8, 8)  # 默认值
    
    print("dp", args.difference_privacy)
    print("MIA evaluation enabled:", args.enable_mia)
    print("RL-DP enabled:", getattr(args, 'enable_rl_dp', False))
    print("Fine-grained RL enabled:", getattr(args, 'enable_fine_rl', False))
    
    if getattr(args, 'enable_rl_dp', False):
        print(f"Privacy budget: {args.privacy_budget}")
        print(f"RL learning rate: {args.rl_lr}")
        print(f"RL training interval: {args.rl_train_interval}")
        
        if getattr(args, 'enable_fine_rl', False):
            print(f"Fine-grained granularity: {args.fine_granularity}")
            print(f"Block size: {args.block_size}")
            print(f"Importance method: {args.importance_method}")
            print(f"Noise visualization: {args.visualize_noise}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    print("dp_number",args.difference_privacy_number)
    # torch.cuda.set_device(int(args.device_id))
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)