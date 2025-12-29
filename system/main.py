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

            server = FedCP(args, i)
            if not args.difference_privacy:
                print("[Main] Using standard FedCP (no DP)")
            else:
                print("[Main] Using FedCP with differential privacy")
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
    parser.add_argument('-dp', '--difference_privacy', action='store_true',
                        help="Enable differential privacy")
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
    parser.add_argument('-al', "--alpha", type=float, default=1)
    parser.add_argument('-lam', "--lamda", type=float, default=0.0)
    parser.add_argument('-mia', "--enable_mia", action='store_true',
                        help="Enable MIA evaluation during training")

    # GIA (Gradient Inversion Attack) parameters
    parser.add_argument('-gia', "--enable_gia", action='store_true',
                        help="Enable GIA evaluation after training")
    parser.add_argument('--gia_stylegan_path', type=str,
                        default='Gradient_Inversion_Attack/pretrained_models/stylegan_xl_cifar10.pkl',
                        help="Path to StyleGAN-XL pretrained model")
    parser.add_argument('--gia_num_samples', type=int, default=5,
                        help="Number of samples to reconstruct per client (GIA)")
    parser.add_argument('--gia_iterations', type=int, default=10000,
                        help="Number of optimization iterations for GIA")
    parser.add_argument('--gia_lr', type=float, default=0.01,
                        help="Learning rate for GIA latent optimization")
    parser.add_argument('--gia_save_visuals', action='store_true', default=True,
                        help="Save GIA visualization results")
    parser.add_argument('--gia_test_defense', action='store_true',
                        help="Test defense mechanisms against GIA")

    # DP阈值参数
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
                        help="Apply noise to all parameters globally (no threshold-based selection)")

    # Layer-wise noise selection parameters
    parser.add_argument('--layer_noise_mode', type=str, default='threshold',
                        choices=['threshold', 'global', 'feature_only', 'classifier_only',
                                'front_half', 'back_half', 'conv_only', 'fc_only'],
                        help="Layer selection mode for noise addition")
    parser.add_argument('--layer_noise_ratio', type=float, default=1.0,
                        help="Ratio of layers to apply noise (for front_half/back_half modes)")

    # Hessian computation parameters
    parser.add_argument('--hessian_compute_interval', type=int, default=5,
                        help="Compute Hessian (second-order gradient) every N rounds (default: 5)")

    args = parser.parse_args()

    
    print("dp", args.difference_privacy)
    print("MIA evaluation enabled:", args.enable_mia)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    print("dp_number",args.difference_privacy_number)
    # torch.cuda.set_device(int(args.device_id))
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)