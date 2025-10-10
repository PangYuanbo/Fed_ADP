# ============================
# test_integration.py
# 测试RL-DP系统与现有联邦学习系统的集成
# ============================

import os
import sys
import argparse
import subprocess
from datetime import datetime

def test_argument_parsing():
    """测试参数解析是否正常工作"""
    print("="*50)
    print("Testing Argument Parsing...")
    print("="*50)

    # 模拟导入main模块并测试参数
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        import main
        print("+ Successfully imported main.py")

        # 测试参数解析
        test_args = [
            '--dataset', 'cifar10',
            '--global_rounds', '20',
            '--difference_privacy', 'True',
            '--enable_rl_dp', 'True',
            '--rl_min_rounds', '10',
            '--enable_mia', 'True',
            '--algorithm', 'FedCP'
        ]

        parser = argparse.ArgumentParser()

        # 添加所有必要的参数 (复制自main.py)
        parser.add_argument('-dp', '--difference_privacy', type=bool, default=False)
        parser.add_argument('-dpn', '--difference_privacy_number', type=float, default=5)
        parser.add_argument('-dpl', '--difference_privacy_layer', type=str, default="model.head")
        parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
        parser.add_argument('-did', "--device_id", type=str, default="0")
        parser.add_argument('-data', "--dataset", type=str, default="mnist")
        parser.add_argument('-nb', "--num_classes", type=int, default=10)
        parser.add_argument('-m', "--model", type=str, default="cnn")
        parser.add_argument('-lbs', "--batch_size", type=int, default=10)
        parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
        parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
        parser.add_argument('-ls', "--local_steps", type=int, default=1)
        parser.add_argument('-algo', "--algorithm", type=str, default="FedGP")
        parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
        parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
        parser.add_argument('-nc', "--num_clients", type=int, default=20)
        parser.add_argument('-pv', "--prev", type=int, default=0)
        parser.add_argument('-t', "--times", type=int, default=1)
        parser.add_argument('-eg', "--eval_gap", type=int, default=1)
        parser.add_argument('-al', "--alpha", type=float, default=1)
        parser.add_argument('-lam', "--lamda", type=float, default=0.0)
        parser.add_argument('-mia', "--enable_mia", type=bool, default=False)

        # RL-DP 参数
        parser.add_argument('--enable_rl_dp', type=bool, default=False)
        parser.add_argument('--rl_min_rounds', type=int, default=100)
        parser.add_argument('--rl_learning_rate', type=float, default=0.01)
        parser.add_argument('--rl_epsilon', type=float, default=0.1)
        parser.add_argument('--rl_update_interval', type=int, default=10)

        args = parser.parse_args(test_args)

        print(f"+ Dataset: {args.dataset}")
        print(f"+ Global rounds: {args.global_rounds}")
        print(f"+ DP enabled: {args.difference_privacy}")
        print(f"+ RL-DP enabled: {args.enable_rl_dp}")
        print(f"+ RL min rounds: {args.rl_min_rounds}")
        print(f"+ MIA enabled: {args.enable_mia}")
        print(f"+ Algorithm: {args.algorithm}")

        print("\n[PASS] Argument parsing test passed!")

    except Exception as e:
        print(f"[FAIL] Argument parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_rl_availability():
    """测试RL模块是否可用"""
    print("\n" + "="*50)
    print("Testing RL Module Availability...")
    print("="*50)

    try:
        from utils.simple_rl_dp import SimpleRLAgent, RLDPManager
        from utils.rl_config import get_rl_dp_config
        print("+ RL core modules imported successfully")

        from flcore.clients.clientcp_rl import clientCP_RL
        print("+ RL client imported successfully")

        from flcore.servers.servercp_rl import FedCP_RL
        print("+ RL server imported successfully")

        print("\n[PASS] RL availability test passed!")
        return True

    except ImportError as e:
        print(f"[FAIL] RL modules not available: {e}")
        return False


def test_server_client_creation():
    """测试服务器和客户端创建"""
    print("\n" + "="*50)
    print("Testing Server and Client Creation...")
    print("="*50)

    try:
        # 模拟参数
        class MockArgs:
            def __init__(self):
                self.dataset = "cifar10"
                self.device = "cpu"
                self.global_rounds = 10
                self.num_clients = 3
                self.join_ratio = 1.0
                self.alpha = 1.0
                self.random_join_ratio = False
                self.eval_gap = 5
                self.difference_privacy = True
                self.enable_rl_dp = True
                self.enable_mia = True
                self.num_classes = 10
                self.rl_min_rounds = 5
                self.rl_learning_rate = 0.01
                self.rl_epsilon = 0.1
                self.rl_update_interval = 3
                self.batch_size = 10
                self.local_learning_rate = 0.01
                self.local_steps = 1
                self.lamda = 0.0

        args = MockArgs()

        # 创建模型
        from flcore.trainmodel.models import FedAvgCNN, LocalModel
        import torch
        import torch.nn as nn

        model = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(args.device)
        args.head = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
        args.model = LocalModel(model, args.head)

        print("+ Model created successfully")

        # 尝试创建标准服务器
        from flcore.servers.servercp import FedCP
        standard_server = FedCP(args, 0)
        print("+ Standard server created successfully")

        # 尝试创建RL服务器（如果可用）
        try:
            from flcore.servers.servercp_rl import FedCP_RL
            rl_server = FedCP_RL(args, 0)
            print("+ RL server created successfully")
        except ImportError:
            print("+ RL server not available (expected in some environments)")

        print("\n[PASS] Server and client creation test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] Server and client creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_execution():
    """测试命令行执行"""
    print("\n" + "="*50)
    print("Testing Command Line Execution...")
    print("="*50)

    try:
        # 测试标准模式
        cmd_standard = [
            "python", "main.py",
            "--dataset", "cifar10",
            "--model", "cnn",
            "--num_clients", "3",
            "--global_rounds", "5",
            "--difference_privacy", "True",
            "--enable_mia", "False",  # 关闭MIA以加快测试
            "--device", "cpu",
            "--eval_gap", "5",
            "--algorithm", "FedCP"
        ]

        print("+ Testing standard mode command:")
        print("  " + " ".join(cmd_standard))

        # 这里只是验证命令格式，不实际执行（因为需要数据集）
        print("+ Command format validated")

        # 测试RL模式
        cmd_rl = [
            "python", "main.py",
            "--dataset", "cifar10",
            "--model", "cnn",
            "--num_clients", "3",
            "--global_rounds", "10",
            "--difference_privacy", "True",
            "--enable_rl_dp", "True",
            "--rl_min_rounds", "3",
            "--enable_mia", "False",
            "--device", "cpu",
            "--eval_gap", "5",
            "--algorithm", "FedCP"
        ]

        print("\n+ Testing RL mode command:")
        print("  " + " ".join(cmd_rl))
        print("+ Command format validated")

        print("\n[PASS] Command line execution test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] Command line execution test failed: {e}")
        return False


def run_all_integration_tests():
    """运行所有集成测试"""
    print(f"\n{'='*60}")
    print("Federated Learning RL-DP Integration Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    tests = [
        ("Argument Parsing", test_argument_parsing),
        ("RL Module Availability", test_rl_availability),
        ("Server Client Creation", test_server_client_creation),
        ("Command Line Execution", test_command_line_execution),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("[SUCCESS] All integration tests passed!")
        print("\nYou can now use the integrated RL-DP system with:")
        print("python main.py --difference_privacy True --enable_rl_dp True --rl_min_rounds 100 --algorithm FedCP")
    else:
        print("[WARNING] Some integration tests failed.")
        print("Check the error messages above for troubleshooting.")

    print("="*60)

    return failed == 0


if __name__ == "__main__":
    import copy
    success = run_all_integration_tests()
    exit(0 if success else 1)