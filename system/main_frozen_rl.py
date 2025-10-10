#!/usr/bin/env python
# ============================
# main_frozen_rl.py
# 冻结模型 + RL噪声参数搜索
#
# 工作流程：
# 1. 训练100轮得到基准模型（无噪声）
# 2. 冻结模型权重
# 3. RL搜索最优噪声参数（只测试，不更新模型）
# ============================

import copy
import torch
import argparse
import os
import numpy as np
from flcore.servers.servercp import FedCP
from utils.mia_attack_wrapper import FederatedMIAEvaluator
from utils.simple_rl_dp import SimpleRLAgent, RLDPManager
from utils.rl_config import RLDPConfig
import json

def run_frozen_rl_experiment(args):
    """
    运行冻结模型RL噪声搜索实验

    阶段1: 训练基准模型（0-100轮，无噪声）
    阶段2: RL噪声搜索（101-N轮，模型冻结，只测试不同噪声）
    """

    print("\n" + "="*80)
    print("Frozen Model + RL Noise Parameter Search Experiment")
    print("="*80)

    # ==================== Phase 1: Train Baseline Model ====================
    print("\n[Phase 1] Training Baseline Model (0-100 rounds, no noise)")
    print("-"*80)

    # Load model if necessary
    if isinstance(args.model, str):
        if args.model == "cnn":
            from flcore.trainmodel.models import FedAvgCNN
            if "cifar" in args.dataset.lower():
                args.model = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(args.device)
            elif args.dataset == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(args.device)

    # 创建无噪声训练配置
    baseline_args = copy.deepcopy(args)
    baseline_args.difference_privacy = False  # 关闭差分隐私
    baseline_args.global_rounds = 100  # 固定100轮

    # 初始化服务器
    server = FedCP(baseline_args, times=1)

    # 训练基准模型
    for round_idx in range(100):
        print(f"\n============= Round {round_idx+1}/100 =============")
        server.train()

        if (round_idx + 1) % args.eval_gap == 0:
            print(f"\nEvaluating round {round_idx+1}...")
            server.test()

    # Save baseline model at round 100
    baseline_model_path = "pretrain/frozen_baseline_model.pth"
    os.makedirs("pretrain", exist_ok=True)
    torch.save(server.global_modules.state_dict(), baseline_model_path)

    # Record baseline performance
    baseline_accuracy = server.rs_test_acc[-1] if server.rs_test_acc else 0.0
    print(f"\n[Phase 1 Complete] Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Model saved to: {baseline_model_path}")

    # ==================== Phase 2: RL Noise Search ====================
    print("\n[Phase 2] RL Noise Parameter Search (Model Frozen)")
    print("-"*80)

    # Initialize RL components
    rl_config = RLDPConfig()
    rl_agent = SimpleRLAgent(
        state_dim=3,
        learning_rate=args.rl_learning_rate,
        epsilon=args.rl_epsilon,
        epsilon_decay=args.rl_epsilon_decay,
        device=args.device
    )

    rl_manager = RLDPManager(
        agent=rl_agent,
        update_interval=args.rl_update_interval,
        min_rounds_before_rl=0  # Enable RL immediately
    )

    # Initialize MIA evaluator
    mia_evaluator = None
    if args.enable_mia:
        try:
            mia_evaluator = FederatedMIAEvaluator(
                dataset_name=args.dataset,
                num_clients=args.num_clients
            )
            print("[MIA] 评估器初始化成功")
        except Exception as e:
            print(f"[MIA] 初始化失败: {e}")

    # RL search loop
    search_results = []
    frozen_model_state = torch.load(baseline_model_path)

    for search_round in range(args.rl_search_rounds):
        print(f"\n============= RL Search Round {search_round+1}/{args.rl_search_rounds} =============")

        # 1. Reset all client models to frozen state
        for client in server.clients:
            client.model.load_state_dict(frozen_model_state)
            client.model.eval()  # Set to evaluation mode

        # 2. RL selects noise parameters
        current_round = 100 + search_round
        state = rl_manager.get_state(
            accuracy=baseline_accuracy,
            mia_f_score=0.5,  # Initial default value
            round_num=current_round,
            total_rounds=100 + args.rl_search_rounds
        )

        action, (threshold_high, threshold_low) = rl_manager.select_action(
            state, current_round
        )

        print(f"[RL] Selected action {action}: thresholds ({threshold_high}, {threshold_low})")

        # 3. Only add noise to model copy (no training)
        test_accuracies = []
        mia_scores = []

        for client in server.clients:
            # Create model copy
            noisy_model = copy.deepcopy(client.model)

            # Apply differential privacy noise
            for name, param in noisy_model.named_parameters():
                if 'head' in name:  # Only add noise to head layer
                    # Calculate gradient norm percentiles (simulated)
                    grad_norms = torch.randn(100).abs()
                    high_percentile = torch.quantile(grad_norms, threshold_high)
                    low_percentile = torch.quantile(grad_norms, threshold_low)

                    # Add Gaussian noise
                    noise = torch.randn_like(param.data) * args.difference_privacy_number * 0.01
                    param.data.add_(noise)

            # 4. Test noisy performance (no update to original model)
            client.model = noisy_model
            test_acc, _ = client.test_metrics()
            test_accuracies.append(test_acc)

        avg_accuracy = np.mean(test_accuracies)

        # 5. MIA evaluation (if enabled)
        mia_f_score = 0.5  # Default value
        if mia_evaluator and (search_round + 1) % 5 == 0:
            try:
                # Prepare noisy models for MIA
                noisy_models = []
                for client in server.clients:
                    noisy_model = copy.deepcopy(client.model)
                    noisy_models.append(noisy_model)

                mia_result = mia_evaluator.evaluate_round(
                    round_num=current_round,
                    client_models=noisy_models
                )
                mia_f_score = mia_result.get('avg_f_score', 0.5)
                print(f"[MIA] F-Score: {mia_f_score:.4f}")
            except Exception as e:
                print(f"[MIA] 评估失败: {e}")

        # 6. Calculate reward
        reward = rl_manager.calculate_reward(
            accuracy=avg_accuracy,
            mia_f_score=mia_f_score,
            baseline_accuracy=baseline_accuracy,
            round_num=current_round
        )

        print(f"[Result] Accuracy: {avg_accuracy:.4f} | MIA F-Score: {mia_f_score:.4f} | Reward: {reward:.4f}")

        # 7. Store results
        search_results.append({
            'round': current_round,
            'action': action,
            'threshold_high': threshold_high,
            'threshold_low': threshold_low,
            'accuracy': avg_accuracy,
            'accuracy_drop': baseline_accuracy - avg_accuracy,
            'mia_f_score': mia_f_score,
            'reward': reward
        })

        # 8. RL learning (using virtual next state)
        next_state = rl_manager.get_state(
            accuracy=avg_accuracy,
            mia_f_score=mia_f_score,
            round_num=current_round + 1,
            total_rounds=100 + args.rl_search_rounds
        )

        # Store experience
        rl_agent.remember(state, action, reward, next_state, done=False)

        # Update RL policy
        if search_round > 0 and search_round % args.rl_update_interval == 0:
            loss = rl_agent.train_step()
            if loss is not None:
                print(f"[RL Training] Loss: {loss:.4f}")

        # Update epsilon
        rl_agent.update_epsilon()
        print(f"[RL] Epsilon: {rl_agent.epsilon:.4f}")

    # ==================== Results Analysis ====================
    print("\n" + "="*80)
    print("Experiment Results Analysis")
    print("="*80)

    # 找到最佳配置
    best_result = max(search_results, key=lambda x: x['reward'])

    print(f"\nBaseline Model Performance (No Noise):")
    print(f"  Accuracy: {baseline_accuracy:.4f}")

    print(f"\nBest Noise Configuration:")
    print(f"  Action: {best_result['action']}")
    print(f"  thresholds: ({best_result['threshold_high']}, {best_result['threshold_low']})")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Accuracy下降: {best_result['accuracy_drop']:.4f}")
    print(f"  MIA F-Score: {best_result['mia_f_score']:.4f}")
    print(f"  Reward: {best_result['reward']:.4f}")

    # 保存结果
    results_path = "frozen_rl_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_accuracy,
            'search_results': search_results,
            'best_result': best_result
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # 绘制结果
    plot_frozen_rl_results(search_results, baseline_accuracy)

    return best_result


def plot_frozen_rl_results(results, baseline_accuracy):
    """绘制RL搜索结果"""
    import matplotlib.pyplot as plt

    rounds = [r['round'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    mia_scores = [r['mia_f_score'] for r in results]
    rewards = [r['reward'] for r in results]
    actions = [r['action'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy变化
    axes[0, 0].plot(rounds, accuracies, 'b-', label='加噪后Accuracy')
    axes[0, 0].axhline(y=baseline_accuracy, color='r', linestyle='--', label='基准Accuracy')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy变化')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MIA风险
    axes[0, 1].plot(rounds, mia_scores, 'g-')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('MIA F-Score')
    axes[0, 1].set_title('MIA攻击风险')
    axes[0, 1].grid(True)

    # Reward变化
    axes[1, 0].plot(rounds, rewards, 'orange')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('RLReward变化')
    axes[1, 0].grid(True)

    # Action分布
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    axes[1, 1].bar(action_counts.keys(), action_counts.values())
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('选择次数')
    axes[1, 1].set_title('Action选择分布')

    plt.tight_layout()
    plt.savefig('frozen_rl_search_results.png', dpi=150)
    print("\n可视化Results saved to: frozen_rl_search_results.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-nc', "--num_clients", type=int, default=10)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCP")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
    parser.add_argument('-eg', "--eval_gap", type=int, default=5)
    parser.add_argument('-al', "--alpha", type=int, default=1)
    parser.add_argument('-lam', "--lamda", type=float, default=0.0)
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)

    # 差分隐私参数
    parser.add_argument('-dp', '--difference_privacy', type=bool, default=True)
    parser.add_argument('-dpn', '--difference_privacy_number', type=float, default=5.0)

    # RL参数
    parser.add_argument('--rl_search_rounds', type=int, default=200,
                        help="RL搜索轮次（在冻结模型上测试）")
    parser.add_argument('--rl_learning_rate', type=float, default=0.01)
    parser.add_argument('--rl_epsilon', type=float, default=0.2)
    parser.add_argument('--rl_epsilon_decay', type=float, default=0.995)
    parser.add_argument('--rl_update_interval', type=int, default=10)

    # MIA参数
    parser.add_argument('--enable_mia', type=bool, default=True)

    args = parser.parse_args()

    # Model loading is done inside run_frozen_rl_experiment()
    run_frozen_rl_experiment(args)
