# ============================
# test_rl_dp_system.py
# RL-DP系统测试脚本
# 用于验证RL自适应差分隐私系统的基本功能
# ============================

import torch
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.simple_rl_dp import SimpleRLAgent, RLDPManager
from utils.rl_config import get_rl_dp_config, RLDPPresets


class MockClient:
    """模拟客户端，用于测试"""
    def __init__(self, client_id):
        self.id = client_id
        self.round = 0
        self.current_accuracy = 0.5
        self.current_mia_f_score = 0.5


def test_simple_rl_agent():
    """测试SimpleRLAgent基本功能"""
    print("="*50)
    print("Testing SimpleRLAgent...")
    print("="*50)

    # 创建RL智能体
    agent = SimpleRLAgent(
        state_dim=3,
        learning_rate=0.01,
        epsilon=0.1,
        device="cpu"  # 测试时使用CPU
    )

    print("+ RL Agent created successfully")

    # 测试状态构建
    state = agent.get_state(accuracy=0.8, mia_f_score=0.3, round_num=10, max_rounds=100)
    print(f"+ State created: {state}")

    # 测试动作选择
    action = agent.select_action(state, training=True)
    print(f"+ Action selected: {action}")

    # 测试阈值获取
    thresholds = agent.get_thresholds(action)
    print(f"+ Thresholds: {thresholds}")

    # 测试奖励计算
    reward = agent.calculate_reward(accuracy=0.8, mia_f_score=0.3)
    print(f"+ Reward calculated: {reward:.4f}")

    # 测试经验存储
    next_state = agent.get_state(accuracy=0.82, mia_f_score=0.28, round_num=11, max_rounds=100)
    agent.store_experience(state, action, reward, next_state, False)
    print("+ Experience stored")

    # 测试统计信息
    stats = agent.get_statistics()
    print(f"+ Statistics: {stats}")

    print("\n[PASS] SimpleRLAgent test passed!\n")


def test_rl_dp_manager():
    """测试RLDPManager功能"""
    print("="*50)
    print("Testing RLDPManager...")
    print("="*50)

    # 创建RL智能体和管理器
    agent = SimpleRLAgent(state_dim=3, device="cpu")
    manager = RLDPManager(agent, update_interval=5, min_rounds_before_rl=10)

    print("+ RLDP Manager created successfully")

    # 模拟训练过程
    accuracies = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.77, 0.78, 0.8, 0.82]
    mia_scores = [0.6, 0.58, 0.55, 0.57, 0.53, 0.5, 0.48, 0.45, 0.43, 0.4, 0.38, 0.36, 0.34, 0.32]

    for round_num in range(1, len(accuracies) + 1):
        accuracy = accuracies[round_num - 1]
        mia_score = mia_scores[round_num - 1]

        # 获取阈值
        thresholds = manager.get_thresholds(round_num, accuracy, mia_score)
        print(f"Round {round_num:2d}: Acc={accuracy:.3f}, MIA={mia_score:.3f}, Thresholds={thresholds}")

        # 测试RL是否在正确时间启用
        should_use_rl = manager.should_use_rl(round_num)
        if round_num == manager.min_rounds_before_rl:
            assert should_use_rl, f"RL should be enabled at round {round_num}"
            print(f"+ RL enabled at round {round_num}")

    # 获取摘要
    summary = manager.get_summary()
    print(f"\n+ Training summary: {summary}")

    print("\n[PASS] RLDPManager test passed!\n")


def test_rl_config():
    """测试RL配置系统"""
    print("="*50)
    print("Testing RL Configuration...")
    print("="*50)

    # 测试默认配置
    config = get_rl_dp_config()
    print("+ Default config loaded")

    # 测试预设配置
    presets = {
        "conservative": RLDPPresets.conservative_exploration(),
        "aggressive": RLDPPresets.aggressive_exploration(),
        "privacy_focused": RLDPPresets.privacy_focused(),
    }

    for name, preset in presets.items():
        print(f"+ {name.capitalize()} preset: epsilon={preset.rl_epsilon}, min_rounds={preset.rl_min_rounds}")

    print("\n[PASS] RL Configuration test passed!\n")


def test_action_space():
    """测试动作空间的完整性"""
    print("="*50)
    print("Testing Action Space...")
    print("="*50)

    agent = SimpleRLAgent(device="cpu")

    # 验证所有动作都有效
    for action in range(agent.action_dim):
        thresholds = agent.get_thresholds(action)
        assert 0 <= thresholds[0] <= 1, f"Invalid threshold_high for action {action}: {thresholds[0]}"
        assert 0 <= thresholds[1] <= 1, f"Invalid threshold_low for action {action}: {thresholds[1]}"
        print(f"+ Action {action}: {thresholds}")

    # 验证默认动作（当前策略）
    default_action = 1
    default_thresholds = agent.get_thresholds(default_action)
    assert default_thresholds == (0.6, 0.4), f"Default thresholds should be (0.6, 0.4), got {default_thresholds}"
    print(f"+ Default action {default_action} has correct thresholds: {default_thresholds}")

    print("\n[PASS] Action Space test passed!\n")


def test_integration():
    """集成测试 - 模拟完整的训练流程"""
    print("="*50)
    print("Integration Test - Simulated Training...")
    print("="*50)

    # 创建模拟环境
    agent = SimpleRLAgent(state_dim=3, learning_rate=0.02, epsilon=0.2, device="cpu")
    manager = RLDPManager(agent, update_interval=5, min_rounds_before_rl=5)

    # 模拟更复杂的训练场景
    np.random.seed(42)
    num_rounds = 30

    print("Simulating federated learning with RL-DP...")
    print("Round | Accuracy | MIA F-score | RL Active | Action | Thresholds")
    print("-" * 65)

    for round_num in range(1, num_rounds + 1):
        # 模拟渐进改善的准确率
        base_acc = 0.5 + 0.3 * (round_num / num_rounds)
        accuracy = base_acc + np.random.normal(0, 0.02)
        accuracy = max(0.3, min(0.95, accuracy))

        # 模拟MIA风险随DP强度的变化
        base_mia = 0.7 - 0.4 * (round_num / num_rounds)
        mia_f_score = base_mia + np.random.normal(0, 0.03)
        mia_f_score = max(0.1, min(0.9, mia_f_score))

        # 获取RL决策
        thresholds = manager.get_thresholds(round_num, accuracy, mia_f_score)
        is_rl_active = manager.should_use_rl(round_num)
        current_action = manager.current_action if hasattr(manager, 'current_action') else 1

        print(f"{round_num:5d} | {accuracy:8.3f} | {mia_f_score:11.3f} | {is_rl_active:9} | {current_action:6d} | {thresholds}")

        # 每10轮显示一次统计
        if round_num % 10 == 0:
            stats = agent.get_statistics()
            print(f"      Stats: epsilon={stats['epsilon']:.3f}, total_actions={stats['total_actions']}")

    # 最终统计
    final_summary = manager.get_summary()
    print(f"\n+ Final summary: {len(manager.history['rounds'])} rounds completed")
    print(f"+ Average accuracy: {np.mean(manager.history['accuracies']):.3f}")
    print(f"+ Average MIA F-score: {np.mean(manager.history['mia_f_scores']):.3f}")

    print("\n[PASS] Integration test passed!\n")


def test_checkpoint_functionality():
    """测试检查点保存和加载功能"""
    print("="*50)
    print("Testing Checkpoint Functionality...")
    print("="*50)

    # 创建临时目录
    test_dir = "temp_test_checkpoints"
    os.makedirs(test_dir, exist_ok=True)

    try:
        # 创建并训练一个RL系统
        agent = SimpleRLAgent(device="cpu")
        manager = RLDPManager(agent, update_interval=3, min_rounds_before_rl=2)

        # 模拟一些训练
        for round_num in range(1, 8):
            accuracy = 0.5 + round_num * 0.05
            mia_score = 0.6 - round_num * 0.04
            manager.get_thresholds(round_num, accuracy, mia_score)

        # 保存检查点
        checkpoint_path = os.path.join(test_dir, "test_checkpoint.json")
        manager.save_checkpoint(checkpoint_path)
        print("+ Checkpoint saved")

        # 创建新的管理器并加载检查点
        new_agent = SimpleRLAgent(device="cpu")
        new_manager = RLDPManager(new_agent, update_interval=3, min_rounds_before_rl=2)
        success = new_manager.load_checkpoint(checkpoint_path)

        assert success, "Checkpoint loading failed"
        print("+ Checkpoint loaded")

        # 验证状态是否正确恢复
        assert len(new_manager.history['rounds']) > 0, "History not restored"
        print("+ History restored correctly")

    finally:
        # 清理
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    print("\n[PASS] Checkpoint functionality test passed!\n")


def run_all_tests():
    """运行所有测试"""
    print(f"\n{'='*60}")
    print("RL-DP System Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    tests = [
        test_simple_rl_agent,
        test_rl_dp_manager,
        test_rl_config,
        test_action_space,
        test_integration,
        test_checkpoint_functionality
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] Test failed: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed! RL-DP system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RL-DP System")
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'agent', 'manager', 'config', 'action_space', 'integration', 'checkpoint'],
                        help="Which test to run")

    args = parser.parse_args()

    if args.test == 'all':
        success = run_all_tests()
        exit(0 if success else 1)
    elif args.test == 'agent':
        test_simple_rl_agent()
    elif args.test == 'manager':
        test_rl_dp_manager()
    elif args.test == 'config':
        test_rl_config()
    elif args.test == 'action_space':
        test_action_space()
    elif args.test == 'integration':
        test_integration()
    elif args.test == 'checkpoint':
        test_checkpoint_functionality()