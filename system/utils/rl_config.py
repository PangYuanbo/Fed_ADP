# ============================
# rl_config.py
# RL-DP系统配置文件
# ============================

import argparse


class RLDPConfig:
    """
    RL差分隐私配置类
    包含所有RL相关的超参数和设置
    """

    def __init__(self):
        # RL智能体配置
        self.rl_learning_rate = 0.01
        self.rl_epsilon = 0.1              # 初始探索率
        self.rl_epsilon_decay = 0.995      # 探索率衰减
        self.rl_epsilon_min = 0.05         # 最小探索率
        self.rl_discount_factor = 0.95     # 折扣因子
        self.rl_memory_size = 1000         # 经验回放缓存大小
        self.rl_batch_size = 32            # 训练批次大小

        # RL管理器配置
        self.rl_update_interval = 10       # RL策略更新间隔轮次
        self.rl_min_rounds = 20            # 启用RL前的最小轮次数
        self.rl_target_network_update = 20 # 目标网络更新间隔

        # 奖励函数配置
        self.reward_accuracy_weight = 1.0   # 准确率权重
        self.reward_mia_penalty = 2.0       # MIA惩罚权重
        self.reward_improvement_bonus = 0.5 # 改进奖励权重

        # 动作空间配置 (threshold_high, threshold_low)
        self.action_space = {
            0: (0.5, 0.3),   # 更激进的噪声添加
            1: (0.6, 0.4),   # 当前策略 (初始默认)
            2: (0.7, 0.5),   # 更保守的噪声添加
            3: (0.8, 0.2),   # 极端策略：只选择很高和很低的梯度
            4: (0.4, 0.6),   # 极端策略：主要选择中等梯度
        }

        # 检查点和日志配置
        self.rl_checkpoint_save_interval = 50  # 检查点保存间隔
        self.rl_statistics_print_interval = 50 # 统计信息打印间隔

        # MIA评估配置
        self.mia_evaluation_for_rl_interval = 10  # 为RL提供MIA评估的间隔

    def add_rl_args(self, parser: argparse.ArgumentParser):
        """
        向ArgumentParser添加RL相关参数

        Args:
            parser: ArgumentParser对象
        """
        # RL启用开关
        parser.add_argument('--enable_rl_dp', type=bool, default=True,
                            help="Enable RL-based adaptive differential privacy")

        # RL智能体参数
        parser.add_argument('--rl_learning_rate', type=float, default=self.rl_learning_rate,
                            help="Learning rate for RL agent")
        parser.add_argument('--rl_epsilon', type=float, default=self.rl_epsilon,
                            help="Initial exploration rate for RL agent")
        parser.add_argument('--rl_epsilon_decay', type=float, default=self.rl_epsilon_decay,
                            help="Exploration rate decay for RL agent")
        parser.add_argument('--rl_epsilon_min', type=float, default=self.rl_epsilon_min,
                            help="Minimum exploration rate for RL agent")
        parser.add_argument('--rl_discount_factor', type=float, default=self.rl_discount_factor,
                            help="Discount factor for RL agent")
        parser.add_argument('--rl_memory_size', type=int, default=self.rl_memory_size,
                            help="Memory size for experience replay")
        parser.add_argument('--rl_batch_size', type=int, default=self.rl_batch_size,
                            help="Batch size for RL training")

        # RL管理器参数
        parser.add_argument('--rl_update_interval', type=int, default=self.rl_update_interval,
                            help="Interval for RL policy updates (in rounds)")
        parser.add_argument('--rl_min_rounds', type=int, default=self.rl_min_rounds,
                            help="Minimum rounds before enabling RL")

        # 奖励函数参数
        parser.add_argument('--reward_accuracy_weight', type=float, default=self.reward_accuracy_weight,
                            help="Weight for accuracy in reward function")
        parser.add_argument('--reward_mia_penalty', type=float, default=self.reward_mia_penalty,
                            help="Penalty weight for MIA risk in reward function")
        parser.add_argument('--reward_improvement_bonus', type=float, default=self.reward_improvement_bonus,
                            help="Bonus weight for improvements in reward function")

    def update_from_args(self, args):
        """
        从ArgumentParser参数更新配置

        Args:
            args: ArgumentParser解析后的参数
        """
        # 更新所有RL相关配置
        for attr in dir(self):
            if not attr.startswith('_') and hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))

    def print_config(self):
        """打印当前配置"""
        print("\n" + "="*50)
        print("RL-DP System Configuration:")
        print("="*50)

        print("RL Agent Configuration:")
        print(f"  Learning Rate: {self.rl_learning_rate}")
        print(f"  Exploration Rate: {self.rl_epsilon} -> {self.rl_epsilon_min} (decay: {self.rl_epsilon_decay})")
        print(f"  Discount Factor: {self.rl_discount_factor}")
        print(f"  Memory Size: {self.rl_memory_size}")
        print(f"  Batch Size: {self.rl_batch_size}")

        print("\nRL Manager Configuration:")
        print(f"  Update Interval: {self.rl_update_interval} rounds")
        print(f"  Min Rounds Before RL: {self.rl_min_rounds}")

        print("\nReward Function Configuration:")
        print(f"  Accuracy Weight: {self.reward_accuracy_weight}")
        print(f"  MIA Penalty: {self.reward_mia_penalty}")
        print(f"  Improvement Bonus: {self.reward_improvement_bonus}")

        print("\nAction Space:")
        for action, thresholds in self.action_space.items():
            print(f"  Action {action}: {thresholds}")

        print("="*50)

    def get_config_dict(self):
        """返回配置字典"""
        config = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                config[attr] = getattr(self, attr)
        return config


# 全局配置实例
rl_dp_config = RLDPConfig()


def get_rl_dp_config():
    """获取RL-DP配置实例"""
    return rl_dp_config


# 配置预设
class RLDPPresets:
    """
    RL-DP配置预设
    包含不同场景下的推荐配置
    """

    @staticmethod
    def conservative_exploration():
        """保守探索配置 - 更注重稳定性"""
        config = RLDPConfig()
        config.rl_epsilon = 0.05
        config.rl_epsilon_decay = 0.99
        config.rl_min_rounds = 30
        config.rl_update_interval = 15
        config.reward_mia_penalty = 3.0
        return config

    @staticmethod
    def aggressive_exploration():
        """激进探索配置 - 更注重学习效率"""
        config = RLDPConfig()
        config.rl_epsilon = 0.2
        config.rl_epsilon_decay = 0.99
        config.rl_min_rounds = 10
        config.rl_update_interval = 5
        config.reward_mia_penalty = 1.5
        return config

    @staticmethod
    def balanced():
        """平衡配置 - 默认推荐"""
        config = RLDPConfig()
        return config

    @staticmethod
    def privacy_focused():
        """隐私优先配置 - 更注重隐私保护"""
        config = RLDPConfig()
        config.reward_accuracy_weight = 0.5
        config.reward_mia_penalty = 4.0
        config.rl_min_rounds = 40
        return config

    @staticmethod
    def accuracy_focused():
        """准确率优先配置 - 更注重模型性能"""
        config = RLDPConfig()
        config.reward_accuracy_weight = 2.0
        config.reward_mia_penalty = 1.0
        config.rl_min_rounds = 15
        return config


def create_rl_args_parser():
    """
    创建包含RL参数的ArgumentParser

    Returns:
        配置好的ArgumentParser
    """
    parser = argparse.ArgumentParser(description="RL-Adaptive Differential Privacy Federated Learning")

    # 添加原有的联邦学习参数
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

    # 添加RL相关参数
    rl_dp_config.add_rl_args(parser)

    return parser


if __name__ == "__main__":
    # 测试配置系统
    print("Testing RL-DP Configuration System")

    # 测试默认配置
    config = get_rl_dp_config()
    config.print_config()

    # 测试不同预设
    print("\n" + "="*50)
    print("Testing Different Presets:")
    print("="*50)

    presets = {
        "Conservative": RLDPPresets.conservative_exploration(),
        "Aggressive": RLDPPresets.aggressive_exploration(),
        "Privacy-Focused": RLDPPresets.privacy_focused(),
        "Accuracy-Focused": RLDPPresets.accuracy_focused()
    }

    for name, preset in presets.items():
        print(f"\n{name} Preset:")
        print(f"  Epsilon: {preset.rl_epsilon}, Min Rounds: {preset.rl_min_rounds}")
        print(f"  Accuracy Weight: {preset.reward_accuracy_weight}, MIA Penalty: {preset.reward_mia_penalty}")

    # 测试ArgumentParser
    parser = create_rl_args_parser()
    test_args = parser.parse_args([
        '--enable_rl_dp', 'True',
        '--rl_learning_rate', '0.02',
        '--dataset', 'cifar10'
    ])

    print(f"\nParsed Args Test:")
    print(f"  RL Enabled: {test_args.enable_rl_dp}")
    print(f"  RL Learning Rate: {test_args.rl_learning_rate}")
    print(f"  Dataset: {test_args.dataset}")