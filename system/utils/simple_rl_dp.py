# ============================
# simple_rl_dp.py
# 简化的强化学习自适应差分隐私系统
# ============================

import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Tuple, Dict, List, Optional
from collections import deque
import random


class SimpleRLAgent:
    """
    简化的强化学习智能体，用于学习最优的梯度百分位阈值选择策略
    使用epsilon-greedy Q-learning算法
    """

    def __init__(self,
                 state_dim: int = 3,
                 learning_rate: float = 0.01,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05,
                 discount_factor: float = 0.95,
                 memory_size: int = 1000,
                 device: str = "auto"):
        """
        初始化简化RL智能体

        Args:
            state_dim: 状态维度 (accuracy, mia_f_score, round_progress)
            learning_rate: 学习率
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            discount_factor: 折扣因子
            memory_size: 经验回放缓存大小
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = 5  # 5个离散动作
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = discount_factor

        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 定义动作空间：(threshold_high, threshold_low) 百分位对
        self.action_space = {
            0: (0.5, 0.3),   # 更激进的噪声添加
            1: (0.6, 0.4),   # 当前策略 (初始默认)
            2: (0.7, 0.5),   # 更保守的噪声添加
            3: (0.8, 0.2),   # 极端策略：只选择很高和很低的梯度
            4: (0.4, 0.6),   # 极端策略：主要选择中等梯度
        }

        # 简单的Q网络
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # 经验回放缓存
        self.memory = deque(maxlen=memory_size)

        # 统计信息
        self.training_history = []
        self.action_counts = {i: 0 for i in range(self.action_dim)}

        # 初始化目标网络
        self.update_target_network()

        print(f"[SimpleRL] Initialized RL agent with {self.action_dim} actions")
        print(f"[SimpleRL] Action space: {self.action_space}")
        print(f"[SimpleRL] Device: {self.device}")

    def _build_network(self) -> nn.Module:
        """构建简单的Q网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )

    def get_state(self, accuracy: float, mia_f_score: float, round_num: int, max_rounds: int = 1000) -> torch.Tensor:
        """
        构建状态向量

        Args:
            accuracy: 当前模型准确率
            mia_f_score: MIA攻击F-score (0-1)
            round_num: 当前轮次
            max_rounds: 最大轮次数

        Returns:
            状态向量 [accuracy, mia_f_score, round_progress]
        """
        round_progress = min(round_num / max_rounds, 1.0)
        state = torch.tensor([accuracy, mia_f_score, round_progress],
                           dtype=torch.float32, device=self.device)
        return state

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """
        选择动作 (epsilon-greedy策略)

        Args:
            state: 当前状态
            training: 是否为训练模式

        Returns:
            选择的动作索引
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            action = random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最高的动作
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()

        # 统计动作选择
        self.action_counts[action] += 1

        return action

    def get_thresholds(self, action: int) -> Tuple[float, float]:
        """
        根据动作获取阈值对

        Args:
            action: 动作索引

        Returns:
            (threshold_high, threshold_low) 百分位阈值对
        """
        return self.action_space[action]

    def calculate_reward(self,
                        accuracy: float,
                        mia_f_score: float,
                        prev_accuracy: float = None,
                        prev_mia_f_score: float = None,
                        alpha: float = 1.0,
                        beta: float = 2.0) -> float:
        """
        计算奖励函数

        Args:
            accuracy: 当前准确率
            mia_f_score: 当前MIA F-score
            prev_accuracy: 前一轮准确率
            prev_mia_f_score: 前一轮MIA F-score
            alpha: 准确率权重
            beta: MIA惩罚权重

        Returns:
            计算得到的奖励值
        """
        # 基础奖励：准确率 - MIA风险
        base_reward = alpha * accuracy - beta * mia_f_score

        # 如果有历史数据，考虑改进量
        if prev_accuracy is not None and prev_mia_f_score is not None:
            accuracy_improvement = accuracy - prev_accuracy
            mia_improvement = prev_mia_f_score - mia_f_score  # MIA下降是好事
            improvement_bonus = 0.5 * (accuracy_improvement + mia_improvement)
            base_reward += improvement_bonus

        return base_reward

    def store_experience(self, state: torch.Tensor, action: int, reward: float,
                        next_state: torch.Tensor, done: bool):
        """存储经验到回放缓存"""
        self.memory.append((state.cpu(), action, reward, next_state.cpu(), done))

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        执行一步训练

        Args:
            batch_size: 批次大小

        Returns:
            损失值，如果没有足够经验则返回None
        """
        if len(self.memory) < batch_size:
            return None

        # 采样批次
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([exp[0] for exp in batch]).to(self.device)
        actions = torch.tensor([exp[1] for exp in batch], device=self.device)
        rewards = torch.tensor([exp[2] for exp in batch], device=self.device)
        next_states = torch.stack([exp[3] for exp in batch]).to(self.device)
        dones = torch.tensor([exp[4] for exp in batch], device=self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'action_counts': self.action_counts,
            'action_space': self.action_space
        }
        torch.save(checkpoint, filepath)
        print(f"[SimpleRL] Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_history = checkpoint.get('training_history', [])
            self.action_counts = checkpoint.get('action_counts', {i: 0 for i in range(self.action_dim)})
            print(f"[SimpleRL] Model loaded from {filepath}")
            return True
        else:
            print(f"[SimpleRL] Model file not found: {filepath}")
            return False

    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        total_actions = sum(self.action_counts.values())
        action_probabilities = {
            k: v/total_actions if total_actions > 0 else 0
            for k, v in self.action_counts.items()
        }

        return {
            'epsilon': self.epsilon,
            'total_actions': total_actions,
            'action_counts': self.action_counts,
            'action_probabilities': action_probabilities,
            'memory_size': len(self.memory),
            'training_steps': len(self.training_history)
        }

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n[SimpleRL] Training Statistics:")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Total actions: {stats['total_actions']}")
        print(f"  Memory size: {stats['memory_size']}")
        print(f"  Action distribution:")
        for action, prob in stats['action_probabilities'].items():
            thresholds = self.action_space[action]
            print(f"    Action {action} {thresholds}: {prob:.3f} ({stats['action_counts'][action]} times)")


class RLDPManager:
    """
    RL-DP管理器，负责协调RL智能体和差分隐私策略
    """

    def __init__(self,
                 agent: SimpleRLAgent,
                 update_interval: int = 10,
                 min_rounds_before_rl: int = 20):
        """
        初始化RL-DP管理器

        Args:
            agent: RL智能体
            update_interval: RL策略更新间隔轮次
            min_rounds_before_rl: 启用RL前的最小轮次数
        """
        self.agent = agent
        self.update_interval = update_interval
        self.min_rounds_before_rl = min_rounds_before_rl

        # 历史记录
        self.history = {
            'rounds': [],
            'accuracies': [],
            'mia_f_scores': [],
            'actions': [],
            'thresholds': [],
            'rewards': []
        }

        # 当前状态
        self.current_state = None
        self.current_action = 1  # 默认使用动作1 (0.6, 0.4)
        self.last_accuracy = None
        self.last_mia_f_score = None

        print(f"[RLDP Manager] Initialized with update interval: {update_interval}")
        print(f"[RLDP Manager] RL will be enabled after round {min_rounds_before_rl}")

    def should_use_rl(self, round_num: int) -> bool:
        """判断是否应该使用RL策略"""
        return round_num >= self.min_rounds_before_rl

    def get_thresholds(self, round_num: int, accuracy: float, mia_f_score: float = 0.5) -> Tuple[float, float]:
        """
        获取当前轮次的阈值策略

        Args:
            round_num: 当前轮次
            accuracy: 当前准确率
            mia_f_score: MIA F-score

        Returns:
            (threshold_high, threshold_low) 阈值对
        """
        if not self.should_use_rl(round_num):
            # 使用默认策略
            return self.agent.action_space[1]  # (0.6, 0.4)

        # 构建当前状态
        current_state = self.agent.get_state(accuracy, mia_f_score, round_num)

        # 训练RL智能体 (如果有历史数据)
        if self.current_state is not None and len(self.history['rounds']) > 0:
            # 计算奖励
            reward = self.agent.calculate_reward(
                accuracy, mia_f_score,
                self.last_accuracy, self.last_mia_f_score
            )

            # 存储经验
            done = False  # 在联邦学习中，每轮都不是终态
            self.agent.store_experience(
                self.current_state, self.current_action, reward, current_state, done
            )

            # 记录历史
            self.history['rewards'].append(reward)

            # 训练
            if round_num % self.update_interval == 0:
                loss = self.agent.train_step()
                if loss is not None:
                    print(f"[RLDP] Round {round_num}: RL training loss = {loss:.4f}")

                # 定期更新目标网络
                if round_num % (self.update_interval * 2) == 0:
                    self.agent.update_target_network()

        # 选择动作
        self.current_action = self.agent.select_action(current_state, training=True)
        thresholds = self.agent.get_thresholds(self.current_action)

        # 更新状态历史
        self.current_state = current_state
        self.last_accuracy = accuracy
        self.last_mia_f_score = mia_f_score

        # 记录历史
        self.history['rounds'].append(round_num)
        self.history['accuracies'].append(accuracy)
        self.history['mia_f_scores'].append(mia_f_score)
        self.history['actions'].append(self.current_action)
        self.history['thresholds'].append(thresholds)

        if round_num % 20 == 0:  # 每20轮打印一次
            print(f"[RLDP] Round {round_num}: Action {self.current_action}, Thresholds {thresholds}, Acc {accuracy:.4f}, MIA {mia_f_score:.4f}")

        return thresholds

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'history': self.history,
            'current_action': self.current_action,
            'last_accuracy': self.last_accuracy,
            'last_mia_f_score': self.last_mia_f_score
        }

        # 保存RL智能体
        agent_path = filepath.replace('.json', '_agent.pth')
        self.agent.save_model(agent_path)

        # 保存管理器状态
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        print(f"[RLDP] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)

            self.history = checkpoint.get('history', self.history)
            self.current_action = checkpoint.get('current_action', 1)
            self.last_accuracy = checkpoint.get('last_accuracy', None)
            self.last_mia_f_score = checkpoint.get('last_mia_f_score', None)

            # 加载RL智能体
            agent_path = filepath.replace('.json', '_agent.pth')
            self.agent.load_model(agent_path)

            print(f"[RLDP] Checkpoint loaded from {filepath}")
            return True
        else:
            print(f"[RLDP] Checkpoint file not found: {filepath}")
            return False

    def get_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.history['rounds']:
            return {'error': 'No training history available'}

        return {
            'total_rounds': len(self.history['rounds']),
            'latest_round': max(self.history['rounds']) if self.history['rounds'] else 0,
            'avg_accuracy': np.mean(self.history['accuracies']) if self.history['accuracies'] else 0,
            'avg_mia_f_score': np.mean(self.history['mia_f_scores']) if self.history['mia_f_scores'] else 0,
            'avg_reward': np.mean(self.history['rewards']) if self.history['rewards'] else 0,
            'agent_stats': self.agent.get_statistics(),
            'action_distribution': {
                str(k): f"{v[0]:.1f}, {v[1]:.1f}"
                for k, v in self.agent.action_space.items()
            }
        }