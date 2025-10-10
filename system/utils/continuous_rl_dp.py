# ============================
# continuous_rl_dp.py
# 连续动作空间的强化学习自适应差分隐私系统
# 使用DDPG (Deep Deterministic Policy Gradient) 算法
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Tuple, Dict, List, Optional
from collections import deque
import random


class Actor(nn.Module):
    """
    Actor网络：输出连续的动作值
    输出: (threshold_high, threshold_low) 两个0-1之间的值
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用sigmoid确保输出在0-1之间
        action = torch.sigmoid(self.fc3(x))
        return action


class Critic(nn.Module):
    """
    Critic网络：评估(state, action)对的Q值
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck噪声，用于探索"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ContinuousRLAgent:
    """
    连续动作空间的RL智能体，使用DDPG算法
    动作空间：(threshold_high, threshold_low) 两个连续值，范围[0, 1]
    """

    def __init__(self,
                 state_dim: int = 3,
                 action_dim: int = 2,  # threshold_high, threshold_low
                 actor_lr: float = 0.0001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.95,
                 tau: float = 0.005,  # 软更新参数
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 noise_scale: float = 0.1,
                 device: str = "auto"):
        """
        初始化连续RL智能体

        Args:
            state_dim: 状态维度 (accuracy, mia_f_score, round_progress)
            action_dim: 动作维度 (threshold_high, threshold_low)
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            memory_size: 经验回放缓存大小
            batch_size: 训练批次大小
            noise_scale: 探索噪声缩放
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale

        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 创建Actor网络（当前和目标）
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 创建Critic网络（当前和目标）
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 探索噪声
        self.noise = OUNoise(action_dim)

        # 经验回放缓存
        self.memory = deque(maxlen=memory_size)

        # 统计信息
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'q_values': []
        }

        print(f"[ContinuousRL] Initialized DDPG agent with continuous action space")
        print(f"[ContinuousRL] Action dim: {action_dim} (threshold_high, threshold_low)")
        print(f"[ContinuousRL] Action range: [0, 1] for each dimension")
        print(f"[ContinuousRL] Device: {self.device}")

    def get_state(self, accuracy: float, mia_f_score: float, round_num: int, max_rounds: int = 1000):
        """构建状态向量"""
        round_progress = min(round_num / max_rounds, 1.0)
        state = torch.tensor([accuracy, mia_f_score, round_progress],
                           dtype=torch.float32, device=self.device)
        return state

    def select_action(self, state: torch.Tensor, add_noise: bool = True):
        """
        选择动作（连续值）

        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声

        Returns:
            action: numpy array [threshold_high, threshold_low]
        """
        self.actor.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        # 添加探索噪声
        if add_noise:
            noise = self.noise.sample() * self.noise_scale
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def get_thresholds(self, action: np.ndarray) -> Tuple[float, float]:
        """
        从动作向量转换为阈值

        Args:
            action: [threshold_high, threshold_low]

        Returns:
            (threshold_high, threshold_low) 百分位阈值对
        """
        # 确保 threshold_high > threshold_low
        threshold_high = max(action[0], action[1])
        threshold_low = min(action[0], action[1])

        # 防止两个阈值太接近
        if threshold_high - threshold_low < 0.1:
            threshold_high = min(threshold_low + 0.1, 1.0)

        return float(threshold_high), float(threshold_low)

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓存"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """训练一步（从经验回放缓存中采样并更新网络）"""
        if len(self.memory) < self.batch_size:
            return None, None

        # 采样batch
        batch = random.sample(self.memory, self.batch_size)

        states = torch.stack([item[0] for item in batch]).to(self.device)
        actions = torch.FloatTensor([item[1] for item in batch]).to(self.device)
        rewards = torch.FloatTensor([item[2] for item in batch]).unsqueeze(1).to(self.device)
        next_states = torch.stack([item[3] for item in batch]).to(self.device)
        dones = torch.FloatTensor([item[4] for item in batch]).unsqueeze(1).to(self.device)

        # ============= 更新Critic =============
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ============= 更新Actor =============
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============= 软更新目标网络 =============
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # 记录统计信息
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['q_values'].append(current_q.mean().item())

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, local_model, target_model):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_history = checkpoint.get('training_history', {'actor_loss': [], 'critic_loss': [], 'q_values': []})

    def print_statistics(self):
        """打印训练统计信息"""
        if len(self.training_history['actor_loss']) > 0:
            print(f"\n[ContinuousRL] Training Statistics:")
            print(f"  Recent Actor Loss: {np.mean(self.training_history['actor_loss'][-10:]):.4f}")
            print(f"  Recent Critic Loss: {np.mean(self.training_history['critic_loss'][-10:]):.4f}")
            print(f"  Recent Q-value: {np.mean(self.training_history['q_values'][-10:]):.4f}")
            print(f"  Total training steps: {len(self.training_history['actor_loss'])}")


class ContinuousRLDPManager:
    """
    连续RL差分隐私管理器
    管理连续动作空间的RL训练和推理流程
    """

    def __init__(self,
                 agent: ContinuousRLAgent,
                 update_interval: int = 10,
                 min_rounds_before_rl: int = 20,
                 reward_accuracy_weight: float = 1.0,
                 reward_mia_penalty: float = 2.0,
                 reward_improvement_bonus: float = 0.5):
        """
        初始化连续RL-DP管理器

        Args:
            agent: ContinuousRLAgent实例
            update_interval: RL策略更新间隔轮次
            min_rounds_before_rl: 启用RL前的最小轮次数
            reward_accuracy_weight: 准确率权重
            reward_mia_penalty: MIA惩罚权重
            reward_improvement_bonus: 改进奖励权重
        """
        self.agent = agent
        self.update_interval = update_interval
        self.min_rounds_before_rl = min_rounds_before_rl

        # 奖励函数权重
        self.reward_accuracy_weight = reward_accuracy_weight
        self.reward_mia_penalty = reward_mia_penalty
        self.reward_improvement_bonus = reward_improvement_bonus

        # 状态管理
        self.current_state = None
        self.current_action = None
        self.last_accuracy = 0.0
        self.last_mia_f_score = 0.5

        # 历史记录
        self.history = {
            'rounds': [],
            'accuracies': [],
            'mia_scores': [],
            'rewards': [],
            'actions': []
        }

        print(f"[ContinuousRLDP Manager] Initialized with continuous action space")
        print(f"[ContinuousRLDP Manager] Update interval: {update_interval}")
        print(f"[ContinuousRLDP Manager] RL will be enabled after round {min_rounds_before_rl}")

    def should_use_rl(self, round_num: int) -> bool:
        """判断是否应该使用RL"""
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
            return (0.6, 0.4)

        # 构建当前状态
        current_state = self.agent.get_state(accuracy, mia_f_score, round_num)

        # 训练RL智能体 (如果有历史数据)
        if self.current_state is not None and self.current_action is not None:
            # 计算奖励
            reward = self.calculate_reward(accuracy, mia_f_score, self.last_accuracy, self.last_mia_f_score)

            # 存储经验
            done = False
            self.agent.remember(self.current_state, self.current_action, reward, current_state, done)

            # 记录历史
            self.history['rewards'].append(reward)

            # 训练
            if round_num % self.update_interval == 0:
                actor_loss, critic_loss = self.agent.train_step()
                if actor_loss is not None:
                    print(f"[ContinuousRLDP] Round {round_num}: Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")

        # 选择动作（连续值）
        add_noise = self.should_use_rl(round_num)  # 只在RL阶段添加噪声探索
        action = self.agent.select_action(current_state, add_noise=add_noise)
        thresholds = self.agent.get_thresholds(action)

        # 更新状态历史
        self.current_state = current_state
        self.current_action = action
        self.last_accuracy = accuracy
        self.last_mia_f_score = mia_f_score

        # 记录历史
        self.history['rounds'].append(round_num)
        self.history['accuracies'].append(accuracy)
        self.history['mia_scores'].append(mia_f_score)
        self.history['actions'].append(action.tolist())

        return thresholds

    def calculate_reward(self, accuracy: float, mia_f_score: float,
                        prev_accuracy: float, prev_mia_f_score: float) -> float:
        """
        计算奖励

        奖励 = accuracy_weight * accuracy
               - mia_penalty * mia_f_score
               + improvement_bonus * (accuracy_improvement - mia_increase)
        """
        # 基础奖励：准确率 - MIA风险
        base_reward = (self.reward_accuracy_weight * accuracy -
                      self.reward_mia_penalty * mia_f_score)

        # 改进奖励
        accuracy_improvement = accuracy - prev_accuracy
        mia_decrease = prev_mia_f_score - mia_f_score
        improvement_reward = self.reward_improvement_bonus * (accuracy_improvement + mia_decrease)

        total_reward = base_reward + improvement_reward

        return total_reward

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        # 保存智能体
        agent_path = filepath.replace('.json', '_agent.pth')
        self.agent.save(agent_path)

        # 保存管理器状态
        checkpoint = {
            'update_interval': self.update_interval,
            'min_rounds_before_rl': self.min_rounds_before_rl,
            'last_accuracy': self.last_accuracy,
            'last_mia_f_score': self.last_mia_f_score,
            'history': self.history,
            'reward_weights': {
                'accuracy': self.reward_accuracy_weight,
                'mia_penalty': self.reward_mia_penalty,
                'improvement_bonus': self.reward_improvement_bonus
            }
        }

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        # 加载智能体
        agent_path = filepath.replace('.json', '_agent.pth')
        if os.path.exists(agent_path):
            self.agent.load(agent_path)

        # 加载管理器状态
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)

            self.update_interval = checkpoint.get('update_interval', self.update_interval)
            self.min_rounds_before_rl = checkpoint.get('min_rounds_before_rl', self.min_rounds_before_rl)
            self.last_accuracy = checkpoint.get('last_accuracy', 0.0)
            self.last_mia_f_score = checkpoint.get('last_mia_f_score', 0.5)
            self.history = checkpoint.get('history', {'rounds': [], 'accuracies': [], 'mia_scores': [], 'rewards': [], 'actions': []})

            weights = checkpoint.get('reward_weights', {})
            self.reward_accuracy_weight = weights.get('accuracy', self.reward_accuracy_weight)
            self.reward_mia_penalty = weights.get('mia_penalty', self.reward_mia_penalty)
            self.reward_improvement_bonus = weights.get('improvement_bonus', self.reward_improvement_bonus)
