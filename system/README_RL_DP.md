# RL自适应差分隐私联邦学习系统 (RL-Adaptive Differential Privacy FL)

## 系统概述

本系统在原有的FedCP联邦学习框架基础上，集成了强化学习（RL）来自适应学习最优的差分隐私噪声添加策略。系统通过学习梯度百分位阈值的选择，在模型准确性和隐私保护之间找到最优平衡。

## 核心特性

### 1. **简化的RL框架**
- 基于PyTorch的轻量级Q-learning实现
- 无需复杂的RL库依赖
- 仅约150行核心RL代码

### 2. **智能阈值学习**
- 5个离散动作空间，包含当前的(0.6, 0.4)阈值作为初始策略
- 自适应学习最优的梯度百分位选择
- 结合一阶梯度分析和二阶Hessian曲率信息

### 3. **多维奖励机制**
- 准确率提升奖励
- MIA攻击抵抗奖励
- 策略稳定性考量

### 4. **完整的MIA集成**
- 实时MIA风险评估
- RL智能体反馈机制
- 隐私风险可视化

## 文件结构

```
system/
├── utils/
│   ├── simple_rl_dp.py           # 核心RL智能体和管理器
│   ├── rl_config.py              # RL配置和超参数管理
│   └── mia_attack_wrapper.py     # MIA攻击评估器
├── flcore/
│   ├── clients/
│   │   └── clientcp_rl.py        # 集成RL的客户端
│   └── servers/
│       └── servercp_rl.py        # 支持RL的服务器
├── main_rl.py                    # RL增强的主程序
├── test_rl_dp_system.py          # 系统测试脚本
├── run_rl_dp_test.sh             # 快速测试脚本
└── README_RL_DP.md               # 本文档
```

## 快速开始

### 1. 系统测试

```bash
# 运行基础组件测试
python test_rl_dp_system.py --test all

# 测试特定组件
python test_rl_dp_system.py --test agent      # 测试RL智能体
python test_rl_dp_system.py --test manager    # 测试RL管理器
python test_rl_dp_system.py --test integration # 集成测试
```

### 2. 快速训练测试

```bash
# 使用平衡配置运行快速测试
python main_rl.py --preset balanced --global_rounds 50 --num_clients 5 --device cpu

# 使用隐私优先配置
python main_rl.py --preset privacy_focused --global_rounds 100

# 使用准确率优先配置
python main_rl.py --preset accuracy_focused --global_rounds 100
```

### 3. 完整训练示例

```bash
# CIFAR-10数据集，启用RL-DP和MIA评估
python main_rl.py \
    --dataset cifar10 \
    --model cnn \
    --num_clients 10 \
    --global_rounds 1000 \
    --difference_privacy True \
    --enable_rl_dp True \
    --enable_mia True \
    --preset balanced
```

## 配置参数

### RL相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_rl_dp` | True | 启用RL自适应差分隐私 |
| `--rl_learning_rate` | 0.01 | RL智能体学习率 |
| `--rl_epsilon` | 0.1 | 初始探索率 |
| `--rl_epsilon_decay` | 0.995 | 探索率衰减 |
| `--rl_min_rounds` | 20 | 启用RL前的最小轮次 |
| `--rl_update_interval` | 10 | RL策略更新间隔 |

### 奖励函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reward_accuracy_weight` | 1.0 | 准确率权重 |
| `--reward_mia_penalty` | 2.0 | MIA风险惩罚权重 |
| `--reward_improvement_bonus` | 0.5 | 改进奖励权重 |

### 预设配置

使用 `--preset` 参数选择预设配置：

- `balanced`: 平衡配置（默认推荐）
- `conservative`: 保守探索，更注重稳定性
- `aggressive`: 激进探索，更注重学习效率
- `privacy_focused`: 隐私优先，强调隐私保护
- `accuracy_focused`: 准确率优先，强调模型性能

## 动作空间设计

RL智能体可以选择5种不同的阈值策略：

| 动作 | 阈值对 (high, low) | 策略描述 |
|------|-------------------|----------|
| 0 | (0.5, 0.3) | 更激进的噪声添加 |
| 1 | (0.6, 0.4) | 当前策略（初始默认） |
| 2 | (0.7, 0.5) | 更保守的噪声添加 |
| 3 | (0.8, 0.2) | 极端策略：高+低梯度 |
| 4 | (0.4, 0.6) | 极端策略：中等梯度 |

## 训练流程

### 阶段1: 预热阶段（前N轮）
- 使用默认阈值 (0.6, 0.4)
- 收集基础性能数据
- 建立MIA基线

### 阶段2: RL学习阶段
- RL智能体开始探索不同策略
- 基于准确率和MIA风险调整策略
- 每N轮更新Q网络

### 阶段3: 策略优化阶段
- 逐渐减少随机探索
- 倾向于选择最优策略
- 持续监控和微调

## 输出文件

### 训练日志
- `results/`: 准确率记录
- `results_after/`: 加噪后准确率
- `logs/`: 详细的梯度范数日志

### RL相关输出
- `rl_checkpoints/`: RL智能体检查点
- `rl_summaries/`: 训练摘要
- `clip_value/`: 裁剪值记录

### MIA评估结果
- `mia_results/`: MIA攻击评估结果
- MIA趋势可视化图表

### 模型保存
- `pretrain/`: 训练完成的模型参数

## 性能监控

### 1. 实时监控
训练过程中会显示：
- 每轮的准确率和MIA风险
- RL选择的动作和阈值
- 探索率和奖励值

### 2. 定期统计
- 每20轮显示RL决策信息
- 每50轮保存检查点和统计
- 每100轮显示总体RL状态

### 3. 最终报告
训练结束后生成：
- RL训练摘要
- MIA评估汇总
- 动作分布统计
- 性能趋势分析

## 故障排除

### 常见问题

1. **RL不启用**
   ```
   检查: --enable_rl_dp True 和 --difference_privacy True
   确保: 轮次数 > rl_min_rounds
   ```

2. **MIA评估失败**
   ```
   检查: Membership_Inference_Attack/ 目录存在
   确保: 攻击模型已训练 (attack_model*.pth)
   ```

3. **内存不足**
   ```
   减少: --num_clients 或 --rl_memory_size
   使用: --device cpu 如果GPU内存不足
   ```

4. **收敛慢**
   ```
   调整: --rl_learning_rate (增加到0.02-0.05)
   减少: --rl_min_rounds (提前启用RL)
   ```

### 调试模式

```bash
# 启用详细日志
python main_rl.py --preset balanced --global_rounds 20 --eval_gap 1

# 只测试RL组件
python test_rl_dp_system.py --test all
```

## 扩展开发

### 添加新动作
在 `utils/rl_config.py` 中修改 `action_space`:

```python
self.action_space = {
    0: (0.5, 0.3),
    1: (0.6, 0.4),
    2: (0.7, 0.5),
    # 添加新动作
    5: (0.9, 0.1),   # 新的极端策略
}
```

### 自定义奖励函数
在 `utils/simple_rl_dp.py` 中修改 `calculate_reward()`:

```python
def calculate_reward(self, accuracy, mia_f_score, ...):
    # 自定义奖励逻辑
    custom_reward = your_formula(accuracy, mia_f_score)
    return custom_reward
```

### 添加新的状态特征
扩展 `get_state()` 方法以包含更多特征：

```python
def get_state(self, accuracy, mia_f_score, round_num, new_feature):
    state = torch.tensor([accuracy, mia_f_score, round_progress, new_feature])
    return state
```

## 论文和引用

如果您在研究中使用了此系统，请考虑引用相关工作。

## 联系信息

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目链接]
- 邮件: [联系邮箱]

---

**注意**: 此系统仅用于研究目的。在生产环境中使用前，请充分测试和验证隐私保证。