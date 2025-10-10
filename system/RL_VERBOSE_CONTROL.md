# RL-DP 日志控制说明

## 问题
在不使用RL时，系统会显示大量的RL相关日志输出：
```
[Client 3] Round 1: Using RL thresholds (0.60, 0.40)
[SimpleRL] Model saved to rl_checkpoints/...
[RLDP] Checkpoint saved to rl_checkpoints/...
```

## 解决方案
添加了 `rl_verbose` 参数来控制RL相关日志的显示。

## 使用方法

### 方法1：通过命令行参数（推荐）

在运行脚本时添加 `--rl_verbose False` 参数：

```bash
python main_rl.py \
    --dataset cifar-10-shadow \
    --enable_rl_dp True \
    --rl_verbose False \
    --global_rounds 10 \
    --num_clients 5
```

### 方法2：在代码中设置

在创建 args 对象时添加参数：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rl_verbose', type=bool, default=False,
                    help="Enable verbose logging for RL-DP system")
args = parser.parse_args()
```

### 方法3：直接修改默认值

修改 `utils/rl_config.py` 中的默认值：

```python
parser.add_argument('--rl_verbose', type=bool, default=False,  # 改为 False
                    help="Enable verbose logging for RL-DP system")
```

## 日志级别说明

### `rl_verbose=False` (默认，安静模式)
- ✅ 不显示 RL 阈值选择信息
- ✅ 不显示 SimpleRL 初始化、保存、加载信息
- ✅ 不显示 RLDP 检查点保存信息
- ✅ 不显示 RL 训练损失信息
- ❌ 仍然显示错误信息（如初始化失败）

### `rl_verbose=True` (详细模式)
- ✅ 显示所有 RL 相关日志
- ✅ 显示每轮的阈值选择
- ✅ 显示模型保存/加载
- ✅ 显示训练损失和统计信息

## 示例输出对比

### 安静模式 (`rl_verbose=False`)
```
Round 1 - Training...
[Client 0] Training completed
[Client 1] Training completed
...
Round 1 - Aggregating...
```

### 详细模式 (`rl_verbose=True`)
```
Round 1 - Training...
[Client 0] Round 1: Using RL thresholds (0.60, 0.40)
[SimpleRL] Initialized RL agent with 5 actions
[RLDP] Round 1: Action 1, Thresholds (0.6, 0.4), Acc 0.7500, MIA 0.5000
[SimpleRL] Model saved to rl_checkpoints/cifar-10-shadow\client_0_rl_checkpoint_agent.pth
[RLDP] Checkpoint saved to rl_checkpoints/cifar-10-shadow\client_0_rl_checkpoint.json
...
```

## 注意事项

1. **错误信息不受影响**：即使在安静模式下，错误信息仍会显示，确保调试能力
2. **性能无影响**：关闭日志不影响 RL-DP 功能的正常运行
3. **建议设置**：
   - 开发/调试阶段：使用 `rl_verbose=True`
   - 生产/批量实验：使用 `rl_verbose=False`
4. **RL 功能默认关闭**：默认 `enable_rl_dp=False`，如需使用 RL 功能，需明确指定 `--enable_rl_dp True`

## 相关文件

- `system/flcore/clients/clientcp_rl.py:81` - 客户端 verbose 控制
- `system/utils/simple_rl_dp.py:31` - SimpleRLAgent verbose 参数
- `system/utils/simple_rl_dp.py:312` - RLDPManager verbose 参数
- `system/utils/rl_config.py:61` - 配置参数定义
