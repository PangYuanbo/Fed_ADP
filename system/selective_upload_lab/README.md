# Selective Upload LAB

独立实验模块：测试选择性上传+噪声添加的MIA防御方案

## 方案概述

本实验测试一种新的隐私保护策略，结合了**选择性上传**和**差分隐私噪声**：

### 核心思想

从第50轮开始（可配置），对客户端参数进行分类处理：

1. **大梯度+高曲率参数** → **本地保留**，不上传到服务器
   - 这些参数对模型性能影响大，保留可以维持准确率
   - 不上传可以降低隐私泄露风险

2. **其他参数** → **添加DP噪声后上传**
   - 使用现有的差分隐私机制添加高斯噪声
   - 复用epsilon、delta、clip_value等参数

3. **本地平滑机制**
   - 客户端收到全局模型后，对保留的参数位置进行平滑融合：
   - `new_param = (本地保留值 + 全局新值) / 2`
   - 避免本地参数与全局模型偏差过大

4. **服务器稀疏聚合**
   - 只对收到更新的参数进行加权平均
   - 如果所有客户端都保留某参数，该位置维持全局模型原值

### 预期效果

- **隐私保护**：MIA F-score降低（攻击成功率下降）
- **模型性能**：准确率保持相对稳定
- **收敛速度**：可能略慢，但在可接受范围内

---

## 文件结构

```
selective_upload_lab/
├── __init__.py              # 模块初始化
├── client_selective.py      # 继承clientCP，实现选择性上传逻辑
├── server_selective.py      # 继承FedCP，实现稀疏聚合逻辑
├── run_selective_test.py    # 独立运行脚本
└── README.md               # 本文档
```

---

## 使用方法

### 1. 选择性上传 + 全局MIA

```bash
cd system
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal -gr 200 -dp -mia \
    --enable_selective_upload \
    --selective_upload_round 50 \
    --gradient_keep_ratio 0.3
```

### 2. 标准FedAvg + 全局MIA（对比baseline）

```bash
cd system
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal -gr 200 -dp -mia
```

注意：不加`--enable_selective_upload`参数时，默认使用标准FedAvg（但仍然进行全局MIA评估）

### 3. 完整参数示例

```bash
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal \
    -gr 200 \
    -dp \
    -mia \
    --enable_selective_upload \
    --selective_upload_round 50 \
    --gradient_keep_ratio 0.3 \
    --noise_multiplier 1.0 \
    --epsilon 0.8 \
    --clip_value 0.005 \
    -nc 20 \
    -al 1.0
```

### 4. 参数说明

#### 核心参数（新增）

- `--enable_selective_upload`：启用选择性上传机制（默认False）
- `--selective_upload_round`：开始应用选择性上传的轮次（默认50，仅在enable_selective_upload时有效）
- `--gradient_keep_ratio`：保留的大梯度参数比例（默认0.3，即top 30%，仅在enable_selective_upload时有效）
- `--noise_multiplier`：DP噪声倍率（默认1.0，值越大噪声越大，适用于调节隐私-效用平衡）

#### 基础参数

- `-data`：数据集名称（如cifar-10-normal）
- `-gr`：全局训练轮次（默认200）
- `-dp`：启用差分隐私
- `-mia`：启用MIA评估（每10轮评估一次）
- `-nc`：客户端数量（默认20）
- `-al`：数据异构性参数alpha（默认1.0，越小越异构）

#### DP参数

- `--epsilon`：隐私预算（默认0.8）
- `--delta`：隐私参数（默认1e-5）
- `--clip_value`：梯度裁剪值（默认0.005）
- `--threshold_high/low`：阈值参数（默认0.6/0.4）

---

## 对比实验

### 实验1: 标准FedAvg + 全局MIA（无DP）

```bash
cd system
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal -gr 200 -mia
```

### 实验2: 标准FedAvg + 全局MIA（有DP）

```bash
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal -gr 200 -dp -mia
```

### 实验3: 选择性上传 + 全局MIA（有DP）

```bash
python selective_upload_lab/run_selective_test.py \
    -data cifar-10-normal -gr 200 -dp -mia \
    --enable_selective_upload \
    --selective_upload_round 50 \
    --gradient_keep_ratio 0.3
```

**对比目标**：
- 准确率：实验3 ≈ 实验2 > 实验1
- MIA F-score：实验3 < 实验2 < 实验1（越低越好）

---

## 评估指标

### 1. MIA F-score（主要指标）

- **位置**：每10轮自动评估
- **含义**：成员推断攻击的成功率
- **期望**：选择性上传方案的F-score < 标准DP < 无DP
- **结果保存**：`mia_results/selective_upload/`

### 2. 模型准确率

- **位置**：每轮记录
- **含义**：全局模型在测试集上的准确率
- **期望**：选择性上传方案的准确率 ≈ 标准DP，略高于标准DP
- **结果保存**：`results/results_cifar-10-normal_200_0.0050_selective_dp.txt`

### 3. 收敛速度

- **含义**：达到目标准确率所需的轮次
- **期望**：选择性上传方案收敛速度略慢，但可接受

---

## 技术细节

### 1. 参数保留判断

```python
# 计算梯度幅度阈值
grad_threshold = torch.quantile(param_diff.abs().view(-1), 1.0 - gradient_keep_ratio)

# 结合Hessian曲率（如可用）
high_gradient_mask = param_diff.abs() >= grad_threshold
# 可选：额外考虑Hessian高曲率

# 保留掩码
kept_mask = high_gradient_mask
```

### 2. 本地平滑机制

```python
# 客户端接收全局模型后
for name, param in model.named_parameters():
    if kept_mask[name]:  # 该参数上轮被保留
        # 平滑融合
        param.data = (kept_params[name] + global_param.data) / 2
    else:
        param.data = global_param.data  # 直接使用全局值
```

### 3. 服务器稀疏聚合

```python
# 对每个参数位置
for param_name in global_model.state_dict():
    valid_clients = [c for c in clients if not c.kept_mask[param_name]]

    if len(valid_clients) > 0:
        # 加权平均有效更新
        global_param = weighted_avg(valid_clients)
    else:
        # 所有客户端都保留，维持原值
        global_param = old_global_param
```

---

## 预期输出示例

```
============================================================
Round 50 - Selective Upload (enabled from round 50)
============================================================

[Client 0 Round 50] feature_extractor.conv1.weight: kept 1824/1920 (95.00%)
[Client 0 Round 50] feature_extractor.conv2.weight: kept 30720/32000 (96.00%)
...

[Server Round 50] Parameter Upload Statistics:
  feature_extractor.conv1.weight: 5/20 clients uploaded
  feature_extractor.conv2.weight: 8/20 clients uploaded
  ...

[MIA Evaluation] Round 50
[MIA] Average F-score: 0.65

Round 50 - Test Accuracy: 0.7845
```

---

## 实验建议

### 参数调优建议

1. **gradient_keep_ratio**
   - 起始值：0.3（保留top 30%）
   - 可尝试：0.2, 0.3, 0.4
   - 影响：越大保留越多，准确率越高，但隐私保护可能减弱

2. **selective_upload_round**
   - 起始值：50
   - 可尝试：30, 50, 70
   - 影响：越早开始，隐私保护越早生效，但可能影响初期收敛

3. **epsilon**
   - 起始值：0.8
   - 可尝试：0.5, 0.8, 1.2
   - 影响：越小隐私保护越强，但噪声越大

4. **noise_multiplier**
   - 起始值：1.0（标准噪声）
   - 可尝试：0.5, 1.0, 2.0, 5.0
   - 影响：直接控制噪声幅度，越大隐私保护越强，但准确率可能下降
   - 建议：与epsilon配合使用，noise_multiplier=2.0 相当于 epsilon减半

### 实验组合建议

```bash
# 实验1：不同保留比例
for ratio in 0.2 0.3 0.4; do
    python selective_upload_lab/run_selective_test.py \
        -data cifar-10-normal -gr 200 -dp -mia \
        --gradient_keep_ratio $ratio
done

# 实验2：不同启动轮次
for round in 30 50 70; do
    python selective_upload_lab/run_selective_test.py \
        -data cifar-10-normal -gr 200 -dp -mia \
        --selective_upload_round $round
done

# 实验3：不同epsilon
for eps in 0.5 0.8 1.2; do
    python selective_upload_lab/run_selective_test.py \
        -data cifar-10-normal -gr 200 -dp -mia \
        --epsilon $eps
done

# 实验4：不同噪声倍率
for multiplier in 0.5 1.0 2.0 5.0; do
    python selective_upload_lab/run_selective_test.py \
        -data cifar-10-normal -gr 200 -dp -mia \
        --enable_selective_upload \
        --noise_multiplier $multiplier
done
```

---

## 故障排查

### 问题1：MIA评估失败

**现象**：`[MIA] Warning: Evaluation failed`

**解决**：
1. 检查预训练攻击模型是否存在：`Membership_Inference_Attack/attack_model{0-9}.pth`
2. 确认GPU内存足够
3. 尝试减少`--batch_size`

### 问题2：准确率异常低

**现象**：准确率< 20%

**解决**：
1. 检查`gradient_keep_ratio`是否过大（>0.5）
2. 检查`selective_upload_round`是否过早（<30）
3. 确认DP参数合理（epsilon不要太小）

### 问题3：CUDA OOM

**现象**：`CUDA out of memory`

**解决**：
```bash
# 减少batch size
python selective_upload_lab/run_selective_test.py -data cifar-10-normal -lbs 5

# 或使用CPU
python selective_upload_lab/run_selective_test.py -data cifar-10-normal -dev cpu
```

---

## 引用与致谢

本实验基于FedCP框架，复用了以下模块：
- `clientCP`：客户端训练和DP逻辑
- `FedCP`：服务器聚合和MIA评估
- `FederatedMIAEvaluator`：MIA攻击评估器

---

## 许可

本LAB与主项目使用相同的许可协议。
