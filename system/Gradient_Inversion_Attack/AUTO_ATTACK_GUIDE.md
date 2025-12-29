# GIA 自动攻击工具使用指南

## 📋 功能概述

这个工具可以自动扫描 `pretrain` 目录中已训练好的模型，并对每个模型执行梯度反演攻击（GI-SMN），生成详细的可视化结果和评估报告。

**类似于 MIA 可视化工具**，但针对梯度反演攻击进行了优化。

---

## 🚀 快速开始

### 方法 1: 一键快速攻击（推荐）

最简单的方式，自动攻击所有预训练模型：

```bash
cd system
python Gradient_Inversion_Attack/quick_gia_attack.py
```

该脚本会：
1. 自动扫描 `pretrain/` 目录中的所有模型
2. 显示找到的模型列表
3. 询问确认后执行攻击
4. 自动生成可视化和报告

**预设参数：**
- 每个模型重建 5 个样本
- 每个样本优化 5000 次迭代（快速模式）
- 自动保存所有可视化结果

---

### 方法 2: 完整控制攻击

如果需要自定义参数：

```bash
cd system

# 攻击特定目录的模型
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --pretrain_dir pretrain/cifar-10-normal/1.00 \
    --num_samples 5 \
    --iterations 10000

# 攻击所有模型
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --scan_all \
    --num_samples 5 \
    --iterations 10000 \
    --save_visuals
```

---

## 📁 目录结构

### 输入（预训练模型）

模型应保存在以下结构：

```
pretrain/
├── cifar-10-normal/
│   ├── 1.00/
│   │   ├── results_client0_200.pt
│   │   ├── results_client1_200.pt
│   │   └── ...
│   ├── 0.50/
│   │   └── ...
│   └── 0.80/
│       └── ...
└── mnist/
    └── ...
```

### 输出（攻击结果）

```
gia_auto_results/
├── quick_attack_gia_results_20241105_220530.json   # 完整结果JSON
├── gia_summary_report_20241105_220530.md           # Markdown报告
├── cifar-10-normal_alpha1.00/                      # 按数据集和alpha分组
│   ├── client_0_round_200/
│   │   ├── sample_0.png                            # 原图 vs 重建对比
│   │   ├── sample_0_loss.png                       # 损失曲线
│   │   ├── sample_1.png
│   │   └── ...
│   ├── client_1_round_200/
│   │   └── ...
│   └── ...
└── cifar-10-normal_alpha0.50/
    └── ...
```

---

## 🎯 命令行参数

### `gia_auto_attack.py` 参数

#### 模型扫描参数
- `--pretrain_dir <path>`: 指定预训练目录（例如：`pretrain/cifar-10-normal/1.00`）
- `--scan_all`: 扫描并攻击所有模型
- `--base_dir <path>`: 扫描的基础目录（默认：`pretrain`）

#### GIA 攻击参数
- `--stylegan_path <path>`: StyleGAN-XL 模型路径
  - 默认：`Gradient_Inversion_Attack/pretrained_models/stylegan_xl_cifar10.pkl`
- `--num_samples <int>`: 每个模型重建的样本数（默认：5）
- `--iterations <int>`: 优化迭代次数（默认：10000）
  - 快速测试：1000-5000
  - 标准质量：10000
  - 高质量：20000+
- `--lr <float>`: 学习率（默认：0.01）
- `--device <str>`: 计算设备（`cuda` 或 `cpu`，默认：`cuda`）

#### 输出参数
- `--results_dir <path>`: 结果保存目录（默认：`gia_auto_results`）
- `--save_visuals`: 保存可视化图片（默认：True）

---

## 📊 输出文件说明

### 1. JSON 结果文件

`quick_attack_gia_results_YYYYMMDD_HHMMSS.json`

包含每个模型的详细攻击结果：

```json
[
  {
    "model_info": {
      "model_path": "pretrain/cifar-10-normal/1.00/results_client0_200.pt",
      "dataset": "cifar-10-normal",
      "alpha": 1.0,
      "client_id": 0,
      "rounds": 200
    },
    "num_samples": 5,
    "avg_psnr": 28.45,
    "avg_ssim": 0.87,
    "high_quality_count": 2,
    "medium_quality_count": 3,
    "low_quality_count": 0,
    "samples": [
      {
        "sample_idx": 0,
        "psnr": 31.2,
        "ssim": 0.91,
        "lpips": 0.08,
        "quality_level": "high"
      }
    ],
    "status": "success"
  }
]
```

### 2. Markdown 报告

`gia_summary_report_YYYYMMDD_HHMMSS.md`

包含：
- 总体统计（成功/失败数量）
- 按数据集分组的统计
- 详细结果表格

### 3. 可视化图片

#### 对比图 (`sample_X.png`)
- 左侧：原始训练图像
- 右侧：GI-SMN 重建图像
- 显示 PSNR、SSIM、LPIPS 指标
- 质量等级（High/Medium/Low）

#### 损失曲线 (`sample_X_loss.png`)
- Total Loss（总损失）
- Gradient Matching Loss（梯度匹配损失）
- TV Loss（总变差损失）
- L2 Loss（L2 正则化）
- Group Loss（组一致性损失）

---

## 💡 使用场景

### 场景 1: 训练后评估

训练完成后，立即评估模型的隐私风险：

```bash
# 1. 训练模型
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200

# 2. 立即执行 GIA 攻击
python Gradient_Inversion_Attack/quick_gia_attack.py
```

### 场景 2: 对比不同 Alpha 值

评估不同 Non-IID 程度下的隐私风险：

```bash
# 攻击 alpha=1.0 的模型
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --pretrain_dir pretrain/cifar-10-normal/1.00

# 攻击 alpha=0.5 的模型
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --pretrain_dir pretrain/cifar-10-normal/0.50

# 对比两者的重建质量
```

### 场景 3: 批量评估所有模型

一次性评估项目中的所有预训练模型：

```bash
python Gradient_Inversion_Attack/gia_auto_attack.py --scan_all
```

### 场景 4: 防御机制测试

评估防御机制（DP、梯度裁剪）的有效性：

```bash
# 1. 训练无防御模型
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200

# 2. 训练 DP 防御模型
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200 -dp --epsilon 0.5

# 3. 对比两者的 GIA 攻击结果
python Gradient_Inversion_Attack/quick_gia_attack.py
```

---

## 📈 结果解读

### 质量指标

#### PSNR (峰值信噪比)
- **> 30 dB**: 高质量重建，隐私风险高 ⚠️
- **25-30 dB**: 中等质量，有一定风险
- **< 25 dB**: 低质量，隐私相对安全 ✅

#### SSIM (结构相似度)
- **> 0.9**: 结构非常相似，隐私风险高 ⚠️
- **0.7-0.9**: 结构较相似
- **< 0.7**: 结构差异大，隐私相对安全 ✅

#### LPIPS (感知相似度)
- **< 0.1**: 感知上非常相似，隐私风险高 ⚠️
- **0.1-0.3**: 感知上有一定相似
- **> 0.3**: 感知差异大，隐私相对安全 ✅

### 质量等级

- **High Quality**: PSNR > 30 且 SSIM > 0.9
  - 重建图像几乎与原图一致
  - **严重隐私泄露风险** ⚠️⚠️⚠️

- **Medium Quality**: PSNR 25-30 或 SSIM 0.7-0.9
  - 重建图像可识别但有失真
  - **中等隐私风险** ⚠️

- **Low Quality**: PSNR < 25 且 SSIM < 0.7
  - 重建图像难以识别
  - **隐私相对安全** ✅

---

## 🔧 故障排除

### 问题 1: 找不到模型

```
[Error] No models found in pretrain!
```

**解决方案：**
1. 确认已经训练过模型
2. 检查目录结构是否正确
3. 确认模型文件名格式：`results_client{id}_{rounds}.pt`

### 问题 2: StyleGAN 模型未找到

```
[Warning] Model file not found: ...stylegan_xl_cifar10.pkl
[Warning] Falling back to placeholder generator
```

**解决方案：**
```bash
cd system/Gradient_Inversion_Attack
bash setup.sh  # 下载 StyleGAN-XL 模型
```

### 问题 3: CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 减少样本数和迭代次数
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --num_samples 2 \
    --iterations 5000

# 或使用 CPU
python Gradient_Inversion_Attack/gia_auto_attack.py --device cpu
```

### 问题 4: 梯度错误

```
element 0 of tensors does not require grad
```

**解决方案：**
这个问题已在最新版本修复。如果仍然出现，请确认：
1. `stylegan_wrapper.py` 中 `generate_from_latent()` 没有使用 `torch.no_grad()`
2. 潜在代码 `z` 已设置 `requires_grad=True`

---

## 🎨 可视化示例

### 重建对比图

```
┌─────────────────────────────────────┐
│  Original Image  │  Reconstructed   │
│                  │                  │
│  [训练数据图像]   │  [GIA重建图像]    │
│                  │                  │
├─────────────────────────────────────┤
│  PSNR: 31.2 dB                     │
│  SSIM: 0.91                        │
│  LPIPS: 0.08                       │
│  Quality: HIGH ⚠️                  │
└─────────────────────────────────────┘
```

### 损失曲线

```
Loss History
│
│  Total ────
│  Grad Match ----
│  TV ······
│  L2 - - -
│  Group ─ · ─
│
└──────────────> Iterations
```

---

## 📖 与 MIA 可视化的对比

| 特性 | MIA 可视化 | GIA 自动攻击 |
|------|-----------|-------------|
| **攻击类型** | 成员推断 | 梯度反演 |
| **输出** | F-score 趋势图 | 重建图像对比 |
| **指标** | F-score, TPR, FPR | PSNR, SSIM, LPIPS |
| **可视化** | 折线图、箱线图 | 图像对比、损失曲线 |
| **目标** | 判断样本是否在训练集 | 重建训练样本内容 |
| **自动扫描** | ✅ | ✅ |
| **批量处理** | ✅ | ✅ |
| **报告生成** | ✅ (Markdown) | ✅ (Markdown) |

---

## 🔗 相关文件

- **核心攻击脚本**: `gia_auto_attack.py`
- **快速启动脚本**: `quick_gia_attack.py`
- **GI-SMN 实现**: `core/gi_smn_attack.py`
- **评估指标**: `evaluation/metrics.py`
- **可视化工具**: `evaluation/visualizer.py`
- **完整文档**: `README.md`

---

## 💬 常见问题

**Q: 攻击需要多长时间？**

A: 取决于参数设置：
- 5个样本 × 5000次迭代 ≈ 5-10分钟/模型（GPU）
- 5个样本 × 10000次迭代 ≈ 10-20分钟/模型（GPU）

**Q: 可以并行处理多个模型吗？**

A: 当前版本是串行处理。如需并行，可以手动启动多个进程，分别处理不同目录。

**Q: 重建质量不好怎么办？**

A: 尝试：
1. 增加迭代次数（`--iterations 20000`）
2. 调整学习率（`--lr 0.005`）
3. 确保使用真实的 StyleGAN-XL 模型（不是 placeholder）
4. 确认训练数据质量

**Q: 如何只攻击特定客户端？**

A: 修改扫描逻辑，或手动指定模型路径：

```python
# 在 gia_auto_attack.py 中
model_list = [m for m in model_list if m['client_id'] in [0, 1, 2]]
```

---

## 📝 更新日志

**v1.0** (2024-11-05)
- ✅ 首次发布
- ✅ 自动扫描预训练模型
- ✅ 批量 GIA 攻击
- ✅ 可视化生成
- ✅ Markdown 报告

---

## 📧 反馈与支持

如有问题或建议，请：
1. 查看 `README.md` 完整文档
2. 检查 `TEST_RESULTS.md` 测试结果
3. 参考 `IMPLEMENTATION_SUMMARY.md` 实现细节

---

**祝您使用愉快！🎉**
