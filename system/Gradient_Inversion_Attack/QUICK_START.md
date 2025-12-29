# GIA 自动攻击工具 - 快速开始

## ✅ 工具已经创建完成！

我已经为您创建了一个类似 MIA 可视化工具的 GIA 自动攻击系统。

---

## 📦 已创建的文件

1. **`gia_auto_attack.py`** - 完整的自动攻击脚本（主程序）
2. **`quick_gia_attack.py`** - 快速启动脚本（简化版）
3. **`run_auto_gia.bat`** - Windows 一键启动批处理文件
4. **`AUTO_ATTACK_GUIDE.md`** - 详细使用指南
5. **`test_auto_attack.py`** - 测试脚本
6. **`QUICK_START.md`** - 本文件（快速开始指南）

---

## 🚀 3种使用方式

### 方式 1: Windows 批处理文件（最简单）

双击或运行：
```bash
Gradient_Inversion_Attack\run_auto_gia.bat
```

会出现菜单让您选择：
- 快速攻击（5000次迭代）
- 完整攻击（10000次迭代）
- 自定义参数

### 方式 2: 快速Python脚本

```bash
cd system
python Gradient_Inversion_Attack/quick_gia_attack.py
```

这会自动：
1. 扫描 `pretrain/` 目录
2. 显示找到的模型
3. 询问确认后开始攻击
4. 生成所有可视化和报告

### 方式 3: 完整控制（自定义参数）

```bash
cd system

# 攻击特定目录
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --pretrain_dir pretrain/cifar-10-normal/1.00 \
    --num_samples 5 \
    --iterations 10000 \
    --save_visuals

# 攻击所有模型
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --scan_all \
    --num_samples 5 \
    --iterations 10000
```

---

## 📁 输入输出

### 输入：预训练模型

工具会自动扫描以下目录结构：

```
pretrain/
├── cifar-10-normal/
│   ├── 1.00/
│   │   ├── results_client0_200.pt  ← 扫描这些文件
│   │   ├── results_client1_200.pt
│   │   └── ...
│   └── 0.50/
│       └── ...
└── mnist/
    └── ...
```

### 输出：攻击结果

```
gia_auto_results/
├── quick_attack_gia_results_TIMESTAMP.json  ← 完整结果
├── gia_summary_report_TIMESTAMP.md          ← Markdown报告
└── cifar-10-normal_alpha1.00/               ← 按数据集分组
    ├── client_0_round_200/
    │   ├── sample_0.png                     ← 对比图
    │   ├── sample_0_loss.png                ← 损失曲线
    │   └── ...
    └── ...
```

---

## 🎯 典型工作流程

### 1. 训练模型

```bash
cd system
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200
```

这会在 `pretrain/cifar-10-normal/1.00/` 创建模型文件。

### 2. 执行GIA攻击

```bash
# 方式 A: 使用批处理文件（推荐）
Gradient_Inversion_Attack\run_auto_gia.bat

# 方式 B: 使用Python脚本
python Gradient_Inversion_Attack/quick_gia_attack.py
```

### 3. 查看结果

打开 `gia_auto_results/` 目录：
- 查看 `.md` 报告文件（可以用任何文本编辑器打开）
- 查看 `.png` 可视化图片
- 查看 `.json` 详细数据

---

## 📊 输出内容解释

### 1. 重建对比图 (sample_X.png)

显示原始图像和重建图像的并排对比：

```
┌──────────────────────────────────┐
│  Original    │  Reconstructed    │
│  [训练图像]   │  [GIA重建图像]    │
├──────────────────────────────────┤
│  PSNR: 31.2 dB                  │
│  SSIM: 0.91                     │
│  LPIPS: 0.08                    │
│  Quality: HIGH ⚠️               │
└──────────────────────────────────┘
```

### 2. 损失曲线 (sample_X_loss.png)

显示优化过程中各种损失的变化：
- **Total Loss**: 总损失
- **Gradient Matching**: 梯度匹配损失
- **TV Loss**: 总变差正则化
- **L2 Loss**: L2 正则化
- **Group Loss**: 组一致性损失

### 3. JSON 结果文件

包含每个样本的详细指标：
```json
{
  "model_info": {...},
  "avg_psnr": 28.45,
  "avg_ssim": 0.87,
  "high_quality_count": 2,
  "samples": [...]
}
```

### 4. Markdown 报告

包含：
- 总体统计
- 按数据集分组的结果
- 详细结果表格

---

## ⚙️ 参数调整建议

### 快速测试（5分钟）

```bash
--num_samples 2 --iterations 1000
```

### 标准质量（10-20分钟）

```bash
--num_samples 5 --iterations 5000
```

### 高质量（30-60分钟）

```bash
--num_samples 5 --iterations 10000
```

### 论文级质量（1-2小时）

```bash
--num_samples 10 --iterations 20000
```

---

## 🔍 质量指标说明

### PSNR (峰值信噪比)
- **> 30 dB**: 重建质量极高，隐私严重泄露 ⚠️⚠️⚠️
- **25-30 dB**: 中等质量
- **< 25 dB**: 低质量，隐私相对安全 ✅

### SSIM (结构相似度)
- **> 0.9**: 结构几乎一致，隐私严重泄露 ⚠️⚠️⚠️
- **0.7-0.9**: 中等相似
- **< 0.7**: 差异较大，隐私相对安全 ✅

### 质量等级
- **High**: PSNR > 30 且 SSIM > 0.9（危险 ⚠️）
- **Medium**: 25 < PSNR < 30 或 0.7 < SSIM < 0.9
- **Low**: PSNR < 25 且 SSIM < 0.7（安全 ✅）

---

## 🆚 与 MIA 可视化对比

| 特性 | MIA 可视化 | GIA 自动攻击 |
|------|------------|-------------|
| **输入** | 训练历史 | 预训练模型 |
| **输出** | F-score图表 | 重建图像对比 |
| **指标** | F-score, TPR, FPR | PSNR, SSIM, LPIPS |
| **目标** | 成员推断 | 数据重建 |
| **自动扫描** | ✅ | ✅ |
| **批处理** | ✅ | ✅ |
| **可视化** | 折线图 | 图像对比 |

两者都支持：
- ✅ 自动扫描目录
- ✅ 批量处理
- ✅ 生成Markdown报告
- ✅ 保存可视化结果

---

## 💡 使用场景

### 场景 1: 评估模型隐私风险

训练后立即评估：
```bash
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200
python Gradient_Inversion_Attack/quick_gia_attack.py
```

### 场景 2: 对比不同Alpha值

```bash
# 训练不同 alpha 的模型
python main.py -algo FedCP -data cifar-10-normal -al 1.0 -gr 200
python main.py -algo FedCP -data cifar-10-normal -al 0.5 -gr 200

# 对比攻击效果
python Gradient_Inversion_Attack/quick_gia_attack.py
```

### 场景 3: 评估防御机制

```bash
# 无防御
python main.py -algo FedCP -data cifar-10-normal -gr 200

# DP防御
python main.py -algo FedCP -data cifar-10-normal -gr 200 -dp --epsilon 0.5

# 对比结果
python Gradient_Inversion_Attack/quick_gia_attack.py
```

---

## 🐛 常见问题

### Q: 找不到模型文件

**A:** 确保已经训练过模型：
```bash
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200
```

### Q: StyleGAN 模型未下载

**A:** 工具会自动使用 placeholder，但真实攻击需要：
```bash
cd Gradient_Inversion_Attack
bash setup.sh
```

### Q: CUDA 内存不足

**A:** 减少参数或使用 CPU：
```bash
python Gradient_Inversion_Attack/gia_auto_attack.py --device cpu --num_samples 2
```

### Q: 重建质量不好

**A:** 增加迭代次数：
```bash
python Gradient_Inversion_Attack/gia_auto_attack.py --iterations 20000
```

---

## 📚 更多信息

- **详细使用指南**: `AUTO_ATTACK_GUIDE.md`
- **GIA实现文档**: `README.md`
- **测试结果**: `tests/TEST_RESULTS.md`
- **实施总结**: `../IMPLEMENTATION_SUMMARY.md`

---

## 🎉 总结

您现在有了一个功能完整的 GIA 自动攻击工具！

**最简单的使用方式：**
```bash
cd system
python Gradient_Inversion_Attack/quick_gia_attack.py
```

工具会自动：
1. ✅ 扫描所有预训练模型
2. ✅ 执行梯度反演攻击
3. ✅ 生成可视化对比图
4. ✅ 创建详细报告
5. ✅ 保存所有结果

**祝您使用愉快！** 🚀
