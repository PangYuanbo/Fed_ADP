# GIA 自动攻击工具 - 创建完成！

## 🎉 任务完成总结

我已经为您创建了一个完整的 **GIA 自动攻击和可视化工具**，类似于您之前的 MIA 可视化脚本，能够自动扫描 `pretrain` 目录中的模型并执行梯度反演攻击。

---

## 📦 已创建的文件清单

### 核心脚本（3个）

1. **`system/Gradient_Inversion_Attack/gia_auto_attack.py`** (670行)
   - 完整的自动攻击脚本
   - 支持自动扫描、批量攻击、生成报告
   - 可自定义所有参数

2. **`system/Gradient_Inversion_Attack/quick_gia_attack.py`** (180行)
   - 快速启动脚本（简化版）
   - 预设推荐参数
   - 交互式确认流程

3. **`system/Gradient_Inversion_Attack/run_auto_gia.bat`** (80行)
   - Windows 一键启动批处理文件
   - 菜单式选择（快速/完整/自定义）
   - 自动检查环境

### 文档（3个）

4. **`system/Gradient_Inversion_Attack/AUTO_ATTACK_GUIDE.md`** (500行)
   - 详细使用指南
   - 所有命令行参数说明
   - 故障排除和最佳实践

5. **`system/Gradient_Inversion_Attack/QUICK_START.md`** (400行)
   - 快速开始指南
   - 3种使用方式说明
   - 常见场景示例

6. **`GIA_AUTO_ATTACK_SUMMARY.md`** (本文件)
   - 项目总结
   - 快速参考

### 测试工具（1个）

7. **`system/Gradient_Inversion_Attack/test_auto_attack.py`** (250行)
   - 自动化测试套件
   - 验证所有功能模块

### 修复（1个）

8. **`system/Gradient_Inversion_Attack/models/stylegan_wrapper.py`**
   - 修复：移除 `torch.no_grad()`，保留梯度用于优化
   - 现在 GIA 攻击可以正常运行 ✅

---

## 🚀 快速使用指南

### 最简单的方式（推荐）

```bash
cd system
python Gradient_Inversion_Attack/quick_gia_attack.py
```

这会：
1. 自动扫描 `pretrain/` 目录中的所有模型
2. 显示找到的模型列表
3. 询问确认后开始攻击
4. 自动生成所有可视化和报告

### Windows 用户（最方便）

双击或运行：
```bash
system\Gradient_Inversion_Attack\run_auto_gia.bat
```

选择菜单项：
- `1` - 快速攻击（5000次迭代）
- `2` - 完整攻击（10000次迭代）
- `3` - 自定义参数

### 高级用法（完整控制）

```bash
cd system

# 攻击特定目录的模型
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

## 📁 目录结构

### 输入（自动扫描）

```
pretrain/
├── cifar-10-normal/
│   ├── 1.00/
│   │   ├── results_client0_200.pt  ← 自动找到这些
│   │   ├── results_client1_200.pt
│   │   └── ...
│   ├── 0.50/
│   │   └── ...
│   └── 0.80/
│       └── ...
└── mnist/
    └── ...
```

### 输出（自动生成）

```
gia_auto_results/
├── quick_attack_gia_results_20241105_220530.json  ← 完整结果
├── gia_summary_report_20241105_220530.md          ← Markdown报告
└── cifar-10-normal_alpha1.00/                     ← 按数据集分组
    ├── client_0_round_200/
    │   ├── sample_0.png                           ← 原图 vs 重建对比
    │   ├── sample_0_loss.png                      ← 损失曲线
    │   ├── sample_1.png
    │   └── ...
    ├── client_1_round_200/
    │   └── ...
    └── ...
```

---

## 🎯 主要功能特性

### ✅ 自动化功能

- [x] 自动扫描 `pretrain/` 目录
- [x] 自动识别数据集、alpha、客户端ID、轮次
- [x] 自动加载对应的训练数据
- [x] 自动提取梯度并执行攻击
- [x] 自动计算质量指标（PSNR、SSIM、LPIPS）
- [x] 自动生成可视化对比图
- [x] 自动保存损失曲线
- [x] 自动生成 Markdown 报告
- [x] 自动保存 JSON 详细结果

### ✅ 支持的模型

- [x] CIFAR-10
- [x] MNIST
- [x] 可扩展到其他数据集

### ✅ 可视化输出

- [x] 原图 vs 重建图对比
- [x] PSNR/SSIM/LPIPS 指标显示
- [x] 质量等级标注（High/Medium/Low）
- [x] 损失曲线图（5种损失）
- [x] 按数据集和客户端分组保存

### ✅ 报告生成

- [x] JSON 格式详细结果
- [x] Markdown 格式总结报告
- [x] 按数据集分组统计
- [x] 详细结果表格

---

## 🆚 与 MIA 可视化对比

| 特性 | MIA 可视化 | GIA 自动攻击 |
|------|------------|-------------|
| **自动扫描** | ✅ | ✅ |
| **批量处理** | ✅ | ✅ |
| **可视化** | 折线图、箱线图 | 图像对比、损失曲线 |
| **报告格式** | Markdown | Markdown + JSON |
| **交互式确认** | ❌ | ✅ |
| **Windows批处理** | ❌ | ✅ |
| **参数自定义** | 有限 | 完全自定义 |
| **测试套件** | ❌ | ✅ |

两者设计理念一致：
- 自动发现和处理
- 批量评估
- 美观的可视化
- 详细的报告

---

## 📊 输出内容示例

### 1. 重建对比图 (sample_X.png)

```
┌──────────────────────────────────────────┐
│  Original Image  │  Reconstructed Image  │
│                  │                       │
│  [训练数据]       │  [GIA重建图像]        │
│                  │                       │
├──────────────────────────────────────────┤
│  Metrics:                               │
│    PSNR: 31.2 dB                        │
│    SSIM: 0.91                           │
│    LPIPS: 0.08                          │
│    Quality: HIGH ⚠️                     │
└──────────────────────────────────────────┘
```

### 2. 损失曲线 (sample_X_loss.png)

显示优化过程中的5种损失：
- Total Loss（总损失）
- Gradient Matching（梯度匹配）
- TV Loss（总变差）
- L2 Loss（L2正则化）
- Group Loss（组一致性）

### 3. Markdown 报告

```markdown
# GIA Auto Attack Summary Report

## Overall Statistics
- Total models evaluated: 10
- Successful attacks: 10
- Failed attacks: 0

## Results by Dataset
### cifar-10-normal
- Models evaluated: 10
- Average PSNR: 29.84 dB
- Average SSIM: 0.87
- Quality: High=3, Medium=5, Low=2

## Detailed Results
| Model | Dataset | Alpha | Client | PSNR | SSIM | Quality |
|-------|---------|-------|--------|------|------|---------|
| ...   | ...     | ...   | ...    | ...  | ...  | ...     |
```

---

## ⚙️ 参数推荐

### 快速测试（5分钟）
```bash
--num_samples 2 --iterations 1000
```

### 标准质量（15分钟）
```bash
--num_samples 5 --iterations 5000
```

### 高质量（30分钟）
```bash
--num_samples 5 --iterations 10000
```

### 论文级（1小时+）
```bash
--num_samples 10 --iterations 20000
```

---

## 💡 典型工作流程

### 场景 1: 训练后立即评估

```bash
# 1. 训练模型
cd system
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200

# 2. 执行 GIA 攻击
python Gradient_Inversion_Attack/quick_gia_attack.py

# 3. 查看结果
cd gia_auto_results
# 查看 .md 报告和 .png 图片
```

### 场景 2: 对比不同配置

```bash
# 训练多个配置
python main.py -algo FedCP -data cifar-10-normal -al 1.0 -gr 200
python main.py -algo FedCP -data cifar-10-normal -al 0.5 -gr 200
python main.py -algo FedCP -data cifar-10-normal -al 0.8 -gr 200

# 一次性评估所有
python Gradient_Inversion_Attack/quick_gia_attack.py

# 对比结果
```

### 场景 3: 评估防御效果

```bash
# 训练无防御模型
python main.py -algo FedCP -data cifar-10-normal -gr 200

# 训练 DP 防御模型
python main.py -algo FedCP -data cifar-10-normal -gr 200 -dp --epsilon 0.5

# 对比攻击效果
python Gradient_Inversion_Attack/quick_gia_attack.py

# 查看防御是否降低了重建质量
```

---

## 🔧 已修复的问题

### ✅ 梯度保留问题

**问题：** 之前 GIA 攻击失败，错误信息：
```
element 0 of tensors does not require grad and does not have a grad_fn
```

**原因：** `StyleGANWrapper.generate_from_latent()` 使用了 `with torch.no_grad()`

**修复：** 移除 `torch.no_grad()`，保留梯度用于优化

**位置：** `system/Gradient_Inversion_Attack/models/stylegan_wrapper.py:150-163`

**现在：** GIA 攻击可以正常运行 ✅

---

## 📚 相关文档

1. **快速开始**: `system/Gradient_Inversion_Attack/QUICK_START.md`
2. **详细指南**: `system/Gradient_Inversion_Attack/AUTO_ATTACK_GUIDE.md`
3. **GIA实现**: `system/Gradient_Inversion_Attack/README.md`
4. **测试结果**: `system/Gradient_Inversion_Attack/tests/TEST_RESULTS.md`
5. **项目总结**: `system/IMPLEMENTATION_SUMMARY.md`

---

## 🎓 质量指标说明

### PSNR (峰值信噪比)
- **> 30 dB**: 重建质量极高，隐私严重泄露 ⚠️⚠️⚠️
- **25-30 dB**: 中等质量，有一定隐私风险 ⚠️
- **< 25 dB**: 低质量，隐私相对安全 ✅

### SSIM (结构相似度)
- **> 0.9**: 结构几乎一致，隐私严重泄露 ⚠️⚠️⚠️
- **0.7-0.9**: 中等相似，有一定隐私风险 ⚠️
- **< 0.7**: 差异较大，隐私相对安全 ✅

### LPIPS (感知相似度)
- **< 0.1**: 感知上非常相似，隐私严重泄露 ⚠️⚠️⚠️
- **0.1-0.3**: 中等相似 ⚠️
- **> 0.3**: 差异较大，隐私相对安全 ✅

---

## 🐛 常见问题

### Q: 找不到模型？

**A:** 确保已经训练过模型：
```bash
python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200
```

### Q: StyleGAN 模型未下载？

**A:** 工具会自动使用 placeholder，但真实攻击需要：
```bash
cd system/Gradient_Inversion_Attack
bash setup.sh
```

### Q: CUDA 内存不足？

**A:** 减少样本数或使用 CPU：
```bash
python Gradient_Inversion_Attack/gia_auto_attack.py \
    --device cpu \
    --num_samples 2 \
    --iterations 5000
```

---

## ✨ 核心优势

### 1. 完全自动化
- 无需手动指定模型路径
- 无需手动加载数据
- 无需手动生成可视化
- 一键完成所有步骤

### 2. 类似 MIA 设计
- 熟悉的使用方式
- 一致的目录结构
- 相同的报告格式
- 批量处理支持

### 3. 灵活可配置
- 支持自定义所有参数
- 支持快速/标准/高质量模式
- 支持特定目录或全局扫描
- 支持 CPU/GPU 切换

### 4. 完善的文档
- 详细使用指南
- 快速开始文档
- 测试套件
- 故障排除

---

## 🎉 立即开始！

### 最简单的使用方式：

```bash
cd system
python Gradient_Inversion_Attack/quick_gia_attack.py
```

### 或者使用批处理文件（Windows）：

```bash
system\Gradient_Inversion_Attack\run_auto_gia.bat
```

---

## 📞 需要帮助？

查看文档：
- **快速开始**: `QUICK_START.md`
- **详细指南**: `AUTO_ATTACK_GUIDE.md`
- **完整文档**: `README.md`

---

**祝您使用愉快！** 🚀🎉

您现在有了一个功能完整、自动化的 GIA 攻击工具，就像您的 MIA 可视化工具一样方便！

---

**创建日期**: 2024-11-05
**版本**: 1.0
**状态**: ✅ 完成并可用
