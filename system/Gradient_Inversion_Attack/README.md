# GI-SMN: Gradient Inversion Attack for Fed_ADP

本模块实现了 **GI-SMN (Gradient Inversion attack based on Style Migration Network)**，这是一种先进的梯度反演攻击方法，用于评估联邦学习系统的隐私脆弱性。

## 📚 参考文献

**Jin Qian, Kaimin Wei, Yongdong Wu, Jilian Zhang, Jinpeng Chen, Huan Bao**
*"GI-SMN: Gradient Inversion Attack against Federated Learning without Prior Knowledge"*
arXiv:2405.03516, 2024
[论文链接](https://arxiv.org/abs/2405.03516)

## 🌟 核心特性

- ✅ **基于StyleGAN-XL的图像重建** - 在64维潜在空间优化（vs 3072维像素空间）
- ✅ **两阶段损失策略** - 纯梯度匹配 + 正则化
- ✅ **无需先验知识** - 不依赖批归一化统计或预训练模型
- ✅ **标准评估指标** - PSNR, SSIM, LPIPS图像质量指标
- ✅ **防御机制测试** - DP噪声、梯度裁剪、梯度剪枝
- ✅ **可视化工具** - 自动生成对比图和评估报告

## 📂 目录结构

```
Gradient_Inversion_Attack/
├── models/
│   ├── __init__.py
│   └── stylegan_wrapper.py      # StyleGAN-XL封装
├── core/
│   ├── __init__.py
│   ├── loss_functions.py        # 损失函数（梯度匹配+正则化）
│   └── gi_smn_attack.py         # GI-SMN核心攻击算法
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # PSNR/SSIM/LPIPS指标
│   ├── defense_tester.py        # 防御机制测试
│   └── visualizer.py            # 可视化工具
├── utils/
│   ├── __init__.py
│   ├── gradient_utils.py        # 梯度提取和处理
│   └── config.py                # 配置管理
├── configs/
│   └── default.yaml             # 默认配置文件
├── pretrained_models/           # StyleGAN-XL预训练模型目录
├── gia_evaluator.py             # 联邦学习评估器（主接口）
├── run_gia_standalone.py        # 独立评估脚本
├── setup.sh                     # 安装脚本
└── README.md                    # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

运行安装脚本（推荐）：

```bash
cd system/Gradient_Inversion_Attack
bash setup.sh
```

或手动安装：

```bash
# 安装Python依赖
pip install lpips piq pillow matplotlib seaborn pyyaml

# 下载StyleGAN-XL模型
mkdir -p pretrained_models
wget https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl \
     -O pretrained_models/stylegan_xl_cifar10.pkl
```

### 2. 集成到联邦学习训练

在训练命令中添加 `-gia` 参数：

```bash
python main.py \
    -algo FedCP \
    -data cifar-10-normal \
    -m cnn \
    -nc 10 \
    -gr 200 \
    -al 1.0 \
    -gia \
    --gia_num_samples 5 \
    --gia_save_visuals
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-gia`, `--enable_gia` | False | 启用GIA评估（训练结束后） |
| `--gia_stylegan_path` | `pretrained_models/stylegan_xl_cifar10.pkl` | StyleGAN-XL模型路径 |
| `--gia_num_samples` | 5 | 每个客户端重建样本数 |
| `--gia_iterations` | 10000 | 优化迭代次数 |
| `--gia_lr` | 0.01 | 潜在编码优化学习率 |
| `--gia_save_visuals` | True | 保存可视化结果 |
| `--gia_test_defense` | False | 测试防御机制 |

### 3. 独立评估已有模型

```bash
python system/Gradient_Inversion_Attack/run_gia_standalone.py \
    --model_path results/your_model.pt \
    --num_samples 10 \
    --test_defense \
    --iterations 5000
```

## 📊 评估指标

### 图像质量指标

| 指标 | 范围 | 越高越好/越低越好 | 质量标准 |
|------|------|-------------------|---------|
| **PSNR** (峰值信噪比) | [0, ∞) dB | 越高越好 | >30 dB: 高质量<br>20-30 dB: 中等质量<br><20 dB: 低质量 |
| **SSIM** (结构相似度) | [-1, 1] | 越高越好 | >0.9: 高度相似<br>0.7-0.9: 较为相似<br><0.7: 差异较大 |
| **LPIPS** (感知相似度) | [0, 1] | 越低越好 | <0.1: 高度相似<br>0.1-0.3: 中等相似<br>>0.3: 差异较大 |

### 输出示例

```
========== GIA (Gradient Inversion Attack) Evaluation ==========
[Server] Running GIA evaluation on all clients...

[GIA Evaluator] Evaluating client 0
  Sample 1/5: PSNR=28.45, SSIM=0.87, LPIPS=0.15 (medium quality)
  Sample 2/5: PSNR=31.22, SSIM=0.91, LPIPS=0.08 (high quality)
  ...

[GIA] Evaluation complete!
[GIA] Successful evaluations: 10/10 clients
[GIA] Average PSNR: 29.84 dB
[GIA] PSNR range: [25.12, 34.56] dB
[GIA] High quality reconstructions: 3 clients
[GIA] Medium quality reconstructions: 5 clients
[GIA] Low quality reconstructions: 2 clients
================================================================
```

## 📁 输出结果

GIA评估会生成以下文件结构：

```
gia_results/
└── cifar-10-normal_alpha1.0/
    ├── gia_results_round_200_2024-01-01_12-00-00.json  # 详细结果
    ├── client_0/
    │   ├── sample_0_round_200.png           # 原始 vs 重建对比图
    │   ├── sample_1_round_200.png
    │   └── ...
    ├── client_1/
    │   └── ...
    ├── visualizations/
    │   ├── quality_distribution.png         # 重建质量分布
    │   └── defense_impact.png               # 防御影响曲线
    └── defense_tests/                       # 防御测试结果（如启用）
        ├── dp_noise_impact.png
        └── gradient_clipping_impact.png
```

## 🛡️ 防御机制测试

启用防御测试以评估DP噪声和梯度裁剪的有效性：

```bash
python main.py -algo FedCP -data cifar-10-normal -gia --gia_test_defense
```

将测试以下防御参数：

- **DP噪声级别**: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
- **梯度裁剪值**: [0.001, 0.005, 0.01, 0.05, 0.1]

## ⚙️ 高级配置

### 自定义配置文件

复制并修改默认配置：

```bash
cp configs/default.yaml configs/my_config.yaml
# 编辑 my_config.yaml
```

```yaml
# configs/my_config.yaml
num_iterations: 20000      # 增加迭代次数以提高质量
learning_rate: 0.005       # 降低学习率以提高稳定性
lambda_tv: 0.005           # 增强TV正则化
```

### Python API使用

```python
from Gradient_Inversion_Attack.gia_evaluator import FederatedGIAEvaluator
from Gradient_Inversion_Attack.utils.config import GIAConfig

# 创建自定义配置
config = GIAConfig(
    num_iterations=15000,
    learning_rate=0.008,
    num_samples_per_client=10
)

# 初始化评估器
evaluator = FederatedGIAEvaluator(
    stylegan_model_path='pretrained_models/stylegan_xl_cifar10.pkl',
    device='cuda',
    config=config,
    results_dir='my_gia_results'
)

# 评估所有客户端
results = evaluator.evaluate_all_clients(
    clients=client_list,
    round_num=100,
    dataset_name='cifar-10-normal'
)
```

## 📈 性能优化

### GPU内存优化

如遇到OOM（内存不足）错误：

1. 减少样本数：`--gia_num_samples 3`
2. 降低迭代次数：`--gia_iterations 5000`
3. 关闭可视化：`--gia_save_visuals false`（注意：需设置为false而非使用action）

### 加速评估

对于快速测试，使用较少迭代次数：

```bash
python main.py -gia --gia_iterations 1000 --gia_num_samples 3
```

## 🔬 技术细节

### GI-SMN工作原理

1. **潜在空间优化**：在StyleGAN-XL的64维潜在空间 z 中优化（而非3072维像素空间）
2. **两阶段策略**：
   - 阶段1 (0 < t < 4T/9): 纯梯度匹配损失
   - 阶段2 (4T/9 < t < T): 梯度匹配 + TV/L2/Group正则化
3. **损失函数**：
   ```
   L = ||∇w* - ∇w||²_F + λ_TV·R_TV + λ_L2·R_L2 + λ_group·R_group
   ```

### 与传统方法对比

| 对比维度 | 传统方法 (DLG/iDLG) | GI-SMN |
|----------|---------------------|---------|
| 优化空间 | 3072维像素 | 64维潜在编码 |
| 先验知识 | 需要BN统计或预训练模型 | 无需先验知识 |
| 批量处理 | 困难 (batch_size=1) | 支持 (batch_size≤16) |
| 防御鲁棒性 | 易被防御 | 能绕过弱DP防御 |

## 🐛 故障排除

### 常见问题

1. **StyleGAN-XL模型未找到**
   ```
   FileNotFoundError: Model file not found
   ```
   **解决方案**：运行 `bash setup.sh` 或手动下载模型

2. **LPIPS不可用**
   ```
   lpips package not found
   ```
   **解决方案**：`pip install lpips`

3. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**：减少`--gia_num_samples`或`--gia_iterations`

4. **导入错误**
   ```
   ModuleNotFoundError: No module named 'Gradient_Inversion_Attack'
   ```
   **解决方案**：确保在`system/`目录下运行

## 📖 引用

如果您在研究中使用了本实现，请引用原论文：

```bibtex
@article{qian2024gismn,
  title={GI-SMN: Gradient Inversion Attack against Federated Learning without Prior Knowledge},
  author={Qian, Jin and Wei, Kaimin and Wu, Yongdong and Zhang, Jilian and Chen, Jinpeng and Bao, Huan},
  journal={arXiv preprint arXiv:2405.03516},
  year={2024}
}
```

## 📞 支持

如有问题或建议，请：

1. 查阅本README和代码注释
2. 检查[GI-SMN论文](https://arxiv.org/abs/2405.03516)
3. 在Fed_ADP项目中提交issue

## 📄 许可证

本模块作为Fed_ADP项目的一部分，遵循项目许可证。

---

**最后更新**: 2024年
**作者**: Fed_ADP Project
**版本**: 0.1.0 (基础框架)
