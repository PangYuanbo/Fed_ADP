# Dataset Setup Guide

This guide explains how to set up the datasets for running FedCP with Membership Inference Attack (MIA) and Reinforcement Learning Differential Privacy (RL-DP) experiments.

## 📦 Required Datasets

The project requires the following datasets, which are **NOT included in this repository** due to their large size (~16GB):

### 1. CIFAR-10 Dataset

**Location**: `dataset/`

The CIFAR-10 dataset should be organized for federated learning with different data distribution settings (IID and Non-IID).

**Structure**:
```
dataset/
├── cifar-10-normal/
│   ├── train/
│   └── test/
└── cifar-10-shadow/  (for MIA experiments)
    ├── train/
    └── test/
```

### 2. Public CIFAR-10 Data (5% IID)

**Location**: `public_cifar10_data_iid_5percent/`

A small subset of CIFAR-10 data for public auxiliary training.

---

## 🔧 How to Generate Datasets

### Option 1: Automatic Download (Recommended)

The datasets will be automatically downloaded when you first run the training scripts:

```bash
cd system
sh run_me.sh
```

The script will check for missing datasets and download them automatically.

### Option 2: Manual Setup

You can generate the federated learning datasets using [PFLlib](https://github.com/TsingZ0/PFLlib):

```bash
# Clone PFLlib
git clone https://github.com/TsingZ0/PFLlib.git

# Generate CIFAR-10 with different settings
cd PFLlib
python generate_data.py --dataset cifar10 --split_type noniid --beta 0.1

# Copy the generated data to this project
cp -r ./dataset ../Fed_ADP/dataset/
```

---

## 📂 Dataset Parameters

### Non-IID Settings

The project uses **Dirichlet distribution** to simulate Non-IID data:

- **α (alpha) = 1.0**: Default setting (moderate heterogeneity)
- **α = 0.1**: High heterogeneity (more challenging)
- **α = 10.0**: Low heterogeneity (close to IID)

### MIA-Specific Datasets

For Membership Inference Attack experiments, you need:

1. **Normal Model Dataset**: Used for training the target federated model
2. **Shadow Model Dataset**: Used for training the MIA attack model

Both datasets are variations of CIFAR-10 with different train/test splits.

---

## 💾 Storage Requirements

- **CIFAR-10 Base**: ~160MB
- **Federated Learning Splits**: ~15GB (multiple clients, train/test)
- **MIA Shadow Models**: ~1GB
- **Total**: ~16GB

---

## 🔍 Verify Dataset Setup

After setting up the datasets, verify the structure:

```bash
# Check dataset directories exist
ls -lh dataset/
ls -lh public_cifar10_data_iid_5percent/

# Check number of client data directories
find dataset/cifar-10-normal/train/ -type d | wc -l
```

Expected output: Should show directories for each client (e.g., 10-100 clients depending on your setup).

---

## ⚙️ Configuration

Update dataset paths in configuration files if needed:

**File**: `system/utils/data_utils.py`

```python
# Default dataset paths
DATASET_PATH = "../dataset/"
PUBLIC_DATA_PATH = "../public_cifar10_data_iid_5percent/"
```

---

## 🚀 Quick Start

Once datasets are set up:

```bash
cd system

# Run standard FedCP
sh run_me.sh

# Run with RL-DP
sh run_rl_dp_test.sh

# Run with MIA evaluation
sh run_with_mia.sh
```

---

## 📚 References

- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **PFLlib**: https://github.com/TsingZ0/PFLlib
- **FedCP Paper**: https://arxiv.org/pdf/2307.01217v2.pdf

---

## 💡 Notes

- Datasets are stored locally and **NOT pushed to GitHub** (see `.gitignore`)
- First run may take longer due to dataset download/generation
- Ensure you have at least 20GB free disk space
- For faster setup, download pre-processed datasets from [PFLlib releases](https://github.com/TsingZ0/PFLlib/releases)
