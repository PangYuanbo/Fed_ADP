# RL模式下的训练数据加载说明

## 概述

在RL模式下，训练数据的加载方式与普通模式完全相同，都使用相同的数据加载函数和路径结构。

---

## 数据加载流程

### 1. Client初始化时的数据加载

#### RL Client (`clientcp_rl.py`)
```python
class clientCP_RL:
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, alpha=self.alpha)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, alpha=self.alpha)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)
```

#### 普通 Client (`clientcp.py`)
```python
class clientCP:
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, alpha=self.alpha)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, alpha=self.alpha)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)
```

**结论**: RL Client和普通Client使用**完全相同**的数据加载方法！

---

### 2. 底层数据加载函数

#### `utils/data_utils.py` 中的 `read_client_data()`

```python
def read_client_data(dataset, idx, is_train=True, alpha=1):
    """
    读取指定client的数据

    Args:
        dataset: 数据集名称（如'cifar10'）
        idx: client ID
        is_train: True表示训练集，False表示测试集
        alpha: 数据分布参数（Dirichlet alpha）

    Returns:
        list of (x, y) tuples
    """
    if is_train:
        train_data = read_data(dataset, idx, is_train, alpha)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train, alpha)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
```

#### `read_data()` 函数

```python
def read_data(dataset, idx, is_train=True, alpha=1):
    """
    从文件系统读取.npz数据文件

    路径格式:
    - 训练集: ../dataset/{alpha}/{dataset}/train/train{idx}_.npz
    - 测试集: ../dataset/{alpha}/{dataset}/test/test{idx}_.npz
    """
    if is_train:
        train_data_dir = os.path.join(f'../dataset/{alpha}', dataset, 'train/')
        train_file = train_data_dir + 'train' + str(idx) + '_.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data
    else:
        test_data_dir = os.path.join(f'../dataset/{alpha}', dataset, 'test/')
        test_file = test_data_dir + 'test' + str(idx) + '_.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data
```

---

## 数据文件路径结构

### 标准路径格式

```
Fed_ADP/
└── dataset/
    └── {alpha}/          # alpha值 (如 1, 0.5, 0.7等)
        └── {dataset}/    # 数据集名称
            ├── train/
            │   ├── train0_.npz
            │   ├── train1_.npz
            │   ├── train2_.npz
            │   └── ...
            └── test/
                ├── test0_.npz
                ├── test1_.npz
                ├── test2_.npz
                └── ...
```

### 实际示例

对于 `alpha=1`, `dataset=cifar-10-dp`:

```
Fed_ADP/
└── dataset/
    └── 1/
        └── cifar-10-dp/
            ├── train/
            │   ├── train0_.npz   # Client 0的训练数据
            │   ├── train1_.npz   # Client 1的训练数据
            │   ├── train2_.npz
            │   └── ...
            └── test/
                ├── test0_.npz    # Client 0的测试数据
                ├── test1_.npz    # Client 1的测试数据
                ├── test2_.npz
                └── ...
```

### 当前可用的数据集

根据目录结构，当前可用的数据集包括：

```bash
dataset/
├── 0.5/          # alpha=0.5 (高度非IID)
├── 0.7/          # alpha=0.7 (中度非IID)
├── 0.8/          # alpha=0.8
├── 1/            # alpha=1.0 (轻度非IID)
├── 10/           # alpha=10 (接近IID)
├── cifar-10-dp/       # DP训练用数据
├── cifar-10-normal/   # 普通训练用数据
├── cifar-10-shadow/   # MIA影子模型用数据
├── mnist-0.1-normal/  # MNIST数据
└── mnist-0.1-npz/
```

---

## RL模式 vs 普通模式的区别

| 特性 | RL模式 | 普通模式 |
|------|--------|----------|
| **数据加载函数** | ✅ 相同 (`read_client_data`) | ✅ 相同 |
| **数据路径** | ✅ 相同 (`../dataset/{alpha}/{dataset}/`) | ✅ 相同 |
| **DataLoader创建** | ✅ 相同 | ✅ 相同 |
| **Batch size** | ✅ 相同 | ✅ 相同 |
| **数据格式** | ✅ 相同 (.npz) | ✅ 相同 |
| **唯一区别** | ⚡ 添加了RL智能体决策差分隐私阈值 | 使用固定的DP阈值 |

**重要**: RL模式和普通模式使用**完全相同的数据**，区别仅在于差分隐私噪声添加策略的选择方式！

---

## 数据加载时序图

```
训练开始
    │
    ├─> Server初始化
    │       │
    │       └─> 创建Clients (RL或普通)
    │               │
    │               └─> 每个Client初始化
    │                       │
    │                       └─> 记录 train_samples, test_samples
    │                           (但此时并未实际加载数据！)
    │
    ├─> 训练轮次开始
    │       │
    │       ├─> Client.train_cs_model() 被调用
    │       │       │
    │       │       └─> trainloader = self.load_train_data()
    │       │               │
    │       │               └─> read_client_data(dataset, id, is_train=True, alpha)
    │       │                       │
    │       │                       └─> 从 ../dataset/{alpha}/{dataset}/train/train{id}_.npz 读取
    │       │
    │       └─> Client.test_metrics_before() 被调用
    │               │
    │               └─> testloader = self.load_test_data()
    │                       │
    │                       └─> read_client_data(dataset, id, is_train=False, alpha)
    │                               │
    │                               └─> 从 ../dataset/{alpha}/{dataset}/test/test{id}_.npz 读取
    │
    └─> 训练结束
```

**关键点**: 数据是**按需加载**的（lazy loading），只有在调用`load_train_data()`或`load_test_data()`时才真正从磁盘读取！

---

## MIA评估时的数据加载

在MIA评估时，同样使用client的数据加载方法：

```python
# mia_attack_wrapper.py
def evaluate_client_mia(self, client, target_labels=None):
    # 使用client自己的数据加载器
    train_loader = client.load_train_data(batch_size=self.batch_size)
    test_loader = client.load_test_data(batch_size=self.batch_size)

    # 进行MIA评估...
```

这确保了：
- ✅ MIA评估使用的是client实际训练用的数据
- ✅ 数据路径和格式完全一致
- ✅ 不需要重新实现数据加载逻辑

---

## 数据文件格式

### .npz文件结构

```python
# 读取示例
with open('train0_.npz', 'rb') as f:
    data = np.load(f, allow_pickle=True)['data'].tolist()

# data 结构:
{
    'x': numpy.ndarray,  # shape: (num_samples, channels, height, width)
                        # 例如 CIFAR-10: (500, 3, 32, 32)
    'y': numpy.ndarray   # shape: (num_samples,)
                        # 例如 CIFAR-10: (500,) 标签范围 0-9
}
```

### 数据转换流程

```
.npz文件
    ↓
numpy.ndarray (read_data)
    ↓
torch.Tensor (read_client_data)
    ↓
list of (x, y) tuples
    ↓
DataLoader
    ↓
训练/评估批次
```

---

## 配置示例

### 运行RL训练使用alpha=1的数据

```bash
python main_rl.py \
    --dataset cifar-10-dp \
    --alpha 1.0 \
    --num_clients 10 \
    --global_rounds 100 \
    --enable_rl_dp True \
    --difference_privacy True
```

数据加载路径: `../dataset/1/cifar-10-dp/train/train{0-9}_.npz`

### 运行RL训练使用alpha=0.5的数据（更高非IID）

```bash
python main_rl.py \
    --dataset cifar-10-dp \
    --alpha 0.5 \
    --num_clients 10 \
    --global_rounds 100 \
    --enable_rl_dp True \
    --difference_privacy True
```

数据加载路径: `../dataset/0.5/cifar-10-dp/train/train{0-9}_.npz`

---

## 常见问题

### Q1: RL模式下数据加载有什么特殊的地方吗？
**A**: 没有！RL模式和普通模式使用完全相同的数据加载方式。RL只影响差分隐私噪声的添加策略，不影响数据本身。

### Q2: 如何验证数据加载路径是否正确？
**A**: 检查日志输出，或者在`read_data()`函数中添加打印语句：
```python
def read_data(dataset, idx, is_train=True, alpha=1):
    if is_train:
        train_file = f'../dataset/{alpha}/{dataset}/train/train{idx}_.npz'
        print(f"[Data] Loading: {train_file}")  # 添加这行
        ...
```

### Q3: 如何查看某个client的数据量？
**A**:
```python
import numpy as np

# 读取client 0的训练数据
file_path = '../dataset/1/cifar-10-dp/train/train0_.npz'
with open(file_path, 'rb') as f:
    data = np.load(f, allow_pickle=True)['data'].tolist()

print(f"Train samples: {data['x'].shape[0]}")
print(f"Data shape: {data['x'].shape}")
print(f"Label distribution: {np.bincount(data['y'])}")
```

### Q4: MIA评估使用的数据和训练数据是同一份吗？
**A**: 是的！MIA评估调用`client.load_train_data()`和`client.load_test_data()`，这与训练时使用的完全相同。

### Q5: 为什么要有不同的alpha值？
**A**: Alpha是Dirichlet分布的参数，控制数据的非IID程度：
- **alpha=10**: 接近IID，每个client的数据分布相似
- **alpha=1**: 轻度非IID
- **alpha=0.5**: 中度非IID
- **alpha=0.1**: 高度非IID，每个client的数据分布差异很大

---

## 数据流总结图

```
┌─────────────────────────────────────────────────────────┐
│                    Fed_ADP/dataset/                     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ alpha=0.5│  │ alpha=1.0│  │ alpha=10 │  ...       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘            │
│        │             │              │                  │
│        └─────────────┴──────────────┘                  │
│                      │                                  │
│              ┌───────┴────────┐                        │
│              │  cifar-10-dp   │                        │
│              └───────┬────────┘                        │
│                      │                                  │
│            ┌─────────┴─────────┐                       │
│            │                   │                       │
│      ┌─────▼─────┐      ┌─────▼─────┐                │
│      │   train/  │      │   test/   │                │
│      │ train*.npz│      │ test*.npz │                │
│      └─────┬─────┘      └─────┬─────┘                │
└────────────┼──────────────────┼──────────────────────┘
             │                  │
             │ read_client_data │
             ▼                  ▼
    ┌────────────────────────────────┐
    │  Client (RL or Normal)         │
    │  - load_train_data()           │
    │  - load_test_data()            │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │  DataLoader                    │
    │  - Batching                    │
    │  - Shuffling (if enabled)      │
    └────────┬───────────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────┐    ┌──────────┐
│ Training│    │   MIA    │
│         │    │Evaluation│
└─────────┘    └──────────┘
```

---

## 总结

1. **RL模式和普通模式使用相同的数据加载机制**
2. **数据路径**: `../dataset/{alpha}/{dataset}/train|test/{train|test}{client_id}_.npz`
3. **数据格式**: .npz文件，包含`x`和`y`两个numpy数组
4. **按需加载**: 数据在调用`load_train_data()`/`load_test_data()`时才真正读取
5. **MIA评估**: 使用与训练完全相同的数据加载方法

**核心原则**: 无论是RL模式还是普通模式，数据加载都是透明且一致的！

---

**相关文件**:
- `utils/data_utils.py` - 数据加载函数
- `flcore/clients/clientcp.py` - 普通Client
- `flcore/clients/clientcp_rl.py` - RL Client
- `utils/mia_attack_wrapper.py` - MIA评估器
