# ============================
# train_attack_on_shadow_models.py
# 使用刚训练的shadow模型来训练MIA攻击模型
# ============================

import torch
import copy
import os
from model import FedAvgCNN, LocalModel, GradientMIA
from train_attack_model import train_attack_model
from utils.attack_feature_config import attack_checkpoint_name

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 50
LR = 5e-4  # 降低学习率以提高训练稳定性 (原来1e-3)
NUM_CLASSES = 10
NUM_CLIENTS = 5  # 你训练了5个客户端模型
ALPHA = 1.0
ATTACK_FEATURES = ['conv1', 'conv2', 'fc1', 'fc', 'softmax']

# 构建模型结构
def get_fresh_model():
    """构建FedAvgCNN模型结构"""
    base = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(DEVICE)
    head = copy.deepcopy(base.fc)
    base.fc = torch.nn.Identity()
    return LocalModel(base, head)

if __name__ == "__main__":
    print("="*60)
    print("Training MIA Attack Models on Shadow Models")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Number of Clients: {NUM_CLIENTS}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Shadow Model Path: system/pretrain/cifar-10-shadow/1.00/")
    print(f"Data Path: dataset/{ALPHA}/cifar-10-shadow/")
    print("="*60)

    # 验证数据路径
    data_dir = f"dataset/{int(ALPHA)}/cifar-10-shadow"
    print(f"\nVerifying data directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("Please ensure you're using the same data used for training the shadow models")
        exit(1)

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # 检查训练和测试数据文件
    train_files_needed = [f"train{i}_.npz" for i in range(NUM_CLIENTS)]
    test_files_needed = [f"test{i}_.npz" for i in range(NUM_CLIENTS)]

    all_data_exists = True
    for i in range(NUM_CLIENTS):
        train_file = os.path.join(train_dir, f"train{i}_.npz")
        test_file = os.path.join(test_dir, f"test{i}_.npz")

        if not os.path.exists(train_file):
            print(f"  ✗ Missing train file for client {i}: {train_file}")
            all_data_exists = False
        if not os.path.exists(test_file):
            print(f"  ✗ Missing test file for client {i}: {test_file}")
            all_data_exists = False

    if not all_data_exists:
        print("\n[ERROR] Some data files are missing!")
        exit(1)

    print(f"✓ All data files found for {NUM_CLIENTS} clients")
    print("="*60)

    # 构造shadow模型文件路径
    shadow_model_dir = "../pretrain/cifar-10-shadow/1.00"
    shadow_client_files = [
        os.path.join(shadow_model_dir, f"results_client{i}_200.pt")
        for i in range(NUM_CLIENTS)
    ]

    # 检查文件是否存在
    print("\nChecking shadow model files...")
    all_exist = True
    for i, file_path in enumerate(shadow_client_files):
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} Client {i}: {file_path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n[ERROR] Some shadow model files are missing!")
        print("Please ensure all shadow models are trained and saved.")
        exit(1)

    print("\n[INFO] All shadow model files found!")

    # 训练所有类别的攻击模型
    print(f"\nStarting attack model training for {NUM_CLASSES} labels...")
    print("-"*60)

    for target_label in range(NUM_CLASSES):
        print(f"\n{'='*60}")
        print(f"Training Attack Model for Label: {target_label}")
        print(f"{'='*60}")

        # 创建新的shadow模型实例
        shadow_model = get_fresh_model()

        # 训练攻击模型
        train_attack_model(
            shadow_model=shadow_model,
            shadow_client_files=shadow_client_files,
            target_label=target_label,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LR,
            num_clients=NUM_CLIENTS,
            alpha=ALPHA,
            attack_features=ATTACK_FEATURES,
        )

        # 检查是否成功保存
        attack_model_path = attack_checkpoint_name(target_label, ATTACK_FEATURES)
        legacy_path = attack_checkpoint_name(target_label, ATTACK_FEATURES, include_suffix=False)
        if os.path.exists(attack_model_path):
            print(f"✓ Attack model for label {target_label} saved to: {attack_model_path}")
        elif os.path.exists(legacy_path):
            print(f"✓ Attack model for label {target_label} saved to legacy path: {legacy_path}")
        else:
            print(f"✗ Warning: Attack model for label {target_label} may not have been saved properly")

    print("\n" + "="*60)
    print("Attack Model Training Complete!")
    print("="*60)

    # 列出所有生成的攻击模型
    print("\nGenerated attack models:")
    for target_label in range(NUM_CLASSES):
        attack_model_path = attack_checkpoint_name(target_label, ATTACK_FEATURES)
        legacy_path = attack_checkpoint_name(target_label, ATTACK_FEATURES, include_suffix=False)
        if os.path.exists(attack_model_path):
            size_mb = os.path.getsize(attack_model_path) / (1024 * 1024)
            print(f"  ✓ {attack_model_path} ({size_mb:.2f} MB)")
        elif os.path.exists(legacy_path):
            size_mb = os.path.getsize(legacy_path) / (1024 * 1024)
            print(f"  ✓ {legacy_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {attack_checkpoint_name(target_label, ATTACK_FEATURES)} (missing)")

    print("\n[INFO] You can now use these attack models to evaluate membership inference attacks")
    print("[INFO] on target models using the whitebox_mia_pipeline.py")
