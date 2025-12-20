# ============================
# train_attack_model.py
# ============================

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils.mia_attack_model import GradientMIA
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from data_utils import read_client_data, filter_by_label


def train_attack_model(shadow_model, shadow_client_files, target_label, batch_size, device, epochs, lr, num_clients,alpha):
    attack_model = GradientMIA().to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 添加学习率调度器：当验证loss不再下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print(len(shadow_client_files), len(shadow_client_files[0]))

    shadow_train_datasets = read_client_data(is_train=True, is_shadow=True,num_clients=num_clients,alpha=alpha)
    shadow_holdout_datasets = read_client_data(is_train=False, is_shadow=True,num_clients=num_clients,alpha=alpha)

    print("[INFO] Original dataset sizes:")
    for idx, (train_ds, holdout_ds) in enumerate(zip(shadow_train_datasets, shadow_holdout_datasets)):
        print(f"  Client {idx}: Train={len(train_ds)}, Holdout={len(holdout_ds)}")

    # 按目标标签过滤
    shadow_train_datasets_filtered = [filter_by_label(d, target_label) for d in shadow_train_datasets]
    shadow_holdout_datasets_filtered = [filter_by_label(d, target_label) for d in shadow_holdout_datasets]

    print(f"\n[INFO] Filtered dataset sizes for label {target_label}:")
    for idx, (train_ds, holdout_ds) in enumerate(zip(shadow_train_datasets_filtered, shadow_holdout_datasets_filtered)):
        print(f"  Client {idx}: Train={len(train_ds)}, Holdout={len(holdout_ds)}")

    # 创建DataLoader (不再平衡数据量)
    shadow_train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in shadow_train_datasets_filtered if len(ds) > 0]

    # 🔑 关键改动：合并所有客户端的holdout数据为一个共享池
    from torch.utils.data import ConcatDataset
    all_holdout_data = [ds for ds in shadow_holdout_datasets_filtered if len(ds) > 0]
    if len(all_holdout_data) > 0:
        combined_holdout_dataset = ConcatDataset(all_holdout_data)
        shared_holdout_loader = DataLoader(combined_holdout_dataset, batch_size=batch_size, shuffle=False)
        print(f"\n[INFO] Combined holdout dataset size: {len(combined_holdout_dataset)}")
    else:
        print(f"[SKIP] No holdout data for label {target_label}")
        return

    X_grad_conv1, X_grad_conv2, X_grad_fc1, X_grad_fc, X_softmax, y_inout = [], [], [], [], [], []

    # 遍历每个客户端模型，使用其训练数据(label=1) + 共享的holdout数据(label=0)
    for client_idx, (model_file, train_loader) in enumerate(zip(shadow_client_files[:len(shadow_train_loaders)], shadow_train_loaders)):
        if len(train_loader) == 0:
            print(f"[DEBUG] Empty train dataset for client {client_idx}. Skipping.")
            continue

        print(f"\n[INFO] Processing Client {client_idx}...")
        shadow_model.load_state_dict(torch.load(model_file, map_location=device))
        shadow_model.to(device)

        # 1. 处理该客户端的训练数据 (成员数据, label=1)
        outputs, labels, head_grads, feat_grads = get_model_outputs_labels_and_grads(shadow_model, train_loader, device)
        g1, g2, g3, g4, softmax = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        X_grad_conv1.append(g1)
        X_grad_conv2.append(g2)
        X_grad_fc1.append(g3)
        X_grad_fc.append(g4)
        X_softmax.append(softmax)
        y_inout.append(torch.full((g1.size(0),), 1))  # 成员数据
        print(f"  Train data: {g1.size(0)} samples (label=1)")

        # 2. 处理共享的holdout数据 (非成员数据, label=0)
        outputs, labels, head_grads, feat_grads = get_model_outputs_labels_and_grads(shadow_model, shared_holdout_loader, device)
        g1, g2, g3, g4, softmax = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        X_grad_conv1.append(g1)
        X_grad_conv2.append(g2)
        X_grad_fc1.append(g3)
        X_grad_fc.append(g4)
        X_softmax.append(softmax)
        y_inout.append(torch.full((g1.size(0),), 0))  # 非成员数据
        print(f"  Holdout data: {g1.size(0)} samples (label=0)")



    if len(X_grad_conv1) == 0:
        print(f"[SKIP] No data collected for label {target_label}.")
        return

    # 拼接张量
    X_grad_conv1 = torch.cat(X_grad_conv1, dim=0)
    X_grad_conv2 = torch.cat(X_grad_conv2, dim=0)
    X_grad_fc1 = torch.cat(X_grad_fc1, dim=0)
    X_grad_fc = torch.cat(X_grad_fc, dim=0)
    X_softmax = torch.cat(X_softmax, dim=0)
    y_inout = torch.cat(y_inout, dim=0)

    # 🔍 统一维度检查（debug friendly）
    lens = [X_grad_conv1.shape[0], X_grad_conv2.shape[0], X_grad_fc1.shape[0],
            X_grad_fc.shape[0], X_softmax.shape[0], y_inout.shape[0]]

    if len(set(lens)) != 1:
        print(f"[ERROR] Shape mismatch at label {target_label}:")
        print(f"  conv1:    {X_grad_conv1.shape}")
        print(f"  conv2:    {X_grad_conv2.shape}")
        print(f"  fc1:      {X_grad_fc1.shape}")
        print(f"  fc:       {X_grad_fc.shape}")
        print(f"  softmax:  {X_softmax.shape}")
        print(f"  y_inout:  {y_inout.shape}")
        return  # ⛔️ 停止该 label 的训练

    dataset = TensorDataset(X_grad_conv1, X_grad_conv2, X_grad_fc1, X_grad_fc, X_softmax, y_inout)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[Training] Attack model for label {target_label}")

    # Early stopping 参数
    best_loss = float('inf')
    best_f_score = 0.0
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        attack_model.train()
        total_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0  # For F-score calculation

        for g1, g2, g3, g4, softmax, labels in loader:
            g1, g2, g3, g4, softmax = g1.to(device), g2.to(device), g3.to(device), g4.to(device), softmax.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            preds = attack_model(g1, g2, g3, g4, softmax)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(attack_model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() * labels.size(0)

            # Calculate confusion matrix elements for F-score
            # 注意：preds是logits，需要通过sigmoid转为概率
            pred_probs = torch.sigmoid(preds)
            pred_binary = (pred_probs > 0.5).float()
            tp += ((pred_binary == 1) & (labels == 1)).sum().item()
            fp += ((pred_binary == 1) & (labels == 0)).sum().item()
            tn += ((pred_binary == 0) & (labels == 0)).sum().item()
            fn += ((pred_binary == 0) & (labels == 1)).sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(loader.dataset)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} F-score: {f_score:.4f} Acc: {acc:.4f}")

        # 学习率调度
        scheduler.step(avg_loss)

        # 保存最佳模型
        if f_score > best_f_score:
            best_f_score = f_score
            best_loss = avg_loss
            torch.save(attack_model.state_dict(), f"attack_model{target_label}.pth")
            patience_counter = 0
            print(f"  → Best model saved! (F-score: {best_f_score:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"[Early Stop] No improvement for {early_stop_patience} epochs. Stopping training.")
            break

    print(f"[Final] Best F-score: {best_f_score:.4f}, Best Loss: {best_loss:.4f}")
