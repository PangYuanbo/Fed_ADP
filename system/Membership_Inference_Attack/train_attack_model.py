# ============================
# train_attack_model.py
# ============================

import os
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.mia_attack_model import GradientMIA
from utils.attack_feature_config import (
    attack_checkpoint_name,
    normalize_attack_features,
)
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from data_utils import read_client_data, filter_by_label


def train_attack_model(shadow_model,
                       shadow_client_files,
                       target_label,
                       batch_size,
                       device,
                       epochs,
                       lr,
                       num_clients,
                       alpha,
                       attack_features=None,
                       checkpoint_dir=".",
                       num_classes=None):
    enabled_features = normalize_attack_features(attack_features)
    if num_classes is None:
        num_classes = getattr(getattr(shadow_model, 'head', None), 'out_features', 10)
    os.makedirs(checkpoint_dir, exist_ok=True)
    primary_checkpoint = os.path.join(checkpoint_dir, attack_checkpoint_name(target_label, enabled_features))
    legacy_checkpoint = os.path.join(checkpoint_dir, attack_checkpoint_name(target_label, enabled_features, include_suffix=False))
    attack_model = GradientMIA(enabled_features=enabled_features, num_classes=num_classes).to(device)
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
        feature_storage = None
        y_inout = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    feature_storage = {name: [] for name in enabled_features}
    y_inout = []

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
        member_features = prepare_attack_model_inputs(
            outputs,
            head_grads,
            feat_grads,
            enabled_features=enabled_features,
            return_dict=True,
        )
        member_count = next(iter(member_features.values())).size(0)
        for name in enabled_features:
            feature_storage[name].append(member_features[name])
        y_inout.append(torch.full((member_count,), 1))
        print(f"  Train data: {member_count} samples (label=1)")

        # 2. 处理共享的holdout数据 (非成员数据, label=0)
        outputs, labels, head_grads, feat_grads = get_model_outputs_labels_and_grads(shadow_model, shared_holdout_loader, device)
        non_member_features = prepare_attack_model_inputs(
            outputs,
            head_grads,
            feat_grads,
            enabled_features=enabled_features,
            return_dict=True,
        )
        non_member_count = next(iter(non_member_features.values())).size(0)
        for name in enabled_features:
            feature_storage[name].append(non_member_features[name])
        y_inout.append(torch.full((non_member_count,), 0))
        print(f"  Holdout data: {non_member_count} samples (label=0)")
        del outputs, labels, head_grads, feat_grads, member_features, non_member_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



    if all(len(v) == 0 for v in feature_storage.values()):
        print(f"[SKIP] No data collected for label {target_label}.")
        feature_storage = None
        y_inout = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    # 拼接张量
    concatenated_features = [torch.cat(feature_storage[name], dim=0) for name in enabled_features]
    y_inout = torch.cat(y_inout, dim=0)

    # 🔍 统一维度检查（debug friendly）
    lens = [tensor.shape[0] for tensor in concatenated_features] + [y_inout.shape[0]]

    if len(set(lens)) != 1:
        print(f"[ERROR] Shape mismatch at label {target_label}:")
        for name, tensor in zip(enabled_features, concatenated_features):
            print(f"  {name}: {tensor.shape}")
        print(f"  y_inout:  {y_inout.shape}")
        return  # ⛔️ 停止该 label 的训练

    dataset = TensorDataset(*concatenated_features, y_inout)
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

        for batch in loader:
            *feature_batches, labels = batch
            feature_inputs = {
                name: tensor.to(device)
                for name, tensor in zip(enabled_features, feature_batches)
            }
            labels = labels.float().unsqueeze(1).to(device)
            preds = attack_model(feature_inputs)
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
            torch.save(attack_model.state_dict(), primary_checkpoint)
            if legacy_checkpoint != primary_checkpoint:
                torch.save(attack_model.state_dict(), legacy_checkpoint)
            patience_counter = 0
            print(f"  → Best model saved! (F-score: {best_f_score:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"[Early Stop] No improvement for {early_stop_patience} epochs. Stopping training.")
            break

    print(f"[Final] Best F-score: {best_f_score:.4f}, Best Loss: {best_loss:.4f}")
    del dataset, loader, concatenated_features, y_inout, feature_storage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
