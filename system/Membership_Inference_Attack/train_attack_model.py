# ============================
# train_attack_model.py
# ============================

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from model import GradientMIA
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from data_utils import read_client_data, filter_by_label


def train_attack_model(shadow_model, shadow_client_files, target_label, batch_size, device, epochs, lr, num_clients,alpha):
    attack_model = GradientMIA().to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(len(shadow_client_files), len(shadow_client_files[0]))

    shadow_train_datasets = read_client_data(is_train=True, is_shadow=True,num_clients=num_clients,alpha=alpha)
    shadow_holdout_datasets = read_client_data(is_train=False, is_shadow=True,num_clients=num_clients,alpha=alpha)
    for idx, (train_ds, holdout_ds) in enumerate(zip(shadow_train_datasets, shadow_holdout_datasets)):
        print(f"[DEBUG] Client {idx}, Train size: {len(train_ds)}, Holdout size: {len(holdout_ds)}")


    shadow_train_datasets_filtered = [filter_by_label(d, target_label) for d in shadow_train_datasets]
    shadow_holdout_datasets_filtered = [filter_by_label(d, target_label) for d in shadow_holdout_datasets]

    # 这一段替换你原来的 shadow_train_datasets_filtered = [...]
    # 它同时生成新的 train 和 holdout 子集，并且两者都裁剪到相同的最小长度 k
    balanced_train = []
    balanced_holdout = []
    for train_ds, holdout_ds in zip(shadow_train_datasets_filtered,
                                    shadow_holdout_datasets_filtered):
        k = min(len(train_ds), len(holdout_ds))
        if k == 0:
            continue
        # 取前 k 个索引（也可随机打乱后选前 k）
        idx = list(range(k))
        balanced_train.append(Subset(train_ds, idx))
        balanced_holdout.append(Subset(holdout_ds, idx))

    shadow_train_datasets_filtered = balanced_train
    shadow_holdout_datasets_filtered = balanced_holdout

    shadow_train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in shadow_train_datasets_filtered]
    shadow_holdout_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in shadow_holdout_datasets_filtered]
    for idx, (train_ds, holdout_ds) in enumerate(zip(shadow_train_datasets_filtered, shadow_holdout_datasets_filtered)):
        print(f"[DEBUG] Client {idx}, Train size: {len(train_ds)}, Holdout size: {len(holdout_ds)}")

    X_grad_conv1, X_grad_conv2, X_grad_fc1, X_grad_fc, X_softmax, y_inout = [], [], [], [], [], []

    for model_file, train_loader, holdout_loader in zip(shadow_client_files, shadow_train_loaders, shadow_holdout_loaders):
        if len(train_loader) == 0 or len(holdout_loader) == 0:
            print(f"[DEBUG] Empty dataset for client {model_file}. Skipping.")
            continue
        shadow_model.load_state_dict(torch.load(model_file, map_location=device))
        shadow_model.to(device)

        for loader, label_flag in [(train_loader, 1), (holdout_loader, 0)]:
            outputs, labels, head_grads, feat_grads = get_model_outputs_labels_and_grads(shadow_model, loader, device)
            g1, g2, g3, g4, softmax = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
            X_grad_conv1.append(g1)
            X_grad_conv2.append(g2)
            X_grad_fc1.append(g3)
            X_grad_fc.append(g4)
            X_softmax.append(softmax)
            y_inout.append(torch.full((g1.size(0),), label_flag))



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
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            
            # Calculate confusion matrix elements for F-score
            pred_binary = (preds > 0.5).float()
            tp += ((pred_binary == 1) & (labels == 1)).sum().item()
            fp += ((pred_binary == 1) & (labels == 0)).sum().item()
            tn += ((pred_binary == 0) & (labels == 0)).sum().item()
            fn += ((pred_binary == 0) & (labels == 1)).sum().item()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(loader.dataset):.4f} F-score: {f_score:.4f} Acc: {acc:.4f}")

    torch.save(attack_model.state_dict(), f"attack_model{target_label}.pth")
