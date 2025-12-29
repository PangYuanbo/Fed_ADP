# =========================
# whitebox_mia_pipeline.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import copy
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from utils.attack_feature_config import DEFAULT_ATTACK_FEATURES, FEATURE_SPECS
from Membership_Inference_Attack.defense_trainer import train_defense_layers
from data_utils import read_client_data, filter_by_label


def _get_attack_feature_order(attack_model):
    return getattr(attack_model, 'enabled_features', DEFAULT_ATTACK_FEATURES)


def _build_attack_dataset(feature_map, label_value, attack_features):
    tensors = [feature_map[name] for name in attack_features]
    if not tensors:
        raise ValueError("No attack features were provided.")
    count = tensors[0].size(0)
    label_tensor = torch.full((count,), label_value, dtype=torch.long)
    return TensorDataset(*tensors, label_tensor)


def _batch_to_attack_inputs(batch, attack_features, device):
    *feature_batches, labels = batch
    inputs = {
        name: tensor.to(device)
        for name, tensor in zip(attack_features, feature_batches)
    }
    return inputs, labels.to(device)


# ============================
# 2. 攻击训练接口
# ============================
def train_attack_model(attack_model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    attack_features = _get_attack_feature_order(attack_model)
    for epoch in range(epochs):
        attack_model.train()
        total_loss, correct = 0, 0
        for batch in dataloader:
            feature_inputs, labels = _batch_to_attack_inputs(batch, attack_features, device)
            labels = labels.float().unsqueeze(1)
            preds = attack_model(feature_inputs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += ((preds > 0.5).int() == labels.int()).sum().item()
        acc = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# ============================
# 3. 白盒攻击评估接口
# ============================
def whitebox_membership_inference_attack_pipeline(
    client_files,
    target_model,
    target_label,
    BATCH_SIZE,
    DEVICE,
    attack_model,
    num_clients=5,
    alpha=1,
    heatmap_cfg=None,
    member_alignment_cfg=None,
    defense_cfg=None,
):
    attack_features = _get_attack_feature_order(attack_model)
    # ========= 在 whitebox_membership_inference_attack_pipeline 里，先放一个工具函数 =========
    def collect_scores_labels(loader):
        """收集攻击模型的概率分数(scores)与标签(labels)"""
        attack_model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                feature_inputs, lbl = _batch_to_attack_inputs(batch, attack_features, DEVICE)
                lbl = lbl.long()
                scores = attack_model(feature_inputs).squeeze(1)  # 假定 attack_model 已含 sigmoid
                all_scores.append(scores.cpu())
                all_labels.append(lbl.cpu())
        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        return scores, labels

    def metrics_from_scores(scores, labels, thr=0.5):
        """基于混合集(scores, labels)计算 Attack F1(正类), TPR(=Recall+), FPR, Acc"""
        import numpy as np
        yhat = (scores > thr).astype(int)
        y = labels.astype(int)

        # 混淆矩阵
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_pos = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / max(1, (tp + fp + tn + fn))
        tpr = recall  # 对正类的召回
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # 负类被误判为正类的比例

        return {
            'precision_pos': precision,
            'recall_pos': recall,
            'f1_pos': f1_pos,
            'acc': acc,
            'tpr': tpr,
            'fpr': fpr,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        }

    """
    白盒 MIA 攻击评估流程：利用目标模型在训练集与 holdout 集上的梯度 + softmax 构造攻击样本，
    返回目标模型在这些子集上的分类准确率，以及对它们的攻击 F-score。
    """
    # 1.0) 读入原始数据
    train_datasets = read_client_data(is_train=True, is_shadow=False,
                                      num_clients=num_clients, alpha=alpha)
    holdout_datasets = read_client_data(is_train=False, is_shadow=False,
                                        num_clients=num_clients, alpha=alpha)

    # 2) 仅保留指定 label 的子集
    train_datasets_filtered = [filter_by_label(ds, target_label) for ds in train_datasets]
    holdout_datasets_filtered = [filter_by_label(ds, target_label) for ds in holdout_datasets]

    # 3) 确保 train 与 holdout 样本数一致
    # train_datasets_filtered = [
    #     Subset(train_ds, list(range(min(len(train_ds), len(holdout_ds)))))
    #     for train_ds, holdout_ds in zip(train_datasets_filtered, holdout_datasets_filtered)
    # ]
    # holdout_datasets_filtered = [
    #     Subset(holdout_ds, list(range(min(len(train_ds), len(holdout_ds)))))
    #     for train_ds, holdout_ds in zip(train_datasets_filtered, holdout_datasets_filtered)
    # ]

    # 4) 构造 DataLoaders
    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
                     for ds in train_datasets_filtered]
    holdout_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
                       for ds in holdout_datasets_filtered]

    def eval_classification_acc(model, loader):
        """返回 model 在 loader 上的分类准确率"""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def eval_attack_fscore(loader):
        """
        返回 membership inference 的 F-score
        """
        attack_model.eval()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                feature_inputs, lbl = _batch_to_attack_inputs(batch, attack_features, DEVICE)
                lbl = lbl.float().unsqueeze(1)
                out = attack_model(feature_inputs)
                preds_list.append((out > 0.5).float().cpu())
                labels_list.append(lbl.cpu())
        p = torch.cat(preds_list).squeeze()  # shape [N]
        l = torch.cat(labels_list).squeeze() # shape [N]

        # 计算混淆矩阵元素
        tp = int(((p == 1) & (l == 1)).sum().item())
        fp = int(((p == 1) & (l == 0)).sum().item())
        fn = int(((p == 0) & (l == 1)).sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score   = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f_score

    client_results = []
    conv1_weight_shape = None
    try:
        conv1_weight_shape = tuple(target_model.feature_extractor.conv1[0].weight.shape)
    except Exception:
        pass

    for client_idx, (model_file, train_loader, holdout_loader) in enumerate(
        zip(client_files, train_loaders, holdout_loaders)
    ):
        # 跳过空数据
        if len(train_loader.dataset) == 0 or len(holdout_loader.dataset) == 0:
            print(f"[DEBUG] Empty dataset for client {model_file}. Skipping.")
            continue

        # 加载目标模型
        client_model = copy.deepcopy(target_model)
        client_model.load_state_dict(torch.load(model_file, map_location=DEVICE), strict=False)
        client_model.to(DEVICE)
        heatmap_note = None

        if defense_cfg and defense_cfg.get('enabled'):
            layer_opts = defense_cfg.get('layer_options', {})
            client_model.feature_extractor.install_defense_layers(**layer_opts)
            train_defense_layers(
                client_model,
                attack_model,
                train_loader,
                DEVICE,
                attack_features,
                defense_cfg,
                eval_loader=holdout_loader,
            )

        # 5.1.0) 评估分类准确率
        train_acc   = eval_classification_acc(client_model, train_loader)
        holdout_acc = eval_classification_acc(client_model, holdout_loader)

        # 5.2) 构造成员 / 非成员的 attack 数据
        member_outputs, _, member_head_grads, member_feat_grads = get_model_outputs_labels_and_grads(
            client_model, train_loader, DEVICE
        )
        member_target_heatmap = compute_layer_channel0_heatmap(member_feat_grads, layer_name='conv1.0.weight')

        holdout_outputs, _, holdout_head_grads, holdout_feat_grads = get_model_outputs_labels_and_grads(
            client_model, holdout_loader, DEVICE
        )
        non_member_target_heatmap = compute_layer_channel0_heatmap(holdout_feat_grads, layer_name='conv1.0.weight')

        apply_member_gradient_alignment(
            member_head_grads,
            member_feat_grads,
            holdout_head_grads,
            holdout_feat_grads,
            attack_features,
            member_alignment_cfg,
        )

        member_features = prepare_attack_model_inputs(
            member_outputs,
            member_head_grads,
            member_feat_grads,
            enabled_features=attack_features,
            return_dict=True,
        )
        data_in = _build_attack_dataset(member_features, 1, attack_features)

        non_member_features = prepare_attack_model_inputs(
            holdout_outputs,
            holdout_head_grads,
            holdout_feat_grads,
            enabled_features=attack_features,
            return_dict=True,
        )
        data_out = _build_attack_dataset(non_member_features, 0, attack_features)

        data_all = DataLoader(ConcatDataset([data_in, data_out]),
                              batch_size=BATCH_SIZE, shuffle=True)
        data_tps = DataLoader(data_in, batch_size=BATCH_SIZE, shuffle=True)
        data_fps = DataLoader(data_out, batch_size=BATCH_SIZE, shuffle=True)

        # 5.3) 评估攻击模型（在混合集 data_all 上一次性计算）
        scores_all, labels_all = collect_scores_labels(data_all)
        attack_metrics = metrics_from_scores(scores_all, labels_all, thr=0.5)

        attack_fscore = float(attack_metrics['f1_pos'])  # Attack F1（正类）
        tps_recall = float(attack_metrics['tpr'])  # TPR：成员识别率
        fps_err = float(attack_metrics['fpr'])  # FPR：非成员误报率

        defense_active = defense_cfg and defense_cfg.get('enabled')
        if defense_active:
            heatmap_note = "skipped (defense enabled)"
        if heatmap_cfg and not defense_active:
            enabled_clients = heatmap_cfg.get('enabled_clients')
            should_generate_heatmap = enabled_clients is None or client_idx in enabled_clients
            if should_generate_heatmap:
                save_root = Path(heatmap_cfg.get('save_root', 'Membership_Inference_Attack/view_heatmaps'))
                prefix_template = heatmap_cfg.get('name_template', 'client{client}_label{label}')
                prefix_name = prefix_template.format(client=client_idx, label=target_label)
                generate_mia_saliency_visuals(
                    member_loader=data_tps,
                    non_member_loader=data_fps,
                    attack_model=attack_model,
                    device=DEVICE,
                    cfg=heatmap_cfg,
                    client_idx=client_idx,
                    target_label=target_label,
                    attack_features=attack_features,
                    target_member_heatmap=member_target_heatmap,
                    target_non_member_heatmap=non_member_target_heatmap,
                    conv1_weight_shape=conv1_weight_shape,
                )
                heatmap_note = f"{save_root / f'label_{target_label}'} (prefix '{prefix_name}')"

        # 6) 汇总结果（建议别把 train_acc 与 holdout_acc 平均；分别保留）
        client_results.append({
            'client': model_file,
            'train_acc': train_acc,
            'holdout_acc': holdout_acc,
            'f_score': attack_fscore,
            'tpr': tps_recall,
            'fpr': fps_err,
            'heatmap_info': heatmap_note,
        })
    return client_results


# ============================
# 4. 可视化模块
# ============================


def plot_attack_results_per_client(results_by_part, part_names,
                                   save_root='view'):
    import matplotlib
    matplotlib.use('Agg')  # 服务器 / 无 GUI
    import matplotlib.pyplot as plt
    from pathlib import Path
    """
    results_by_part: list (part) -> list (client) -> list (label) -> dict
    每个 dict 现在包含:
      - 'train_acc'
      - 'holdout_acc'
      - 'f_score'  # Attack F-score
      - 'tpr'      # True Positive Rate (Recall for members)
      - 'fpr'      # False Positive Rate
    """
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    for part_idx, (part_results, part_name) in enumerate(zip(results_by_part, part_names)):
        for client_idx, label_results in enumerate(part_results):

            # --------- 收集 10 个 label 的数值 ---------
            train_vals  = [r['train_acc']     for r in label_results]
            fscore_vals = [r['f_score']       for r in label_results]
            tpr_vals    = [r['tpr']           for r in label_results]
            fpr_vals    = [r['fpr']           for r in label_results]
            labels_x    = list(range(len(label_results)))  # 0…9

            # --------- 打印 train_acc 到控制台 ---------
            print(f"[Plot] {part_name} | Client {client_idx} | Train Acc per label: {train_vals}")

            # --------- 一张图绘四条折线 ---------
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(labels_x, train_vals,  marker='d', linestyle=':',  label='Train Acc')
            ax.plot(labels_x, fscore_vals, marker='o', linestyle='-',  label='Attack F-score')
            ax.plot(labels_x, tpr_vals,    marker='s', linestyle='--', label='TPR (Member Recall)')
            ax.plot(labels_x, fpr_vals,    marker='^', linestyle='-.', label='FPR')

            ax.set_xticks(labels_x)
            ax.set_xlabel("Target Label")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score / Rate")
            ax.set_title(f"{part_name}  |  Client {client_idx}")
            ax.grid(True)
            ax.legend(loc='best')
            fig.tight_layout()

            out = save_root / f"{part_name}_c{client_idx}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)

def plot_attack_results_last_vs_fscore(results_by_part, part_names):
    """
    For each part:
       • Figure 1.0.0: average Train Acc & F-score of all clients except the last one
       • Figure 2:     Train Acc & F-score of the last client alone

    Parameters
    ----------
    results_by_part : list(part) -> list(client) -> list(label) -> dict (must contain 'train_acc' and 'f_score')
    part_names      : list(str)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    for part_results, part_name in zip(results_by_part, part_names):
        num_clients = len(part_results)
        if num_clients == 0:
            print(f"[WARN] '{part_name}' 无客户端数据，跳过。")
            continue
        if num_clients == 1:
            print(f"[WARN] '{part_name}' 只有 1.0 个客户端，Avg-Others 图将省略。")

        # ---------- 1.0) 计算“其余客户端平均” ----------
        if num_clients > 1:
            num_labels = len(part_results[0])
            train_sum  = np.zeros(num_labels)
            fscore_sum = np.zeros(num_labels)

            for client_metrics in part_results[:-1]:  # 不含最后一个
                for lbl, m in enumerate(client_metrics):
                    train_sum[lbl]  += m['train_acc']
                    fscore_sum[lbl] += m['f_score']

            denom      = num_clients - 1
            train_avg  = train_sum  / denom
            fscore_avg = fscore_sum / denom

            # 画 Avg-Others 图
            labels_x = list(range(num_labels))
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(labels_x, train_avg,  marker='d', linestyle=':',  label='Avg Train Acc')
            ax1.plot(labels_x, fscore_avg, marker='o', linestyle='-',  label='Avg F-score')

            ax1.set_xticks(labels_x)
            ax1.set_xlabel("Target Label")
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Accuracy / F-score")
            ax1.set_title(f"{part_name} | Avg Across Clients (except last)")
            ax1.grid(True)
            ax1.legend(loc='best')
            fig1.tight_layout()

        # ---------- 2) 计算“最后一个客户端” ----------
        last_client = part_results[-1]
        num_labels  = len(last_client)
        last_train  = np.array([m['train_acc'] for m in last_client])
        last_fscore = np.array([m['f_score']    for m in last_client])

        labels_x = list(range(num_labels))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(labels_x, last_train,  marker='d', linestyle=':',  label='Train Acc')
        ax2.plot(labels_x, last_fscore, marker='o', linestyle='-',  label='F-score')

        ax2.set_xticks(labels_x)
        ax2.set_xlabel("Target Label")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Accuracy / F-score")
        ax2.set_title(f"{part_name} | Last Client (index {num_clients-1})")
        ax2.grid(True)
        ax2.legend(loc='best')
        fig2.tight_layout()

        # ---------- 显示 ----------
        plt.show()


def compute_attack_saliency(loader, attack_model, device, attack_features, target_value, max_batches=None):
    attack_model.eval()
    saliency = {}
    sample_total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        *feature_batches, _ = batch
        tensors = []
        for name, tensor in zip(attack_features, feature_batches):
            tensor = tensor.to(device)
            tensor.requires_grad_(True)
            tensors.append((name, tensor))

        attack_model.zero_grad()
        logits = attack_model({name: tensor for name, tensor in tensors})
        target = torch.full_like(logits, fill_value=target_value)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        for name, tensor in tensors:
            if tensor.grad is None:
                continue
            grad_map = tensor.grad.detach().abs().sum(dim=0, keepdim=True).cpu()
            saliency[name] = saliency.get(name, 0) + grad_map

        sample_total += tensors[0][1].size(0)

        for _, tensor in tensors:
            tensor.detach_()

    if sample_total == 0:
        return {}

    for key in saliency:
        saliency[key] = saliency[key] / sample_total

    return saliency


def compute_attack_input_statistics(loader, attack_features, max_batches=None):
    stats = {}
    sample_total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        *feature_batches, _ = batch
        tensors = list(zip(attack_features, feature_batches))

        batch_size = feature_batches[0].size(0)
        sample_total += batch_size

        for name, tensor in tensors:
            stats[name] = stats.get(name, 0) + tensor.abs().sum(dim=0, keepdim=True)

    if sample_total == 0:
        return {}

    for key in stats:
        stats[key] = stats[key] / sample_total

    return stats


def _grad_list_abs_mean(grad_list):
    values = []
    for g in grad_list:
        if g is None:
            continue
        if isinstance(g, torch.Tensor):
            arr = g.detach()
        else:
            arr = torch.tensor(g)
        if arr.numel() == 0:
            continue
        values.append(arr.abs().mean().item())
    if not values:
        return 0.0
    return float(np.mean(values))


def apply_member_gradient_alignment(member_head_grads,
                                    member_feat_grads,
                                    ref_head_grads,
                                    ref_feat_grads,
                                    attack_features,
                                    cfg):
    if not cfg or not cfg.get('enabled'):
        return
    epsilon = cfg.get('epsilon', 1e-8)
    min_scale = cfg.get('min_scale', 1e-2)
    max_scale = cfg.get('max_scale', 1e2)
    verbose = cfg.get('verbose', False)

    for feature_name in attack_features:
        spec = FEATURE_SPECS.get(feature_name)
        if spec is None:
            continue
        if spec['grad_source'] == 'softmax':
            continue
        src_member = member_feat_grads if spec['grad_source'] == 'feature' else member_head_grads
        src_ref = ref_feat_grads if spec['grad_source'] == 'feature' else ref_head_grads
        key = spec['grad_key']
        member_list = src_member.get(key)
        ref_list = src_ref.get(key)
        if not member_list or not ref_list:
            continue

        member_mean = _grad_list_abs_mean(member_list)
        ref_mean = _grad_list_abs_mean(ref_list)
        if member_mean <= 0:
            continue
        scale = ref_mean / max(member_mean, epsilon) if ref_mean > 0 else 1.0
        scale = max(min_scale, min(max_scale, scale))

        if abs(scale - 1.0) < 1e-6:
            continue

        for idx, grad in enumerate(member_list):
            if grad is None:
                continue
            member_list[idx] = grad * scale

        if verbose:
            print(f"[GradAlign] Feature={feature_name} mean_member={member_mean:.4e} "
                  f"mean_ref={ref_mean:.4e} scale={scale:.4f}")


def compute_layer_channel0_heatmap(grad_dict, layer_name):
    grads = grad_dict.get(layer_name)
    if not grads:
        return None
    tensors = []
    for g in grads:
        if g is None:
            continue
        if isinstance(g, torch.Tensor):
            tensors.append(g.detach().cpu())
        else:
            tensors.append(torch.tensor(g))
    if not tensors:
        return None
    stacked = torch.stack(tensors, dim=0).float()
    heat = stacked.abs().mean(dim=0)  # [out, in, kH, kW]
    if heat.dim() != 4 or heat.size(0) == 0:
        return None
    return heat.mean(dim=1).cpu()


def extract_channel0_from_saliency(saliency_tensor, conv_shape):
    if saliency_tensor is None or conv_shape is None:
        return None
    out_c, in_c, k_h, k_w = conv_shape
    required = out_c * in_c * k_h * k_w
    arr = saliency_tensor.detach().cpu().squeeze().flatten()
    if arr.numel() < required:
        pad = torch.zeros(required - arr.numel())
        arr = torch.cat([arr, pad])
    else:
        arr = arr[:required]
    conv = arr.view(out_c, in_c, k_h, k_w).abs()
    if conv.size(0) == 0:
        return None
    return conv.mean(dim=1).cpu()


def log_saliency_layer_means(member_saliency, non_member_saliency, member_label='member_target', non_member_label='non_member_target'):
    def mean_or_none(tensor):
        if tensor is None:
            return None
        arr = tensor.detach().cpu()
        if arr.numel() == 0:
            return None
        return float(arr.mean().item())

    layer_names = sorted(set(member_saliency.keys()) | set(non_member_saliency.keys()))
    for name in layer_names:
        mem_avg = mean_or_none(member_saliency.get(name))
        non_avg = mean_or_none(non_member_saliency.get(name))
        print(
            f"[Heatmap-Avg] Layer={name} {member_label}={mem_avg if mem_avg is not None else 'N/A'} "
            f"{non_member_label}={non_avg if non_avg is not None else 'N/A'}"
        )


def save_saliency_heatmaps(member_saliency,
                          non_member_saliency,
                          diff_saliency,
                          prefix,
                          save_root,
                          extra_heatmaps=None,
                          target_label=None):
    save_root = Path(save_root)
    # 如果提供了 target_label，创建按 label 分类的子文件夹
    if target_label is not None:
        save_root = save_root / f"label_{target_label}"
    save_root.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def tensor_to_array(tensor):
        arr = tensor.squeeze().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def plot_single(name, tensor):
        arr = tensor_to_array(tensor)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(arr, cmap='magma')
        ax.set_title(f"{prefix} | {name}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = save_root / f"{prefix}_{name}.png"
        fig.savefig(out_path, dpi=200)
        print(f"[Heatmap] Saved {out_path}")
        plt.close(fig)

    for key, tensor in member_saliency.items():
        plot_single(f"{key}_member", tensor)
    for key, tensor in non_member_saliency.items():
        plot_single(f"{key}_non_member", tensor)
    for key, tensor in diff_saliency.items():
        plot_single(f"{key}_diff", tensor)

    if extra_heatmaps:
        for name, tensor in extra_heatmaps.items():
            if tensor is not None:
                plot_single(name, tensor)

    print(f"[Heatmap] Completed saliency visualization set '{prefix}' under {save_root}")


def generate_mia_saliency_visuals(member_loader,
                                 non_member_loader,
                                 attack_model,
                                 device,
                                 cfg,
                                 client_idx,
                                 target_label,
                                 attack_features,
                                 target_member_heatmap=None,
                                 target_non_member_heatmap=None,
                                 conv1_weight_shape=None):
    max_batches = cfg.get('max_batches')
    save_root = cfg.get('save_root', 'Membership_Inference_Attack/view_heatmaps')
    prefix_template = cfg.get('name_template', 'client{client}_label{label}')
    prefix = prefix_template.format(client=client_idx, label=target_label)

    member_input_stats = compute_attack_input_statistics(
        member_loader,
        attack_features,
        max_batches=max_batches,
    )
    non_member_input_stats = {}
    if non_member_loader is not None:
        non_member_input_stats = compute_attack_input_statistics(
            non_member_loader,
            attack_features,
            max_batches=max_batches,
        )

    log_saliency_layer_means(
        member_input_stats,
        non_member_input_stats,
        member_label='member_input_magnitude',
        non_member_label='non_member_input_magnitude',
    )
    input_diff = {}
    for key, tensor in member_input_stats.items():
        other = non_member_input_stats.get(key)
        if other is None:
            input_diff[key] = tensor.clone()
        else:
            input_diff[key] = (tensor - other).abs()

    save_saliency_heatmaps(
        member_input_stats,
        non_member_input_stats,
        input_diff,
        f"{prefix}_raw_inputs",
        save_root,
        extra_heatmaps=None,
        target_label=target_label,
    )

    member_actual = compute_attack_saliency(member_loader, attack_model, device, attack_features, target_value=1, max_batches=max_batches)
    member_flipped = compute_attack_saliency(member_loader, attack_model, device, attack_features, target_value=0, max_batches=max_batches)
    log_saliency_layer_means(member_actual, member_flipped, member_label='member_actual', non_member_label='member_flipped')

    diff_saliency = {}
    for key, tensor in member_actual.items():
        other = member_flipped.get(key)
        if other is None:
            diff_saliency[key] = tensor.clone()
        else:
            diff_saliency[key] = (tensor - other).abs()

    extra_maps = {}

    def add_aggregated_map(name, tensor):
        if tensor is None:
            return
        agg = tensor.mean(dim=0, keepdim=True)
        extra_maps[name] = agg

    add_aggregated_map("target_conv1_member", target_member_heatmap)
    add_aggregated_map("target_conv1_non_member", target_non_member_heatmap)

    # NOTE: Disable per-channel attack conv1 export for now; re-enable when needed.
    # if conv1_weight_shape is not None:
    #     member_channels = extract_channel0_from_saliency(member_actual.get('conv1'), conv1_weight_shape)
    #     flipped_channels = extract_channel0_from_saliency(member_flipped.get('conv1'), conv1_weight_shape)
    #     add_aggregated_map("attack_conv1_member", member_channels)
    #     add_aggregated_map("attack_conv1_non_member", flipped_channels)

    save_saliency_heatmaps(member_actual,
                           member_flipped,
                           diff_saliency,
                           prefix,
                           save_root,
                           extra_heatmaps=extra_maps,
                           target_label=target_label)

    if non_member_loader is not None:
        holdout_actual = compute_attack_saliency(non_member_loader, attack_model, device, attack_features, target_value=0, max_batches=max_batches)
        holdout_flipped = compute_attack_saliency(non_member_loader, attack_model, device, attack_features, target_value=1, max_batches=max_batches)
        log_saliency_layer_means(holdout_actual, holdout_flipped, member_label='holdout_actual', non_member_label='holdout_flipped')
        diff_holdout = {}
        for key, tensor in holdout_actual.items():
            other = holdout_flipped.get(key)
            if other is None:
                diff_holdout[key] = tensor.clone()
            else:
                diff_holdout[key] = (tensor - other).abs()
        save_saliency_heatmaps(holdout_actual,
                               holdout_flipped,
                               diff_holdout,
                               f"{prefix}_holdout_flip",
                               save_root,
                               extra_heatmaps=None,
                               target_label=target_label)

        dataset_diff = {}
        for key, tensor in member_actual.items():
            other = holdout_actual.get(key)
            if other is None:
                dataset_diff[key] = tensor.clone()
            else:
                dataset_diff[key] = (tensor - other).abs()
        log_saliency_layer_means(member_actual, holdout_actual, member_label='member_actual', non_member_label='holdout_actual')
        save_saliency_heatmaps(member_actual,
                               holdout_actual,
                               dataset_diff,
                               f"{prefix}_dataset",
                               save_root,
                               extra_heatmaps=None,
                               target_label=target_label)
