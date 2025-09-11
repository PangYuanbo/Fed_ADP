# =========================
# whitebox_mia_pipeline.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from data_utils import read_client_data, filter_by_label


# ============================
# 2. 攻击训练接口
# ============================
def train_attack_model(attack_model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        attack_model.train()
        total_loss, correct = 0, 0
        for g1, g2, g3, g4, softmax, labels in dataloader:
            g1, g2, g3, g4, softmax = g1.to(device), g2.to(device), g3.to(device), g4.to(device), softmax.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            preds = attack_model(g1, g2, g3, g4, softmax)
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
):
    # ========= 在 whitebox_membership_inference_attack_pipeline 里，先放一个工具函数 =========
    def collect_scores_labels(loader):
        """收集攻击模型的概率分数(scores)与标签(labels)"""
        attack_model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for a, b, c, d, sm, lbl in loader:
                a, b, c, d, sm = [t.to(DEVICE) for t in (a, b, c, d, sm)]
                lbl = lbl.to(DEVICE).long()
                scores = attack_model(a, b, c, d, sm).squeeze(1)  # 假定 attack_model 已含 sigmoid
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
            for a, b, c, d, sm, lbl in loader:
                a, b, c, d, sm = [t.to(DEVICE) for t in (a, b, c, d, sm)]
                lbl = lbl.to(DEVICE).float().unsqueeze(1)
                out = attack_model(a, b, c, d, sm)
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

    for model_file, train_loader, holdout_loader in zip(client_files, train_loaders, holdout_loaders):
        # 跳过空数据
        if len(train_loader.dataset) == 0 or len(holdout_loader.dataset) == 0:
            print(f"[DEBUG] Empty dataset for client {model_file}. Skipping.")
            continue

        # 加载目标模型
        target_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        target_model.to(DEVICE)

        # 5.1.0) 评估分类准确率
        train_acc   = eval_classification_acc(target_model, train_loader)
        holdout_acc = eval_classification_acc(target_model, holdout_loader)

        # 5.2) 构造成员 / 非成员的 attack 数据
        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(
            target_model, train_loader, DEVICE
        )
        g1, g2, g3, g4, sm_in = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        data_in = TensorDataset(g1, g2, g3, g4, sm_in, torch.ones(g1.size(0)).long())

        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(
            target_model, holdout_loader, DEVICE
        )
        g1, g2, g3, g4, sm_out = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        data_out = TensorDataset(g1, g2, g3, g4, sm_out, torch.zeros(g1.size(0)).long())

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

        # 6) 汇总结果（建议别把 train_acc 与 holdout_acc 平均；分别保留）
        client_results.append({
            'client': model_file,
            'train_acc': train_acc,
            'holdout_acc': holdout_acc,
            'f_score': attack_fscore,
            'tpr': tps_recall,
            'fpr': fps_err,
        })
        print(
            f"[Client: {model_file}] "
            f"Train Acc: {train_acc:.4f} | Holdout Acc: {holdout_acc:.4f} | "
            f"Attack F1: {attack_fscore:.4f} | TPR: {tps_recall:.4f} | FPR: {fps_err:.4f}"
        )

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
