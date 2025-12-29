import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.mia_attack_model import GradientMIA
from utils.attack_feature_config import attack_checkpoint_name

# ============================
# run_mia_pipeline.py
# ============================
if __name__ == "__main__":
    import torch
    import copy
    from Membership_Inference_Attack.model import FedAvgCNN, LocalModel
    from Membership_Inference_Attack.whitebox_mia_pipeline import  whitebox_membership_inference_attack_pipeline, plot_attack_results_last_vs_fscore
    from Membership_Inference_Attack.train_attack_model import train_attack_model
    from Membership_Inference_Attack.evaluate_client_accuracy import evaluate_all_clients_accuracy
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    EPOCHS = 25
    LR = 1e-3
    NUM_CLASSES = 10
    NUM_CLIENTS = 10
    ALPHA = 1
    TARGET_LABELS = [0, 1, 2, 3, 4, 5]  # 扩展为 label 0~5
    TRAIN_ATTACK_MODELS = True
    ATTACK_FEATURES = ['conv1', 'conv2', 'fc']  # 更换攻击模型分支组合
    HEATMAP_CFG = {
        "save_root": "Membership_Inference_Attack/view_heatmaps",
        "enabled_clients": None,
        "max_batches": 64,
        "name_template": "client{client}_label{label}",
        "include_member_saliency": True,
        "include_non_member_saliency": False,
        "include_diff_saliency": False ,
        "include_target_gradient_maps": True,
        "target_gradient_max_batches": 64,
    }
    MEMBER_GRAD_ALIGNMENT = {
        "enabled": True,
        "epsilon": 1e-8,
        "min_scale": 0.1,
        "max_scale": 10.0,
        "verbose": False,
    }
    DEFENSE_CFG = {
        "enabled": True,
        "epochs": 20,
        "lr": 5e-4,
        "max_batches": 128,
        "lambda_defense": 0.2,
        "lambda_classification": 0.8,
        "layer_options": {
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
            "init_std": 1e-3,
        },
    }
    # 构建目标模型结构（全复制）
    def get_fresh_model():
        base = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(DEVICE)
        head = copy.deepcopy(base.fc)
        base.fc = torch.nn.Identity()
        return LocalModel(base, head)

    # 构造 shadow 模型文件名
    shadow_client_files = [
        # f"shadow_model/{ALPHA}/client_{i}_model_50_C1.0_tau0.1.pt"
                f"shadow_model/{ALPHA}/results_cifar-10-shadow_client{i}_1000_0.0050.pt"
        # f"shadow_model/{ALPHA}/results_client{i}_500.pt"
        for i in range(NUM_CLIENTS)
    ]



    # # #
    # if TRAIN_ATTACK_MODELS:
    #     for target_label in TARGET_LABELS:
    #         print(f"==== Training Attack Model for Label: {target_label} ====")
    #         shadow_model = get_fresh_model()
    #         train_attack_model(
    #             shadow_model,
    #             shadow_client_files,
    #             target_label,
    #             batch_size=BATCH_SIZE,
    #             device=DEVICE,
    #             epochs=EPOCHS,
    #             lr=LR,
    #             num_clients=NUM_CLIENTS,
    #             alpha=ALPHA,
    #             attack_features=ATTACK_FEATURES,
    #         )

    # 构造 target 模型文件名（多个版本用于不同结构对比）
    target_model_names = [""]
    target_client_files = {
        name: [
            # f"dp_model/{ALPHA}/client_{i}_model_50_C1.0_tau0.1.pt"
            f"dp_model/{ALPHA}/results_client{i}_500{name}.pt"
            # f"normal_model/{ALPHA}/results_cifar-10-normal_client{i}_1000_0.0050.pt"
            for i in range(NUM_CLIENTS)
        ] for name in target_model_names
    }

    # 测试每个客户端模型的准确率
    for part_idx, name in enumerate(target_model_names):
        print(f"\n=== Accuracy for Target Model Set: {name} ===")
        accs = evaluate_all_clients_accuracy(
            client_model_files=target_client_files[name],
            get_model_fn=get_fresh_model,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            num_clients=NUM_CLIENTS
        )
        print(f"Average accuracy across {NUM_CLIENTS} clients: {sum(accs) / len(accs):.4f}")

    # Step 2: 执行白盒攻击评估
    all_results_by_part = [[] for _ in range(len(target_model_names))]  # 每个 part 一组结果

    # ---------------------------------------------
    # 0) 预先给三维列表占位： part × client × label
    #    先建空 list，后面逐层 append
    all_results_by_part = [
        [[] for _ in range(NUM_CLIENTS)]  # 每个 client 再存所有 label 的结果
        for _ in target_model_names
    ]
    # ---------------------------------------------
    for target_label in TARGET_LABELS:
        print(f"==== Evaluating Attack on Target Label: {target_label} ====")
        attack_model = GradientMIA(enabled_features=ATTACK_FEATURES).to(DEVICE)
        checkpoint_name = attack_checkpoint_name(target_label, ATTACK_FEATURES)
        fallback_name = attack_checkpoint_name(target_label, ATTACK_FEATURES, include_suffix=False)
        if os.path.exists(checkpoint_name):
            attack_model.load_state_dict(torch.load(checkpoint_name))
        else:
            attack_model.load_state_dict(torch.load(fallback_name))
        attack_model.eval()

        for part_idx, name in enumerate(target_model_names):
            target_model = get_fresh_model()

            # 返回值应该是  list[dict]  ，长度 = NUM_CLIENTS
            client_results = whitebox_membership_inference_attack_pipeline(
                target_client_files[name],
                target_model,
                target_label,
                BATCH_SIZE,
                DEVICE,
                attack_model,
                num_clients=NUM_CLIENTS,
                alpha=ALPHA,
                heatmap_cfg=HEATMAP_CFG,
                member_alignment_cfg=MEMBER_GRAD_ALIGNMENT,
                defense_cfg=DEFENSE_CFG,
            )

            # 按 client 聚合，再按 label 压入同一 client 的列表里
            for c_idx, res in enumerate(client_results):
                all_results_by_part[part_idx][c_idx].append(res)

    # all_results_by_part 结构：
    # part_idx ─┬─ client_idx ─┬─ label_idx ─ dict{'f_score', 'tpr', 'fpr', ...}
    #           │              └─ ...
    #           └─ ...

    # plot_attack_results_per_client(all_results_by_part, target_model_names)
    plot_attack_results_last_vs_fscore(all_results_by_part, target_model_names)

    for part_idx, part_name in enumerate(target_model_names):
        print(f"\n=== Summary for Target Model Set: {part_name} ===")
        for client_idx, label_results in enumerate(all_results_by_part[part_idx]):
            for result in label_results:
                heatmap_note = result.get('heatmap_info')
                heatmap_str = f" | Heatmap: {heatmap_note}" if heatmap_note else ""
                print(
                    f"[Client {client_idx}] Label {TARGET_LABELS[label_results.index(result)]} | "
                    f"Train Acc: {result['train_acc']:.4f} | "
                    f"Holdout Acc: {result['holdout_acc']:.4f} | "
                    f"Attack F1: {result['f_score']:.4f} | "
                    f"TPR: {result['tpr']:.4f} | "
                    f"FPR: {result['fpr']:.4f}"
                    f"{heatmap_str}"
                )
