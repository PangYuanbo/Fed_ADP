# ============================
# mia_attack_wrapper.py
# MIA攻击封装器 - 专门用于联邦学习环境
# ============================

import os
import sys
import torch
import copy
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

# 添加MIA模块路径
mia_path = os.path.join(os.path.dirname(__file__), '..', 'Membership_Inference_Attack')
sys.path.append(mia_path)

# 全局变量标记是否可用
MIA_AVAILABLE = False

try:
    from model import FedAvgCNN, LocalModel, GradientMIA
    from mia_attack_utils import (
        get_model_outputs_labels_and_grads,
        prepare_attack_model_inputs,
        get_model_outputs_labels_and_grads_gpu,     # GPU优化版本（生成器）
        prepare_attack_model_inputs_gpu,             # GPU优化版本（批量）
        prepare_attack_model_inputs_single_gpu       # GPU优化版本（单batch流式）
    )
    from data_utils import read_client_data, filter_by_label
    MIA_AVAILABLE = True
    print(f"[MIA Wrapper] Successfully imported MIA modules from: {mia_path}")
except ImportError as e:
    print(f"[MIA Wrapper] Warning: Could not import MIA modules: {e}")
    print(f"[MIA Wrapper] MIA path: {mia_path}")
    print("[MIA Wrapper] MIA evaluation will be disabled")
    # 创建占位符类避免导入错误
    class FedAvgCNN: pass
    class LocalModel: pass
    class GradientMIA: pass
    def get_model_outputs_labels_and_grads(*args, **kwargs): pass
    def prepare_attack_model_inputs(*args, **kwargs): pass
    def get_model_outputs_labels_and_grads_gpu(*args, **kwargs): pass
    def prepare_attack_model_inputs_gpu(*args, **kwargs): pass
    def prepare_attack_model_inputs_single_gpu(*args, **kwargs): pass
    def read_client_data(*args, **kwargs): pass
    def filter_by_label(*args, **kwargs): pass


class FederatedMIAEvaluator:
    """
    专门用于联邦学习环境的MIA攻击评估器
    """

    def __init__(self,
                 attack_model_dir: str = "Membership_Inference_Attack",
                 num_classes: int = 10,
                 device: str = "auto",
                 batch_size: int = 1,
                 alpha: float = 1.0,
                 max_samples_per_client: int = None,
                 use_global_test: bool = True,
                 use_gpu_optimization: bool = True):  # 🔑 新增GPU优化选项
        """
        初始化联邦学习MIA评估器

        Args:
            attack_model_dir: 攻击模型目录
            num_classes: 类别数量
            device: 计算设备
            batch_size: 批次大小
            alpha: 数据分布参数
            max_samples_per_client: 每个客户端最大样本数（None表示使用全部数据）
            use_global_test: 是否使用全局test数据集作为non-member（默认True）
            use_gpu_optimization: 是否使用GPU优化模式（数据保持在GPU上，适合5090等大显存GPU）
        """
        # 检查MIA模块是否可用
        if not MIA_AVAILABLE:
            raise ImportError("MIA modules are not available. Please check PyTorch installation and MIA module paths.")

        self.attack_model_dir = attack_model_dir
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_samples_per_client = max_samples_per_client
        self.use_global_test = use_global_test
        self.use_gpu_optimization = use_gpu_optimization and self.device.type == 'cuda'  # 只在CUDA设备上启用

        print(f"[MIA Wrapper] Initializing with attack_model_dir: {attack_model_dir}")
        print(f"[MIA Wrapper] Device: {self.device}")
        print(f"[MIA Wrapper] Num classes: {num_classes}")
        print(f"[MIA Wrapper] Use global test data: {use_global_test}")
        print(f"[MIA Wrapper] GPU Optimization: {self.use_gpu_optimization}")  # 显示GPU优化状态

        # 🔑 强制GPU优化警告
        if not self.use_gpu_optimization:
            print("[MIA Wrapper] WARNING: GPU optimization is disabled. This will significantly slow down MIA evaluation!")
            print("[MIA Wrapper] Make sure you are using a CUDA device for best performance.")
        else:
            print("[MIA Wrapper] ✓ GPU optimization enabled - all MIA computations will run on GPU")

        # 加载攻击模型
        self.attack_models = {}
        self._load_attack_models()

        # 结果历史记录
        self.evaluation_history = []

    def _load_attack_models(self):
        """加载预训练的攻击模型"""
        successful_loads = 0

        for label in range(self.num_classes):
            model_path = os.path.join(self.attack_model_dir, f"attack_model{label}.pth")

            if os.path.exists(model_path):
                try:
                    attack_model = GradientMIA().to(self.device)
                    attack_model.load_state_dict(torch.load(model_path, map_location=self.device))
                    attack_model.eval()
                    self.attack_models[label] = attack_model
                    successful_loads += 1
                except Exception as e:
                    print(f"[MIA Wrapper] Failed to load attack model for label {label}: {e}")
            else:
                print(f"[MIA Wrapper] Attack model for label {label} not found at {model_path}")

        print(f"[MIA Wrapper] Successfully loaded {successful_loads}/{self.num_classes} attack models")

        if successful_loads == 0:
            print("[MIA Wrapper] WARNING: No attack models loaded. MIA evaluation will be disabled.")

    def _get_fresh_model_template(self):
        """创建模型模板"""
        base = FedAvgCNN(in_features=3, num_classes=self.num_classes, dim=1600).to(self.device)
        head = copy.deepcopy(base.fc)
        base.fc = torch.nn.Identity()
        return LocalModel(base, head)

    def _create_global_test_loader(self, clients):
        """
        创建全局test数据加载器（合并所有clients的test数据）

        Args:
            clients: 客户端列表

        Returns:
            DataLoader: 包含所有clients test数据的DataLoader
        """
        print(f"[MIA Wrapper] Creating global test dataset from {len(clients)} clients...")

        all_test_samples_x = []
        all_test_samples_y = []

        for client in clients:
            try:
                # 获取该client的test数据
                test_loader = client.load_test_data(batch_size=self.batch_size)

                # 收集所有test数据
                for batch_x, batch_y in test_loader:
                    all_test_samples_x.append(batch_x.cpu())
                    all_test_samples_y.append(batch_y.cpu())

            except Exception as e:
                print(f"[MIA Wrapper] Warning: Failed to load test data for client {client.id}: {e}")
                continue

        if not all_test_samples_x:
            print(f"[MIA Wrapper] Warning: No test data collected from clients")
            return None

        # 合并所有数据
        global_test_x = torch.cat(all_test_samples_x, dim=0)
        global_test_y = torch.cat(all_test_samples_y, dim=0)

        print(f"[MIA Wrapper] Global test dataset created: {len(global_test_x)} total samples")

        # 创建TensorDataset和DataLoader
        from torch.utils.data import TensorDataset
        global_test_dataset = TensorDataset(global_test_x, global_test_y)
        global_test_loader = DataLoader(global_test_dataset, batch_size=self.batch_size, shuffle=False)

        return global_test_loader

    def evaluate_client_mia(self,
                           client,
                           target_labels: Optional[List[int]] = None,
                           global_test_loader: Optional[DataLoader] = None) -> Dict:
        """
        评估单个客户端的MIA攻击成功率

        Args:
            client: 客户端对象，必须有以下属性/方法：
                - client.id: 客户端ID
                - client.model: 客户端模型
                - client.load_train_data(): 返回训练数据DataLoader
                - client.load_test_data(): 返回测试数据DataLoader（仅在global_test_loader=None时使用）
            target_labels: 目标标签列表，None表示所有标签
            global_test_loader: 全局test数据加载器（所有clients的test data合并），
                               如果提供则使用全局test data作为non-member，
                               否则使用client自己的test data

        Returns:
            Dict: MIA评估结果
        """
        if not self.attack_models:
            print(f"[MIA Wrapper] Client {client.id}: No attack models available")
            return {
                'client_id': client.id,
                'status': 'failed',
                'error': 'No attack models available'
            }

        # 确定要评估的标签
        if target_labels is None:
            target_labels = list(self.attack_models.keys())
        else:
            target_labels = [label for label in target_labels if label in self.attack_models]

        if not target_labels:
            print(f"[MIA Wrapper] Client {client.id}: No valid target labels")
            return {
                'client_id': client.id,
                'status': 'failed',
                'error': 'No valid target labels'
            }

        try:
            # 🔑 关键修复: 评估前清理模型梯度和缓存
            if hasattr(client, 'model') and client.model is not None:
                client.model.eval()  # 确保评估模式
                client.model.zero_grad()  # 清理梯度
                # 清理所有参数的梯度
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # 获取train data loader
            train_loader = client.load_train_data(batch_size=self.batch_size)

            # 根据是否提供global_test_loader决定使用哪个test data
            if global_test_loader is not None:
                print(f"[MIA Wrapper] Client {client.id}: Using GLOBAL test data as non-member")
                test_loader = global_test_loader
            else:
                print(f"[MIA Wrapper] Client {client.id}: Using client's own test data as non-member")
                test_loader = client.load_test_data(batch_size=self.batch_size)

            print(f"[MIA Wrapper] Client {client.id}: Data loaders created successfully")

            client_results = {
                'client_id': client.id,
                'status': 'success',
                'summary': {}
            }

            # 🔑 优化：使用滚动统计，避免保存所有label的结果
            label_stats = {
                'sum_f_score': 0.0, 'sum_f_score_sq': 0.0,
                'sum_tpr': 0.0, 'sum_fpr': 0.0, 'sum_accuracy': 0.0,
                'sum_model_accuracy': 0.0,  # 🔑 NEW: 累加模型在各label上的准确率
                'max_f_score': -float('inf'), 'min_f_score': float('inf'),
                'labels_evaluated': 0
            }

            for idx, label in enumerate(target_labels):
                label_result = self._evaluate_single_label_with_loaders(
                    client.model, train_loader, test_loader, label
                )

                if label_result['status'] == 'success':
                    # 🔑 只提取关键指标，不保存完整结果
                    f_score = label_result['f_score']
                    model_acc = label_result.get('model_accuracy', 0.0)  # 🔑 NEW: 获取模型准确率
                    label_stats['sum_f_score'] += f_score
                    label_stats['sum_f_score_sq'] += f_score ** 2
                    label_stats['sum_tpr'] += label_result['tpr']
                    label_stats['sum_fpr'] += label_result['fpr']
                    label_stats['sum_accuracy'] += label_result['accuracy']
                    label_stats['sum_model_accuracy'] += model_acc  # 🔑 NEW: 累加模型准确率
                    label_stats['max_f_score'] = max(label_stats['max_f_score'], f_score)
                    label_stats['min_f_score'] = min(label_stats['min_f_score'], f_score)
                    label_stats['labels_evaluated'] += 1

                    # 🔑 NEW: 同时打印F-score和模型在该label上的准确率
                    print(f"[MIA Wrapper]   Label {label} | F-score: {f_score:.4f} | Model Acc on this label: {model_acc:.4f}")

                # 🔑 立即删除label结果，强制清理
                del label_result
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 🔑 从滚动统计计算总体统计
            if label_stats['labels_evaluated'] > 0:
                n = label_stats['labels_evaluated']
                avg_f_score = label_stats['sum_f_score'] / n
                variance = (label_stats['sum_f_score_sq'] / n) - (avg_f_score ** 2)

                client_results['summary'] = {
                    'avg_f_score': avg_f_score,
                    'avg_tpr': label_stats['sum_tpr'] / n,
                    'avg_fpr': label_stats['sum_fpr'] / n,
                    'avg_accuracy': label_stats['sum_accuracy'] / n,
                    'avg_model_accuracy': label_stats['sum_model_accuracy'] / n,  # 🔑 NEW: 平均模型准确率
                    'max_f_score': label_stats['max_f_score'],
                    'min_f_score': label_stats['min_f_score'],
                    'labels_evaluated': n,
                    'privacy_risk': self._assess_privacy_risk(avg_f_score)
                }
            else:
                client_results['status'] = 'failed'
                client_results['error'] = 'No successful label evaluations'

            # **关键修复**: 清理DataLoader和统计器
            del train_loader, test_loader
            del label_stats

            # 强制垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return client_results

        except Exception as e:
            print(f"[MIA Wrapper] Client {client.id}: Exception during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()

            # 清理内存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'client_id': client.id,
                'status': 'failed',
                'error': f"Exception during evaluation: {str(e)}"
            }

    def _evaluate_single_label_with_loaders(self, client_model, train_loader, test_loader, target_label) -> Dict:
        """
        使用DataLoader评估单个标签的MIA攻击（使用完整数据集，强制内存回收）
        """
        try:
            # 🔑 关键修复: 评估前清理模型状态
            client_model.eval()
            client_model.zero_grad()
            # 从DataLoader中过滤特定标签的数据（内存优化：直接在CPU上操作）
            train_samples_x, train_samples_y = [], []
            test_samples_x, test_samples_y = [], []

            # 收集训练数据中的目标标签样本
            for batch_x, batch_y in train_loader:
                # 确保在CPU上进行过滤，避免GPU内存累积
                batch_x = batch_x.cpu()
                batch_y = batch_y.cpu()

                mask = (batch_y == target_label)
                if mask.any():
                    train_samples_x.append(batch_x[mask])
                    train_samples_y.append(batch_y[mask])

            # 收集测试数据中的目标标签样本
            for batch_x, test_y in test_loader:
                # 确保在CPU上进行过滤
                batch_x = batch_x.cpu()
                test_y = test_y.cpu()

                mask = (test_y == target_label)
                if mask.any():
                    test_samples_x.append(batch_x[mask])
                    test_samples_y.append(test_y[mask])

            # 合并所有批次
            if not train_samples_x or not test_samples_x:
                # 清理内存
                del train_samples_x, train_samples_y, test_samples_x, test_samples_y
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return {
                    'status': 'failed',
                    'error': f'Insufficient data for label {target_label}'
                }

            train_x = torch.cat(train_samples_x, dim=0)
            train_y = torch.cat(train_samples_y, dim=0)
            test_x = torch.cat(test_samples_x, dim=0)
            test_y = torch.cat(test_samples_y, dim=0)

            # 立即清理临时列表
            del train_samples_x, train_samples_y, test_samples_x, test_samples_y

            print(f"[MIA Wrapper] Label {target_label}: Train samples={len(train_x)}, Test samples={len(test_x)}")

            # 创建临时数据集用于评估（使用完整数据集）
            from torch.utils.data import TensorDataset
            train_filtered = TensorDataset(train_x, train_y)
            test_filtered = TensorDataset(test_x, test_y)

            if len(train_filtered) == 0 or len(test_filtered) == 0:
                return {
                    'status': 'failed',
                    'error': f'Insufficient data for label {target_label}'
                }

            # 创建数据加载器
            train_loader = DataLoader(train_filtered, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_filtered, batch_size=self.batch_size, shuffle=False)

            # 🔑 流式处理：逐batch处理，避免累积所有梯度
            attack_model = self.attack_models[target_label]
            all_pred_scores = []
            all_true_labels = []

            # 🔑 NEW: 计算原始模型在该label上的分类准确率
            model_correct_count = 0
            model_total_count = 0

            # 处理训练数据 (member = 1)
            for batch_outputs, batch_labels, batch_head_grads, batch_feat_grads in \
                    get_model_outputs_labels_and_grads_gpu(client_model, train_loader, self.device):

                # 🔑 使用单batch版本的输入准备函数
                batch_inputs = prepare_attack_model_inputs_single_gpu(
                    batch_outputs, batch_head_grads, batch_feat_grads, self.device
                )

                # 攻击模型预测
                with torch.no_grad():
                    batch_preds = attack_model(*batch_inputs)
                    batch_scores = batch_preds.squeeze(1)  # 攻击模型输出已经经过sigmoid，不需要再做一次

                    # 只保存预测分数，不保存梯度
                    all_pred_scores.append(batch_scores.detach())
                    all_true_labels.append(torch.ones(len(batch_scores), device=self.device))

                    # 🔑 NEW: 计算原始模型的分类准确率
                    model_predictions = torch.argmax(batch_outputs, dim=1)
                    model_correct_count += (model_predictions == batch_labels).sum().item()
                    model_total_count += len(batch_labels)

                # 🔑 立即清理当前batch
                del batch_outputs, batch_labels, batch_head_grads, batch_feat_grads
                del batch_inputs, batch_preds, batch_scores

            # 处理测试数据 (non-member = 0)
            for batch_outputs, batch_labels, batch_head_grads, batch_feat_grads in \
                    get_model_outputs_labels_and_grads_gpu(client_model, test_loader, self.device):

                # 🔑 使用单batch版本的输入准备函数
                batch_inputs = prepare_attack_model_inputs_single_gpu(
                    batch_outputs, batch_head_grads, batch_feat_grads, self.device
                )

                # 攻击模型预测
                with torch.no_grad():
                    batch_preds = attack_model(*batch_inputs)
                    batch_scores = torch.sigmoid(batch_preds).squeeze(1)  # 只移除最后一维，保留batch维度

                    # 只保存预测分数
                    all_pred_scores.append(batch_scores.detach())
                    all_true_labels.append(torch.zeros(len(batch_scores), device=self.device))

                    # 🔑 NEW: 计算原始模型的分类准确率（测试集）
                    model_predictions = torch.argmax(batch_outputs, dim=1)
                    model_correct_count += (model_predictions == batch_labels).sum().item()
                    model_total_count += len(batch_labels)

                # 🔑 立即清理当前batch
                del batch_outputs, batch_labels, batch_head_grads, batch_feat_grads
                del batch_inputs, batch_preds, batch_scores

            # 🔑 检查是否有数据
            if not all_pred_scores or not all_true_labels:
                return {
                    'status': 'failed',
                    'error': f'No predictions generated for label {target_label}'
                }

            # 合并所有预测结果
            pred_scores = torch.cat(all_pred_scores)
            true_labels = torch.cat(all_true_labels)

            # 清理临时列表
            del all_pred_scores, all_true_labels

            # 计算指标（GPU上计算）
            metrics = self._calculate_metrics(pred_scores, true_labels, use_gpu=True, label=target_label)

            # 🔑 NEW: 计算原始模型在该label上的分类准确率
            model_accuracy = model_correct_count / model_total_count if model_total_count > 0 else 0.0

            # 清理最终变量
            del pred_scores, true_labels
            del train_x, train_y, test_x, test_y
            del train_filtered, test_filtered
            del train_loader, test_loader  # 🔑 新增: 清理DataLoader

            # 🔑 关键修复: 清理模型梯度
            client_model.zero_grad()
            for param in client_model.parameters():
                if param.grad is not None:
                    param.grad = None

            # 强制垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'status': 'success',
                'label': target_label,
                'train_samples': metrics.get('tp', 0) + metrics.get('fn', 0),
                'test_samples': metrics.get('tn', 0) + metrics.get('fp', 0),
                'model_accuracy': model_accuracy,  # 🔑 NEW: 添加模型分类准确率
                **metrics
            }

        except Exception as e:
            # 🔑 打印完整异常信息
            print(f"[MIA Wrapper]   EXCEPTION in label {target_label}: {str(e)}")
            import traceback
            traceback.print_exc()

            # 异常时也要清理内存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'status': 'failed',
                'error': f"Exception in label {target_label}: {str(e)}"
            }

    def _calculate_metrics(self, pred_scores, true_labels, threshold=0.5, use_gpu=False, label=None):
        """
        计算MIA攻击指标

        Args:
            pred_scores: 预测分数 (CPU或GPU tensor)
            true_labels: 真实标签 (CPU或GPU tensor)
            threshold: 分类阈值
            use_gpu: 是否使用GPU计算（True则输入已在GPU上）
            label: 当前评估的label（用于debug输出）
        """
        # 🔍 DEBUG: 预测分数统计
        pred_min = pred_scores.min().item()
        pred_max = pred_scores.max().item()
        pred_mean = pred_scores.mean().item()

        pred_binary = (pred_scores > threshold).int()
        true_binary = true_labels.int()

        # 混淆矩阵（GPU上计算更快）
        tp = ((pred_binary == 1) & (true_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (true_binary == 0)).sum().item()
        tn = ((pred_binary == 0) & (true_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (true_binary == 1)).sum().item()

        # 🔍 DEBUG: 详细统计
        total = len(pred_scores)
        pred_member_count = (pred_binary == 1).sum().item()
        pred_non_member_count = (pred_binary == 0).sum().item()
        true_member_count = (true_binary == 1).sum().item()
        true_non_member_count = (true_binary == 0).sum().item()

        # 计算指标
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        tpr = recall  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

        # 🔍 DEBUG: 输出详细信息（只在F-score=0时）
        if f_score == 0.0 and label is not None:
            print(f"[MIA DEBUG] Label {label}:")
            print(f"  Pred scores: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
            print(f"  Predictions: member={pred_member_count}, non-member={pred_non_member_count} (total={total})")
            print(f"  Ground truth: member={true_member_count}, non-member={true_non_member_count}")
            print(f"  Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  Accuracy: {accuracy:.4f} ({tp+tn}/{total} correct)")
            if pred_member_count == 0:
                print(f"  → Attack model always predicts non-member (all scores <= 0.5)")
                print(f"  → High accuracy ({accuracy:.2%}) is misleading - just predicting majority class!")
            elif pred_non_member_count == 0:
                print(f"  → Attack model always predicts member (all scores > 0.5)")
                print(f"  → High accuracy ({accuracy:.2%}) is misleading - just predicting majority class!")

        return {
            'accuracy': accuracy,
            'f_score': f_score,
            'precision': precision,
            'recall': recall,
            'tpr': tpr,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

    def _assess_privacy_risk(self, f_score):
        """评估隐私风险级别"""
        if f_score > 0.8:
            return 'high'
        elif f_score > 0.6:
            return 'medium'
        else:
            return 'low'

    def evaluate_all_clients(self,
                           clients,
                           round_num: int,
                           dataset_name: str,
                           save_results: bool = True,
                           results_dir: str = "mia_results",
                           training_config: Optional[Dict] = None) -> Dict:
        """
        评估所有客户端的MIA攻击成功率

        Args:
            clients: 客户端列表，每个client必须有：
                - client.id: 客户端ID
                - client.model: 客户端模型
                - client.load_train_data(): 返回训练数据DataLoader
                - client.load_test_data(): 返回测试数据DataLoader
            round_num: 当前轮次
            dataset_name: 数据集名称
            save_results: 是否保存结果
            results_dir: 结果保存目录

        Returns:
            Dict: 所有客户端的MIA评估结果
        """
        if not self.attack_models:
            return {
                'round': round_num,
                'status': 'failed',
                'error': 'No attack models available'
            }

        print(f"[MIA Wrapper] Evaluating {len(clients)} clients for round {round_num}")
        print(f"[MIA Wrapper] Available attack models: {len(self.attack_models)}")
        print(f"[MIA Wrapper] Attack model labels: {list(self.attack_models.keys())}")

        # 🔑 关键修改: 创建全局test data loader（合并所有clients的test data）
        global_test_loader = None
        if self.use_global_test:
            print(f"[MIA Wrapper] Creating global test dataset from all clients...")
            global_test_loader = self._create_global_test_loader(clients)
            if global_test_loader is not None:
                print(f"[MIA Wrapper] ✓ Using GLOBAL test data as non-member for all clients")
            else:
                print(f"[MIA Wrapper] Warning: Failed to create global test loader, falling back to per-client test data")
        else:
            print(f"[MIA Wrapper] Using each client's own test data as non-member")

        all_results = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'clients': {},
            'summary': {},
            'status': 'success'
        }

        successful_evaluations = 0
        # 🔑 优化：使用滚动统计而不是列表累积，减少内存
        stats_accumulator = {
            'sum_f_score': 0.0, 'sum_f_score_sq': 0.0,
            'sum_tpr': 0.0, 'sum_fpr': 0.0, 'sum_accuracy': 0.0,
            'max_f_score': -float('inf'), 'min_f_score': float('inf'),
            'high_risk': 0, 'medium_risk': 0, 'low_risk': 0
        }

        for idx, client in enumerate(clients):
            print(f"\n[MIA Wrapper] ({'='*60})")
            print(f"[MIA Wrapper] Evaluating client {client.id} ({idx+1}/{len(clients)})...")

            # 🔑 内存监控：评估前GPU状态
            if torch.cuda.is_available():
                gpu_mem_before = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                print(f"[MIA Wrapper] GPU Memory before: {gpu_mem_before:.2f} GB")

            # **关键修复**: 在评估前清理模型梯度
            if hasattr(client, 'model') and client.model is not None:
                client.model.zero_grad()
                # 确保模型处于评估模式
                client.model.eval()

            client_result = self.evaluate_client_mia(client, global_test_loader=global_test_loader)  # 传入全局test loader

            all_results['clients'][client.id] = client_result

            if client_result['status'] == 'success':
                successful_evaluations += 1
                summary = client_result['summary']
                f_score = summary['avg_f_score']

                # 🔑 滚动统计更新
                stats_accumulator['sum_f_score'] += f_score
                stats_accumulator['sum_f_score_sq'] += f_score ** 2
                stats_accumulator['sum_tpr'] += summary['avg_tpr']
                stats_accumulator['sum_fpr'] += summary['avg_fpr']
                stats_accumulator['sum_accuracy'] += summary['avg_accuracy']
                stats_accumulator['max_f_score'] = max(stats_accumulator['max_f_score'], f_score)
                stats_accumulator['min_f_score'] = min(stats_accumulator['min_f_score'], f_score)

                # 风险统计
                if f_score > 0.8:
                    stats_accumulator['high_risk'] += 1
                elif f_score > 0.6:
                    stats_accumulator['medium_risk'] += 1
                else:
                    stats_accumulator['low_risk'] += 1

                print(f"[MIA Wrapper] Client {client.id} evaluation successful, F-score: {f_score:.4f}")
            else:
                print(f"[MIA Wrapper] Client {client.id} evaluation failed: {client_result.get('error', 'Unknown error')}")

            # **关键增强**: 评估后立即清理模型梯度和GPU缓存
            if hasattr(client, 'model') and client.model is not None:
                client.model.zero_grad()
                # 清理未使用的张量
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # 🔑 强制垃圾回收和GPU缓存清理
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 🔑 从滚动统计计算总体统计（不使用列表）
        if successful_evaluations > 0:
            n = successful_evaluations
            avg_f_score = stats_accumulator['sum_f_score'] / n
            # 计算标准差: std = sqrt(E[X^2] - (E[X])^2)
            variance = (stats_accumulator['sum_f_score_sq'] / n) - (avg_f_score ** 2)
            std_f_score = np.sqrt(max(0, variance))  # max(0, ...) 防止数值误差导致负数

            all_results['summary'] = {
                'successful_clients': successful_evaluations,
                'total_clients': len(clients),
                'success_rate': successful_evaluations / len(clients),
                'avg_f_score': avg_f_score,
                'std_f_score': std_f_score,
                'avg_tpr': stats_accumulator['sum_tpr'] / n,
                'avg_fpr': stats_accumulator['sum_fpr'] / n,
                'avg_accuracy': stats_accumulator['sum_accuracy'] / n,
                'max_f_score': stats_accumulator['max_f_score'],
                'min_f_score': stats_accumulator['min_f_score'],
                'high_risk_clients': stats_accumulator['high_risk'],
                'medium_risk_clients': stats_accumulator['medium_risk'],
                'low_risk_clients': stats_accumulator['low_risk']
            }
        else:
            all_results['status'] = 'failed'
            all_results['error'] = 'No successful client evaluations'

        # 清理统计累加器
        del stats_accumulator

        # 保存结果
        if save_results:
            self._save_results(all_results, results_dir, training_config)

        # **关键修复**: 只保存轻量级的历史记录，不保存完整的all_results
        # 避免evaluation_history无限增长导致内存爆炸
        lightweight_entry = {
            'round': round_num,
            'timestamp': all_results['timestamp'],
            'status': all_results['status'],
            'summary': all_results.get('summary', {})
        }
        self.evaluation_history.append(lightweight_entry)

        # 限制历史记录长度（只保留最近50轮）
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]

        # **关键修复**: 清理all_results中的详细数据，只返回摘要
        # 这样可以避免调用者持有大量数据
        lightweight_results = {
            'round': round_num,
            'timestamp': all_results['timestamp'],
            'dataset': dataset_name,
            'summary': all_results.get('summary', {}),
            'status': all_results['status'],
            'clients': {}  # 只返回client F-score，不返回详细结果
        }

        # 只保留每个client的F-score，删除详细的label_results
        for client_id, client_result in all_results['clients'].items():
            if client_result['status'] == 'success' and 'summary' in client_result:
                lightweight_results['clients'][client_id] = {
                    'status': 'success',
                    'summary': {'avg_f_score': client_result['summary']['avg_f_score']}
                }

        # 🔑 清理原始的all_results和global_test_loader
        del all_results
        if global_test_loader is not None:
            del global_test_loader
            print(f"[MIA Wrapper] Global test loader cleaned up")

        # 🔑 强制垃圾回收和GPU缓存清理
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_gpu_mem = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"[MIA Wrapper] Final GPU Memory: {final_gpu_mem:.2f} GB")

        return lightweight_results

    def _save_results(self, results, results_dir, training_config=None):
        """
        保存MIA评估结果（使用MIAManager进行参数化管理）

        Args:
            results: MIA评估结果字典
            results_dir: 结果保存目录（如果使用MIAManager则会被覆盖）
            training_config: 训练配置字典（用于创建参数化目录）
        """
        try:
            # 如果提供了训练配置，使用MIAManager创建参数化目录
            actual_results_dir = results_dir

            if training_config:
                try:
                    from utils.mia_manager import MIAManager

                    manager = MIAManager()

                    # 从配置中提取参数
                    dataset = training_config.get('dataset', 'unknown')
                    alpha = training_config.get('alpha', 1.0)
                    dp_noise = training_config.get('dp_noise', 0.0)
                    enable_rl = training_config.get('enable_rl', False)

                    # 检查是否需要创建新目录（第一轮评估时）
                    if results['round'] == 10:  # 通常第一次评估是第10轮
                        # 创建参数化运行目录
                        actual_results_dir = manager.create_run_directory(
                            dataset=dataset,
                            alpha=alpha,
                            dp_noise=dp_noise,
                            enable_rl=enable_rl
                        )

                        # 保存训练配置
                        manager.save_config(actual_results_dir, training_config)

                        print(f"[MIA Wrapper] Created new run directory: {actual_results_dir}")
                    else:
                        # 查找现有运行目录
                        found_dir = manager.find_run_by_params(
                            dataset=dataset,
                            alpha=alpha,
                            dp_noise=dp_noise,
                            enable_rl=enable_rl,
                            return_latest=True
                        )

                        if found_dir:
                            actual_results_dir = found_dir
                            print(f"[MIA Wrapper] Using existing run directory: {actual_results_dir}")
                        else:
                            # 如果没找到，创建新的
                            actual_results_dir = manager.create_run_directory(
                                dataset=dataset,
                                alpha=alpha,
                                dp_noise=dp_noise,
                                enable_rl=enable_rl
                            )
                            manager.save_config(actual_results_dir, training_config)
                            print(f"[MIA Wrapper] Created new run directory: {actual_results_dir}")

                except ImportError:
                    print("[MIA Wrapper] Warning: MIAManager not available, using default directory")
                except Exception as e:
                    print(f"[MIA Wrapper] Warning: Failed to use MIAManager: {e}")
                    print("[MIA Wrapper] Falling back to default directory")

            # 确保目录存在
            os.makedirs(actual_results_dir, exist_ok=True)

            # 保存详细结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_file = f"mia_detailed_round_{results['round']}_{timestamp}.json"
            detailed_path = os.path.join(actual_results_dir, detailed_file)

            with open(detailed_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # 保存简化的历史记录（包含每个client的F-score）
            history_file = os.path.join(actual_results_dir, "mia_history.json")

            # 提取每个client的F-score
            client_f_scores = {}
            if 'clients' in results:
                for client_id, client_result in results['clients'].items():
                    if client_result['status'] == 'success' and 'summary' in client_result:
                        client_f_scores[str(client_id)] = client_result['summary']['avg_f_score']

            history_entry = {
                'round': results['round'],
                'timestamp': results['timestamp'],
                'summary': results.get('summary', {}),
                'client_f_scores': client_f_scores  # 新增：每个client的F-score
            }

            # 读取现有历史或创建新的
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                except:
                    history = []

            history.append(history_entry)

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)

            print(f"[MIA Wrapper] Results saved to: {detailed_path}")
            print(f"[MIA Wrapper] History updated: {history_file}")

        except Exception as e:
            print(f"[MIA Wrapper] Failed to save results: {e}")
            import traceback
            traceback.print_exc()

    def get_f_score_trends(self) -> Dict:
        """获取F-score趋势数据用于绘图"""
        if not self.evaluation_history:
            return {'error': 'No evaluation history available'}

        trends = {
            'rounds': [],
            'avg_f_scores': [],
            'client_f_scores': {},  # client_id -> [f_scores]
            'risk_distribution': []  # [high, medium, low] counts per round
        }

        for result in self.evaluation_history:
            if result['status'] == 'success' and 'summary' in result:
                round_num = result['round']
                summary = result['summary']

                trends['rounds'].append(round_num)
                trends['avg_f_scores'].append(summary.get('avg_f_score', 0))

                # 风险分布
                risk_dist = [
                    summary.get('high_risk_clients', 0),
                    summary.get('medium_risk_clients', 0),
                    summary.get('low_risk_clients', 0)
                ]
                trends['risk_distribution'].append(risk_dist)

                # 各客户端的F-score
                for client_id, client_result in result['clients'].items():
                    if client_result['status'] == 'success':
                        if client_id not in trends['client_f_scores']:
                            trends['client_f_scores'][client_id] = []
                        trends['client_f_scores'][client_id].append(
                            client_result['summary']['avg_f_score']
                        )

        return trends

    def export_client_f_scores_to_csv(self, results_dir: str = "mia_results", output_filename: str = "client_f_scores.csv"):
        """
        将每轮每个client的F-score导出为CSV文件，方便可视化

        Args:
            results_dir: 结果目录
            output_filename: 输出CSV文件名

        Returns:
            str: CSV文件路径，如果失败返回None
        """
        history_file = os.path.join(results_dir, "mia_history.json")

        if not os.path.exists(history_file):
            print(f"[MIA Wrapper] History file not found: {history_file}")
            return None

        try:
            # 读取历史记录
            with open(history_file, 'r') as f:
                history = json.load(f)

            if not history:
                print(f"[MIA Wrapper] No history data available")
                return None

            # 收集所有client ID
            all_client_ids = set()
            for entry in history:
                if 'client_f_scores' in entry:
                    all_client_ids.update(entry['client_f_scores'].keys())

            all_client_ids = sorted(all_client_ids, key=lambda x: int(x))

            # 创建CSV内容
            import csv
            csv_path = os.path.join(results_dir, output_filename)

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # 写入表头
                header = ['round'] + [f'client_{cid}' for cid in all_client_ids]
                writer.writerow(header)

                # 写入数据
                for entry in history:
                    round_num = entry['round']
                    client_f_scores = entry.get('client_f_scores', {})

                    row = [round_num]
                    for cid in all_client_ids:
                        f_score = client_f_scores.get(str(cid), '')  # 如果该client没有数据，留空
                        row.append(f_score)

                    writer.writerow(row)

            print(f"[MIA Wrapper] Client F-scores exported to: {csv_path}")
            print(f"[MIA Wrapper] Total rounds: {len(history)}, Total clients: {len(all_client_ids)}")

            return csv_path

        except Exception as e:
            print(f"[MIA Wrapper] Failed to export CSV: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_client_data_custom(self, dataset_base_path, is_train=True, is_shadow=True, num_clients=5, alpha=1):
        """
        自定义数据加载函数，使用指定的数据集路径
        """
        import numpy as np
        from torch.utils.data import TensorDataset

        data_list = []
        for i in range(num_clients):
            if is_shadow:
                file_name = os.path.join(
                    dataset_base_path, str(alpha),
                    f"{'cifar-10-shadow/train/train' if is_train else 'cifar-10-shadow/test/test'}{i}_.npz"
                )
            else:
                file_name = os.path.join(
                    dataset_base_path, str(alpha),
                    f"{'cifar-10-normal/train/train' if is_train else 'cifar-10-normal/test/test'}{i}_.npz"
                )

            if not os.path.exists(file_name):
                print(f"[MIA Wrapper] Warning: File {file_name} not found, skipping client {i}")
                continue

            try:
                with open(file_name, 'rb') as f:
                    single_data = np.load(f, allow_pickle=True)['data'].tolist()

                X = torch.Tensor(single_data['x']).float()
                y = torch.Tensor(single_data['y']).long()
                client_dataset = TensorDataset(X, y)
                data_list.append(client_dataset)

                print(f"[MIA Wrapper] Loaded client {i} data from {file_name}")

            except Exception as e:
                print(f"[MIA Wrapper] Error loading {file_name}: {e}")
                continue

        return data_list