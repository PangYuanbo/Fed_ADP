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
    from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
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
                 max_samples_per_client: int = 50):
        """
        初始化联邦学习MIA评估器

        Args:
            attack_model_dir: 攻击模型目录
            num_classes: 类别数量
            device: 计算设备
            batch_size: 批次大小
            alpha: 数据分布参数
            max_samples_per_client: 每个客户端最大样本数（用于加速）
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

        print(f"[MIA Wrapper] Initializing with attack_model_dir: {attack_model_dir}")
        print(f"[MIA Wrapper] Device: {self.device}")
        print(f"[MIA Wrapper] Num classes: {num_classes}")

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

    def evaluate_client_mia(self,
                           client_model,
                           client_id: int,
                           dataset_name: str,
                           target_labels: Optional[List[int]] = None) -> Dict:
        """
        评估单个客户端的MIA攻击成功率

        Args:
            client_model: 客户端模型
            client_id: 客户端ID
            dataset_name: 数据集名称
            target_labels: 目标标签列表，None表示所有标签

        Returns:
            Dict: MIA评估结果
        """
        if not self.attack_models:
            print(f"[MIA Wrapper] Client {client_id}: No attack models available")
            return {
                'client_id': client_id,
                'status': 'failed',
                'error': 'No attack models available'
            }

        # 确定要评估的标签
        if target_labels is None:
            target_labels = list(self.attack_models.keys())
        else:
            target_labels = [label for label in target_labels if label in self.attack_models]

        if not target_labels:
            print(f"[MIA Wrapper] Client {client_id}: No valid target labels")
            return {
                'client_id': client_id,
                'status': 'failed',
                'error': 'No valid target labels'
            }

        try:
            # 加载客户端数据 - 使用MIA模块的数据加载方式
            print(f"[MIA Wrapper] Client {client_id}: Loading data for dataset {dataset_name}, alpha={self.alpha}")

            # 数据集名称映射 - 将联邦学习的数据集名称映射到MIA的名称
            dataset_mapping = {
                'cifar10': 'cifar-10-normal',
                'cifar-10': 'cifar-10-normal',
                'cifar-10-normal': 'cifar-10-normal'
            }

            # 确保我们使用正确的数据集名称
            mapped_dataset = dataset_mapping.get(dataset_name, 'cifar-10-normal')
            print(f"[MIA Wrapper] Mapped dataset '{dataset_name}' to '{mapped_dataset}'")

            # 使用自定义数据加载函数，指向新的数据路径
            project_root = os.path.join(os.path.dirname(__file__), '..', '..')
            project_root = os.path.abspath(project_root)
            dataset_base_path = os.path.join(project_root, 'dataset')

            print(f"[MIA Wrapper] Using dataset path: {dataset_base_path}")

            # MIA模块的read_client_data返回所有客户端的数据列表，我们需要选择特定的客户端
            is_shadow = False  # 使用正常模型数据，不是影子模型数据
            num_clients = 10   # 假设有10个客户端

            # 使用自定义数据加载函数
            train_data_list = self._read_client_data_custom(
                dataset_base_path, is_train=True, is_shadow=is_shadow,
                num_clients=num_clients, alpha=self.alpha
            )
            test_data_list = self._read_client_data_custom(
                dataset_base_path, is_train=False, is_shadow=is_shadow,
                num_clients=num_clients, alpha=self.alpha
            )

            # 获取特定客户端的数据
            if client_id >= len(train_data_list) or client_id >= len(test_data_list):
                raise IndexError(f"Client {client_id} not found in data (available: {len(train_data_list)} clients)")

            train_data = train_data_list[client_id]
            test_data = test_data_list[client_id]

            print(f"[MIA Wrapper] Client {client_id}: Train data size: {len(train_data)}, Test data size: {len(test_data)}")

            client_results = {
                'client_id': client_id,
                'status': 'success',
                'label_results': {},
                'summary': {}
            }

            all_f_scores = []
            all_tprs = []
            all_fprs = []
            all_accuracies = []

            for label in target_labels:
                label_result = self._evaluate_single_label(
                    client_model, train_data, test_data, label
                )

                if label_result['status'] == 'success':
                    client_results['label_results'][label] = label_result
                    all_f_scores.append(label_result['f_score'])
                    all_tprs.append(label_result['tpr'])
                    all_fprs.append(label_result['fpr'])
                    all_accuracies.append(label_result['accuracy'])

            # 计算总体统计
            if all_f_scores:
                client_results['summary'] = {
                    'avg_f_score': np.mean(all_f_scores),
                    'avg_tpr': np.mean(all_tprs),
                    'avg_fpr': np.mean(all_fprs),
                    'avg_accuracy': np.mean(all_accuracies),
                    'max_f_score': np.max(all_f_scores),
                    'min_f_score': np.min(all_f_scores),
                    'labels_evaluated': len(all_f_scores),
                    'privacy_risk': self._assess_privacy_risk(np.mean(all_f_scores))
                }
            else:
                client_results['status'] = 'failed'
                client_results['error'] = 'No successful label evaluations'

            return client_results

        except Exception as e:
            print(f"[MIA Wrapper] Client {client_id}: Exception during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'client_id': client_id,
                'status': 'failed',
                'error': f"Exception during evaluation: {str(e)}"
            }

    def _evaluate_single_label(self, client_model, train_data, test_data, target_label) -> Dict:
        """评估单个标签的MIA攻击"""
        try:
            # 过滤特定标签的数据
            train_filtered = filter_by_label(train_data, target_label)
            test_filtered = filter_by_label(test_data, target_label)

            # 限制样本数量以加速评估
            if len(train_filtered) > self.max_samples_per_client:
                train_indices = torch.randperm(len(train_filtered))[:self.max_samples_per_client]
                train_filtered = torch.utils.data.Subset(train_filtered, train_indices)

            if len(test_filtered) > self.max_samples_per_client:
                test_indices = torch.randperm(len(test_filtered))[:self.max_samples_per_client]
                test_filtered = torch.utils.data.Subset(test_filtered, test_indices)

            if len(train_filtered) == 0 or len(test_filtered) == 0:
                return {
                    'status': 'failed',
                    'error': f'Insufficient data for label {target_label}'
                }

            # 创建数据加载器
            train_loader = DataLoader(train_filtered, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_filtered, batch_size=self.batch_size, shuffle=False)

            # 获取模型输出和梯度
            train_outputs, train_labels, train_head_grads, train_feat_grads = \
                get_model_outputs_labels_and_grads(client_model, train_loader, self.device)

            test_outputs, test_labels, test_head_grads, test_feat_grads = \
                get_model_outputs_labels_and_grads(client_model, test_loader, self.device)

            # 准备攻击模型输入
            train_inputs = prepare_attack_model_inputs(train_outputs, train_head_grads, train_feat_grads)
            test_inputs = prepare_attack_model_inputs(test_outputs, test_head_grads, test_feat_grads)

            # 使用攻击模型进行预测
            attack_model = self.attack_models[target_label]

            with torch.no_grad():
                # 成员数据（训练集）预测
                train_preds = []
                for i in range(len(train_inputs[0])):
                    inputs = [inp[i:i+1].to(self.device) for inp in train_inputs]
                    pred = attack_model(*inputs).cpu()
                    train_preds.append(pred)
                train_preds = torch.cat(train_preds, dim=0)

                # 非成员数据（测试集）预测
                test_preds = []
                for i in range(len(test_inputs[0])):
                    inputs = [inp[i:i+1].to(self.device) for inp in test_inputs]
                    pred = attack_model(*inputs).cpu()
                    test_preds.append(pred)
                test_preds = torch.cat(test_preds, dim=0)

            # 计算攻击指标
            train_scores = torch.sigmoid(train_preds).squeeze()
            test_scores = torch.sigmoid(test_preds).squeeze()

            # 真实标签：训练数据为1（成员），测试数据为0（非成员）
            true_labels = torch.cat([
                torch.ones(len(train_scores)),
                torch.zeros(len(test_scores))
            ])

            # 预测分数
            pred_scores = torch.cat([train_scores, test_scores])

            # 计算指标
            metrics = self._calculate_metrics(pred_scores, true_labels)

            return {
                'status': 'success',
                'label': target_label,
                'train_samples': len(train_filtered),
                'test_samples': len(test_filtered),
                **metrics
            }

        except Exception as e:
            return {
                'status': 'failed',
                'error': f"Exception in label {target_label}: {str(e)}"
            }

    def _calculate_metrics(self, pred_scores, true_labels, threshold=0.5):
        """计算MIA攻击指标"""
        pred_binary = (pred_scores > threshold).int()
        true_binary = true_labels.int()

        # 混淆矩阵
        tp = ((pred_binary == 1) & (true_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (true_binary == 0)).sum().item()
        tn = ((pred_binary == 0) & (true_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (true_binary == 1)).sum().item()

        # 计算指标
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        tpr = recall  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

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
                           results_dir: str = "mia_results") -> Dict:
        """
        评估所有客户端的MIA攻击成功率

        Args:
            clients: 客户端列表
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

        all_results = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'clients': {},
            'summary': {},
            'status': 'success'
        }

        successful_evaluations = 0
        all_f_scores = []
        all_tprs = []
        all_fprs = []
        all_accuracies = []

        for client in clients:
            print(f"[MIA Wrapper] Evaluating client {client.id}...")
            client_result = self.evaluate_client_mia(
                client.model,
                client.id,
                dataset_name
            )

            all_results['clients'][client.id] = client_result

            if client_result['status'] == 'success':
                successful_evaluations += 1
                summary = client_result['summary']
                all_f_scores.append(summary['avg_f_score'])
                all_tprs.append(summary['avg_tpr'])
                all_fprs.append(summary['avg_fpr'])
                all_accuracies.append(summary['avg_accuracy'])
                print(f"[MIA Wrapper] Client {client.id} evaluation successful, F-score: {summary['avg_f_score']:.4f}")
            else:
                print(f"[MIA Wrapper] Client {client.id} evaluation failed: {client_result.get('error', 'Unknown error')}")

        # 计算总体统计
        if all_f_scores:
            all_results['summary'] = {
                'successful_clients': successful_evaluations,
                'total_clients': len(clients),
                'success_rate': successful_evaluations / len(clients),
                'avg_f_score': np.mean(all_f_scores),
                'std_f_score': np.std(all_f_scores),
                'avg_tpr': np.mean(all_tprs),
                'avg_fpr': np.mean(all_fprs),
                'avg_accuracy': np.mean(all_accuracies),
                'max_f_score': np.max(all_f_scores),
                'min_f_score': np.min(all_f_scores),
                'high_risk_clients': sum(1 for score in all_f_scores if score > 0.8),
                'medium_risk_clients': sum(1 for score in all_f_scores if 0.6 < score <= 0.8),
                'low_risk_clients': sum(1 for score in all_f_scores if score <= 0.6)
            }
        else:
            all_results['status'] = 'failed'
            all_results['error'] = 'No successful client evaluations'

        # 保存结果
        if save_results:
            self._save_results(all_results, results_dir)

        # 添加到历史记录
        self.evaluation_history.append(all_results)

        return all_results

    def _save_results(self, results, results_dir):
        """保存MIA评估结果"""
        try:
            os.makedirs(results_dir, exist_ok=True)

            # 保存详细结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_file = f"mia_detailed_round_{results['round']}_{timestamp}.json"
            detailed_path = os.path.join(results_dir, detailed_file)

            with open(detailed_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # 保存简化的历史记录
            history_file = os.path.join(results_dir, "mia_history.json")
            history_entry = {
                'round': results['round'],
                'timestamp': results['timestamp'],
                'summary': results.get('summary', {})
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

        except Exception as e:
            print(f"[MIA Wrapper] Failed to save results: {e}")

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