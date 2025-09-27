"""
MIA Evaluator Module
A modular component for evaluating membership inference attacks during federated learning
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import json
from typing import Dict, List, Optional, Tuple
import copy
from utils.mia_attack_model import GradientMIA

class MIAEvaluator:
    """
    Modular MIA evaluator for integration into federated learning training loop
    """
    
    def __init__(self, 
                 attack_model_dir: str = "Membership_Inference_Attack",
                 num_classes: int = 10,
                 device: str = "cuda",
                 batch_size: int = 32,
                 evaluate_interval: int = 10,  # Evaluate MIA every N rounds
                 save_results: bool = True,
                 results_dir: str = "mia_results"):
        """
        Initialize MIA Evaluator
        
        Args:
            attack_model_dir: Directory containing pre-trained attack models
            num_classes: Number of classes in the dataset
            device: Device to run evaluation on
            batch_size: Batch size for MIA evaluation
            evaluate_interval: Frequency of MIA evaluation (every N rounds)
            save_results: Whether to save MIA results to file
            results_dir: Directory to save results
        """
        self.attack_model_dir = attack_model_dir
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.evaluate_interval = evaluate_interval
        self.save_results = save_results
        self.results_dir = results_dir
        
        # Create results directory if needed
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Store attack models for each label
        self.attack_models = {}
        self._load_attack_models()
        
        # Store historical results
        self.history = {
            'rounds': [],
            'clients': {},
            'average': []
        }
    
    def _load_attack_models(self):
        """Load pre-trained attack models for each label"""
        for label in range(self.num_classes):
            model_path = os.path.join(self.attack_model_dir, f"attack_model{label}.pth")
            if os.path.exists(model_path):
                model = GradientMIA().to(self.device)
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    self.attack_models[label] = model
                    print(f"[MIA Evaluator] Loaded attack model for label {label}")
                except Exception as e:
                    print(f"[MIA Evaluator] Error loading attack model for label {label}: {e}")
            else:
                print(f"[MIA Evaluator] Warning: Attack model not found for label {label} at {model_path}")
                # Optionally create a new untrained model
                # self.attack_models[label] = GradientMIA().to(self.device)
    
    def should_evaluate(self, round_num: int) -> bool:
        """Check if MIA should be evaluated at this round"""
        return round_num % self.evaluate_interval == 0 and round_num > 0
    
    def evaluate_client(self, 
                       client_model,
                       client_id: int,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       target_label: Optional[int] = None) -> Dict:
        """
        Evaluate MIA attack success on a single client
        
        Args:
            client_model: The client's trained model
            client_id: Client identifier
            train_loader: DataLoader for training data (members)
            test_loader: DataLoader for test data (non-members)
            target_label: Specific label to evaluate (None = all labels)
            
        Returns:
            Dictionary containing MIA metrics
        """
        if target_label is not None:
            labels_to_eval = [target_label] if target_label in self.attack_models else []
        else:
            labels_to_eval = list(self.attack_models.keys())
        
        if not labels_to_eval:
            return {'error': 'No attack models available for evaluation'}
        
        results = {
            'client_id': client_id,
            'label_results': {}
        }
        
        client_model.eval()
        
        for label in labels_to_eval:
            # Get attack model for this label
            attack_model = self.attack_models[label]
            
            # Prepare attack inputs from member data (train)
            member_features = self._extract_attack_features(
                client_model, train_loader, label, is_member=True
            )
            
            # Prepare attack inputs from non-member data (test)
            nonmember_features = self._extract_attack_features(
                client_model, test_loader, label, is_member=False
            )
            
            if member_features is None or nonmember_features is None:
                continue
            
            # Evaluate attack performance
            metrics = self._compute_attack_metrics(
                attack_model, member_features, nonmember_features
            )
            
            results['label_results'][label] = metrics
        
        # Compute average metrics across all labels
        if results['label_results']:
            avg_metrics = self._average_metrics(results['label_results'])
            results['average'] = avg_metrics
        
        return results
    
    def _extract_attack_features(self, 
                                 model,
                                 dataloader: DataLoader,
                                 target_label: int,
                                 is_member: bool,
                                 max_samples: int = 100) -> Optional[TensorDataset]:
        """
        直接复制您的get_model_outputs_labels_and_grads + prepare_attack_model_inputs实现
        """
        # == 第1步：完全按照您的get_model_outputs_labels_and_grads实现 ==
        model.eval()
        outputs_list, labels_list = [], []
        head_grads, feat_grads = {}, {}

        for name, param in model.head.named_parameters():
            head_grads[name] = []
        for name, param in model.feature_extractor.named_parameters():
            feat_grads[name] = []

        loss_fn = nn.CrossEntropyLoss()

        for batch_x, batch_y in dataloader:
            # Filter for target label
            mask = (batch_y == target_label)
            if not mask.any():
                continue
                
            batch_x = batch_x[mask].to(self.device)
            batch_y = batch_y[mask].to(self.device)
            
            model.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()

            softmax = nn.functional.softmax(logits.detach().cpu(), dim=1)
            outputs_list.append(softmax.numpy())
            labels_list.append(batch_y.detach().cpu().numpy())

            for name, param in model.head.named_parameters():
                grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
                head_grads[name].append(grad)

            for name, param in model.feature_extractor.named_parameters():
                grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
                feat_grads[name].append(grad)

        if not outputs_list:
            return None
            
        outputs = np.concatenate(outputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # == 第2步：完全按照您的prepare_attack_model_inputs实现 ==
        # 关键理解：在您的原始实现中，softmax_out需要与gradient tensor的batch数匹配
        # 我们需要将样本级别的outputs聚合为batch级别
        outputs_tensor = torch.from_numpy(outputs).float()
        
        # 将outputs按batch分组并求平均，匹配梯度tensor的batch数量
        num_batches = len(outputs_list)  # 这等于梯度的batch数
        batch_sizes = [arr.shape[0] for arr in outputs_list]  # 每个batch的样本数
        
        # 重新组织softmax输出以匹配batch结构
        batch_softmax_list = []
        start_idx = 0
        for i, batch_size in enumerate(batch_sizes):
            end_idx = start_idx + batch_size
            batch_outputs = outputs_tensor[start_idx:end_idx]  # 当前batch的所有样本
            batch_mean = torch.mean(batch_outputs, dim=0, keepdim=True)  # [1, num_classes]
            batch_softmax_list.append(batch_mean)
            start_idx = end_idx
            
        softmax_out = torch.cat(batch_softmax_list, dim=0)  # [num_batches, num_classes]
        softmax_out = nn.functional.softmax(softmax_out, dim=1)

        def stack_grads(grad_list):
            grads = [torch.tensor(g, dtype=torch.float32) for g in grad_list if g is not None]
            return torch.stack(grads, dim=0)

        grad_conv1 = stack_grads(feat_grads['conv1.0.weight'])
        grad_conv2 = stack_grads(feat_grads['conv2.0.weight'])
        grad_fc1 = stack_grads(feat_grads['fc1.0.weight'])
        grad_fc = stack_grads(head_grads['weight'])

        def reshape_tensor(tensor):
            B = tensor.shape[0]
            flat = tensor.view(B, -1)
            side = int(flat.shape[1] ** 0.5)
            if side * side != flat.shape[1]:
                pad = side * side - flat.shape[1]
                flat = nn.functional.pad(flat, (0, pad), value=0)
            return flat.view(B, 1, side, side)

        g1 = reshape_tensor(grad_conv1)
        g2 = reshape_tensor(grad_conv2) 
        g3 = reshape_tensor(grad_fc1)
        g4 = reshape_tensor(grad_fc)
        
        # == 第3步：完全按照您的train_attack_model.py中的标签创建逻辑 ==
        # y_inout.append(torch.full((g1.size(0),), label_flag))
        y_inout = torch.full((g1.size(0),), 1 if is_member else 0, dtype=torch.long)
        
        
        return TensorDataset(g1, g2, g3, g4, softmax_out, y_inout)
            
    
    
    def _compute_attack_metrics(self, 
                               attack_model,
                               member_data: TensorDataset,
                               nonmember_data: TensorDataset) -> Dict:
        """Compute MIA attack metrics"""
        # Combine member and non-member data
        combined_data = ConcatDataset([member_data, nonmember_data])
        dataloader = DataLoader(combined_data, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                grad_conv1, grad_conv2, grad_fc1, grad_fc, softmax, labels = batch
                
                # Move to device
                grad_conv1 = grad_conv1.to(self.device)
                grad_conv2 = grad_conv2.to(self.device)
                grad_fc1 = grad_fc1.to(self.device)
                grad_fc = grad_fc.to(self.device)
                softmax = softmax.to(self.device)
                labels = labels.to(self.device)
                
                # Get attack predictions
                preds = attack_model(grad_conv1, grad_conv2, grad_fc1, grad_fc, softmax)
                
                all_preds.append(preds.squeeze())
                all_labels.append(labels)
        
        if not all_preds:
            return {'error': 'No predictions generated'}
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        # Calculate metrics
        predictions = (all_preds > 0.5).astype(int)
        
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        metrics = {
            'accuracy': (tp + tn) / max(1, tp + fp + tn + fn),
            'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn),
            'f1_score': 2 * tp / max(1, 2 * tp + fp + fn),
            'tpr': tp / max(1, tp + fn),  # True Positive Rate
            'fpr': fp / max(1, fp + tn),  # False Positive Rate
            'member_acc': tp / max(1, tp + fn),  # Accuracy on members
            'nonmember_acc': tn / max(1, tn + fp),  # Accuracy on non-members
            'total_samples': len(all_labels),
            'member_samples': int(np.sum(all_labels == 1)),
            'nonmember_samples': int(np.sum(all_labels == 0))
        }
        
        return metrics
    
    def _average_metrics(self, label_results: Dict) -> Dict:
        """Compute average metrics across all labels"""
        metrics_sum = {}
        count = 0
        
        for label, metrics in label_results.items():
            if 'error' in metrics:
                continue
            
            for key, value in metrics.items():
                if key not in ['total_samples', 'member_samples', 'nonmember_samples']:
                    if key not in metrics_sum:
                        metrics_sum[key] = 0
                    metrics_sum[key] += value
            count += 1
        
        if count == 0:
            return {}
        
        avg_metrics = {key: value / count for key, value in metrics_sum.items()}
        return avg_metrics
    
    def evaluate_all_clients(self,
                            clients: List,
                            round_num: int,
                            dataset_name: str = "cifar10",
                            alpha: float = 1.0) -> Dict:
        """
        Evaluate MIA on all clients
        
        Args:
            clients: List of client objects
            round_num: Current training round
            dataset_name: Name of the dataset
            alpha: Data heterogeneity parameter
            
        Returns:
            Dictionary containing evaluation results for all clients
        """
        print(f"[MIA] Evaluating at round {round_num}")
        
        round_results = {
            'round': round_num,
            'clients': {},
            'average': {}
        }
        
        all_client_metrics = []
        
        for client in clients:
            # Get client's data loaders
            train_loader = client.load_train_data(batch_size=self.batch_size)
            test_loader = client.load_test_data(batch_size=self.batch_size)
            
            # Evaluate MIA on this client
            client_results = self.evaluate_client(
                client.model,
                client.id,
                train_loader,
                test_loader
            )
            
            round_results['clients'][client.id] = client_results
            
            # Collect average metrics if available
            if 'average' in client_results:
                all_client_metrics.append(client_results['average'])
        
        # Compute global average across all clients
        if all_client_metrics:
            global_avg = {}
            for key in all_client_metrics[0].keys():
                values = [m[key] for m in all_client_metrics if key in m]
                global_avg[key] = np.mean(values)
            
            round_results['average'] = global_avg
            
            print(f"[MIA] Round {round_num} - Attack Acc: {global_avg['accuracy']:.3f}, "
                  f"F1: {global_avg['f1_score']:.3f}")
        
        # Store in history
        self.history['rounds'].append(round_num)
        for client_id, results in round_results['clients'].items():
            if client_id not in self.history['clients']:
                self.history['clients'][client_id] = []
            self.history['clients'][client_id].append(results)
        self.history['average'].append(round_results['average'])
        
        # Save results if configured
        if self.save_results:
            self._save_round_results(round_results, dataset_name, alpha)
        
        return round_results
    
    def _save_round_results(self, results: Dict, dataset_name: str, alpha: float):
        """Save MIA evaluation results for this round"""
        filename = f"mia_{dataset_name}_alpha{alpha:.2f}_round{results['round']:04d}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save the complete history
        history_file = os.path.join(self.results_dir, f"mia_history_{dataset_name}_alpha{alpha:.2f}.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of MIA evaluation history"""
        if not self.history['average']:
            return {'error': 'No evaluation history available'}
        
        summary = {
            'total_rounds_evaluated': len(self.history['rounds']),
            'rounds': self.history['rounds'],
            'metrics_over_time': {}
        }
        
        # Extract metrics over time
        for metrics in self.history['average']:
            for key, value in metrics.items():
                if key not in summary['metrics_over_time']:
                    summary['metrics_over_time'][key] = []
                summary['metrics_over_time'][key].append(value)
        
        # Compute statistics
        for key, values in summary['metrics_over_time'].items():
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
            summary[f'{key}_final'] = values[-1] if values else None
        
        return summary