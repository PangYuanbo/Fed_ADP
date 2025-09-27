"""
Simple MIA Evaluator that directly calls your existing implementation
"""

import torch
import os
import sys
import numpy as np
from typing import Dict, List, Optional

# Add Membership_Inference_Attack to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimpleMIAEvaluator:
    """
    Simple wrapper that directly uses your existing MIA implementation
    """
    
    def __init__(self, 
                 attack_model_dir: str = "Membership_Inference_Attack",
                 num_classes: int = 10,
                 device: str = "cuda",
                 batch_size: int = 1,
                 evaluate_interval: int = 10,
                 save_results: bool = True,
                 results_dir: str = "mia_results",
                 max_samples_per_label: int = 100):  # Limit samples for faster evaluation
        """Initialize MIA Evaluator"""
        self.attack_model_dir = attack_model_dir
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.evaluate_interval = evaluate_interval
        self.save_results = save_results
        self.results_dir = results_dir
        self.max_samples_per_label = max_samples_per_label  # Limit to avoid memory issues
        
        # Import your existing MIA implementation
        from Membership_Inference_Attack.mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
        from Membership_Inference_Attack.model import GradientMIA
        
        self.get_model_outputs_labels_and_grads = get_model_outputs_labels_and_grads
        self.prepare_attack_model_inputs = prepare_attack_model_inputs
        self.GradientMIA = GradientMIA
        
        # Load attack models
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
                model = self.GradientMIA().to(self.device)
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    self.attack_models[label] = model
                    print(f"[MIA Evaluator] Loaded attack model for label {label}")
                except Exception as e:
                    print(f"[MIA Evaluator] Error loading attack model for label {label}: {e}")
    
    def should_evaluate(self, round_num: int) -> bool:
        """Check if MIA should be evaluated at this round"""
        return round_num % self.evaluate_interval == 0 and round_num > 0
    
    def evaluate_client(self, 
                       client_model,
                       client_id: int,
                       train_loader,
                       test_loader,
                       target_label: Optional[int] = None) -> Dict:
        """
        Evaluate MIA attack success on a single client using your original implementation
        Note: This directly calls your existing MIA code from Membership_Inference_Attack/
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
            print(f"[MIA] Processing label {label} for client {client_id}...")
            try:
                # Get attack model for this label
                attack_model = self.attack_models[label]
                
                # Create filtered data loaders for this specific label
                # This ensures gradients are computed only for the target label
                from torch.utils.data import DataLoader, Subset
                
                print(f"[MIA] Filtering train data for label {label}...")
                # Filter train data for target label
                train_indices = []
                for idx, (_, y) in enumerate(train_loader.dataset):
                    if y == label:
                        train_indices.append(idx)
                        # Limit samples for performance
                        if len(train_indices) >= self.max_samples_per_label:
                            break
                
                if not train_indices:
                    print(f"[MIA] No train samples found for label {label}, skipping...")
                    continue
                    
                member_subset = Subset(train_loader.dataset, train_indices)
                member_loader = DataLoader(member_subset, batch_size=self.batch_size, shuffle=False)
                print(f"[MIA] Found {len(train_indices)} train samples for label {label} (limited to {self.max_samples_per_label})")
                
                print(f"[MIA] Filtering test data for label {label}...")
                # Filter test data for target label  
                test_indices = []
                for idx, (_, y) in enumerate(test_loader.dataset):
                    if y == label:
                        test_indices.append(idx)
                        # Limit samples for performance
                        if len(test_indices) >= self.max_samples_per_label:
                            break
                        
                if not test_indices:
                    print(f"[MIA] No test samples found for label {label}, skipping...")
                    continue
                    
                nonmember_subset = Subset(test_loader.dataset, test_indices)
                nonmember_loader = DataLoader(nonmember_subset, batch_size=self.batch_size, shuffle=False)
                print(f"[MIA] Found {len(test_indices)} test samples for label {label} (limited to {self.max_samples_per_label})")
                
                # Use your original implementation to get features
                print(f"[MIA] Computing gradients for member data (label {label})...")
                # For member data (train)
                member_outputs, member_labels, member_head_grads, member_feat_grads = \
                    self.get_model_outputs_labels_and_grads(client_model, member_loader, self.device)
                
                print(f"[MIA] Preparing attack inputs for members (label {label})...")
                # Prepare attack inputs for members
                g1_m, g2_m, g3_m, g4_m, softmax_m = self.prepare_attack_model_inputs(
                    member_outputs, member_head_grads, member_feat_grads
                )
                
                print(f"[MIA] Computing gradients for non-member data (label {label})...")
                # For non-member data (test)
                nonmember_outputs, nonmember_labels, nonmember_head_grads, nonmember_feat_grads = \
                    self.get_model_outputs_labels_and_grads(client_model, nonmember_loader, self.device)
                
                print(f"[MIA] Preparing attack inputs for non-members (label {label})...")
                # Prepare attack inputs for non-members
                g1_n, g2_n, g3_n, g4_n, softmax_n = self.prepare_attack_model_inputs(
                    nonmember_outputs, nonmember_head_grads, nonmember_feat_grads
                )
                
                # Evaluate attack
                with torch.no_grad():
                    # Member predictions
                    member_preds = attack_model(g1_m.to(self.device), g2_m.to(self.device), 
                                               g3_m.to(self.device), g4_m.to(self.device), 
                                               softmax_m.to(self.device))
                    
                    # Non-member predictions
                    nonmember_preds = attack_model(g1_n.to(self.device), g2_n.to(self.device),
                                                  g3_n.to(self.device), g4_n.to(self.device),
                                                  softmax_n.to(self.device))
                
                # Calculate metrics
                member_preds = member_preds.squeeze().cpu().numpy()
                nonmember_preds = nonmember_preds.squeeze().cpu().numpy()
                
                # Make sure they are 1D arrays
                if member_preds.ndim == 0:
                    member_preds = np.array([member_preds])
                if nonmember_preds.ndim == 0:
                    nonmember_preds = np.array([nonmember_preds])
                
                member_correct = np.sum(member_preds > 0.5)
                nonmember_correct = np.sum(nonmember_preds <= 0.5)
                total = len(member_preds) + len(nonmember_preds)
                
                metrics = {
                    'accuracy': (member_correct + nonmember_correct) / max(1, total),
                    'tpr': member_correct / max(1, len(member_preds)),
                    'fpr': 1 - (nonmember_correct / max(1, len(nonmember_preds))),
                    'member_acc': member_correct / max(1, len(member_preds)),
                    'nonmember_acc': nonmember_correct / max(1, len(nonmember_preds)),
                    'total_samples': total,
                    'member_samples': len(member_preds),
                    'nonmember_samples': len(nonmember_preds)
                }
                
                # Calculate F1 score
                tp = member_correct
                fp = len(nonmember_preds) - nonmember_correct
                fn = len(member_preds) - member_correct
                
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1_score = 2 * precision * recall / max(0.001, precision + recall)
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1_score
                
                results['label_results'][label] = metrics
                
            except Exception as e:
                print(f"[MIA] Error evaluating label {label}: {e}")
                import traceback
                traceback.print_exc()
                results['label_results'][label] = {'error': str(e)}
        
        # Compute average metrics across all labels
        if results['label_results']:
            avg_metrics = self._average_metrics(results['label_results'])
            results['average'] = avg_metrics
        
        return results
    
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
        """
        print(f"[MIA] Evaluating at round {round_num}")
        print(f"[MIA] Number of clients to evaluate: {len(clients)}")
        print(f"[MIA] Available attack models: {len(self.attack_models)} for labels {list(self.attack_models.keys())}")
        
        round_results = {
            'round': round_num,
            'clients': {},
            'average': {}
        }
        
        all_client_metrics = []
        
        for i, client in enumerate(clients):
            print(f"[MIA] Processing client {client.id} ({i+1}/{len(clients)})...")
            
            try:
                # Get client's data loaders
                print(f"[MIA] Loading data for client {client.id}...")
                train_loader = client.load_train_data(batch_size=self.batch_size)
                test_loader = client.load_test_data(batch_size=self.batch_size)
                
                print(f"[MIA] Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
                
                # Evaluate MIA on this client
                print(f"[MIA] Starting MIA evaluation for client {client.id}...")
                client_results = self.evaluate_client(
                    client.model,
                    client.id,
                    train_loader,
                    test_loader
                )
                print(f"[MIA] Completed MIA evaluation for client {client.id}")
                
                round_results['clients'][client.id] = client_results
                
                # Collect average metrics if available
                if 'average' in client_results and client_results['average']:
                    all_client_metrics.append(client_results['average'])
                    print(f"[MIA] Client {client.id} metrics collected")
                else:
                    print(f"[MIA] Client {client.id} - no average metrics available")
                    
            except Exception as e:
                print(f"[MIA] Error processing client {client.id}: {e}")
                import traceback
                traceback.print_exc()
                round_results['clients'][client.id] = {'error': str(e)}
        
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
        self.history['average'].append(round_results.get('average', {}))
        
        return round_results
    
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
            if metrics:  # Check if metrics is not empty
                for key, value in metrics.items():
                    if key not in summary['metrics_over_time']:
                        summary['metrics_over_time'][key] = []
                    summary['metrics_over_time'][key].append(value)
        
        # Compute statistics
        import numpy as np
        for key, values in summary['metrics_over_time'].items():
            if values:  # Only compute if we have values
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                summary[f'{key}_final'] = values[-1] if values else None
        
        return summary