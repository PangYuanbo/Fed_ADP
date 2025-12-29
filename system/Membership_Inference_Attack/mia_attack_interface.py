# ============================
# mia_attack_interface.py
# MIA攻击接口封装 - 可在训练过程中直接调用
# 返回JSON格式结果，无print输出
# ============================

import os
import torch
import copy
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# 导入现有模块
from .model import FedAvgCNN, LocalModel, GradientMIA
from .whitebox_mia_pipeline import whitebox_membership_inference_attack_pipeline
from .train_attack_model import train_attack_model
from .evaluate_client_accuracy import evaluate_all_clients_accuracy
from utils.attack_feature_config import (
    DEFAULT_ATTACK_FEATURES,
    attack_checkpoint_name,
    normalize_attack_features,
)


class MIAAttackInterface:
    """
    MIA攻击接口类，封装所有MIA攻击相关功能
    返回JSON格式结果，无print输出
    """

    def __init__(self, device=None, default_config=None, verbose=False):
        """
        初始化MIA攻击接口

        Args:
            device: 计算设备，默认自动选择
            default_config: 默认配置字典
            verbose: 是否输出详细日志
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_config = default_config or self._get_default_config()
        self.verbose = verbose
        self.logs = []  # 存储日志信息

    def _log(self, message, level='info'):
        """记录日志信息"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        if self.verbose:
            print(f"[{level.upper()}] {message}")

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'batch_size': 1,
            'num_classes': 10,
            'num_clients': 10,
            'alpha': 1,
            'attack_epochs': 50,
            'attack_lr': 1e-3,
            'model_dim': 1600,
            'model_in_features': 3,
            'attack_model_dir': ".",
            'attack_features': DEFAULT_ATTACK_FEATURES.copy(),
        }

    def _get_fresh_model(self):
        """创建新的模型实例"""
        config = self.default_config
        base = FedAvgCNN(
            in_features=config['model_in_features'],
            num_classes=config['num_classes'],
            dim=config['model_dim']
        ).to(self.device)
        head = copy.deepcopy(base.fc)
        base.fc = torch.nn.Identity()
        return LocalModel(base, head)

    def _normalize_attack_features(self, config):
        return normalize_attack_features(config.get('attack_features'))

    def _attack_model_paths(self, label, attack_model_dir, attack_features):
        primary = os.path.join(attack_model_dir, attack_checkpoint_name(label, attack_features))
        legacy = os.path.join(attack_model_dir, attack_checkpoint_name(label, attack_features, include_suffix=False))
        return primary, legacy

    def run_mia_attack(self,
                      target_model_path: str,
                      attack_model_dir: str = ".",
                      target_label: Optional[int] = None,
                      shadow_model_paths: Optional[List[str]] = None,
                      config: Optional[Dict] = None,
                      save_results: bool = False,
                      results_dir: str = "mia_results") -> Dict:
        """
        单次MIA攻击评估 - 可在训练过程中直接调用

        Args:
            target_model_path: 目标模型路径
            attack_model_dir: 攻击模型存储目录
            target_label: 指定攻击的标签，None表示所有标签
            shadow_model_paths: 影子模型路径列表，None则使用默认路径
            config: 配置参数，会与default_config合并
            save_results: 是否保存结果到文件
            results_dir: 结果保存目录

        Returns:
            Dict: 包含攻击结果的字典
        """
        self.logs = []  # 重置日志

        # 合并配置
        merged_config = {**self.default_config, **(config or {})}
        merged_config['attack_model_dir'] = attack_model_dir
        attack_features = self._normalize_attack_features(merged_config)
        merged_config['attack_features'] = attack_features
        attack_model_dir = merged_config.get('attack_model_dir', '.')
        attack_features = self._normalize_attack_features(merged_config)
        merged_config['attack_features'] = attack_features

        # 确定要攻击的标签
        target_labels = [target_label] if target_label is not None else list(range(merged_config['num_classes']))

        # 准备影子模型路径
        if shadow_model_paths is None:
            shadow_model_paths = [
                f"shadow_model/{merged_config['alpha']}/results_cifar-10-shadow_client{i}_1000_0.0050.pt"
                for i in range(merged_config['num_clients'])
            ]

        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'target_model': target_model_path,
            'config': merged_config,
            'attack_results': {},
            'summary': {},
            'logs': [],
            'errors': []
        }

        all_f_scores, all_tprs, all_fprs = [], [], []
        errors = []

        for label in target_labels:
            self._log(f"Running MIA Attack on Label: {label}", 'info')

            try:
                # 检查攻击模型是否存在
                attack_model_path, legacy_path = self._attack_model_paths(label, attack_model_dir, attack_features)
                chosen_path = attack_model_path if os.path.exists(attack_model_path) else legacy_path
                if not os.path.exists(chosen_path):
                    self._log(f"Attack model for label {label} not found at {attack_model_path}", 'warning')
                    self._log(f"Training attack model for label {label}", 'info')
                    train_result = self._train_attack_model_if_needed(label, shadow_model_paths, merged_config, attack_model_dir, attack_features)
                    if not train_result['success']:
                        errors.append({
                            'label': label,
                            'type': 'training_failed',
                            'message': train_result['error']
                        })
                        continue
                    chosen_path = attack_model_path if os.path.exists(attack_model_path) else legacy_path

                # 加载攻击模型
                attack_model = GradientMIA(enabled_features=attack_features, num_classes=merged_config['num_classes']).to(self.device)
                if os.path.exists(chosen_path):
                    attack_model.load_state_dict(torch.load(chosen_path, map_location=self.device))
                    attack_model.eval()
                    self._log(f"Attack model for label {label} loaded successfully", 'info')
                else:
                    error_msg = f"Could not load attack model for label {label}"
                    self._log(error_msg, 'error')
                    errors.append({
                        'label': label,
                        'type': 'model_load_failed',
                        'message': error_msg
                    })
                    continue

                # 执行攻击
                target_model = self._get_fresh_model()

                # 抑制whitebox_mia_pipeline的print输出
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    client_results = whitebox_membership_inference_attack_pipeline(
                        [target_model_path],  # 单个模型路径作为列表
                        target_model,
                        label,
                        merged_config['batch_size'],
                        self.device,
                        attack_model,
                        num_clients=1,  # 单个模型评估
                        alpha=merged_config['alpha'],
                    )

                if client_results:
                    result = client_results[0]  # 获取第一个（也是唯一一个）结果
                    results['attack_results'][f'label_{label}'] = result

                    all_f_scores.append(result['f_score'])
                    all_tprs.append(result['tpr'])
                    all_fprs.append(result['fpr'])

                    self._log(f"Label {label} - F-score: {result['f_score']:.4f}, TPR: {result['tpr']:.4f}, FPR: {result['fpr']:.4f}", 'info')
                else:
                    error_msg = f"No results obtained for label {label}"
                    self._log(error_msg, 'warning')
                    errors.append({
                        'label': label,
                        'type': 'no_results',
                        'message': error_msg
                    })

            except Exception as e:
                error_msg = f"Exception during attack on label {label}: {str(e)}"
                self._log(error_msg, 'error')
                errors.append({
                    'label': label,
                    'type': 'exception',
                    'message': error_msg
                })

        # 计算总体统计
        if all_f_scores:
            results['summary'] = {
                'avg_f_score': sum(all_f_scores) / len(all_f_scores),
                'avg_tpr': sum(all_tprs) / len(all_tprs),
                'avg_fpr': sum(all_fprs) / len(all_fprs),
                'max_f_score': max(all_f_scores),
                'min_f_score': min(all_f_scores),
                'labels_evaluated': len(all_f_scores),
                'labels_requested': len(target_labels),
                'success_rate': len(all_f_scores) / len(target_labels)
            }
        else:
            results['summary'] = {
                'labels_evaluated': 0,
                'labels_requested': len(target_labels),
                'success_rate': 0.0
            }

        # 添加日志和错误信息
        results['logs'] = self.logs
        results['errors'] = errors

        if errors:
            results['status'] = 'partial_success' if all_f_scores else 'failed'

        # 保存结果
        if save_results:
            self._save_results(results, results_dir)

        return results

    def run_full_mia_evaluation(self,
                               target_model_paths: List[str],
                               shadow_model_paths: Optional[List[str]] = None,
                               retrain_attack: bool = False,
                               config: Optional[Dict] = None,
                               save_results: bool = True,
                               results_dir: str = "mia_results") -> Dict:
        """
        完整的MIA攻击评估流程（类似run_mia_pipeline）

        Args:
            target_model_paths: 目标模型路径列表
            shadow_model_paths: 影子模型路径列表
            retrain_attack: 是否重新训练攻击模型
            config: 配置参数
            save_results: 是否保存结果
            results_dir: 结果保存目录

        Returns:
            Dict: 完整的评估结果
        """
        merged_config = {**self.default_config, **(config or {})}

        # 准备影子模型路径
        if shadow_model_paths is None:
            shadow_model_paths = [
                f"shadow_model/{merged_config['alpha']}/results_cifar-10-shadow_client{i}_1000_0.0050.pt"
                for i in range(merged_config['num_clients'])
            ]

        self.logs = []  # 重置日志

        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'target_models': target_model_paths,
            'shadow_models': shadow_model_paths,
            'config': merged_config,
            'retrain_attack': retrain_attack,
            'model_accuracy': {},
            'attack_results': {},
            'summary': {},
            'logs': [],
            'errors': []
        }

        errors = []

        # 1. 评估模型准确率
        self._log("Evaluating Model Accuracy", 'info')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                accuracies = evaluate_all_clients_accuracy(
                    client_model_files=target_model_paths,
                    get_model_fn=self._get_fresh_model,
                    device=self.device,
                    batch_size=merged_config['batch_size'],
                    alpha=merged_config['alpha'],
                    num_clients=len(target_model_paths)
                )
            results['model_accuracy'] = {
                'individual': accuracies,
                'average': sum(accuracies) / len(accuracies) if accuracies else 0.0,
                'num_clients': len(accuracies)
            }
            self._log(f"Model accuracy evaluation completed. Average: {results['model_accuracy']['average']:.4f}", 'info')
        except Exception as e:
            error_msg = f"Failed to evaluate model accuracy: {str(e)}"
            self._log(error_msg, 'error')
            errors.append({
                'type': 'accuracy_evaluation_failed',
                'message': error_msg
            })
            results['model_accuracy'] = {'error': str(e)}

        # 2. 训练攻击模型（如果需要）
        if retrain_attack:
            self._log("Training Attack Models", 'info')
            for target_label in range(merged_config['num_classes']):
                self._log(f"Training attack model for label {target_label}", 'info')
                try:
                    shadow_model = self._get_fresh_model()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        train_attack_model(
                            shadow_model,
                            shadow_model_paths,
                            target_label,
                            batch_size=merged_config['batch_size'],
                            device=self.device,
                            epochs=merged_config['attack_epochs'],
                            lr=merged_config['attack_lr'],
                            num_clients=len(shadow_model_paths),
                            alpha=merged_config['alpha'],
                            attack_features=attack_features,
                            checkpoint_dir=attack_model_dir,
                            num_classes=merged_config['num_classes'],
                        )
                    self._log(f"Attack model for label {target_label} trained successfully", 'info')
                except Exception as e:
                    error_msg = f"Failed to train attack model for label {target_label}: {str(e)}"
                    self._log(error_msg, 'error')
                    errors.append({
                        'type': 'attack_training_failed',
                        'label': target_label,
                        'message': error_msg
                    })

        # 3. 执行MIA攻击评估
        self._log("Running MIA Attack Evaluation", 'info')
        all_results_by_client = [[] for _ in range(len(target_model_paths))]

        for target_label in range(merged_config['num_classes']):
            self._log(f"Evaluating attack on target label: {target_label}", 'info')

            try:
                # 加载攻击模型
                attack_model = GradientMIA(enabled_features=attack_features, num_classes=merged_config['num_classes']).to(self.device)
                attack_model_path, legacy_path = self._attack_model_paths(target_label, attack_model_dir, attack_features)
                chosen_path = attack_model_path if os.path.exists(attack_model_path) else legacy_path
                if os.path.exists(chosen_path):
                    attack_model.load_state_dict(torch.load(chosen_path, map_location=self.device))
                    attack_model.eval()
                    self._log(f"Attack model for label {target_label} loaded", 'info')
                else:
                    error_msg = f"Attack model for label {target_label} not found"
                    self._log(error_msg, 'warning')
                    errors.append({
                        'type': 'attack_model_not_found',
                        'label': target_label,
                        'message': error_msg
                    })
                    continue

                target_model = self._get_fresh_model()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    client_results = whitebox_membership_inference_attack_pipeline(
                        target_model_paths,
                        target_model,
                        target_label,
                        merged_config['batch_size'],
                        self.device,
                        attack_model,
                        num_clients=len(target_model_paths),
                        alpha=merged_config['alpha'],
                    )

                # 按客户端聚合结果
                for c_idx, res in enumerate(client_results):
                    all_results_by_client[c_idx].append(res)

                self._log(f"Attack evaluation for label {target_label} completed", 'info')

            except Exception as e:
                error_msg = f"Exception during attack evaluation for label {target_label}: {str(e)}"
                self._log(error_msg, 'error')
                errors.append({
                    'type': 'attack_evaluation_failed',
                    'label': target_label,
                    'message': error_msg
                })

        results['attack_results'] = {
            f'client_{i}': client_results
            for i, client_results in enumerate(all_results_by_client)
        }

        # 4. 计算总体统计
        self._calculate_summary_statistics(results)

        # 添加日志和错误信息
        results['logs'] = self.logs
        results['errors'] = errors

        if errors:
            results['status'] = 'partial_success' if results['attack_results'] else 'failed'

        # 5. 保存结果
        if save_results:
            self._save_results(results, results_dir)

        return results

    def quick_mia_test(self,
                      target_model_path: str,
                      test_label: int = 0,
                      config: Optional[Dict] = None) -> Dict:
        """
        快速MIA测试 - 用于验证攻击是否正常工作

        Args:
            target_model_path: 目标模型路径
            test_label: 测试标签
            config: 配置参数

        Returns:
            Dict: 测试结果
        """
        merged_config = {**self.default_config, **(config or {})}
        merged_config['num_clients'] = 1  # 快速测试只用一个客户端

        result = self.run_mia_attack(
            target_model_path=target_model_path,
            target_label=test_label,
            config=merged_config,
            save_results=False
        )

        if f'label_{test_label}' in result['attack_results']:
            test_result = result['attack_results'][f'label_{test_label}']
            return {
                'status': 'success',
                'test_label': test_label,
                'f_score': test_result['f_score'],
                'tpr': test_result['tpr'],
                'fpr': test_result['fpr'],
                'train_acc': test_result.get('train_acc', None),
                'holdout_acc': test_result.get('holdout_acc', None),
                'logs': result.get('logs', [])
            }
        else:
            return {
                'status': 'failed',
                'test_label': test_label,
                'error': 'No results obtained',
                'logs': result.get('logs', []),
                'errors': result.get('errors', [])
            }

    def _train_attack_model_if_needed(self, target_label, shadow_model_paths, config, save_dir, attack_features):
        """如果需要，训练攻击模型"""
        try:
            self._log(f"Training attack model for label {target_label}", 'info')
            shadow_model = self._get_fresh_model()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_attack_model(
                    shadow_model,
                    shadow_model_paths,
                    target_label,
                    batch_size=config['batch_size'],
                    device=self.device,
                    epochs=config['attack_epochs'],
                    lr=config['attack_lr'],
                    num_clients=len(shadow_model_paths),
                    alpha=config['alpha'],
                    attack_features=attack_features,
                    checkpoint_dir=save_dir,
                    num_classes=config['num_classes'],
                )

            checkpoint_path = os.path.join(save_dir, attack_checkpoint_name(target_label, attack_features))
            self._log(f"Attack model for label {target_label} saved to {checkpoint_path}", 'info')

            return {'success': True}

        except Exception as e:
            error_msg = f"Failed to train attack model for label {target_label}: {str(e)}"
            self._log(error_msg, 'error')
            return {'success': False, 'error': error_msg}

    def _calculate_summary_statistics(self, results):
        """计算总体统计信息"""
        all_f_scores, all_tprs, all_fprs = [], [], []

        for client_id, client_results in results['attack_results'].items():
            for label_result in client_results:
                all_f_scores.append(label_result['f_score'])
                all_tprs.append(label_result['tpr'])
                all_fprs.append(label_result['fpr'])

        if all_f_scores:
            results['summary'] = {
                'total_evaluations': len(all_f_scores),
                'avg_f_score': sum(all_f_scores) / len(all_f_scores),
                'avg_tpr': sum(all_tprs) / len(all_tprs),
                'avg_fpr': sum(all_fprs) / len(all_fprs),
                'max_f_score': max(all_f_scores),
                'min_f_score': min(all_f_scores),
                'num_clients': len(results['attack_results']),
                'num_labels': len(all_f_scores) // len(results['attack_results']) if results['attack_results'] else 0
            }

    def _save_results(self, results, results_dir):
        """保存结果到文件"""
        try:
            Path(results_dir).mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mia_results_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self._log(f"Results saved to: {filepath}", 'info')
            return filepath
        except Exception as e:
            self._log(f"Failed to save results: {str(e)}", 'error')
            return None


# ============================
# 便捷函数接口
# ============================

def run_mia_attack(target_model_path: str,
                  attack_model_dir: str = ".",
                  target_label: Optional[int] = None,
                  device: str = "auto",
                  **kwargs) -> Dict:
    """
    便捷函数：单次MIA攻击评估

    使用示例：
        result = run_mia_attack("model_epoch_100.pt", target_label=7)
        f_score = result['summary']['avg_f_score']
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interface = MIAAttackInterface(device=device)
    return interface.run_mia_attack(
        target_model_path=target_model_path,
        attack_model_dir=attack_model_dir,
        target_label=target_label,
        **kwargs
    )


def run_full_mia_evaluation(target_model_paths: List[str],
                           device: str = "auto",
                           **kwargs) -> Dict:
    """
    便捷函数：完整MIA评估

    使用示例：
        models = [f"client_{i}_model.pt" for i in range(10)]
        results = run_full_mia_evaluation(models)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interface = MIAAttackInterface(device=device)
    return interface.run_full_mia_evaluation(
        target_model_paths=target_model_paths,
        **kwargs
    )


def quick_mia_test(target_model_path: str,
                  test_label: int = 0,
                  device: str = "auto") -> Dict:
    """
    便捷函数：快速MIA测试

    使用示例：
        result = quick_mia_test("my_model.pt", test_label=5)
        if result['status'] == 'success':
            f_score = result['f_score']
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interface = MIAAttackInterface(device=device)
    return interface.quick_mia_test(
        target_model_path=target_model_path,
        test_label=test_label
    )


# ============================
# 使用示例和测试
# ============================

if __name__ == "__main__":
    # 仅作为模块导入使用，不输出任何内容
    pass
