# ============================
# server_selective.py
# 继承FedCP，添加稀疏聚合逻辑
# ============================

import sys
import os
import copy
import torch

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from flcore.servers.servercp import FedCP
from flcore.clients.clientcp import clientCP
from utils.data_utils import read_client_data

# 导入ClientSelective（避免相对导入问题）
try:
    from .client_selective import ClientSelective
except ImportError:
    # 当作为脚本运行时，使用绝对导入
    from client_selective import ClientSelective


class ServerSelective(FedCP):
    """
    选择性上传服务器（支持全局MIA评估）

    功能：
    1. 根据args.enable_selective_upload选择客户端类型：
       - True: 使用ClientSelective（选择性上传）
       - False: 使用普通clientCP（标准FedAvg）
    2. 实现稀疏聚合（仅在enable_selective_upload=True时）
    3. 始终使用全局MIA评估（评估global model）
    """

    def __init__(self, args, times):
        """
        初始化服务器，根据参数选择客户端类型

        Args:
            args: 训练参数
            times: 运行次数
        """
        # 暂存原始参数
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_modules = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.alpha = args.alpha
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        # 不使用RL客户端（保持简单）
        self.use_rl_clients = False

        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        # 🔑 根据参数选择客户端类型
        self.enable_selective_upload = getattr(args, 'enable_selective_upload', False)
        client_type_name = "ClientSelective" if self.enable_selective_upload else "clientCP"

        print(f"[Server] Creating {self.num_clients} {client_type_name} clients...")
        print(f"[Server] Selective Upload: {'Enabled' if self.enable_selective_upload else 'Disabled'}")

        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, alpha=self.alpha)
            test_data = read_client_data(self.dataset, i, is_train=False, alpha=self.alpha)

            # 🔑 根据参数选择客户端类型
            if self.enable_selective_upload:
                client = ClientSelective(args,
                                        id=i,
                                        train_samples=len(train_data),
                                        test_samples=len(test_data))
            else:
                client = clientCP(args,
                                 id=i,
                                 train_samples=len(train_data),
                                 test_samples=len(test_data))

            self.clients.append(client)
            filename = f"results_{self.dataset}_{client.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "w") as f:
                pass

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Finished creating server and clients ({client_type_name}).")

        # 初始化全局模型MIA评估器（如果启用）
        self.enable_mia = getattr(args, 'enable_mia', False)
        self.mia_attack_models = {}
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size

        if self.enable_mia:
            try:
                # 加载预训练的MIA攻击模型
                from Membership_Inference_Attack.model import GradientMIA
                attack_model_dir = os.path.join(os.path.dirname(__file__), '..', 'Membership_Inference_Attack')

                for label in range(self.num_classes):
                    model_path = os.path.join(attack_model_dir, f"attack_model{label}.pth")
                    if os.path.exists(model_path):
                        attack_model = GradientMIA().to(self.device)
                        attack_model.load_state_dict(torch.load(model_path, map_location=self.device))
                        attack_model.eval()
                        self.mia_attack_models[label] = attack_model

                print(f"[Server] Loaded {len(self.mia_attack_models)}/{self.num_classes} MIA attack models")
                print(f"[Server] Global Model MIA Evaluator initialized (魔改版)")
            except Exception as e:
                print(f"[Server] Warning: Failed to initialize MIA evaluator: {e}")
                self.enable_mia = False

    def aggregate_parameters(self):
        """
        聚合方法（根据enable_selective_upload选择策略）

        - enable_selective_upload=True: 稀疏聚合（只聚合上传的参数）
        - enable_selective_upload=False: 标准FedAvg（调用父类方法）
        """
        assert (len(self.uploaded_models) > 0)

        # 🔑 如果未启用选择性上传，使用标准FedAvg聚合
        if not self.enable_selective_upload:
            super().aggregate_parameters()
            return

        # 保存旧的全局模型（用于保留未更新的参数）
        old_global_model = copy.deepcopy(self.global_modules)

        # 初始化新的全局模型为零
        self.global_modules = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_modules.parameters():
            param.data = torch.zeros_like(param.data)

        # 获取所有客户端的kept_mask
        client_masks = []
        for client in self.selected_clients:
            if hasattr(client, 'kept_mask') and client.kept_mask:
                client_masks.append(client.kept_mask)
            else:
                # 如果客户端没有kept_mask（可能是早期轮次），所有参数都视为上传
                dummy_mask = {name: torch.zeros_like(param, dtype=torch.bool)
                             for name, param in client.model.named_parameters()}
                client_masks.append(dummy_mask)

        # 对每个参数进行稀疏聚合
        for param_name, global_param in self.global_modules.named_parameters():
            # 收集所有上传了此参数的客户端
            valid_weights = []
            valid_params = []

            for i, (weight, client_model, client_mask) in enumerate(
                    zip(self.uploaded_weights, self.uploaded_models, client_masks)):

                # 检查该客户端是否上传了此参数
                if param_name in client_mask:
                    mask = client_mask[param_name]

                    # 如果整个参数都没有被保留（mask全False），则该客户端上传了此参数
                    if not mask.all():
                        # 部分或全部上传
                        # 对于简化，我们使用参数级别的判断：如果有任何元素被上传，就使用整个参数
                        # 更精细的实现可以逐元素聚合，但会更复杂
                        if not mask.any():
                            # 完全上传（没有保留任何元素）
                            valid_weights.append(weight)
                            valid_params.append(client_model.state_dict()[param_name])
                        else:
                            # 部分保留：我们需要逐元素聚合
                            # 为简化起见，如果超过50%的参数被保留，就不使用该客户端的这个参数
                            kept_ratio = mask.float().mean().item()
                            if kept_ratio < 0.5:
                                valid_weights.append(weight)
                                valid_params.append(client_model.state_dict()[param_name])
                    # 如果mask.all()为True，说明整个参数都被保留，不参与聚合
                else:
                    # 没有mask记录，说明全部上传
                    valid_weights.append(weight)
                    valid_params.append(client_model.state_dict()[param_name])

            # 聚合
            if len(valid_params) > 0:
                # 有客户端上传了此参数，进行加权平均
                total_weight = sum(valid_weights)
                normalized_weights = [w / total_weight for w in valid_weights]

                aggregated_param = torch.zeros_like(global_param.data)
                for w, param in zip(normalized_weights, valid_params):
                    aggregated_param += w * param

                global_param.data = aggregated_param
            else:
                # 所有客户端都保留了此参数，维持原值
                global_param.data = old_global_model.state_dict()[param_name].clone()

        # 打印聚合统计信息（每10轮）
        if hasattr(self, 'current_round') and self.current_round % 10 == 0:
            self._print_aggregation_stats(client_masks)

    def _print_aggregation_stats(self, client_masks):
        """打印聚合统计信息"""
        param_upload_counts = {}

        for param_name in self.global_modules.state_dict().keys():
            upload_count = 0
            for client_mask in client_masks:
                if param_name in client_mask:
                    mask = client_mask[param_name]
                    if not mask.all():  # 有上传
                        upload_count += 1
                else:
                    upload_count += 1  # 没有mask记录，算作上传

            param_upload_counts[param_name] = upload_count

        # 只打印有保留的参数
        print(f"\n[Server Round {self.current_round}] Parameter Upload Statistics:")
        for param_name, count in param_upload_counts.items():
            if count < len(self.selected_clients):
                print(f"  {param_name}: {count}/{len(self.selected_clients)} clients uploaded")

    def train(self, args):
        """
        重写训练方法，添加轮次追踪

        主要是为了在aggregate_parameters中能够访问当前轮次
        """
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        # 🔑 根据enable_selective_upload选择文件名后缀
        mode_name = "selective" if self.enable_selective_upload else "standard"
        if args.difference_privacy:
            filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_{mode_name}_global_mia_dp.txt"
        else:
            filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_{mode_name}_global_mia.txt"
        file_path = os.path.join(result_dir, filename)

        for i in range(self.global_rounds + 1):
            import time
            s_t = time.time()
            self.current_round = i  # 记录当前轮次
            self.selected_clients = self.select_clients()

            # Print round info
            if i > 0:
                print(f"\n{'='*60}")
                if self.enable_selective_upload:
                    print(f"Round {i} - Selective Upload (enabled from round {self.clients[0].selective_upload_round})")
                else:
                    print(f"Round {i} - Standard FedAvg (Global MIA only)")
                print(f"{'='*60}")

            # Evaluate accuracy
            if i % self.eval_gap == 0:
                if i > 0:
                    print("\nEvaluate before local training")
                self.evaluate()

                with open(file_path, "a") as f:
                    f.write(f"Round {i}: ACC = {self.rs_test_acc[-1]:.4f}\n")

                if i > 0:
                    print(f"Round {i} - Test Accuracy: {self.rs_test_acc[-1]:.4f}")
            elif i > 0:
                print(f"Round {i} - Training...")

            # Client training
            for client in self.selected_clients:
                client.train_cs_model(i, args)

            # Prepare upload models (apply selective upload AFTER MIA evaluation)
            # This ensures MIA uses the original trained model state
            # 🔑 只在启用选择性上传时调用
            if self.enable_selective_upload:
                for client in self.selected_clients:
                    if hasattr(client, 'prepare_upload_model'):
                        client.prepare_upload_model(i)

            # Aggregate models
            self.receive_models()
            self.aggregate_parameters()

            # Evaluate global model accuracy (EVERY round)
            if i > 0:
                global_accuracy = self.evaluate_global_model_accuracy()
                print(f"\n[Global Model] Round {i} Accuracy on All Test Data: {global_accuracy:.4f} ({global_accuracy*100:.2f}%)")

                # 保存全局准确率到文件
                global_acc_file = os.path.join(result_dir, f"{filename.replace('.txt', '_global_acc.txt')}")
                with open(global_acc_file, "a") as f:
                    f.write(f"Round {i}: Global_ACC = {global_accuracy:.4f}\n")

            # Global Model MIA evaluation (every 10 rounds, AFTER aggregation)
            # 魔改版：评估全局模型而非客户端模型
            if self.enable_mia and i % 10 == 0 and i > 0:
                mia_results = self.evaluate_global_model_mia(i)
                if mia_results['status'] == 'success':
                    # 可以保存结果到文件
                    mia_dir = "mia_results_global"
                    os.makedirs(mia_dir, exist_ok=True)
                    import json
                    with open(os.path.join(mia_dir, f"round_{i}.json"), "w") as f:
                        json.dump(mia_results, f, indent=2)

            # Send global model to clients (will trigger smoothing for selective clients)
            self.send_models()

            # Print timing
            print(f"Round {i} completed in {time.time() - s_t:.2f}s")

        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)

        # Save final models (to separate directory)
        self.save_models(args)

    def save_models(self, args):
        """
        重写save_models，保存到独立目录
        """
        # 🔑 根据enable_selective_upload选择保存目录
        mode_name = "selective" if self.enable_selective_upload else "standard"
        save_dir = f"pretrain_{mode_name}_global_mia/{args.dataset}/{args.alpha:.2f}"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[Global MIA Lab] Saving models to: {save_dir}")

        # 保存每个客户端的模型
        for c in self.clients:
            filename = f"client{c.id}_round{args.global_rounds}_{mode_name}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(c.model.state_dict(), save_path)
            print(f"  ✓ Client {c.id} model saved")

        # 保存全局模型
        filename = f"global_model_round{args.global_rounds}_{mode_name}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(self.global_modules.state_dict(), save_path)
        print(f"  ✓ Global model saved")

        print(f"[Global MIA Lab] All models saved successfully!")


    def evaluate_global_model_accuracy(self):
        """
        评估全局模型在全局测试集上的准确率

        Returns:
            float: 准确率
        """
        self.global_modules.eval()

        all_test_x = []
        all_test_y = []

        # 收集所有客户端的测试数据
        for client in self.clients:
            test_loader = client.load_test_data(batch_size=self.batch_size)
            for x, y in test_loader:
                all_test_x.append(x.cpu())
                all_test_y.append(y.cpu())

        # 合并数据
        global_test_x = torch.cat(all_test_x, dim=0)
        global_test_y = torch.cat(all_test_y, dim=0)

        # 创建DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(global_test_x, global_test_y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 测试
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.global_modules(x)
                _, predicted = torch.max(output.data, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total
        self.global_modules.train()

        return accuracy

    def evaluate_global_model_mia(self, round_num):
        """
        评估全局模型的MIA攻击成功率（魔改版 - 复用原始MIA实现）

        策略：
        1. 创建一个"全局客户端"包装器，包含global model和合并的数据
        2. 直接调用原始的FederatedMIAEvaluator.evaluate_client_mia()
        3. 完全复用已验证的MIA实现，只改数据源和攻击对象

        Args:
            round_num: 当前轮次

        Returns:
            dict: MIA评估结果
        """
        if not self.enable_mia or len(self.mia_attack_models) == 0:
            return {'status': 'disabled'}

        print(f"\n{'='*60}")
        print(f"Global Model MIA Evaluation - Round {round_num}")
        print(f"{'='*60}")

        try:
            # 导入原始MIA评估器
            from utils.mia_attack_wrapper import FederatedMIAEvaluator

            # 创建"全局客户端"包装器
            global_client = self._create_global_client_wrapper()

            # 初始化MIA评估器（如果还没有）
            if not hasattr(self, 'mia_evaluator'):
                self.mia_evaluator = FederatedMIAEvaluator(
                    attack_model_dir=os.path.join(os.path.dirname(__file__), '..', 'Membership_Inference_Attack'),
                    num_classes=self.num_classes,
                    device=self.device,
                    batch_size=1,  # 🔑 强制使用batch_size=1（MIA模型要求）
                    alpha=self.alpha,
                    use_global_test=False,  # 我们已经在wrapper中处理了全局测试数据
                    use_gpu_optimization=True
                )

            # 🔑 直接调用原始MIA评估方法，传入全局客户端
            result = self.mia_evaluator.evaluate_client_mia(
                client=global_client,
                target_labels=None,  # 评估所有标签
                global_test_loader=None  # 已经在wrapper中处理
            )

            # 打印结果
            if result['status'] == 'success':
                summary = result['summary']
                print(f"\n[Global MIA Results]")
                print(f"  Average F-score: {summary['avg_f_score']:.4f}")
                print(f"  Average Model Accuracy (per label): {summary.get('avg_model_accuracy', 0.0):.4f}")
                print(f"  Average MIA Attack Accuracy: {summary['avg_accuracy']:.4f}")
                print(f"  Average TPR: {summary['avg_tpr']:.4f}")
                print(f"  Average FPR: {summary['avg_fpr']:.4f}")
                print(f"  Labels evaluated: {summary['labels_evaluated']}")
                print(f"{'='*60}\n")

                # 转换为标准格式
                return {
                    'status': 'success',
                    'round': round_num,
                    'f_score': summary['avg_f_score'],
                    'accuracy': summary['avg_accuracy'],
                    'tpr': summary['avg_tpr'],
                    'fpr': summary['avg_fpr'],
                    'labels_evaluated': summary['labels_evaluated']
                }
            else:
                print(f"[Global MIA] Evaluation failed: {result.get('error', 'Unknown error')}")
                return {'status': 'failed', 'message': result.get('error', 'Unknown error')}

        except Exception as e:
            print(f"[Global MIA] Error: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def _create_global_client_wrapper(self):
        """
        创建一个"全局客户端"包装器，用于适配原始MIA接口

        包装器包含：
        - id: -1 (表示全局)
        - model: global_modules
        - load_train_data(): 返回合并的所有客户端训练数据
        - load_test_data(): 返回合并的所有客户端测试数据
        """
        class GlobalClientWrapper:
            def __init__(self, server):
                self.id = -1  # 全局客户端标识
                self.model = server.global_modules
                self.server = server
                self.batch_size = server.batch_size

            def load_train_data(self, batch_size=None):
                """返回合并的所有客户端训练数据"""
                # 🔑 强制使用batch_size=1（MIA模型要求）
                batch_size = 1

                all_train_x = []
                all_train_y = []

                for client in self.server.clients:
                    train_loader = client.load_train_data(batch_size=batch_size)
                    for x, y in train_loader:
                        all_train_x.append(x.cpu())
                        all_train_y.append(y.cpu())

                # 合并所有数据
                global_train_x = torch.cat(all_train_x, dim=0)
                global_train_y = torch.cat(all_train_y, dim=0)

                print(f"[Global Client Wrapper] Loaded {len(global_train_x)} training samples (member)")

                # 创建DataLoader
                from torch.utils.data import TensorDataset, DataLoader
                train_dataset = TensorDataset(global_train_x, global_train_y)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                return train_loader

            def load_test_data(self, batch_size=None):
                """返回合并的所有客户端测试数据"""
                # 🔑 强制使用batch_size=1（MIA模型要求）
                batch_size = 1

                all_test_x = []
                all_test_y = []

                for client in self.server.clients:
                    test_loader = client.load_test_data(batch_size=batch_size)
                    for x, y in test_loader:
                        all_test_x.append(x.cpu())
                        all_test_y.append(y.cpu())

                # 合并所有数据
                global_test_x = torch.cat(all_test_x, dim=0)
                global_test_y = torch.cat(all_test_y, dim=0)

                print(f"[Global Client Wrapper] Loaded {len(global_test_x)} test samples (non-member)")

                # 创建DataLoader
                from torch.utils.data import TensorDataset, DataLoader
                test_dataset = TensorDataset(global_test_x, global_test_y)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                return test_loader

        return GlobalClientWrapper(self)
