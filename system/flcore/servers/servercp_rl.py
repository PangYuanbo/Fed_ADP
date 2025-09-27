# ============================
# servercp_rl.py
# 支持RL自适应差分隐私的FedCP服务器
# 基于原始servercp.py修改，添加RL-MIA集成
# ============================

import copy
import numpy as np
import torch
import time
from flcore.clients.clientcp_rl import clientCP_RL
from utils.data_utils import read_client_data
from threading import Thread
import os
from utils.mia_attack_wrapper import FederatedMIAEvaluator
from utils.rl_config import get_rl_dp_config


class FedCP_RL:
    """
    支持RL自适应差分隐私的FedCP服务器
    在原始FedCP基础上集成RL智能体和MIA评估反馈
    """

    def __init__(self, args, times):
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

        # ============= RL相关配置 =============
        self.enable_rl = getattr(args, 'enable_rl_dp', True)
        self.rl_config = get_rl_dp_config()
        self.rl_config.update_from_args(args)

        print(f"[Server] RL-DP System Status: {'Enabled' if self.enable_rl else 'Disabled'}")
        if self.enable_rl:
            self.rl_config.print_config()

        # 创建RL客户端
        result_dir = "results"
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, alpha=self.alpha)
            test_data = read_client_data(self.dataset, i, is_train=False, alpha=self.alpha)

            if self.enable_rl:
                client = clientCP_RL(args,
                                   id=i,
                                   train_samples=len(train_data),
                                   test_samples=len(test_data))
            else:
                # 如果不启用RL，回退到原始客户端
                from flcore.clients.clientcp import clientCP
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
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.head = None
        self.cs = None

        # ============= MIA评估器初始化 =============
        self.mia_evaluator = None
        self.mia_evaluation_interval = 10  # MIA评估间隔
        self.enable_mia = getattr(args, 'enable_mia', True)
        self.mia_results_history = []

        # RL专用MIA记录 (用于提供更频繁的反馈)
        self.rl_mia_history = {}  # client_id -> [mia_f_scores]

        if self.enable_mia:
            try:
                print(f"[Server] Initializing MIA evaluator for RL feedback...")
                print(f"[Server] Attack model dir: Membership_Inference_Attack")
                print(f"[Server] Device: {args.device}")
                print(f"[Server] Num classes: {args.num_classes}")

                self.mia_evaluator = FederatedMIAEvaluator(
                    attack_model_dir="Membership_Inference_Attack",
                    num_classes=args.num_classes,
                    device=args.device,
                    batch_size=1,
                    alpha=args.alpha,
                    max_samples_per_client=50  # 限制样本数量以加速评估
                )
                print(f"[Server] Federated MIA evaluator initialized successfully")
                print(f"[Server] MIA evaluation will run every {self.mia_evaluation_interval} rounds")
            except ImportError as e:
                print(f"[Server] MIA Import Error: {e}")
                print(f"[Server] This is likely due to missing PyTorch or MIA module dependencies")
                print(f"[Server] Continuing without MIA evaluation...")
                self.mia_evaluator = None
                self.enable_mia = False
            except Exception as e:
                print(f"[Server] Warning: Could not initialize MIA evaluator: {e}")
                print(f"[Server] Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                self.mia_evaluator = None
                self.enable_mia = False

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.set_parameters(self.global_modules)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_modules.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_modules = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_modules.parameters():
            param.data = torch.zeros_like(param.data)
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics_before(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        for c in self.clients:
            ct, ns, auc = c.test_metrics_before()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            filename = f"results_{self.dataset}_{c.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "a") as f:
                f.write(f"Round {c.round}: ACC = {ct*1.0/ns:.4f}\n")

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def test_metrics_after(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        print("after noise_acc")
        result_dir = "results_after"
        os.makedirs(result_dir, exist_ok=True)

        for c in self.clients:
            ct, ns, auc = c.test_metrics_before()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            filename = f"results_{self.dataset}_{c.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "a") as f:
                f.write(f"Round {c.round}: ACC = {ct * 1.0 / ns:.4f}\n")

        for c in self.clients:
            c.test_metrics_after()

    def evaluate(self, acc=None):
        stats = self.test_metrics_before()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

    def update_client_mia_scores(self, mia_results):
        """
        将MIA评估结果更新到客户端，为RL提供反馈

        Args:
            mia_results: MIA评估结果字典
        """
        if not self.enable_rl or mia_results['status'] != 'success':
            return

        for client_id, client_result in mia_results['clients'].items():
            if client_result['status'] == 'success' and client_id < len(self.clients):
                mia_f_score = client_result['summary']['avg_f_score']

                # 更新客户端MIA分数
                if hasattr(self.clients[client_id], 'update_mia_score'):
                    self.clients[client_id].update_mia_score(mia_f_score)

                # 记录RL MIA历史
                if client_id not in self.rl_mia_history:
                    self.rl_mia_history[client_id] = []
                self.rl_mia_history[client_id].append(mia_f_score)

                print(f"[Server] Updated Client {client_id} MIA F-score: {mia_f_score:.4f}")

    def save_rl_summary(self, round_num):
        """保存RL训练摘要"""
        if not self.enable_rl:
            return

        summary_dir = "rl_summaries"
        os.makedirs(summary_dir, exist_ok=True)

        summary = {
            'round': round_num,
            'dataset': self.dataset,
            'client_summaries': {},
            'global_stats': {
                'avg_accuracy': self.rs_test_acc[-1] if self.rs_test_acc else 0,
                'rl_enabled_clients': 0,
                'total_clients': len(self.clients)
            }
        }

        for client in self.clients:
            if hasattr(client, 'rl_manager') and client.rl_manager:
                client_summary = client.rl_manager.get_summary()
                summary['client_summaries'][client.id] = client_summary
                if 'error' not in client_summary:
                    summary['global_stats']['rl_enabled_clients'] += 1

        # 保存摘要
        import json
        summary_file = os.path.join(summary_dir, f"rl_summary_round_{round_num}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[Server] RL summary saved to {summary_file}")

    def train(self, args):
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        if args.difference_privacy:
            if self.enable_rl:
                filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_dp_rl.txt"
            else:
                filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_dp.txt"
        else:
            filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"

        file_path = os.path.join(result_dir, filename)

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # 打印轮次信息
            if i > 0:
                print(f"\n-------------Round number: {i}-------------")

            # 评估准确率
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

            # ============= RL增强的MIA评估 =============
            # 为RL提供更频繁的MIA反馈
            if self.enable_mia and self.mia_evaluator and i > 0 and i % self.mia_evaluation_interval == 0:
                print(f"\n[Server] Running MIA evaluation for RL feedback (round {i})...")
                try:
                    mia_results = self.mia_evaluator.evaluate_all_clients(
                        self.clients,
                        round_num=i,
                        dataset_name=args.dataset,
                        save_results=True,
                        results_dir=f"mia_results/{args.dataset}_alpha{args.alpha}_rl"
                    )

                    self.mia_results_history.append(mia_results)

                    # 将MIA结果反馈给RL客户端
                    if self.enable_rl:
                        self.update_client_mia_scores(mia_results)

                    # 记录MIA结果
                    if mia_results['status'] == 'success' and 'summary' in mia_results:
                        summary = mia_results['summary']
                        with open(file_path, "a") as f:
                            f.write(f"    MIA Round {i} - Avg F-score: {summary['avg_f_score']:.4f}, "
                                   f"Avg TPR: {summary['avg_tpr']:.4f}, Avg FPR: {summary['avg_fpr']:.4f}, "
                                   f"High Risk Clients: {summary['high_risk_clients']}/{summary['total_clients']}\n")

                        print(f"[Server] MIA Results - Avg F-score: {summary['avg_f_score']:.4f}, "
                              f"High Risk: {summary['high_risk_clients']}/{summary['total_clients']} clients")
                    else:
                        print(f"[Server] MIA evaluation failed: {mia_results.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"[Server] MIA evaluation error: {e}")

            # 客户端训练
            for client in self.selected_clients:
                client.round = i
                client.train_cs_model(i, args)

            self.test_metrics_after()
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            # ============= RL相关记录和保存 =============
            if self.enable_rl and i > 0:
                # 定期保存RL摘要
                if i % 50 == 0:
                    self.save_rl_summary(i)

                # 打印RL统计信息
                if i % 100 == 0:
                    print(f"\n[Server] RL Training Statistics (Round {i}):")
                    rl_enabled = sum(1 for c in self.clients if hasattr(c, 'enable_rl') and c.enable_rl)
                    print(f"  RL-enabled clients: {rl_enabled}/{len(self.clients)}")

                    # 打印平均MIA分数
                    if self.rl_mia_history:
                        avg_mia_scores = []
                        for client_id, scores in self.rl_mia_history.items():
                            if scores:
                                avg_mia_scores.append(scores[-1])
                        if avg_mia_scores:
                            print(f"  Average MIA F-score: {np.mean(avg_mia_scores):.4f}")

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        # 训练结束后的总结
        self.save_models(args)
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # ============= RL训练总结 =============
        if self.enable_rl:
            print("\n========== RL-DP Training Summary ==========")

            # 最终RL摘要
            final_summary = {
                'total_rounds': self.global_rounds,
                'best_accuracy': max(self.rs_test_acc) if self.rs_test_acc else 0,
                'final_accuracy': self.rs_test_acc[-1] if self.rs_test_acc else 0,
                'rl_enabled_clients': sum(1 for c in self.clients if hasattr(c, 'enable_rl') and c.enable_rl),
                'total_clients': len(self.clients),
                'mia_evaluations': len(self.mia_results_history)
            }

            print(f"Total Rounds: {final_summary['total_rounds']}")
            print(f"Best Accuracy: {final_summary['best_accuracy']:.4f}")
            print(f"Final Accuracy: {final_summary['final_accuracy']:.4f}")
            print(f"RL-enabled Clients: {final_summary['rl_enabled_clients']}/{final_summary['total_clients']}")
            print(f"MIA Evaluations: {final_summary['mia_evaluations']}")

            # 保存最终摘要
            self.save_rl_summary(self.global_rounds)

            print("============================================\n")

        # 打印MIA评估总结
        if self.enable_mia and self.mia_evaluator and self.mia_results_history:
            print("\n========== MIA Evaluation Summary ==========")

            # 计算汇总统计
            all_f_scores = []
            all_rounds = []
            for result in self.mia_results_history:
                if result['status'] == 'success' and 'summary' in result:
                    all_f_scores.append(result['summary']['avg_f_score'])
                    all_rounds.append(result['round'])

            if all_f_scores:
                print(f"Total MIA evaluation rounds: {len(all_f_scores)}")
                print(f"Evaluated rounds: {all_rounds}")
                print(f"Final F-score: {all_f_scores[-1]:.4f}")
                print(f"Average F-score: {np.mean(all_f_scores):.4f}")
                print(f"Max F-score: {np.max(all_f_scores):.4f}")
                print(f"Min F-score: {np.min(all_f_scores):.4f}")
                print(f"F-score std: {np.std(all_f_scores):.4f}")

                # 生成MIA趋势可视化
                try:
                    from utils.mia_visualizer import MIAVisualizer
                    visualizer = MIAVisualizer()

                    plot_path = visualizer.plot_f_score_trends(
                        self.mia_results_history,
                        save_dir=f"mia_results/{args.dataset}_alpha{args.alpha}_rl",
                        dataset_name=f"{args.dataset}_RL",
                        alpha=args.alpha
                    )
                    print(f"MIA trend visualization saved to: {plot_path}")

                except Exception as e:
                    print(f"Warning: Could not generate MIA visualization: {e}")
            else:
                print("No successful MIA evaluations recorded.")

            print("============================================\n")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def save_models(self, args):
        save_dir = f"pretrain/{args.dataset}/{args.alpha:.2f}"
        if self.enable_rl:
            save_dir += "_rl"
        os.makedirs(save_dir, exist_ok=True)

        for c in self.clients:
            filename = f"results_client{c.id}_{args.global_rounds}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(c.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        filename = f"results_client{self.num_clients}_{args.global_rounds}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(self.global_modules.state_dict(), save_path)
        print(f"Global model saved to {save_path}")