import copy
import numpy as np
import torch
import time
from flcore.clients.clientcp import *

# 尝试导入RL客户端（可选）
try:
    from flcore.clients.clientcp_rl import clientCP_RL
    RL_CLIENT_AVAILABLE = True
except ImportError:
    RL_CLIENT_AVAILABLE = False

# 尝试导入连续RL客户端（可选）
try:
    from flcore.clients.clientcp_continuous_rl import clientCP_ContinuousRL
    CONTINUOUS_RL_CLIENT_AVAILABLE = True
except ImportError:
    CONTINUOUS_RL_CLIENT_AVAILABLE = False
from utils.data_utils import read_client_data
from threading import Thread
import os
from utils.mia_attack_wrapper import FederatedMIAEvaluator


class FedCP:
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

        # 检查使用哪种RL客户端
        use_continuous_rl = getattr(args, 'use_continuous_rl', False)
        self.use_rl_clients = (getattr(args, 'enable_rl_dp', False) and
                              args.difference_privacy)

        result_dir = "results"

        # 只在使用RL时打印RL客户端信息
        if self.use_rl_clients:
            if use_continuous_rl and CONTINUOUS_RL_CLIENT_AVAILABLE:
                print(f"[Server] Using Continuous RL clients (DDPG)")
            elif RL_CLIENT_AVAILABLE:
                print(f"[Server] Using Discrete RL clients (DQN)")
            else:
                print(f"[Server] RL requested but not available, using standard clients")

        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True,alpha=self.alpha)
            test_data = read_client_data(self.dataset, i, is_train=False,alpha=self.alpha)

            # 根据配置创建相应的客户端
            if self.use_rl_clients and use_continuous_rl and CONTINUOUS_RL_CLIENT_AVAILABLE:
                # 使用连续RL客户端（DDPG）
                client = clientCP_ContinuousRL(args,
                                              id=i,
                                              train_samples=len(train_data),
                                              test_samples=len(test_data))
            elif self.use_rl_clients and RL_CLIENT_AVAILABLE:
                # 使用离散RL客户端（DQN）
                client = clientCP_RL(args,
                                   id=i,
                                   train_samples=len(train_data),
                                   test_samples=len(test_data))
            else:
                # 使用标准客户端
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
        
        # Initialize MIA evaluator for federated learning
        self.mia_evaluator = None
        self.mia_evaluation_interval = 10  # Evaluate MIA every 10 rounds
        self.enable_mia = getattr(args, 'enable_mia', True)  # Default enabled for testing
        self.mia_results_history = []  # Store MIA results over time

        # 获取实验ID（用于文件隔离）
        self.exp_id = os.environ.get('MIA_EXP_ID', None)
        self.mia_results_base_dir = os.environ.get('MIA_RESULTS_DIR', None)

        if self.enable_mia:
            try:
                print(f"[Server] Initializing MIA evaluator...")
                if self.exp_id:
                    print(f"[Server] Experiment ID: {self.exp_id}")
                    print(f"[Server] MIA Results Dir: {self.mia_results_base_dir}")
                print(f"[Server] Attack model dir: Membership_Inference_Attack")
                print(f"[Server] Device: {args.device}")
                print(f"[Server] Num classes: {args.num_classes}")

                self.mia_evaluator = FederatedMIAEvaluator(
                    attack_model_dir="Membership_Inference_Attack",
                    num_classes=args.num_classes,
                    device=args.device,
                    batch_size=1,  # 与原版MIA一致
                    alpha=args.alpha,
                    max_samples_per_client=None,  # None表示使用完整数据集
                    use_gpu_optimization=True  # 🔑 强制启用GPU优化，所有计算图在GPU上
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

        # 计算训练准确率
        print("\n[Per-Client Train Accuracy]")
        for c in self.clients:
            train_ct, train_ns = c.train_metrics()
            train_acc = train_ct * 1.0 / train_ns if train_ns > 0 else 0.0
            print(f'  Client {c.id}: Train Acc = {train_acc:.4f} ({train_ct}/{train_ns})')

        # 计算测试准确率
        print("\n[Per-Client Test Accuracy]")
        for c in self.clients:
            ct, ns, auc = c.test_metrics_before()
            acc = ct * 1.0 / ns
            print(f'  Client {c.id}: Test Acc = {acc:.4f} ({ct}/{ns}), AUC = {auc:.4f}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            filename=f"results_{self.dataset}_{c.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "a") as f:
                f.write(f"Round {c.round}: Test ACC = {acc:.4f}\n")



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
            # print(f'Client {c.id}: Acc: {ct * 1.0 / ns}, AUC: {auc}')
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            filename = f"results_{self.dataset}_{c.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "a") as f:
                f.write(f"Round {c.round}: ACC = {ct * 1.0 / ns:.4f}\n")
        for c in self.clients:
            c.test_metrics_after()


        return
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


    def train(self,args):
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        if args.difference_privacy:
            filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_dp.txt"
        else:
            filename = f"results_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"
        file_path = os.path.join(result_dir, filename)
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Print round info every round for better visibility
            if i > 0:  # Skip round 0 for cleaner output
                print(f"\n-------------Round number: {i}-------------")

            # Evaluate accuracy based on eval_gap setting
            if i%self.eval_gap == 0:
                if i > 0:  # Don't print "Evaluate before local training" for round 0
                    print("\nEvaluate before local training")
                self.evaluate()

                with open(file_path, "a") as f:
                    f.write(f"Round {i}: ACC = {self.rs_test_acc[-1]:.4f}\n")

                # Print accuracy every eval_gap for console visibility
                if i > 0:
                    print(f"Round {i} - Test Accuracy: {self.rs_test_acc[-1]:.4f}")
            elif i > 0:
                # For non-eval rounds, still show progress but don't evaluate
                print(f"Round {i} - Training...")

            # Evaluate MIA every 10 rounds
            if self.enable_mia and self.mia_evaluator and i > 0 and i % self.mia_evaluation_interval == 0:
                print(f"\n[Server] Running MIA evaluation for round {i}...")
                try:
                    # 使用唯一的MIA结果目录（如果设置了实验ID）
                    if self.mia_results_base_dir:
                        mia_dir = self.mia_results_base_dir
                    else:
                        mia_dir = f"mia_results/{args.dataset}_alpha{args.alpha}"

                    mia_results = self.mia_evaluator.evaluate_all_clients(
                        self.clients,
                        round_num=i,
                        dataset_name=args.dataset,
                        save_results=True,
                        results_dir=mia_dir
                    )

                    self.mia_results_history.append(mia_results)

                    # Log MIA results alongside accuracy
                    if mia_results['status'] == 'success' and 'summary' in mia_results:
                        summary = mia_results['summary']
                        with open(file_path, "a") as f:
                            f.write(f"    MIA Round {i} - Avg F-score: {summary['avg_f_score']:.4f}, "
                                   f"Avg TPR: {summary['avg_tpr']:.4f}, Avg FPR: {summary['avg_fpr']:.4f}, "
                                   f"High Risk Clients: {summary['high_risk_clients']}/{summary['total_clients']}\n")

                        print(f"[Server] MIA Results - Avg F-score: {summary['avg_f_score']:.4f}, "
                              f"High Risk: {summary['high_risk_clients']}/{summary['total_clients']} clients")

                        # 如果使用RL客户端，更新MIA分数以供RL学习
                        if self.use_rl_clients:
                            self.update_rl_clients_mia_scores(mia_results)
                    else:
                        print(f"[Server] MIA evaluation failed: {mia_results.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"[Server] MIA evaluation error: {e}")


            for client in self.selected_clients:
                client.round= i
                client.train_cs_model(i,args)

            self.test_metrics_after()
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()


            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])
        self.save_models(args)
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        # Print MIA evaluation summary and generate visualizations
        if self.enable_mia and self.mia_evaluator and self.mia_results_history:
            print("\n========== MIA Evaluation Summary ==========")

            # Calculate summary statistics
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

                # Generate MIA visualizations
                try:
                    from utils.mia_visualizer import MIAVisualizer
                    visualizer = MIAVisualizer()
                    results_dir = f"mia_results/{args.dataset}_alpha{args.alpha}"

                    # Generate main F-score trend plot (4 subplots)
                    plot_path = visualizer.plot_f_score_trends(
                        self.mia_results_history,
                        save_dir=results_dir,
                        dataset_name=args.dataset,
                        alpha=args.alpha
                    )
                    print(f"[Server] MIA trend visualization saved to: {plot_path}")

                    # Generate per-client detailed plot
                    client_plot_path = visualizer.plot_per_client_f_scores(
                        self.mia_results_history,
                        save_dir=results_dir,
                        dataset_name=args.dataset,
                        alpha=args.alpha
                    )
                    print(f"[Server] Per-client MIA plot saved to: {client_plot_path}")

                    # Generate summary report
                    report_path = visualizer.create_summary_report(
                        self.mia_results_history,
                        save_dir=results_dir,
                        dataset_name=args.dataset,
                        alpha=args.alpha
                    )
                    print(f"[Server] MIA summary report saved to: {report_path}")

                    # Export CSV for external analysis
                    csv_path = self.mia_evaluator.export_client_f_scores_to_csv(
                        results_dir=results_dir,
                        output_filename="client_f_scores.csv"
                    )
                    if csv_path:
                        print(f"[Server] Client F-scores CSV exported to: {csv_path}")

                except Exception as e:
                    print(f"[Server] Warning: Could not generate MIA visualizations: {e}")
                    import traceback
                    traceback.print_exc()

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
        os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

        for c in self.clients:
            filename = f"results_client{c.id}_{args.global_rounds}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(c.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        filename = f"results_client{self.num_clients}_{args.global_rounds}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(self.global_modules.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def update_rl_clients_mia_scores(self, mia_results):
        """
        将MIA评估结果更新到RL客户端，为RL提供反馈
        仅在使用RL客户端时调用

        Args:
            mia_results: MIA评估结果字典
        """
        if not self.use_rl_clients or mia_results['status'] != 'success':
            return

        updated_count = 0
        for client_id, client_result in mia_results['clients'].items():
            if client_result['status'] == 'success' and int(client_id) < len(self.clients):
                mia_f_score = client_result['summary']['avg_f_score']
                client = self.clients[int(client_id)]

                # 更新RL客户端的MIA分数
                if hasattr(client, 'update_mia_score'):
                    client.update_mia_score(mia_f_score)
                    updated_count += 1

        # 只在RL模式下打印
        if updated_count > 0 and self.use_rl_clients:
            print(f"[Server] Updated MIA scores for {updated_count} RL clients")


