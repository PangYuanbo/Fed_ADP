import copy
import numpy as np
import torch
import time
from flcore.clients.clientcp import *
from utils.data_utils import read_client_data
from threading import Thread
import os
from utils.mia_evaluator_simple import SimpleMIAEvaluator


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

        result_dir = "results"
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True,alpha=self.alpha)
            test_data = read_client_data(self.dataset, i, is_train=False,alpha=self.alpha)
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
        
        # Initialize MIA evaluator if attack models are available
        self.mia_evaluator = None
        self.mia_evaluation_interval = 10  # Evaluate MIA every 10 rounds
        self.enable_mia = getattr(args, 'enable_mia', False)  # Add flag to enable/disable MIA
        
        if self.enable_mia:
            try:
                self.mia_evaluator = SimpleMIAEvaluator(
                    attack_model_dir="Membership_Inference_Attack",
                    num_classes=args.num_classes,
                    device=args.device,
                    batch_size=1,  # Use batch_size=1 as in your original implementation
                    evaluate_interval=10,  # Evaluate MIA every 10 rounds
                    save_results=True,
                    results_dir=f"mia_results/{args.dataset}",
                    max_samples_per_label=50  # Limit to 50 samples per label for faster evaluation
                )
                print("[Server] MIA evaluator initialized successfully")
            except Exception as e:
                print(f"[Server] Warning: Could not initialize MIA evaluator: {e}")
                self.mia_evaluator = None


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
            # print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            filename=f"results_{self.dataset}_{c.id}.txt"
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
        
        # Initialize MIA results tracking
        mia_results_history = []
        
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
                
            # Evaluate MIA independently based on its own interval
            if self.mia_evaluator and self.mia_evaluator.should_evaluate(i):
                    print("\n[Server] Running MIA evaluation...")
                    mia_results = self.mia_evaluator.evaluate_all_clients(
                        self.clients,
                        round_num=i,
                        dataset_name=args.dataset,
                        alpha=args.alpha
                    )
                    mia_results_history.append(mia_results)
                    
                    # Log MIA results alongside accuracy
                    if 'average' in mia_results and mia_results['average']:
                        avg = mia_results['average']
                        with open(file_path, "a") as f:
                            f.write(f"    MIA - Acc: {avg['accuracy']:.4f}, F1: {avg['f1_score']:.4f}, "
                                   f"TPR: {avg['tpr']:.4f}, FPR: {avg['fpr']:.4f}\n")


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
        
        # Print MIA evaluation summary if available
        if self.mia_evaluator and mia_results_history:
            print("\n========== MIA Evaluation Summary ==========")
            summary = self.mia_evaluator.get_summary()
            if 'error' not in summary:
                print(f"Total rounds evaluated: {summary['total_rounds_evaluated']}")
                print(f"Final MIA Accuracy: {summary.get('accuracy_final', 'N/A'):.4f}")
                print(f"Final MIA F1 Score: {summary.get('f1_score_final', 'N/A'):.4f}")
                print(f"Final TPR (Member Detection): {summary.get('tpr_final', 'N/A'):.4f}")
                print(f"Final FPR (False Alarms): {summary.get('fpr_final', 'N/A'):.4f}")
                print(f"\nMIA Accuracy - Mean: {summary.get('accuracy_mean', 'N/A'):.4f}, "
                      f"Std: {summary.get('accuracy_std', 'N/A'):.4f}")
                print(f"MIA F1 Score - Mean: {summary.get('f1_score_mean', 'N/A'):.4f}, "
                      f"Std: {summary.get('f1_score_std', 'N/A'):.4f}")
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



