"""
FedSel Server Implementation

Implements federated learning server with sparse aggregation for FedSel:
- Creates FedSel clients (with PE mechanism)
- Aggregates sparse Top-k gradient updates
- Handles dimension-level sparsity
- Optional MIA evaluation integration

Key Features:
- Sparse aggregation: Only aggregates uploaded dimensions
- Unuploaded dimensions: Maintain global model values
- Compatible with existing FedCP evaluation framework
"""

import sys
import os
import copy
import torch
import numpy as np
import time

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from flcore.servers.servercp import FedCP
from utils.data_utils import read_client_data
from fedsel_lab.clients.client_fedsel_pe import ClientFedSelPE


class ServerFedSel(FedCP):
    """
    FedSel server with sparse aggregation for Top-k updates.

    Extends FedCP and overrides:
    - Client creation: Use ClientFedSelPE instead of clientCP
    - Aggregation: Sparse aggregation for Top-k dimension updates
    """

    def __init__(self, args, times):
        """
        Initialize FedSel server.

        Args:
            args: Training arguments
            times: Experiment run number
        """
        # Initialize basic attributes (don't call super().__init__ to avoid double client creation)
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

        # FedSel-specific parameters
        self.fedsel_mechanism = getattr(args, 'fedsel_mechanism', 'PE')
        self.fedsel_k = getattr(args, 'fedsel_k', 100)
        self.fedsel_beta = getattr(args, 'fedsel_beta', 0.9)

        # Disable RL clients for FedSel
        self.use_rl_clients = False

        # Results directory
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"FedSel Server Initialization")
        print(f"{'='*60}")
        print(f"Mechanism: {self.fedsel_mechanism}")
        print(f"Top-k: {self.fedsel_k}")
        print(f"Beta (accumulation): {self.fedsel_beta}")
        print(f"Dataset: {self.dataset}")
        print(f"Clients: {self.num_clients}")
        print(f"Join ratio: {self.join_ratio}")
        print(f"Global rounds: {self.global_rounds}")
        print(f"{'='*60}\n")

        # Create FedSel clients
        print(f"Creating {self.num_clients} FedSel-{self.fedsel_mechanism} clients...")

        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, alpha=self.alpha)
            test_data = read_client_data(self.dataset, i, is_train=False, alpha=self.alpha)

            # Create FedSel-PE client
            client = ClientFedSelPE(
                args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data)
            )

            self.clients.append(client)

            # Create result file
            filename = f"results_{self.dataset}_{client.id}.txt"
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "w") as f:
                pass

        print(f"Finished creating FedSel server and {self.num_clients} clients.\n")

        # Initialize other attributes from parent
        self.Budget = []
        self.head = None
        self.cs = None

        # MIA evaluation (optional)
        self.mia_evaluator = None
        self.mia_evaluation_interval = 10
        self.enable_mia = getattr(args, 'enable_mia', False)
        self.mia_results_history = []

        if self.enable_mia:
            try:
                print(f"[FedSel Server] Initializing MIA evaluator...")
                from utils.mia_attack_wrapper import FederatedMIAEvaluator

                self.mia_evaluator = FederatedMIAEvaluator(
                    attack_model_dir="Membership_Inference_Attack",
                    num_classes=args.num_classes,
                    device=args.device,
                    batch_size=1,
                    alpha=args.alpha,
                    max_samples_per_client=None,
                    use_gpu_optimization=True
                )
                print(f"[FedSel Server] MIA evaluator initialized successfully")
                print(f"[FedSel Server] MIA will evaluate every {self.mia_evaluation_interval} rounds\n")
            except Exception as e:
                print(f"[FedSel Server] Warning: Could not initialize MIA evaluator: {e}")
                print(f"[FedSel Server] Continuing without MIA evaluation\n")
                self.mia_evaluator = None
                self.enable_mia = False

    def aggregate_parameters(self):
        """
        Sparse aggregation for Top-k gradient updates.

        Handles sparsity at dimension level:
        1. For each parameter, collect all client updates
        2. Aggregate dimension-wise (only non-zero values)
        3. Unuploaded dimensions maintain global model values

        Note: FedSel clients upload sparse gradients where only Top-k
        dimensions are non-zero. This method aggregates accordingly.
        """
        assert len(self.uploaded_models) > 0

        # Save old global model
        old_global_state = copy.deepcopy(self.global_modules.state_dict())

        # Initialize new global state
        new_global_state = copy.deepcopy(old_global_state)

        # Track statistics for logging
        total_params = 0
        total_uploaded = 0

        # Aggregate each parameter
        for param_name in new_global_state.keys():
            # Get old parameter value
            old_param = old_global_state[param_name]

            # Collect all client updates for this parameter
            client_updates = []
            client_weights_list = []

            for weight, client_model in zip(self.uploaded_weights, self.uploaded_models):
                client_param = client_model.state_dict()[param_name]

                # Compute update (relative to old global model)
                update = client_param - old_param

                client_updates.append(update)
                client_weights_list.append(weight)

            # Weighted average of updates
            aggregated_update = torch.zeros_like(old_param)
            weight_mask_sum = torch.zeros_like(old_param)

            for update, weight in zip(client_updates, client_weights_list):
                # Only count dimensions that were actually uploaded (non-zero)
                nonzero_mask = update != 0
                aggregated_update[nonzero_mask] += weight * update[nonzero_mask]
                weight_mask_sum[nonzero_mask] += weight

            # Normalize by the sum of weights that actually uploaded each dim
            safe_update = torch.zeros_like(old_param)
            nonzero_weights = weight_mask_sum != 0
            safe_update[nonzero_weights] = (
                aggregated_update[nonzero_weights] / weight_mask_sum[nonzero_weights]
            )

            # Apply aggregated update
            new_global_state[param_name] = old_param + safe_update

            # Track sparsity statistics
            total_params += safe_update.numel()
            total_uploaded += (weight_mask_sum != 0).sum().item()

        # Load aggregated state
        self.global_modules.load_state_dict(new_global_state)

        # Log sparsity statistics (every 10 rounds)
        if hasattr(self, 'current_round') and self.current_round % 10 == 0:
            sparsity = total_uploaded / total_params if total_params > 0 else 0
            print(f"\n[FedSel Server Round {self.current_round}] Aggregation Statistics:")
            print(f"  Total parameters: {total_params}")
            print(f"  Uploaded (non-zero): {total_uploaded}")
            print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
            print(f"  Compression ratio: {1/sparsity:.2f}x\n" if sparsity > 0 else "  Compression ratio: inf\n")

    def train(self, args):
        """
        Main training loop for FedSel.

        Follows standard FL workflow:
        1. Evaluate (periodically)
        2. Select clients
        3. Train clients locally
        4. Receive sparse models
        5. Aggregate with sparse aggregation
        6. Send updated global model

        Args:
            args: Training arguments
        """
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.current_round = i
            self.selected_clients = self.select_clients()

            if i > 0:
                print(f"\n{'='*60}")
                print(f"FedSel Round {i}/{self.global_rounds}")
                print(f"Selected clients: {len(self.selected_clients)}")
                print(f"{'='*60}")

            # Evaluate
            if i % self.eval_gap == 0:
                if i > 0:
                    print(f"\n[Evaluation] Round {i}")
                self.evaluate()

            # MIA evaluation (optional)
            if (self.enable_mia and self.mia_evaluator and
                i > 0 and i % self.mia_evaluation_interval == 0):
                try:
                    print(f"\n[FedSel MIA] Evaluating privacy at round {i}...")
                    mia_results = self.mia_evaluator.evaluate_all_clients_federated(
                        clients=self.clients,
                        round_num=i,
                        dataset_name=args.dataset
                    )

                    self.mia_results_history.append(mia_results)

                    if mia_results.get('status') == 'success':
                        summary = mia_results.get('summary', {})
                        print(f"[FedSel MIA] Avg F-score: {summary.get('avg_f_score', 0):.4f}")
                        print(f"[FedSel MIA] High risk clients: {summary.get('high_risk_clients', 0)}/{summary.get('total_clients', 0)}")
                except Exception as e:
                    print(f"[FedSel MIA] Evaluation error: {e}")

            # Client training
            for client in self.selected_clients:
                client.round = i
                client.train_cs_model(i, args)

            # Aggregation
            self.test_metrics_after()
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            # Log time
            self.Budget.append(time.time() - s_t)
            if i > 0:
                print(f"Round {i} completed in {self.Budget[-1]:.2f}s")

        # Save final model
        self.save_models(args)

        # Print summary
        print(f"\n{'='*60}")
        print(f"FedSel Training Complete!")
        print(f"{'='*60}")
        print(f"Best accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Average round time: {sum(self.Budget[1:])/len(self.Budget[1:]):.2f}s")
        print(f"{'='*60}\n")

        # MIA summary (if enabled)
        if self.enable_mia and self.mia_evaluator and self.mia_results_history:
            print(f"\n{'='*60}")
            print(f"FedSel MIA Evaluation Summary")
            print(f"{'='*60}")

            all_f_scores = []
            for result in self.mia_results_history:
                if result.get('status') == 'success' and 'summary' in result:
                    all_f_scores.append(result['summary']['avg_f_score'])

            if all_f_scores:
                print(f"Total evaluations: {len(all_f_scores)}")
                print(f"Final F-score: {all_f_scores[-1]:.4f}")
                print(f"Average F-score: {np.mean(all_f_scores):.4f}")
                print(f"Max F-score: {np.max(all_f_scores):.4f}")
                print(f"Min F-score: {np.min(all_f_scores):.4f}")
                print(f"{'='*60}\n")
