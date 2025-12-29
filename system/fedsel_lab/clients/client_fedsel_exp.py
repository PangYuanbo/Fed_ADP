"""
FedSel Client with Exponential Mechanism (EXP)

Implements private Top-k dimension selection by sampling
indices proportionally to their gradient ranks (Algorithm 2 in paper).
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.clients.client_fedsel_base import ClientFedSelBase
from fedsel_lab.mechanisms.exponential_mechanism import ExponentialMechanism


class ClientFedSelEXP(ClientFedSelBase):
    """FedSel client using the Exponential Mechanism for selection."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # Ensure mechanism tag stays in sync for logging/budget allocator
        args.fedsel_mechanism = 'EXP'
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if self.id == 0:
            print(f"[FedSel-EXP] Client {self.id} using Exponential mechanism")

    def _select_topk_dimensions(self, gradient_vector, k, epsilon_1):
        selector = ExponentialMechanism(
            epsilon=epsilon_1,
            delta=self.delta,
            k=k
        )
        return selector.select_topk(gradient_vector, k)
