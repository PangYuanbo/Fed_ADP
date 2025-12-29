"""
FedSel Client with Perturbed Sampling (PS) Mechanism.

Implements Algorithm 4 from the FedSel paper by sampling
from Top-k vs non-Top-k pools with privacy-preserving probabilities.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.clients.client_fedsel_base import ClientFedSelBase
from fedsel_lab.mechanisms.perturbed_sampling import PerturbedSampling


class ClientFedSelPS(ClientFedSelBase):
    """FedSel client using Perturbed Sampling for dimension selection."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        args.fedsel_mechanism = 'PS'
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if self.id == 0:
            print(f"[FedSel-PS] Client {self.id} using Perturbed Sampling mechanism")

    def _select_topk_dimensions(self, gradient_vector, k, epsilon_1):
        selector = PerturbedSampling(
            epsilon=epsilon_1,
            delta=self.delta,
            k=k
        )
        return selector.select_topk(gradient_vector, k)
