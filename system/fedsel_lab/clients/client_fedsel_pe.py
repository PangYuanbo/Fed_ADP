"""
FedSel Client with Perturbed Encoding (PE) Mechanism

Implements private Top-k dimension selection using PE mechanism:
1. Encode gradients as k-hot binary vector
2. Add Laplacian noise to encoding
3. Decode Top-k from noisy encoding

Recommended for: 10 < k < 200 (balanced privacy-utility)
"""

import sys
import os
import torch

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fedsel_lab.clients.client_fedsel_base import ClientFedSelBase
from fedsel_lab.mechanisms.perturbed_encoding import PerturbedEncoding


class ClientFedSelPE(ClientFedSelBase):
    """
    FedSel client using Perturbed Encoding (PE) mechanism.

    Extends ClientFedSelBase and implements _select_topk_dimensions()
    using the PE mechanism (encode → perturb → decode).

    Privacy Guarantee: (ε1, δ)-LDP for dimension selection
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        Initialize FedSel-PE client.

        Args:
            args: Training arguments
            id: Client ID
            train_samples: Number of training samples
            test_samples: Number of test samples
            **kwargs: Additional arguments
        """
        # Set mechanism name before calling parent
        args.fedsel_mechanism = 'PE'

        # Call parent constructor
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        if self.id == 0:
            print(f"[FedSel-PE] Client {self.id} using Perturbed Encoding mechanism")

    def _select_topk_dimensions(self, gradient_vector, k, epsilon_1):
        """
        Select Top-k dimensions using PE mechanism.

        Implements the three-stage PE workflow:
        1. Encode: Create k-hot binary vector from gradients
        2. Perturb: Add Laplacian noise (scale = 2k/ε1)
        3. Decode: Extract Top-k indices from noisy encoding

        Args:
            gradient_vector: 1D tensor of accumulated gradients [d]
            k: Number of dimensions to select
            epsilon_1: Privacy budget for dimension selection

        Returns:
            Tensor of selected dimension indices [k]
        """
        # Create PE mechanism with allocated budget
        pe = PerturbedEncoding(
            epsilon=epsilon_1,
            delta=self.delta,
            k=k
        )

        # Run PE pipeline: encode → perturb → decode
        selected_indices = pe.select_topk(gradient_vector, k)

        return selected_indices
