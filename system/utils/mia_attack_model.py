"""
MIA attack model capable of consuming configurable feature sets while remaining
backwards compatible with legacy checkpoints.
"""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from utils.attack_feature_config import FEATURE_SPECS, normalize_attack_features


def _make_conv_branch(out_channels: int, embedding_dim: int):
    return nn.Sequential(
        nn.Conv2d(1, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(out_channels, embedding_dim),
    )


def _make_softmax_branch(num_classes: int, hidden_dim: int, embedding_dim: int):
    return nn.Sequential(
        nn.Linear(num_classes, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embedding_dim),
        nn.ReLU(),
    )


class GradientMIA(nn.Module):
    """
    Attack model that fuses gradients/softmax statistics from selected layers.
    """

    def __init__(self,
                 enabled_features: Optional[Iterable[str]] = None,
                 num_classes: int = 10):
        super().__init__()
        self.enabled_features: List[str] = normalize_attack_features(enabled_features)
        self.num_classes = num_classes
        self._branch_modules: Dict[str, nn.Module] = {}

        fusion_dim = 0

        if 'conv1' in self.enabled_features:
            spec = FEATURE_SPECS['conv1']
            self.cnn1 = _make_conv_branch(spec['hidden_channels'], spec['embedding_dim'])
            self._branch_modules['conv1'] = self.cnn1
            fusion_dim += spec['embedding_dim']
        else:
            self.cnn1 = None

        if 'conv2' in self.enabled_features:
            spec = FEATURE_SPECS['conv2']
            self.cnn2 = _make_conv_branch(spec['hidden_channels'], spec['embedding_dim'])
            self._branch_modules['conv2'] = self.cnn2
            fusion_dim += spec['embedding_dim']
        else:
            self.cnn2 = None

        if 'fc1' in self.enabled_features:
            spec = FEATURE_SPECS['fc1']
            self.cnn_fc1 = _make_conv_branch(spec['hidden_channels'], spec['embedding_dim'])
            self._branch_modules['fc1'] = self.cnn_fc1
            fusion_dim += spec['embedding_dim']
        else:
            self.cnn_fc1 = None

        if 'fc' in self.enabled_features:
            spec = FEATURE_SPECS['fc']
            self.cnn_fc = _make_conv_branch(spec['hidden_channels'], spec['embedding_dim'])
            self._branch_modules['fc'] = self.cnn_fc
            fusion_dim += spec['embedding_dim']
        else:
            self.cnn_fc = None

        if 'softmax' in self.enabled_features:
            spec = FEATURE_SPECS['softmax']
            self.fcn_softmax = _make_softmax_branch(self.num_classes, spec['mlp_hidden'], spec['embedding_dim'])
            self._branch_modules['softmax'] = self.fcn_softmax
            fusion_dim += spec['embedding_dim']
        else:
            self.fcn_softmax = None

        if fusion_dim == 0:
            raise ValueError("At least one attack feature is required.")

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, *feature_args, **feature_kwargs):
        inputs = self._normalize_inputs(feature_args, feature_kwargs)
        embeddings = []
        for name in self.enabled_features:
            branch = self._branch_modules.get(name)
            if branch is None:
                raise ValueError(f"No branch registered for feature '{name}'")
            embeddings.append(branch(inputs[name]))
        fused = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        return self.classifier(fused)

    def _normalize_inputs(self, feature_args, feature_kwargs):
        if feature_kwargs:
            return feature_kwargs
        if len(feature_args) == 1 and isinstance(feature_args[0], dict):
            return feature_args[0]
        if len(feature_args) != len(self.enabled_features):
            raise ValueError(
                f"Expected {len(self.enabled_features)} feature tensors, "
                f"got {len(feature_args)}."
            )
        return {
            name: tensor
            for name, tensor in zip(self.enabled_features, feature_args)
        }
