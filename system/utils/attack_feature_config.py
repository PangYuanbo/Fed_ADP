"""
Configuration helpers for modular MIA attack features.
"""

from typing import Iterable, List, Optional

# Specification describing how each attack feature should be handled.
FEATURE_SPECS = {
    'conv1': {
        'grad_source': 'feature',
        'grad_key': 'conv1.0.weight',
        'branch_type': 'conv',
        'hidden_channels': 64,
        'embedding_dim': 128,
    },
    'conv2': {
        'grad_source': 'feature',
        'grad_key': 'conv2.0.weight',
        'branch_type': 'conv',
        'hidden_channels': 128,
        'embedding_dim': 128,
    },
    'fc1': {
        'grad_source': 'feature',
        'grad_key': 'fc1.0.weight',
        'branch_type': 'conv',
        'hidden_channels': 64,
        'embedding_dim': 128,
    },
    'fc': {
        'grad_source': 'head',
        'grad_key': 'weight',
        'branch_type': 'conv',
        'hidden_channels': 64,
        'embedding_dim': 128,
    },
    'softmax': {
        'grad_source': 'softmax',
        'grad_key': None,
        'branch_type': 'softmax',
        'embedding_dim': 64,
        'mlp_hidden': 32,
    },
}

DEFAULT_ATTACK_FEATURES = ['conv1', 'conv2', 'fc1', 'fc', 'softmax']


def normalize_attack_features(features: Optional[Iterable[str]]) -> List[str]:
    """
    Normalize a feature selection into a validated, ordered list.
    """
    if not features:
        return DEFAULT_ATTACK_FEATURES.copy()
    normalized = []
    for name in features:
        key = str(name).strip().lower()
        if key not in FEATURE_SPECS:
            raise ValueError(f"Unsupported attack feature '{name}'. "
                             f"Valid options: {list(FEATURE_SPECS.keys())}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def attack_feature_suffix(features: Optional[Iterable[str]]) -> str:
    """
    Generate a deterministic suffix (e.g. conv1-conv2-softmax) for checkpoint names.
    """
    normalized = normalize_attack_features(features)
    return "-".join(normalized)


def attack_checkpoint_name(label: int,
                           features: Optional[Iterable[str]],
                           prefix: str = "attack_model",
                           include_suffix: bool = True) -> str:
    """
    Build a checkpoint file name for a given label and feature selection.
    """
    if include_suffix:
        suffix = attack_feature_suffix(features)
        return f"{prefix}_{suffix}_label{label}.pth"
    return f"{prefix}{label}.pth"


def feature_is_softmax(name: str) -> bool:
    return FEATURE_SPECS[name]['grad_source'] == 'softmax'
