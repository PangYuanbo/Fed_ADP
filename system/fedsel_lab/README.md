# FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection

Implementation of the FedSel framework from the paper:
**"FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection"**

## Overview

FedSel is a two-stage LDP framework that reduces dimension dependency in federated learning from O(√d) to O(1/√d):

1. **Stage 1: Dimension Selection** - Privately select Top-k dimensions using LDP mechanisms
2. **Stage 2: Value Perturbation** - Perturb only the selected dimension values with DP noise

## Key Features

- **Gradient Accumulation**: Stabilizes Top-k selection with r_t = β*r_{t-1} + g_t
- **Automatic Budget Allocation**: Intelligently splits privacy budget ε into ε1 (selection) and ε2 (perturbation)
- **PE Mechanism**: Perturbed Encoding for balanced privacy-utility tradeoff (recommended for k=10-200)
- **Comprehensive Testing**: Full unit test coverage for all core components

## Current Implementation Status

### ✓ Phase 1 Completed (Week 1)

**Core Infrastructure:**
- [x] Directory structure and package initialization
- [x] Gradient accumulator (`utils/gradient_accumulator.py`)
- [x] Privacy budget allocator (`mechanisms/budget_allocation.py`)
- [x] PE mechanism (`mechanisms/perturbed_encoding.py`)
- [x] Comprehensive unit tests (21 tests total)

**Test Results:**
```
Gradient Accumulator: 8/8 tests passed ✓
PE Mechanism: 13/13 tests passed ✓
```

### ✓ Phase 2 Completed (Week 2)

**Client Implementation:**
- [x] `client_fedsel_base.py` - Base client with two-stage LDP framework
- [x] `client_fedsel_pe.py` - PE mechanism client
- [x] Integration tests (6 tests total)
- [x] Gradient accumulation integration
- [x] Budget allocation integration
- [x] Sparse gradient upload

**Test Results:**
```
Client Integration: 6/6 tests passed ✓
Total: 27/27 tests passed (100% pass rate) ✓
```

**Implementation Stats:**
```
Production Code: ~654 LOC
Test Code: ~458 LOC
Total: ~1112 LOC (non-comment)
```

### ✓ Phase 3 Completed (Week 3)

**Server Implementation:**
- [x] `server_fedsel.py` - Server with sparse aggregation (318 LOC)
- [x] End-to-end testing (`test_fedsel_end_to_end.py`)
- [x] MIA evaluation integration (optional, in server)
- [x] Sparse aggregation verified (17.33x compression on test)

**Test Results:**
```
End-to-End Test: PASSED ✓
- 5 clients, 3 rounds
- Sparse aggregation: 5.77% sparsity (17.33x compression)
- Budget allocation: Adaptive (eps1=0.3-0.5, eps2=0.5-0.7)
- Training time: ~0.20s per round
```

**Implementation Stats:**
```
Production Code: ~972 LOC (including server)
Test Code: ~668 LOC (including end-to-end)
Total: ~1640 LOC (non-comment)
```

### 📋 Upcoming Phases

**Phase 4 (Week 4-5): CIFAR-10 Experiments**
- [ ] Create `run_fedsel_pe.py` for real CIFAR-10 experiments
- [ ] Baseline comparisons (FedAvg, FedAvg+DP, FedSel)
- [ ] Hyperparameter tuning (k, β, ε)
- [ ] Results collection and analysis

**Phase 5 (Week 6): Polish & Documentation**
- [ ] Code cleanup and optimization
- [ ] Usage examples and tutorials
- [ ] Final experiment runs
- [ ] Performance benchmarking

## Project Structure

```
fedsel_lab/
├── __init__.py                    # Package initialization
├── README.md                      # This file
│
├── clients/                       # Client implementations
│   ├── __init__.py
│   ├── client_fedsel_base.py     # ✓ Base client (two-stage LDP)
│   └── client_fedsel_pe.py       # ✓ PE mechanism client
│
├── servers/                       # Server implementation
│   ├── __init__.py
│   └── server_fedsel.py          # ✓ FedSel server with sparse aggregation
│
├── mechanisms/                    # LDP dimension selection mechanisms
│   ├── __init__.py
│   ├── budget_allocation.py      # ✓ Privacy budget allocation
│   └── perturbed_encoding.py     # ✓ PE mechanism
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   └── gradient_accumulator.py   # ✓ Gradient accumulation
│
├── experiments/                   # Experiment scripts
│   ├── __init__.py
│   └── test_fedsel_end_to_end.py # ✓ End-to-end integration test
│
└── tests/                         # Unit tests
    ├── __init__.py
    ├── test_accumulation.py      # ✓ 8 tests
    ├── test_mechanisms.py        # ✓ 13 tests
    └── test_integration.py       # ✓ 6 tests
```

## Quick Start

### 1. Test Gradient Accumulator

```python
from fedsel_lab.utils.gradient_accumulator import GradientAccumulator
import torch
import torch.nn as nn

# Create a simple model
model = nn.Linear(10, 5)

# Initialize accumulator
accumulator = GradientAccumulator(model, beta=0.9)

# Simulate gradient accumulation
for round_num in range(5):
    current_grads = {
        name: torch.randn_like(param)
        for name, param in model.named_parameters()
    }

    accumulated = accumulator.accumulate(current_grads)
    print(f"Round {round_num}: {accumulated['weight'].norm():.4f}")
```

### 2. Test Budget Allocation

```python
from fedsel_lab.mechanisms.budget_allocation import BudgetAllocator

# Create allocator
allocator = BudgetAllocator(epsilon_total=1.0, delta=1e-6)

# Allocate for d=1000, k=100
eps1, eps2 = allocator.allocate(d=1000, k=100, mechanism='PE')
print(f"Selection budget: {eps1:.4f}")      # 0.4
print(f"Perturbation budget: {eps2:.4f}")  # 0.6

# Compute noise scale
noise_std = allocator.compute_noise_scale(eps2, sensitivity=0.005)
print(f"Noise std: {noise_std:.6f}")
```

### 3. Test PE Mechanism

```python
from fedsel_lab.mechanisms.perturbed_encoding import PerturbedEncoding
import torch

# Create PE mechanism
pe = PerturbedEncoding(epsilon=1.0, delta=1e-6, k=10)

# Simulate gradient vector
gradients = torch.randn(100)

# Select Top-10 dimensions privately
selected_indices = pe.select_topk(gradients, k=10)
print(f"Selected {len(selected_indices)} dimensions: {selected_indices}")

# Verify privacy properties
sensitivity = pe.compute_sensitivity(k=10)        # 20
expected_noise = pe.compute_expected_noise_magnitude(k=10)  # 20
print(f"Sensitivity: {sensitivity}")
print(f"Expected noise: {expected_noise}")
```

## Running Tests

```bash
# Run all tests
cd system
python fedsel_lab/tests/test_accumulation.py
python fedsel_lab/tests/test_mechanisms.py

# Or run specific test
python -m unittest fedsel_lab.tests.test_accumulation.TestGradientAccumulator.test_accumulation_formula
```

## Key Formulas

### Gradient Accumulation
```
r_t = β * r_{t-1} + g_t
```
where β ∈ [0.7, 0.95], typically 0.9

### Privacy Budget Allocation
```
ε_total = ε1 + ε2

PE mechanism: ε1 = 0.4 * ε_total, ε2 = 0.6 * ε_total
```

### PE Mechanism
```
Stage 1 (Encode): z ∈ {0,1}^d where Top-k positions = 1
Stage 2 (Perturb): z̃ = z + Lap(2k/ε1)
Stage 3 (Decode): Select k indices with largest z̃ values
```

### Gaussian Noise (Value Perturbation)
```
σ = sensitivity * √(2 * ln(1.25/δ)) / ε2
```

## Implementation Notes

- **Language**: Code comments in English, documentation bilingual (English + Chinese)
- **Testing**: Focus on CIFAR-10 throughout
- **Priority**: Working prototype first (PE mechanism only)
- **MIA Integration**: Using existing `system/utils/mia_attack_wrapper.py`

## Dependencies

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Python >= 3.7

## References

1. **FedSel Paper**: "Federated SGD under Local Differential Privacy with Top-k Dimension Selection"
   - Algorithm 1: Two-stage LDP framework
   - Algorithm 3: Perturbed Encoding (PE) mechanism

2. **Base Framework**: Inherits from `system/flcore/clients/clientcp.py` and `system/flcore/servers/servercp.py`

## Progress Log

### Week 1 (Completed)
- ✓ Created directory structure
- ✓ Implemented gradient accumulator with 8 comprehensive tests
- ✓ Implemented budget allocator with multi-mechanism support
- ✓ Implemented PE mechanism with 13 comprehensive tests
- ✓ Validated all core utilities

**Lines of Code**: ~600 LOC
**Test Coverage**: 21 tests, 100% pass rate

### Week 2 (Completed)
- ✓ Implemented `ClientFedSelBase` with two-stage LDP framework
- ✓ Implemented `ClientFedSelPE` using PE mechanism
- ✓ Integrated gradient accumulation into training loop
- ✓ Integrated budget allocation for ε1/ε2 split
- ✓ Implemented sparse gradient generation (Top-k selection)
- ✓ Created 6 comprehensive integration tests

**Lines of Code**: ~654 LOC (production) + ~458 LOC (tests)
**Test Coverage**: 27 tests total, 100% pass rate

### Week 3 (Completed)
- ✓ Implemented `ServerFedSel` with sparse aggregation
- ✓ Dimension-level sparsity handling (only aggregate uploaded dims)
- ✓ Sparsity statistics tracking (compression ratio)
- ✓ MIA evaluation integration (optional)
- ✓ Created end-to-end integration test
- ✓ Fixed NPZ dataset format compatibility
- ✓ Fixed Unicode encoding issues for Windows
- ✓ Verified complete FL pipeline (client → server → aggregation)

**Lines of Code**: ~972 LOC (production) + ~668 LOC (tests)
**Test Coverage**: End-to-end test passing, 100% success rate

**Key Achievements**:
- Sparse aggregation: 5.77% sparsity (17.33x compression) on test
- Adaptive budget allocation: eps1 ∈ [0.3, 0.5], eps2 ∈ [0.5, 0.7]
- Fast training: ~0.20s per round on mock dataset
- Full compatibility with existing FedCP evaluation framework

---

**Last Updated**: November 29, 2024
**Version**: 0.3.0 (Phase 3 Complete - Server Implementation & End-to-End Testing)
