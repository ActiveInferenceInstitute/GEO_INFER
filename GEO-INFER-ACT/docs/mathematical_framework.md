# Mathematical Framework

## Introduction

This document provides the mathematical foundations for Active Inference as implemented in GEO-INFER-ACT.

## Notation

| Symbol | Description |
|--------|-------------|
| `s` | Hidden states |
| `o` | Observations |
| `a` | Actions |
| `π` | Policy (sequence of actions) |
| `q(·)` | Approximate posterior |
| `p(·)` | Generative model |
| `F` | Variational free energy |
| `G` | Expected free energy |

## Generative Model

### Discrete State-Space Model

For discrete time steps with hidden states and observations:

```
p(o₀:T, s₀:T) = p(s₀) ∏ₜ p(oₜ|sₜ) p(sₜ|sₜ₋₁, aₜ₋₁)
```

### Key Matrices

**A - Likelihood Matrix**: `p(o|s)`

```
A[i,j] = p(oᵢ | sⱼ)
```

**B - Transition Matrix**: `p(s'|s,a)`

```
B[a][i,j] = p(s'ᵢ | sⱼ, a)
```

**C - Preference Vector**: `p(o)` (preferred observations)

```
C[i] = log p(oᵢ) ∝ utility(oᵢ)
```

**D - Prior Vector**: `p(s₀)` (initial state prior)

```
D[i] = p(s₀ = i)
```

## Variational Free Energy

### Definition

```
F = E_q[log q(s) - log p(o,s)]
```

### Decomposition

```
F = Complexity - Accuracy

Where:
  Complexity = DKL[q(s) || p(s)]
  Accuracy = E_q[log p(o|s)]
```

### In Practice

For discrete models with categorical distributions:

```python
def free_energy(qs, A, B, o, prior):
    """
    qs: posterior beliefs about states
    A: likelihood matrix
    B: transition matrix
    o: observation (one-hot encoded)
    prior: prior beliefs
    """
    # Accuracy (expected log likelihood)
    accuracy = qs @ log(A.T @ o)
    
    # Complexity (KL divergence from prior)
    complexity = kl_divergence(qs, prior)
    
    return complexity - accuracy
```

## Expected Free Energy

### Definition

Expected free energy for policy π at future time τ:

```
G(π,τ) = E_q[log q(sτ|π) - log p(oτ,sτ|π)]
```

### Decomposition

```
G = Epistemic Value + Pragmatic Value

Where:
  Epistemic = E_q[H[p(o|s)]] - H[E_q[p(o|s)]]     # Information gain
  Pragmatic = E_q[log q(o|π) - log p(o)]          # KL from preferences
```

### Implementation

```python
def expected_free_energy(qs, A, B, C, policy, horizon):
    """
    Compute expected free energy for a policy
    """
    G = 0
    qs_pi = qs.copy()
    
    for t in range(horizon):
        action = policy[t]
        
        # Predicted states after action
        qs_pi = B[action] @ qs_pi
        
        # Predicted observations
        qo = A @ qs_pi
        
        # Pragmatic value (preferences)
        pragmatic = qo @ (log(qo) - C)
        
        # Epistemic value (information gain)
        epistemic = entropy(A @ qs_pi) - qs_pi @ entropy(A, axis=0)
        
        G += pragmatic - epistemic
    
    return G
```

## Policy Selection

### Softmax Policy Selection

```
p(π) = σ(-γ × G(π))
```

Where:

- `σ` is softmax function
- `γ` is precision (inverse temperature)
- `G(π)` is expected free energy of policy π

### Action Selection

```python
from geo_infer_act import PolicySelector


def select_action(qs, preferences, policies):
    """Select the lowest-EFE action with the canonical ACT selector."""
    selector = PolicySelector(selection_mode="deterministic")
    result = selector.select_policy(qs, policies, preferences)
    return result["policy"]["action"]
```

## Belief Updating

### State Estimation

Given observation `o`:

```python
def update_beliefs(qs, A, o):
    """
    Update beliefs given observation
    """
    # Likelihood of observation given each state
    likelihood = A.T @ o
    
    # Posterior = prior × likelihood (normalized)
    qs_new = qs * likelihood
    qs_new = qs_new / qs_new.sum()
    
    return qs_new
```

## Spatial Extensions

### H3 State Space

For geospatial agents, states often correspond to H3 cells:

```python
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

# State space = H3 cells at resolution 9
indexer = SpatialIndexingInterface(backend="h3")
states = indexer.polygon_to_cells(boundary, resolution=9)
n_states = len(states)

# Transition matrix encodes spatial adjacency
B = create_spatial_transition_matrix(states)
```

### Multi-Scale Inference

Hierarchical H3 enables multi-scale generative models:

```
p(s) = p(s_fine | s_coarse) × p(s_coarse)
```

## Code Implementation References

### VFE Implementation

The variational free energy formulas above are implemented in:

- **[`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py)**: `FreeEnergyCalculator` class
  - `compute_categorical_free_energy()`
  - `compute_gaussian_free_energy()`
- **[`utils/math.py`](../src/geo_infer_act/utils/math.py)**: `compute_free_energy_categorical()` utility
- **[`core/spatial_agent.py`](../src/geo_infer_act/core/spatial_agent.py)**: `_compute_spatial_free_energy()` for H3 grids

### EFE Implementation

The expected free energy formulas are implemented in:

- **[`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py)**: `compute_expected_free_energy()`
- **[`core/policy_selection.py`](../src/geo_infer_act/core/policy_selection.py)**: `PolicySelector.compute_expected_free_energy()`
- **[`utils/math.py`](../src/geo_infer_act/utils/math.py)**: Standalone utility function

### Working Examples

| Example | Mathematical Concepts Demonstrated |
|---------|-----------------------------------|
| [`spatial_inference_demo.py`](../examples/spatial_inference_demo.py) | Spatial VFE, H3 belief propagation |
| [`modern_active_inference.py`](../examples/modern_active_inference.py) | Full VFE/EFE cycle, hierarchical models |
| [`h3_active_inference.py`](../examples/h3_active_inference.py) | H3 state space, spatial transitions |
| [`simple_model.py`](../examples/simple_model.py) | Basic belief updating, VFE minimization |

## Further Reading

- [Active Inference Overview](./active_inference_overview.md)
- [Free Energy Principle](./free_energy_principle.md)
- [fep_lean Notation Bridge](./fep_lean_notation_bridge.md) — correspondence of the notation and implemented constructs to the fep_lean formalization catalogue
