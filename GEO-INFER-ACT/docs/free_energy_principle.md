# The Free Energy Principle

## Introduction

The **Free Energy Principle (FEP)** is a theoretical framework proposing that all adaptive systems—from cells to societies—minimize variational free energy to maintain their existence and adapt to their environment.

## Historical Context

Developed by neuroscientist **Karl Friston** at University College London, the FEP emerged from efforts to understand brain function through the lens of Bayesian inference and statistical physics.

## Core Formulation

### Variational Free Energy

Free energy is defined as:

```
F = E_q[log q(s) - log p(o,s)]
     = E_q[log q(s)] - E_q[log p(o,s)]
     = -H[q(s)] + E_q[-log p(o,s)]
```

This can be rewritten as:

```
F = DKL[q(s) || p(s|o)] + (-log p(o))
```

Where:

- `DKL` - Kullback-Leibler divergence (a measure of difference between distributions)
- `q(s)` - Approximate posterior (recognition density)
- `p(s|o)` - True posterior
- `p(o)` - Model evidence (marginalized likelihood)

### Key Insight

Since `DKL ≥ 0`, free energy is an **upper bound on surprise**:

```
F ≥ -log p(o) = surprise
```

Minimizing F therefore minimizes surprise.

## Two Routes to Minimization

### 1. Perceptual Inference

Adjust beliefs `q(s)` to better explain observations:

```python
# Gradient descent on beliefs
dq/dt = -∂F/∂q
```

This is equivalent to Bayesian belief updating.

### 2. Active Inference

Select actions that change observations to match predictions:

```python
# Select action minimizing expected free energy
action = argmin_a E[F(o_future | a)]
```

## Mathematical Details

### Generative Model

The joint probability `p(o,s)` factorizes as:

```
p(o,s) = p(o|s) × p(s)
```

Where:

- `p(o|s)` - Likelihood (how states generate observations)
- `p(s)` - Prior (expected states)

### Belief Updating

Under Laplace approximation (Gaussian beliefs):

```
μ = μ + Δt × (∂F/∂μ)
```

Where `μ` is the mean of the approximate posterior.

## Connection to Other Frameworks

| Framework | Relationship |
|-----------|--------------|
| **Bayesian Inference** | FEP generalizes Bayesian updating |
| **Predictive Coding** | Perception as prediction error minimization |
| **Optimal Control** | Action selection with uncertainty |
| **Information Theory** | Free energy involves entropy and KL divergence |
| **Thermodynamics** | Analogy to free energy in physics |

## Implications

### For Neuroscience

- Brain as an inference machine
- Hierarchical predictive processing
- Unified theory of perception, action, learning

### For AI

- Principled approach to agent design
- Natural exploration-exploitation balance
- Robust to uncertainty

### For Geospatial AI

- Agents that actively seek information
- Spatial uncertainty quantification
- Adaptive environmental monitoring

## Code Implementation

### Variational Free Energy (VFE)

VFE is calculated in GEO-INFER-ACT at the following locations:

| Module | Function | Description |
|--------|----------|-------------|
| [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) | `compute_categorical_free_energy()` | VFE for categorical models; returns `FreeEnergyBreakdown` when requested |
| [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) | `compute_gaussian_free_energy()` | VFE for Gaussian models |
| [`core/active_inference.py`](../src/geo_infer_act/core/active_inference.py) | `compute_free_energy()` | Agent-level VFE computation |
| [`core/spatial_agent.py`](../src/geo_infer_act/core/spatial_agent.py) | `_compute_spatial_free_energy()` | Spatial VFE across H3 cells |
| [`utils/math.py`](../src/geo_infer_act/utils/math.py) | `compute_free_energy_categorical()` | Standalone utility function |

### Expected Free Energy (EFE)

EFE is calculated for policy selection at:

| Module | Function | Description |
|--------|----------|-------------|
| [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) | `compute_expected_free_energy()` | Core EFE with epistemic/pragmatic/risk/ambiguity decomposition |
| [`core/policy_selection.py`](../src/geo_infer_act/core/policy_selection.py) | `compute_expected_free_energy()` | Policy-level EFE for action selection; accepts policy-conditioned predictive beliefs |
| [`core/active_inference.py`](../src/geo_infer_act/core/active_inference.py) | `compute_expected_free_energy()` | Agent EFE computation |
| [`core/spatial_agent.py`](../src/geo_infer_act/core/spatial_agent.py) | `spatial_action()` | Spatial EFE across cells for spatial policy |
| [`utils/math.py`](../src/geo_infer_act/utils/math.py) | `compute_expected_free_energy()` | Standalone utility function |

## Example Demonstrations

See VFE and EFE in practice:

| Example | VFE Usage | EFE Usage |
|---------|-----------|-----------|
| [`spatial_inference_demo.py`](../examples/spatial_inference_demo.py) | Package runner records spatial free-energy metrics | Via `SpatialActiveInferenceAgent.spatial_action()` |
| [`modern_active_inference.py`](../examples/modern_active_inference.py) | Package runner records analyzer free-energy traces | Via policy evaluation in `ActiveInferenceModel` |
| [`h3_active_inference.py`](../examples/h3_active_inference.py) | H3 runner records aggregate and per-cell free energy | Via grid policy selection and EFE diagnostics |
| [`simple_model.py`](../examples/simple_model.py) | Agent belief updates | Policy evaluation |
| [`urban_planning.py`](../examples/urban_planning.py) | Planning optimization | Action selection |

## Further Reading

- [Active Inference Overview](./active_inference_overview.md)
- [Mathematical Framework](./mathematical_framework.md)

## References

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
