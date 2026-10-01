---
name: geo-infer-act
description: Canonical GEO-INFER Active Inference implementation. Use when implementing or reviewing free-energy minimization, belief updating, generative models, policy selection, H3/spatial active inference, or typed ACT diagnostics.
prerequisites:
  required:
    - geo-infer-bayes
  recommended:
    - geo-infer-space
    - geo-infer-time
difficulty: advanced
estimated_time: 60min
examples_dir: ./examples/
---

# GEO-INFER-ACT

## Instructions

Use `GEO-INFER-ACT/src/geo_infer_act` as the canonical implementation for
Active Inference inside GEO-INFER. Prefer these public exports:

```python
from geo_infer_act import (
    ActiveInferenceModel,
    ActiveInferenceStepResult,
    FreeEnergyBreakdown,
    FreeEnergyCalculator,
    GenerativeModel,
    H3BeliefUpdateResult,
    H3CellDiagnostics,
    H3EdgeDiagnostics,
    H3GridInferenceResult,
    H3LevelDiagnostics,
    H3SpatialConsistency,
    NestedH3BeliefUpdateResult,
    NestedH3GridInferenceResult,
    NestedH3LevelSummary,
    PolicyEvaluation,
    PolicySelector,
    SpatialActiveInferenceAgent,
    SpatialInferenceTrace,
)
```

## Examples

```python
import numpy as np
from geo_infer_act import ActiveInferenceModel, GenerativeModel

generative_model = GenerativeModel(
    "categorical",
    {"state_dim": 3, "obs_dim": 3},
)

agent = ActiveInferenceModel(
    model_type="categorical",
    policy_selection_mode="deterministic",
    random_seed=42,
)
agent.set_generative_model(generative_model)

result = agent.step(
    np.array([1.0, 0.0, 0.0]),
    available_actions=["survey", "wait"],
    return_result=True,
)

assert isinstance(result, ActiveInferenceStepResult)
```

```python
import numpy as np
from geo_infer_act import ActiveInferenceModel, GenerativeModel

cells = ["89283082803ffff"]
model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
model.enable_nested_h3_spatial([7, 8, 9], cells=cells)

agent = ActiveInferenceModel(
    model_type="categorical",
    policy_selection_mode="deterministic",
)
agent.set_generative_model(model)

nested = agent.infer_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    return_result=True,
)
trace = agent.trace_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    grid_result=nested,
)

assert isinstance(nested, NestedH3GridInferenceResult)
assert isinstance(trace, SpatialInferenceTrace)
```

## Guidelines

### Method Contracts

- `geo_infer_act.utils.pymdp_adapter` is the only production runtime bridge to
  `inferactively-pymdp==1.0.3`; it builds JAX `pymdp.agent.Agent` instances,
  validates the exact installed version, uses explicit RNG-key action sampling,
  and returns normalized posterior, action posterior, negative EFE, and VFE/free
  energy metadata.
- `FreeEnergyCalculator.compute_categorical_free_energy(..., return_breakdown=True)`
  returns `FreeEnergyBreakdown` with `free_energy = complexity - accuracy`.
- `FreeEnergyCalculator.compute_expected_free_energy(..., return_breakdown=True)`
  returns pragmatic, epistemic, risk, ambiguity, and entropy terms.
- `PolicySelector.select_policy(...)` returns selected policy metadata and a
  `PolicyEvaluation` object.
- `ActiveInferenceModel.step(..., return_result=True)` returns an
  `ActiveInferenceStepResult`; the default return is the `(beliefs, action)`
  tuple.
- `GenerativeModel.update_h3_beliefs(..., return_result=True)` returns an
  `H3BeliefUpdateResult` with normalized per-cell beliefs, aggregate free
  energy, and `H3SpatialConsistency`.
- `ActiveInferenceModel.infer_over_h3_grid(..., return_result=True)` and
  `SpatialActiveInferenceAgent.step(..., return_result=True)` return
  `H3GridInferenceResult`; their default dictionary outputs include per-cell
  pymdp metadata.
- `GenerativeModel.compute_h3_cell_diagnostics(...)`,
  `ActiveInferenceModel.trace_over_h3_grid(...)`,
  `ActiveInferenceModel.trace_over_nested_h3_grid(...)`,
  `SpatialActiveInferenceAgent.trace_step(...)`, and
  `SpatialActiveInferenceAgent.trace_nested_step(...)` return
  `SpatialInferenceTrace` with `H3CellDiagnostics`, `H3EdgeDiagnostics`, and
  `H3LevelDiagnostics`.
- `GenerativeModel.enable_nested_h3_spatial(...)` delegates hierarchy
  construction to SPACE and stores parent/child closure for ordered H3
  resolutions.
- `GenerativeModel.update_nested_h3_beliefs(..., return_result=True)` returns
  `NestedH3BeliefUpdateResult` with normalized finest-cell beliefs,
  parent-level aggregate beliefs, `NestedH3LevelSummary` rows, cross-level
  coherence, and finite aggregate free energy.
- `ActiveInferenceModel.infer_over_nested_h3_grid(..., return_result=True)` and
  `SpatialActiveInferenceAgent.step_nested(..., return_result=True)` return
  `NestedH3GridInferenceResult` while preserving existing flat H3 behavior.
- H3 methods must validate real `h3>=4.5.0,<5` cells. Synthetic cells are only
  for explicit `cell_*` unit-test paths.
- Runner `h3` and nested H3 modes emit `data/pymdp_h3_diagnostics.json`,
  `data/pymdp_policy_posteriors.csv`, and
  `visualizations/pymdp_policy_free_energy.html` with manifest-linked sidecars.
- Runner `h3`, `spatial`, and nested H3 modes emit
  `data/spatial_inference_trace.json`, `data/spatial_research_statistics.json`,
  `data/h3_cell_diagnostics.csv`, `data/h3_edge_diagnostics.csv`,
  `visualizations/h3_belief_flux_map.html`, `visualizations/h3_policy_surface.html`,
  `visualizations/h3_policy_transitions.html`,
  `visualizations/h3_spatial_autocorrelation.html`,
  `visualizations/h3_entropy_free_energy_phase.html`, and
  `visualizations/spatial_inference_research_report.html`; nested mode also
  emits `data/nested_h3_parent_child_diagnostics.csv`,
  `data/nested_h3_level_diagnostics.csv`, and
  `visualizations/nested_h3_hierarchy_map.html` plus
  `visualizations/nested_h3_parent_child_residuals.html`.
- Research-profile H3 runs are opt-in with
  `RunConfig.parameters["research_profile"] = True` or
  `geo-infer-act-run --research-profile`. They keep real H3 cells and real
  `inferactively-pymdp==1.0.3` while installing deterministic likelihoods,
  preferences, and action-conditioned transitions that avoid collapsed traces.
- The deterministic visualization gallery is generated with
  `uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py`.
  Use `uv run`; system Python may contain a legacy pymdp distribution and is
  outside the supported ACT/H3 runtime contract.

### Integrations

- AGENT active-inference adapters should call or conform to ACT typed result
  contracts.
- MATH/BAYES convenience surfaces may expose helpers, but ACT remains the
  canonical implementation for Active Inference policy and free-energy logic.
- Optional external backends may be absent; use real local ACT methods or
  explicit `not_available` results.

## Verification

```bash
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_act_geospatial_contract.py
uv run pytest GEO-INFER-ACT/tests/unit/test_spatial_trace_diagnostics.py -q
uv run pytest GEO-INFER-ACT/tests/unit/test_spatial_research_statistics.py -q
uv run pytest GEO-INFER-ACT/tests/unit/test_pymdp_h3_backend.py -q
uv run pytest GEO-INFER-ACT/tests/unit/test_nested_h3_active_inference.py -q
uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py --json
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```

Do not add inert placeholders, fake policy selection, first-policy defaults, or
undocumented public methods. Do not import legacy `pymdp.control` or
`pymdp.inference` in production code.
