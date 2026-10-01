# GEO-INFER-ACT

Advanced Active Inference framework implementing Free Energy Principle for geospatial decision-making, perception, and learning.

## Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `debug_models.py`
- `verify_comprehensive.py`
- `verify_pipeline.py`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- `debug_models.py:main` (function)
- `verify_comprehensive.py:audit_active_inference_model` (function)
- `verify_comprehensive.py:audit_generative_model` (function)
- `verify_comprehensive.py:audit_free_energy_and_policy` (function)
- `verify_comprehensive.py:audit_inference_math` (function)
- `verify_comprehensive.py:audit_spatial_agent` (function)
- `verify_comprehensive.py:audit_domain_models` (function)
- `verify_comprehensive.py:audit_api_interface` (function)
- `verify_comprehensive.py:audit_visualization_methods` (function)
- `verify_comprehensive.py:audit_scenario_outputs` (function)
- `verify_comprehensive.py:audit_docs_and_mermaid` (function)
- `verify_comprehensive.py:parse_args` (function)
- `verify_comprehensive.py:main` (function)
- `verify_pipeline.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-ACT`
- Package: `geo_infer_act`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-act`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT`

## Dependencies

- `matplotlib>=3.4.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `requests>=2.25.0`
- `scipy>=1.7.0`
- `seaborn>=0.11.0`
- `inferactively-pymdp==1.0.3`
- `h3>=4.5.0,<5`
- `geo-infer-bayes>=0.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Implemented H3 Active Inference Contracts

- ACT uses `inferactively-pymdp==1.0.3` through
  `geo_infer_act.utils.pymdp_adapter` for categorical H3 active-inference
  runtime paths. H3 runtime cells are validated with real `h3>=4.5.0,<5`.
- Flat H3 APIs remain available through `GenerativeModel.enable_h3_spatial`,
  `GenerativeModel.update_h3_beliefs`, `ActiveInferenceModel.infer_over_h3_grid`,
  `SpatialActiveInferenceAgent.step`, and `simulate_h3_lattice`.
- Research trace APIs are available through
  `GenerativeModel.compute_h3_cell_diagnostics`,
  `ActiveInferenceModel.trace_over_h3_grid`,
  `ActiveInferenceModel.trace_over_nested_h3_grid`,
  `SpatialActiveInferenceAgent.trace_step`, and
  `SpatialActiveInferenceAgent.trace_nested_step`.
- Nested H3 APIs are opt-in through `enable_nested_h3_spatial`,
  `update_nested_h3_beliefs`, `infer_over_nested_h3_grid`,
  `SpatialActiveInferenceAgent.step_nested`, and
  `MultiAgentModel.simulate_nested_h3_lattice`.
- H3 diagnostics use `H3CellDiagnostics`, `H3EdgeDiagnostics`,
  `H3LevelDiagnostics`, and `SpatialInferenceTrace`; nested results use
  `NestedH3LevelSummary`, `NestedH3BeliefUpdateResult`, and
  `NestedH3GridInferenceResult` from `geo_infer_act`.
- Nested runner mode is enabled with `RunConfig.parameters["nested_h3"] = True`
  and emits `data/h3_hierarchy.csv`, `data/nested_h3_diagnostics.json`,
  `data/nested_h3_parent_child_diagnostics.csv`, and
  `visualizations/nested_h3_hierarchy_map.html`.
- Flat and nested H3 runner outputs include pymdp diagnostics in
  `data/pymdp_h3_diagnostics.json`, `data/pymdp_policy_posteriors.csv`, and
  `visualizations/pymdp_policy_free_energy.html`.
- Flat and nested H3 runner outputs also include
  `data/spatial_inference_trace.json`, `data/spatial_research_statistics.json`,
  `data/h3_cell_diagnostics.csv`, `data/h3_edge_diagnostics.csv`,
  `visualizations/h3_belief_flux_map.html`, `visualizations/h3_policy_surface.html`,
  `visualizations/h3_policy_transitions.html`,
  `visualizations/h3_spatial_autocorrelation.html`,
  `visualizations/h3_entropy_free_energy_phase.html`, and
  `visualizations/spatial_inference_research_report.html`.
- Research-profile runs are opt-in through
  `RunConfig.parameters["research_profile"] = True` or
  `geo-infer-act-run --research-profile`; they keep real H3 geometry and
  `inferactively-pymdp==1.0.3` while using deterministic offline spatial
  fields that avoid collapsed policy and entropy traces.
- Generate the deterministic four-run gallery with
  `uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py`.
  The supported runtime is `uv run`; system Python may contain older pymdp
  distributions and is not a valid H3 runtime contract.
- Optional Python model-source integrations (Bayeux, PyMC, and Pyro) require
  `config["allow_dynamic_code"] = True` and execute in per-call namespaces.
- Core inference utilities validate finite probability inputs, use local RNG
  instances for reproducible categorical sampling, and apply solve-based
  Joseph-form Gaussian updates. `PolicySelector` accepts
  `expected_posterior`/`posterior_beliefs` for KL information gain.
- `VariationalInference.structured_update` consumes explicit categorical factor
  tables with `variables` and `potential`/`values`/`table` fields, and
  `MultiAgentModel.step` runs a perception-action-resource cycle with optional
  movement and harvest fields in action dictionaries.

```python
import numpy as np
from geo_infer_act import ActiveInferenceModel, GenerativeModel

model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
model.enable_nested_h3_spatial([7, 8, 9], cells=["89283082803ffff"])

agent = ActiveInferenceModel(model_type="categorical")
agent.set_generative_model(model)
result = agent.infer_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    return_result=True,
)
trace = agent.trace_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    grid_result=result,
)
```

## Visualization Contracts

- Belief, policy, free-energy, hierarchical, and H3-grid plots validate finite
  aligned inputs, preserve caller-supplied figure sizes, and avoid changing
  process-wide matplotlib or seaborn state.
- H3 static and animated outputs create their parent directories before writing;
  constant-valued grids still receive a valid color scale.

Nested validation command:

```bash
uv run pytest GEO-INFER-ACT/tests/unit/test_nested_h3_active_inference.py -q
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_act_geospatial_contract.py
```

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
