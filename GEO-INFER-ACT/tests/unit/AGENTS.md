# Agent Instructions: GEO-INFER-ACT/tests/unit

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Unit workspace within `GEO-INFER-ACT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_act` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_analysis.py`
- `test_api.py`
- `test_categorical_regressions.py`
- `test_civic_intel.py`
- `test_climate_model.py`
- `test_comprehensive_audit_contracts.py`
- `test_continuous_efe.py`
- `test_continuous_pomdp_filter.py`
- `test_core.py`
- `test_dynamic_causal_model.py`
- `test_ecological_model.py`
- `test_free_energy.py`
- `test_gaussian_contract.py`
- `test_generative_efe.py`
- `test_geospatial_runner_outputs.py`
- `test_gnn_contract.py`
- `test_gnn_factored_contract.py`
- `test_gnn_gaussian_contract.py`
- `test_h3.py`
- `test_h3_active_inference.py`
- `test_h3_adapter.py`
- `test_h3_adapter_nested_grid_resolution.py`
- `test_h3_grid_independence.py`
- `test_h3_validation_logging.py`
- `test_h3_viz_integration.py`
- `test_inference_hardening.py`
- `test_markov_decision_process.py`
- `test_math_and_correlation_contracts.py`
- `test_method_scan_contracts.py`
- `test_model_contracts.py`
- `test_models.py`
- `test_nested_h3_active_inference.py`
- `test_perception_policy_timing.py`
- `test_policy_decomposition.py`
- `test_policy_selection.py`
- `test_pymdp_h3_backend.py`
- `test_runner_contracts.py`
- `test_spatial_agent.py`
- `test_spatial_grid_scoring.py`
- `test_spatial_research_statistics.py`
- `test_spatial_trace_diagnostics.py`
- `test_utils.py`
- `test_variational_inference.py`
- `factored_example.json`

## Validation

```bash
uv run python -m pytest GEO-INFER-ACT/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
