# GEO-INFER-ACT/tests/unit

Unit workspace within `GEO-INFER-ACT`.

## Contents

- `test_analysis.py`
- `test_api.py`
- `test_bayeux_import_boundary.py`
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
- `test_research_policy_contracts.py`
- `test_runner_contracts.py`
- `test_spatial_agent.py`
- `test_spatial_grid_scoring.py`
- `test_spatial_research_statistics.py`
- `test_spatial_trace_diagnostics.py`
- `test_utils.py`
- `test_variational_inference.py`
- `factored_example.json`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-ACT`
- Package: `geo_infer_act`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-act`
- Tests: `uv run python -m pytest GEO-INFER-ACT/tests/unit`

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
- `geo-infer-bayes>=0.4.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-ACT/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
