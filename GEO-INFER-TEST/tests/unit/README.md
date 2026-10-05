# GEO-INFER-TEST/tests/unit

Unit workspace within `GEO-INFER-TEST`.

## Contents

- `test_act_research_oracles.py`
- `test_build_package_wheels.py`
- `test_check_coverage_floor_gate.py`
- `test_ci_workflow_contracts.py`
- `test_crescent_city_bundled_seed_uniqueness.py`
- `test_crescent_city_civic_intel_demo.py`
- `test_crescent_city_geo_intel_contract_sync.py`
- `test_data_domains.py`
- `test_data_quality_contract.py`
- `test_doc_examples.py`
- `test_documentation_census.py`
- `test_execution_contracts.py`
- `test_gnn_interchange_receipts.py`
- `test_log_integration.py`
- `test_manuscript_research.py`
- `test_measure_module_coverage.py`
- `test_module_health.py`
- `test_parametric_load_benchmarks.py`
- `test_performance_monitor.py`
- `test_process_ownership_contracts.py`
- `test_root_pytest_policy.py`
- `test_run_unified_tests.py`
- `test_runtime_metadata.py`
- `test_script_validators.py`
- `test_spatial_functions.py`
- `test_test_contracts_validator.py`
- `test_test_discoverer.py`
- `test_test_orchestrator.py`
- `test_test_runner.py`
- `test_testing_helpers.py`
- `test_unified_runner_extra_test_paths.py`
- `test_validate_doc_imports.py`
- `test_validate_h3_active_inference_contract.py`
- `test_validate_packaging.py`
- `test_validate_repo_contracts.py`
- `test_validate_skills.py`
- `test_validators.py`
- `test_validators_parametric.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python -m pytest GEO-INFER-TEST/tests/unit`

## Dependencies

- `coverage[toml]>=7.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `hypothesis>=6.0.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pytest>=7.0.0`
- `pytest-benchmark>=4.0.0`
- `pytest-cov>=4.0.0`
- `pytest-html>=3.1.0`
- `pytest-mock>=3.10.0`
- `pytest-timeout>=2.1.0`
- `pytest-xdist>=3.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -m pytest GEO-INFER-TEST/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
