# GEO-INFER-TEST/tests/integration

Integration workspace within `GEO-INFER-TEST`.

## Contents

- `test_act_agent_ant_coordination.py`
- `test_ai_space_domain_integration.py`
- `test_cross_module.py`
- `test_cross_module_civic_intel.py`
- `test_cross_module_workflows.py`
- `test_ecosystem_health.py`
- `test_h3_space_time_bayes_risk_act_composition.py`
- `test_module_imports.py`
- `test_sec_api_app_security.py`
- `test_space_time_data_integration.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python -m pytest GEO-INFER-TEST/tests/integration`

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
uv sync --all-packages --all-extras
uv run python -m pytest GEO-INFER-TEST/tests/integration
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
