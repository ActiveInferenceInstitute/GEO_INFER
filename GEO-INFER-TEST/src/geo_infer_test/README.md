# GEO-INFER-TEST/src/geo_infer_test

Geo Infer Test workspace within `GEO-INFER-TEST`.

## Contents

- `core/`
- `models/`
- `__init__.py`
- `testing.py`

## Public Interface

- `testing.py:as_finite_array` (function)
- `testing.py:assert_finite` (function)
- `testing.py:assert_probability` (function)
- `testing.py:assert_stochastic_matrix` (function)
- `testing.py:assert_same_finite_values` (function)
- `testing.py:assert_no_nan_statistics` (function)
- `testing.py:assert_model_contract` (function)
- `testing.py:assert_seed_replay` (function)
- `testing.py:assert_packaged_config_loads` (function)
- `testing.py:assert_visualization_manifest` (function)
- `testing.py:LocalService` (class)
- `testing.py:deterministic_rng` (function)
- `testing.py:local_filesystem` (function)
- `testing.py:sqlite_database` (function)
- `testing.py:local_http_server` (function)
- `testing.py:local_service` (function)

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
