# GEO-INFER-AG/tests

Tests workspace within `GEO-INFER-AG`.

## Contents

- `data/`
- `integration/`
- `performance/`
- `unit/`
- `conftest.py`

## Public Interface

- `conftest.py:sample_field_data` (function)
- `conftest.py:sample_soil_data` (function)
- `conftest.py:sample_weather_data` (function)
- `conftest.py:sample_management_data` (function)
- `conftest.py:sample_time_series_data` (function)
- `conftest.py:management_practices` (function)

## Module Metadata

- Module: `GEO-INFER-AG`
- Package: `geo_infer_ag`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ag`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AG`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `rasterio>=1.2.0`
- `pyproj>=3.0.0`
- `matplotlib>=3.3.0`
- `scipy>=1.6.0`
- `xarray>=0.18.0`
- `joblib>=1.0.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-AG` module's current behavior through unit,
  integration, system, and performance test surfaces.
- Primary marker: tests receive exactly one primary marker from their canonical
  directory; additive domain markers remain allowed.
- Required fixtures: local `tests/conftest.py` fixtures and shared
  `geo_infer_test.testing` fixtures for deterministic RNG, filesystem, HTTP,
  SQLite, service, model, and artifact boundaries.
- Dependencies: required test/runtime dependencies are installed by
  `uv sync --all-packages --all-extras`; missing backends are failures.
- Expected artifacts: JUnit XML under `.geo-infer-test-results/`; model and
  visualization outputs require finite statistics, sidecars, hashes, and a
  manifest.
- Failure triage: `env -u VIRTUAL_ENV uv run pytest -c pyproject.toml -q
  GEO-INFER-AG/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
