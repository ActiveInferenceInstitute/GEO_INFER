# GEO-INFER-HEALTH/tests

Tests workspace within `GEO-INFER-HEALTH`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`
- `test_disease_surveillance_integration.py`

## Public Interface

- `conftest.py:sample_locations` (function)
- `conftest.py:sample_health_facilities` (function)
- `conftest.py:sample_disease_reports` (function)
- `conftest.py:sample_population_data` (function)
- `conftest.py:sample_environmental_data` (function)
- `conftest.py:disease_analyzer` (function)
- `conftest.py:healthcare_analyzer` (function)
- `conftest.py:environmental_analyzer` (function)
- `conftest.py:temp_config_file` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:test_data_dir` (function)
- `conftest.py:setup_test_environment` (function)
- `conftest.py:mock_api_client` (function)
- `conftest.py:pytest_configure` (function)
- `conftest.py:assert_geospatial_objects_equal` (function)
- `conftest.py:create_test_grid` (function)

## Module Metadata

- Module: `GEO-INFER-HEALTH`
- Package: `geo_infer_health`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-health`
- Tests: `uv run python -m pytest GEO-INFER-HEALTH/tests`

## Dependencies

- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`
- `pydantic>=2.5.0`
- `pydantic-settings>=2.1.0`
- `pyyaml>=6.0.0`
- `loguru>=0.7.0`
- `geopandas>=0.13.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-HEALTH` module's current behavior through unit,
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
  GEO-INFER-HEALTH/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-HEALTH/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
