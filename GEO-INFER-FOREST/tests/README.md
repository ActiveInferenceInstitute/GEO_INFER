# GEO-INFER-FOREST/tests

Tests workspace within `GEO-INFER-FOREST`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`
- `test_wildfire_risk.py`

## Public Interface

- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:forest_stand_gdf` (function)
- `conftest.py:biomass_allometric_params` (function)
- `conftest.py:forest_config` (function)

## Module Metadata

- Module: `GEO-INFER-FOREST`
- Package: `geo_infer_forest`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-forest`
- Tests: `uv run python -m pytest GEO-INFER-FOREST/tests`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-FOREST` module's current behavior through unit,
  integration, system, and performance test surfaces.
- Primary marker: tests receive exactly one primary marker from their canonical
  directory; additive domain markers remain allowed.
- Required fixtures: local `tests/conftest.py` fixtures and shared
  `geo_infer_test.testing` fixtures for deterministic RNG, filesystem, HTTP,
  SQLite, service, model, and artifact boundaries.
- Dependencies: required test/runtime dependencies are installed by
  `uv sync --all-packages --all-extras --all-groups`; missing backends are failures.
- Expected artifacts: JUnit XML under `.geo-infer-test-results/`; model and
  visualization outputs require finite statistics, sidecars, hashes, and a
  manifest.
- Failure triage: `env -u VIRTUAL_ENV uv run pytest -c pyproject.toml -q
  GEO-INFER-FOREST/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-FOREST/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
