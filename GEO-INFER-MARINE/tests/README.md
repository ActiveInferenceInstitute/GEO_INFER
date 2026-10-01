# GEO-INFER-MARINE/tests

Tests workspace within `GEO-INFER-MARINE`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-MARINE`
- Package: `geo_infer_marine`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-marine`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE`

## Dependencies

- `numpy>=1.20.0`
- `xarray>=0.19.0`
- `netcdf4>=1.5.8`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-MARINE` module's current behavior through unit,
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
  GEO-INFER-MARINE/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
