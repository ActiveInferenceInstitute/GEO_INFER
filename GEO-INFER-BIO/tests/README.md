# GEO-INFER-BIO/tests

Tests workspace within `GEO-INFER-BIO`.

## Contents

- `integration/`
- `unit/`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-BIO`
- Package: `geo_infer_bio`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-bio`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO`

## Dependencies

- `biopython>=1.79`
- `fastapi>=0.100.0`
- `geopandas>=0.13.0`
- `graphql-core>=3.1.0`
- `matplotlib>=3.4.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `pydantic>=2.0.0`
- `python-multipart>=0.0.7`
- `requests>=2.28`
- `seaborn>=0.11.0`
- `shapely>=2.0.0`
- `strawberry-graphql>=0.96.0`
- `uvicorn>=0.15.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-BIO` module's current behavior through unit,
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
  GEO-INFER-BIO/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
