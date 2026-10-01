# GEO-INFER-SEC/tests

Tests workspace within `GEO-INFER-SEC`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`
- `test_anonymization.py`

## Public Interface

- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:sample_credentials` (function)
- `conftest.py:security_config` (function)
- `conftest.py:audit_log_entries` (function)

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-sec`
- Tests: `uv run python -m pytest GEO-INFER-SEC/tests`

## Dependencies

- `cryptography>=36.0.0`
- `flask>=2.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyjwt>=2.3.0`
- `pyyaml>=6.0`
- `shapely>=2.0.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-SEC` module's current behavior through unit,
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
  GEO-INFER-SEC/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-SEC/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
