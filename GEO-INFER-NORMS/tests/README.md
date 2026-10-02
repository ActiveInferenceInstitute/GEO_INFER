# GEO-INFER-NORMS/tests

Tests workspace within `GEO-INFER-NORMS`.

## Contents

- `integration/`
- `test-outputs/`
- `unit/`
- `conftest.py`

## Public Interface

- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:normative_rules` (function)
- `conftest.py:compliance_data` (function)
- `conftest.py:norms_config` (function)

## Module Metadata

- Module: `GEO-INFER-NORMS`
- Package: `geo_infer_norms`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-norms`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS`

## Dependencies

- `fastapi>=0.95.0,<1`
- `starlette>=0.27.0`
- `geopandas>=0.13.0,<2`
- `matplotlib>=3.7.0,<4`
- `networkx>=2.6.0,<4`
- `numpy>=1.24.0,<3`
- `pandas>=2.0.0,<3`
- `pydantic>=2.0.0,<3`
- `shapely>=2.0.0,<3`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-NORMS` module's current behavior through unit,
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
  GEO-INFER-NORMS/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
