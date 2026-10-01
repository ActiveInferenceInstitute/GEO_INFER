# GEO-INFER-SIM/tests

Tests workspace within `GEO-INFER-SIM`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`
- `test_simulation_engine_integration.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SIM`
- Package: `geo_infer_sim`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-sim`
- Tests: `uv run python -m pytest GEO-INFER-SIM/tests`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-SIM` module's current behavior through unit,
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
  GEO-INFER-SIM/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-SIM/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
