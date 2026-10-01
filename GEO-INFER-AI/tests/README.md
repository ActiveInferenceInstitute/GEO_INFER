# GEO-INFER-AI/tests

Tests workspace within `GEO-INFER-AI`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-AI`
- Package: `geo_infer_ai`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ai`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AI`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `h3>=4.5.0,<5`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-AI` module's current behavior through unit,
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
  GEO-INFER-AI/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AI
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
