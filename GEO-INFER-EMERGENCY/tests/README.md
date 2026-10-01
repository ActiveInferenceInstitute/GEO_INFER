# GEO-INFER-EMERGENCY/tests

Tests workspace within `GEO-INFER-EMERGENCY`.

## Contents

- `integration/`
- `conftest.py`
- `test_acceptance_emergency.py`
- `test_awareness.py`
- `test_coordinator.py`
- `test_evacuation.py`
- `test_evacuation_sar.py`
- `test_geo.py`
- `test_resources.py`
- `test_sar.py`

## Public Interface

- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:hazard_zone_gdf` (function)
- `conftest.py:shelter_locations_gdf` (function)
- `conftest.py:emergency_config` (function)

## Module Metadata

- Module: `GEO-INFER-EMERGENCY`
- Package: `geo_infer_emergency`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-emergency`
- Tests: `uv run python -m pytest GEO-INFER-EMERGENCY/tests`

## Dependencies

- `networkx>=2.6.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-EMERGENCY` module's current behavior through unit,
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
  GEO-INFER-EMERGENCY/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-EMERGENCY/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
