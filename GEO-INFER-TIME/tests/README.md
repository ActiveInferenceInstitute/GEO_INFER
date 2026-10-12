# GEO-INFER-TIME/tests

Tests workspace within `GEO-INFER-TIME`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`
- `test_advanced_forecasting.py`
- `test_analysis_extended.py`
- `test_event_detection.py`
- `test_forecasting.py`
- `test_interpolation.py`
- `test_io_utils_db.py`
- `test_kafka_transport.py`
- `test_statistics_extended.py`
- `test_stream_processing.py`
- `test_stream_processing_adapters.py`
- `test_stream_transport_contracts.py`
- `test_temporal_analysis.py`
- `test_temporal_statistics.py`
- `test_temporal_visualization.py`
- `test_timeseries_model.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-TIME`
- Package: `geo_infer_time`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-time`
- Tests: `uv run python -m pytest GEO-INFER-TIME/tests`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.6.1`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-TIME` module's current behavior through unit,
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
  GEO-INFER-TIME/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python -m pytest GEO-INFER-TIME/tests
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
