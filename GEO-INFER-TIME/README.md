# GEO-INFER-TIME

Temporal analysis, time series processing, forecasting, and spatio-temporal data fusion for dynamic geospatial applications.

## Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `.gitignore`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-TIME`
- Package: `geo_infer_time`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-time`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.6.1`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
```


## Visualization Contracts

- Temporal plotting helpers validate nonempty finite series, aligned timestamps,
  confidence bounds, and anomaly indices before rendering.
- Figure creation and saving remain scoped to each call, preserving reusable
  style configuration without leaking global matplotlib state.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
