# GEO-INFER-TIME/src/geo_infer_time/utils

Utils workspace within `GEO-INFER-TIME`.

## Contents

- `__init__.py`

## Public Interface

- `__init__.py:validate_timeseries` (function)
- `__init__.py:detect_frequency` (function)
- `__init__.py:align_timeseries` (function)
- `__init__.py:create_timeseries` (function)
- `__init__.py:fill_gaps` (function)

## Module Metadata

- Module: `GEO-INFER-TIME`
- Package: `geo_infer_time`
- Version: `0.4.0`
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


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
