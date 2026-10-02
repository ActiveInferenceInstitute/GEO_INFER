# GEO-INFER-TIME/examples

Examples workspace within `GEO-INFER-TIME`.

## Contents

- `basic_temporal_analysis.py`
- `demo_all_methods.py`

## Public Interface

- `basic_temporal_analysis.py:generate_sample_timeseries` (function)
- `basic_temporal_analysis.py:main` (function)
- `demo_all_methods.py:success` (function)
- `demo_all_methods.py:info` (function)
- `demo_all_methods.py:section` (function)

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
