# GEO-INFER-WATER/examples

Examples workspace within `GEO-INFER-WATER`.

## Contents

- `__init__.py`
- `basic_water_analysis.py`
- `water_quality_monitoring.py`

## Public Interface

- `basic_water_analysis.py:main` (function)
- `water_quality_monitoring.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-WATER`
- Package: `geo_infer_water`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-water`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module WATER`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module WATER
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
