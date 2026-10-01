# GEO-INFER-MARINE/examples

Examples workspace within `GEO-INFER-MARINE`.

## Contents

- `__init__.py`
- `basic_marine_analysis.py`
- `marine_ecosystem_analysis.py`

## Public Interface

- `basic_marine_analysis.py:main` (function)
- `marine_ecosystem_analysis.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-MARINE`
- Package: `geo_infer_marine`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-marine`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE`

## Dependencies

- `numpy>=1.20.0`
- `xarray>=0.19.0`
- `netcdf4>=1.5.8`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
