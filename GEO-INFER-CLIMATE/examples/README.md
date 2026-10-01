# GEO-INFER-CLIMATE/examples

Examples workspace within `GEO-INFER-CLIMATE`.

## Contents

- `basic_climate_analysis.py`
- `climate_projection.py`

## Public Interface

- `basic_climate_analysis.py:create_sample_climate_data` (function)
- `basic_climate_analysis.py:main` (function)
- `climate_projection.py:create_synthetic_data` (function)
- `climate_projection.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-CLIMATE`
- Package: `geo_infer_climate`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-climate`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
