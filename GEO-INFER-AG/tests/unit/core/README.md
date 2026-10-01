# GEO-INFER-AG/tests/unit/core

Core workspace within `GEO-INFER-AG`.

## Contents

- `test_agricultural_analysis.py`
- `test_field_boundary.py`
- `test_seasonal_analysis.py`
- `test_sustainability.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-AG`
- Package: `geo_infer_ag`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ag`
- Tests: `uv run python -m pytest GEO-INFER-AG/tests/unit/core`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `rasterio>=1.2.0`
- `pyproj>=3.0.0`
- `matplotlib>=3.3.0`
- `scipy>=1.6.0`
- `xarray>=0.18.0`
- `joblib>=1.0.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-AG/tests/unit/core
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
