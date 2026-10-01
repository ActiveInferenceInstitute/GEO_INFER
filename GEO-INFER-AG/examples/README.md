# GEO-INFER-AG/examples

Examples workspace within `GEO-INFER-AG`.

## Contents

- `basic_agricultural_analysis.py`
- `precision_agriculture.py`

## Public Interface

- `basic_agricultural_analysis.py:make_field_gdf` (function)
- `basic_agricultural_analysis.py:make_ndvi_series` (function)
- `basic_agricultural_analysis.py:main` (function)
- `precision_agriculture.py:make_farm_data` (function)
- `precision_agriculture.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-AG`
- Package: `geo_infer_ag`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ag`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AG`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module AG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
