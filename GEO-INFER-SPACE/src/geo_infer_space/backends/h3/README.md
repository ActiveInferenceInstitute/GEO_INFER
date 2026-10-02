# GEO-INFER-SPACE/src/geo_infer_space/backends/h3

H3 workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `analytics.py`
- `core.py`
- `datasets.py`
- `h3_backend.py`
- `ml_integration.py`
- `operations.py`
- `utils.py`
- `visualization.py`

## Public Interface

- `analytics.py:H3SpatialAnalyzer` (class)
- `analytics.py:H3ClusterAnalyzer` (class)
- `analytics.py:H3DensityAnalyzer` (class)
- `analytics.py:H3NetworkAnalyzer` (class)
- `analytics.py:H3TemporalAnalyzer` (class)
- `core.py:H3Cell` (class)
- `core.py:H3Grid` (class)
- `core.py:H3Analytics` (class)
- `core.py:H3Visualizer` (class)
- `core.py:H3Validator` (class)
- `datasets.py:H3Dataset` (class)
- `datasets.py:H3DataLoader` (class)
- `datasets.py:H3DataExporter` (class)
- `h3_backend.py:H3Backend` (class)
- `ml_integration.py:H3MLFeatureEngine` (class)
- `ml_integration.py:H3DisasterResponse` (class)
- `ml_integration.py:H3PerformanceOptimizer` (class)
- `operations.py:get_resolution_info` (function)
- `operations.py:find_optimal_resolution` (function)
- `operations.py:create_h3_grid_for_bounds` (function)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

- `geo-infer-time>=0.4.0`
- `fastapi>=0.100.0`
- `fiona>=1.8.0`
- `geojson-pydantic>=2.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `networkx>=2.6.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.3.0`
- `python-multipart>=0.0.5`
- `pyyaml>=6.0`
- `requests>=2.28.0`
- `rasterio>=1.3.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `shapely>=2.0.0`
- `uvicorn>=0.15.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
