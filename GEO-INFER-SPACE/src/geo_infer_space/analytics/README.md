# GEO-INFER-SPACE/src/geo_infer_space/analytics

Analytics workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `geostatistics.py`
- `network.py`
- `point_cloud.py`
- `raster.py`
- `spatiotemporal.py`
- `temporal.py`
- `vector.py`

## Public Interface

- `geostatistics.py:spatial_interpolation` (function)
- `geostatistics.py:clustering_analysis` (function)
- `geostatistics.py:hotspot_detection` (function)
- `geostatistics.py:spatial_autocorrelation` (function)
- `geostatistics.py:variogram_analysis` (function)
- `network.py:shortest_path` (function)
- `network.py:service_area` (function)
- `network.py:network_connectivity` (function)
- `network.py:routing_analysis` (function)
- `network.py:accessibility_analysis` (function)
- `point_cloud.py:PointCloud` (class)
- `point_cloud.py:load_point_cloud` (function)
- `point_cloud.py:point_cloud_filtering` (function)
- `point_cloud.py:feature_extraction` (function)
- `point_cloud.py:classification` (function)
- `point_cloud.py:surface_generation` (function)
- `raster.py:terrain_analysis` (function)
- `raster.py:map_algebra` (function)
- `raster.py:focal_statistics` (function)
- `raster.py:zonal_statistics` (function)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

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
