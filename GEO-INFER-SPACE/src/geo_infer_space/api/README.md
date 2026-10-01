# GEO-INFER-SPACE/src/geo_infer_space/api

Api workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `rest_api.py`
- `schemas.py`

## Public Interface

- `rest_api.py:general_exception_handler` (function)
- `rest_api.py:geojson_to_gdf` (function)
- `rest_api.py:gdf_to_geojson` (function)
- `rest_api.py:buffer_analysis_endpoint` (function)
- `rest_api.py:proximity_analysis_endpoint` (function)
- `rest_api.py:interpolation_endpoint` (function)
- `rest_api.py:clustering_endpoint` (function)
- `rest_api.py:hotspot_detection_endpoint` (function)
- `rest_api.py:network_analysis_endpoint` (function)
- `rest_api.py:h3_analysis_endpoint` (function)
- `rest_api.py:health_check` (function)
- `rest_api.py:get_capabilities` (function)
- `schemas.py:SpatialAnalysisRequest` (class)
- `schemas.py:SpatialAnalysisResponse` (class)
- `schemas.py:BufferAnalysisRequest` (class)
- `schemas.py:ProximityAnalysisRequest` (class)
- `schemas.py:InterpolationRequest` (class)
- `schemas.py:ClusteringRequest` (class)
- `schemas.py:HotspotRequest` (class)
- `schemas.py:NetworkAnalysisRequest` (class)

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
