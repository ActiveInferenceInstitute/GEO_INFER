# GEO-INFER-SPACE/src/geo_infer_space/core

Core workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `algorithm_registry.py`
- `analytics.py`
- `api_clients.py`
- `base_module.py`
- `data_integrator.py`
- `dispatcher.py`
- `geolibre_projects.py`
- `geometric_operations.py`
- `h3_policy.py`
- `interfaces.py`
- `place_analyzer.py`
- `sparse_transition.py`
- `spatial_indexing.py`
- `spatial_methods.py`
- `spatial_processor.py`
- `spatiotemporal.py`
- `state_space.py`
- `statistics.py`
- `unified_backend.py`
- `visualization_engine.py`
- `visualization_receipt.py`
- `whitebox_bridge.py`

## Public Interface

- `algorithm_registry.py:ParameterSpec` (class)
- `algorithm_registry.py:ProcessingAlgorithm` (class)
- `algorithm_registry.py:ProcessingContext` (class)
- `algorithm_registry.py:AlgorithmRegistry` (class)
- `algorithm_registry.py:build_reference_registry` (function)
- `analytics.py:SpatialAnalyticsInterface` (class)
- `api_clients.py:BaseAPIManager` (class)
- `api_clients.py:GeneralGeoDataFetcher` (class)
- `base_module.py:BaseAnalysisModule` (class)
- `data_integrator.py:DataIntegrator` (class)
- `dispatcher.py:SpatialBackendDispatcher` (class)
- `dispatcher.py:get_backend_dispatcher` (function)
- `dispatcher.py:configure_backends` (function)
- `dispatcher.py:reset_dispatcher` (function)
- `geolibre_projects.py:default_map_view` (function)
- `geolibre_projects.py:geojson_layer` (function)
- `geolibre_projects.py:tile_layer` (function)
- `geolibre_projects.py:build_project` (function)
- `geolibre_projects.py:dumps_project` (function)
- `geolibre_projects.py:write_project` (function)

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
