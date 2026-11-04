# GEO-INFER-SPACE/src/geo_infer_space/backends/gpu

Gpu workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `gpu_acceleration.py`

## Public Interface

- `gpu_acceleration.py:AcceleratorUnavailableError` (class)
- `gpu_acceleration.py:get_backend_diagnostics` (function)
- `gpu_acceleration.py:get_available_backends` (function)
- `gpu_acceleration.py:is_accelerator_available` (function)
- `gpu_acceleration.py:pairwise_haversine_kernel` (function)
- `gpu_acceleration.py:euclidean_distance_kernel` (function)
- `gpu_acceleration.py:gpu_spatial_join_by_distance` (function)
- `gpu_acceleration.py:spatial_join_kernel` (function)
- `gpu_acceleration.py:h3_grid_distance_kernel` (function)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

- `geo-infer-time>=0.4.0`
- `fastapi>=0.100.0`
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
