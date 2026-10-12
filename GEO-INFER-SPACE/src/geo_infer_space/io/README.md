# GEO-INFER-SPACE/src/geo_infer_space/io

Io workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `format_handlers.py`
- `point_cloud_io.py`
- `raster_io.py`
- `vector_io.py`

## Public Interface

- `format_handlers.py:FormatHandler` (class)
- `format_handlers.py:GeoJSONHandler` (class)
- `format_handlers.py:ShapefileHandler` (class)
- `format_handlers.py:GeoTIFFHandler` (class)
- `format_handlers.py:COGHandler` (class)
- `format_handlers.py:LASHandler` (class)
- `format_handlers.py:NetCDFHandler` (class)
- `format_handlers.py:get_handler_for_path` (function)
- `format_handlers.py:list_supported_formats` (function)
- `point_cloud_io.py:PointCloudReader` (class)
- `point_cloud_io.py:PointCloudWriter` (class)
- `point_cloud_io.py:read_point_cloud_file` (function)
- `point_cloud_io.py:write_point_cloud_file` (function)
- `point_cloud_io.py:supported_point_cloud_formats` (function)
- `raster_io.py:RasterReader` (class)
- `raster_io.py:RasterWriter` (class)
- `raster_io.py:read_raster_file` (function)
- `raster_io.py:write_raster_file` (function)
- `raster_io.py:supported_raster_formats` (function)
- `raster_io.py:detect_raster_format` (function)

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
