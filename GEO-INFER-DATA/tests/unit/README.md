# GEO-INFER-DATA/tests/unit

Unit workspace within `GEO-INFER-DATA`.

## Contents

- `test_api.py`
- `test_api_connectors.py`
- `test_archive_safety.py`
- `test_caching.py`
- `test_cloud_connectors.py`
- `test_compression.py`
- `test_duckdb_provisioning.py`
- `test_duckdb_spatial.py`
- `test_error_handling.py`
- `test_file_connector.py`
- `test_format_detection.py`
- `test_geospatial_validation.py`
- `test_identifiers.py`
- `test_indexing.py`
- `test_ingestion.py`
- `test_optional_backend_boundaries.py`
- `test_performance.py`
- `test_pipeline.py`
- `test_rest_api_error_handling.py`
- `test_schemas.py`
- `test_storage.py`
- `test_stream_connectors.py`
- `test_timestamp_contracts.py`
- `test_validation.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-DATA`
- Package: `geo_infer_data`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-data`
- Tests: `uv run python -m pytest GEO-INFER-DATA/tests/unit`

## Dependencies

- `aiohttp>=3.8.0`
- `aiomqtt>=2.4.0`
- `fastapi>=0.100.0`
- `starlette>=0.27.0`
- `geopandas>=0.13.0`
- `geo-infer-time>=0.4.0`
- `h3>=4.5.0,<5`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `psutil>=5.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.5.0`
- `pyyaml>=6.0.0`
- `requests>=2.31.0`
- `rtree>=1.0.0`
- `shapely>=2.0.0`
- `sqlalchemy>=2.0.0`
- `urllib3>=2.0.6`
- `uvicorn[standard]>=0.23.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-DATA/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
