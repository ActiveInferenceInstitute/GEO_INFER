# GEO-INFER-DATA/src/geo_infer_data/utils

Utils workspace within `GEO-INFER-DATA`.

## Contents

- `caching.py`
- `compression.py`
- `dependencies.py`
- `duckdb_spatial.py`
- `format_detection.py`
- `identifiers.py`
- `indexing.py`
- `performance.py`
- `secure_serialization.py`
- `timestamps.py`
- `validation.py`

## Public Interface

- `caching.py:CacheEntry` (class)
- `caching.py:CacheManager` (class)
- `compression.py:DataCompressor` (class)
- `dependencies.py:MissingOptionalDependency` (class)
- `dependencies.py:require_dependency` (function)
- `duckdb_spatial.py:DuckDBSpatialError` (class)
- `duckdb_spatial.py:provision_spatial_extension` (function)
- `duckdb_spatial.py:read_cloud_native_vector` (function)
- `duckdb_spatial.py:duckdb_status` (function)
- `format_detection.py:FormatDetector` (class)
- `identifiers.py:validate_sql_identifier` (function)
- `indexing.py:SpatialIndexer` (class)
- `indexing.py:TemporalIndexer` (class)
- `performance.py:PerformanceMonitor` (class)
- `performance.py:OperationTracker` (class)
- `performance.py:DataProcessingProfiler` (class)
- `performance.py:StepProfiler` (class)
- `secure_serialization.py:PayloadSecurityError` (class)
- `secure_serialization.py:SigningKeyUnavailableError` (class)
- `secure_serialization.py:MalformedEnvelopeError` (class)

## Module Metadata

- Module: `GEO-INFER-DATA`
- Package: `geo_infer_data`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-data`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
