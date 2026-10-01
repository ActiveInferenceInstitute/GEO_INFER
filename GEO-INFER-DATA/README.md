# GEO-INFER-DATA

Foundational data backbone providing ETL pipelines, storage optimization, and data quality assurance for geospatial datasets.

## Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-DATA`
- Package: `geo_infer_data`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-DATA`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA`

## Dependencies

- `aiohttp>=3.8.0`
- `aiomqtt>=2.4.0`
- `boto3>=1.28.0`
- `fastapi>=0.100.0`
- `starlette>=0.27.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `minio>=7.1.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `psutil>=5.9.0`
- `psycopg2-binary>=2.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.5.0`
- `pyyaml>=6.0.0`
- `rasterio>=1.3.0`
- `redis>=4.5.0`
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
