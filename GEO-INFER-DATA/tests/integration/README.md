# GEO-INFER-DATA/tests/integration

Integration workspace within `GEO-INFER-DATA`.

## Contents

- `test_end_to_end.py`
- `test_pipeline_integration.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-DATA`
- Package: `geo_infer_data`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-data`
- Tests: `uv run python -m pytest GEO-INFER-DATA/tests/integration`

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
uv run python -m pytest GEO-INFER-DATA/tests/integration
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
