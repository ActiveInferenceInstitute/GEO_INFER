# GEO-INFER-API/tests/unit

Unit workspace within `GEO-INFER-API`.

## Contents

- `test_algorithms_router.py`
- `test_config.py`
- `test_exceptions.py`
- `test_geojson_helpers.py`
- `test_geojson_helpers_extended.py`
- `test_geojson_router.py`
- `test_geojson_visualization.py`
- `test_middleware.py`
- `test_models_geojson.py`
- `test_security_config.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-API`
- Package: `geo_infer_api`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-api`
- Tests: `uv run python -m pytest GEO-INFER-API/tests/unit`

## Dependencies

- `starlette>=0.27.0`
- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `uvicorn>=0.21.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-API/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
