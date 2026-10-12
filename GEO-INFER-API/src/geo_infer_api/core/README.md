# GEO-INFER-API/src/geo_infer_api/core

Core workspace within `GEO-INFER-API`.

## Contents

- `__init__.py`
- `config.py`
- `exceptions.py`
- `middleware.py`

## Public Interface

- `config.py:Settings` (class)
- `config.py:get_settings` (function)
- `exceptions.py:APIError` (class)
- `exceptions.py:ValidationError` (class)
- `exceptions.py:NotFoundError` (class)
- `exceptions.py:ConflictError` (class)
- `exceptions.py:GeometryError` (class)
- `exceptions.py:ProcessingError` (class)
- `exceptions.py:BadRequestError` (class)
- `middleware.py:ErrorHandlerMiddleware` (class)
- `middleware.py:RequestLoggingMiddleware` (class)

## Module Metadata

- Module: `GEO-INFER-API`
- Package: `geo_infer_api`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-api`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module API`

## Dependencies

- `starlette>=0.27.0`
- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `uvicorn>=0.21.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
