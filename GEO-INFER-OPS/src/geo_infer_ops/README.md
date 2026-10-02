# GEO-INFER-OPS/src/geo_infer_ops

Geo Infer Ops workspace within `GEO-INFER-OPS`.

## Contents

- `config/`
- `core/`
- `deployment/`
- `health/`
- `utils/`
- `__init__.py`
- `app.py`

## Public Interface

- `app.py:create_app` (function)

## Module Metadata

- Module: `GEO-INFER-OPS`
- Package: `geo_infer_ops`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-ops`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module OPS`

## Dependencies

- `fastapi>=0.100.0`
- `prometheus-client>=0.20.0`
- `prometheus-fastapi-instrumentator>=5.7.0`
- `pydantic>=2.0.0`
- `structlog>=21.1.0`
- `kubernetes>=29.0.0`
- `redis>=4.5.0`
- `PyJWT>=2.0.0`
- `cryptography>=40.0.0`
- `psutil>=5.9.0`
- `PyYAML>=6.0.0`
- `uvicorn>=0.23.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module OPS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
