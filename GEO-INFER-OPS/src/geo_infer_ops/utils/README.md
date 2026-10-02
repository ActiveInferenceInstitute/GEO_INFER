# GEO-INFER-OPS/src/geo_infer_ops/utils

Utils workspace within `GEO-INFER-OPS`.

## Contents

- `__init__.py`
- `config.py`
- `shared_logging.py`

## Public Interface

- `config.py:find_config_file` (function)
- `config.py:load_config` (function)
- `shared_logging.py:configure_logging` (function)
- `shared_logging.py:get_logger` (function)
- `shared_logging.py:LoggingContext` (class)
- `shared_logging.py:setup_module_logging` (function)

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
