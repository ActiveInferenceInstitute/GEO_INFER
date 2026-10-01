# GEO-INFER-OPS/tests/unit

Unit workspace within `GEO-INFER-OPS`.

## Contents

- `test_app.py`
- `test_cache_envelope.py`
- `test_deployment_timeouts.py`
- `test_find_config_file.py`
- `test_health_checks.py`
- `test_logging_hygiene.py`
- `test_orchestrator.py`
- `test_packaged_resources.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-OPS`
- Package: `geo_infer_ops`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ops`
- Tests: `uv run python -m pytest GEO-INFER-OPS/tests/unit`

## Dependencies

- `fastapi>=0.100.0`
- `prometheus-client>=0.12.0`
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
uv run python -m pytest GEO-INFER-OPS/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
