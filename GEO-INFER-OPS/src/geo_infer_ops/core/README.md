# GEO-INFER-OPS/src/geo_infer_ops/core

Core workspace within `GEO-INFER-OPS`.

## Contents

- `__init__.py`
- `cache.py`
- `config.py`
- `deployment.py`
- `logging.py`
- `monitoring.py`
- `orchestrator.py`
- `secure_serialization.py`
- `security.py`
- `testing.py`

## Public Interface

- `cache.py:CacheSerializer` (class)
- `cache.py:CacheManager` (class)
- `config.py:LoggingConfig` (class)
- `config.py:MonitoringConfig` (class)
- `config.py:TestingConfig` (class)
- `config.py:DockerConfig` (class)
- `config.py:KubernetesConfig` (class)
- `config.py:DeploymentConfig` (class)
- `config.py:TLSConfig` (class)
- `config.py:AuthConfig` (class)
- `config.py:SecurityConfig` (class)
- `config.py:RedisConfig` (class)
- `config.py:CacheConfig` (class)
- `config.py:Config` (class)
- `config.py:load_config` (function)
- `config.py:get_config` (function)
- `config.py:update_config` (function)
- `deployment.py:DeploymentManager` (class)
- `logging.py:get_logger` (function)
- `monitoring.py:reset_metrics` (function)

## Module Metadata

- Module: `GEO-INFER-OPS`
- Package: `geo_infer_ops`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ops`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module OPS`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module OPS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
