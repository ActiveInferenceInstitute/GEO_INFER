# GEO-INFER-HEALTH/src/geo_infer_health

Geo Infer Health workspace within `GEO-INFER-HEALTH`.

## Contents

- `api/`
- `core/`
- `models/`
- `utils/`
- `__init__.py`
- `cli.py`
- `health_config.yaml`

## Public Interface

- `cli.py:setup_cli` (function)
- `cli.py:main` (function)
- `cli.py:run_server` (function)
- `cli.py:run_analysis` (function)
- `cli.py:run_hotspot_analysis` (function)
- `cli.py:run_accessibility_analysis` (function)
- `cli.py:run_environment_analysis` (function)
- `cli.py:run_batch_processing` (function)
- `cli.py:run_validation` (function)

## Module Metadata

- Module: `GEO-INFER-HEALTH`
- Package: `geo_infer_health`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-health`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module HEALTH`

## Dependencies

- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`
- `pydantic>=2.5.0`
- `pydantic-settings>=2.1.0`
- `pyyaml>=6.0.0`
- `loguru>=0.7.0`
- `geopandas>=0.13.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module HEALTH
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
