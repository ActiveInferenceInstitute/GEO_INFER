# GEO-INFER-HEALTH/tests/unit

Unit workspace within `GEO-INFER-HEALTH`.

## Contents

- `test_active_inference_failure_flags.py`
- `test_advanced_geospatial.py`
- `test_api_routers.py`
- `test_cli_validation.py`
- `test_config.py`
- `test_config_packaged.py`
- `test_disease_surveillance.py`
- `test_enhanced_disease_surveillance.py`
- `test_environmental_health.py`
- `test_geospatial_utils.py`
- `test_healthcare_accessibility.py`
- `test_hotspot_loader.py`
- `test_models.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-HEALTH`
- Package: `geo_infer_health`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-health`
- Tests: `uv run python -m pytest GEO-INFER-HEALTH/tests/unit`

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
uv run python -m pytest GEO-INFER-HEALTH/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
