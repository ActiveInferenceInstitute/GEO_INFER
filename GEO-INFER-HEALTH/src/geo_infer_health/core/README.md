# GEO-INFER-HEALTH/src/geo_infer_health/core

Core workspace within `GEO-INFER-HEALTH`.

## Contents

- `__init__.py`
- `disease_surveillance.py`
- `enhanced_disease_surveillance.py`
- `environmental_health.py`
- `healthcare_accessibility.py`

## Public Interface

- `disease_surveillance.py:DiseaseHotspotAnalyzer` (class)
- `enhanced_disease_surveillance.py:ActiveInferenceDiseaseAnalyzer` (class)
- `environmental_health.py:EnvironmentalHealthAnalyzer` (class)
- `healthcare_accessibility.py:HealthcareAccessibilityAnalyzer` (class)

## Module Metadata

- Module: `GEO-INFER-HEALTH`
- Package: `geo_infer_health`
- Version: `0.3.0`
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
