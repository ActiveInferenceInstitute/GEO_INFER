# GEO-INFER-HEALTH/src/geo_infer_health/api

Api workspace within `GEO-INFER-HEALTH`.

## Contents

- `__init__.py`
- `api_disease_surveillance.py`
- `api_environmental_health.py`
- `api_healthcare_accessibility.py`
- `.gitkeep`

## Public Interface

- `api_disease_surveillance.py:reset_stores` (function)
- `api_disease_surveillance.py:submit_disease_report` (function)
- `api_disease_surveillance.py:get_all_disease_reports` (function)
- `api_disease_surveillance.py:identify_disease_hotspots` (function)
- `api_disease_surveillance.py:get_local_incidence_rate` (function)
- `api_disease_surveillance.py:add_population_data_area` (function)
- `api_disease_surveillance.py:get_all_population_data` (function)
- `api_environmental_health.py:reset_stores` (function)
- `api_environmental_health.py:submit_environmental_reading` (function)
- `api_environmental_health.py:get_all_environmental_readings` (function)
- `api_environmental_health.py:get_readings_near_location_api` (function)
- `api_environmental_health.py:get_average_exposure_api` (function)
- `api_healthcare_accessibility.py:reset_stores` (function)
- `api_healthcare_accessibility.py:add_health_facility` (function)
- `api_healthcare_accessibility.py:get_all_health_facilities` (function)
- `api_healthcare_accessibility.py:find_nearby_facilities` (function)
- `api_healthcare_accessibility.py:get_nearest_facility_endpoint` (function)
- `api_healthcare_accessibility.py:get_facility_population_ratio` (function)
- `api_healthcare_accessibility.py:add_accessibility_population_data` (function)
- `api_healthcare_accessibility.py:get_accessibility_population_data` (function)

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
