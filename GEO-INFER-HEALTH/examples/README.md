# GEO-INFER-HEALTH/examples

Examples workspace within `GEO-INFER-HEALTH`.

## Contents

- `advanced_health_analysis.py`
- `example_disease_surveillance.py`
- `example_environmental_health.py`
- `example_healthcare_accessibility.py`
- `.gitkeep`

## Public Interface

- `advanced_health_analysis.py:create_sample_disease_data` (function)
- `advanced_health_analysis.py:create_sample_healthcare_data` (function)
- `advanced_health_analysis.py:create_sample_environmental_data` (function)
- `advanced_health_analysis.py:create_sample_population_data` (function)
- `advanced_health_analysis.py:demonstrate_active_inference_disease_analysis` (function)
- `advanced_health_analysis.py:demonstrate_healthcare_accessibility_analysis` (function)
- `advanced_health_analysis.py:demonstrate_environmental_health_analysis` (function)
- `advanced_health_analysis.py:demonstrate_advanced_geospatial_analysis` (function)
- `advanced_health_analysis.py:main` (function)
- `example_disease_surveillance.py:print_response` (function)
- `example_disease_surveillance.py:submit_sample_disease_reports` (function)
- `example_disease_surveillance.py:get_disease_reports` (function)
- `example_disease_surveillance.py:add_sample_population_data` (function)
- `example_disease_surveillance.py:identify_hotspots` (function)
- `example_disease_surveillance.py:get_local_incidence` (function)
- `example_environmental_health.py:print_response` (function)
- `example_environmental_health.py:submit_sample_env_readings` (function)
- `example_environmental_health.py:get_env_readings` (function)
- `example_environmental_health.py:get_readings_near_loc_example` (function)
- `example_environmental_health.py:get_average_exposure_example` (function)

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
