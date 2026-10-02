# GEO-INFER-HEALTH/src/geo_infer_health/utils

Utils workspace within `GEO-INFER-HEALTH`.

## Contents

- `__init__.py`
- `advanced_geospatial.py`
- `config.py`
- `geospatial_utils.py`
- `logging.py`

## Public Interface

- `advanced_geospatial.py:project_to_utm` (function)
- `advanced_geospatial.py:buffer_point` (function)
- `advanced_geospatial.py:spatial_clustering` (function)
- `advanced_geospatial.py:calculate_spatial_statistics` (function)
- `advanced_geospatial.py:validate_geographic_bounds` (function)
- `advanced_geospatial.py:interpolate_points` (function)
- `advanced_geospatial.py:find_centroid` (function)
- `advanced_geospatial.py:calculate_voronoi_regions` (function)
- `advanced_geospatial.py:calculate_spatial_autocorrelation` (function)
- `advanced_geospatial.py:calculate_hotspot_statistics` (function)
- `config.py:HealthConfig` (class)
- `config.py:load_yaml_config` (function)
- `config.py:load_json_config` (function)
- `config.py:validate_config` (function)
- `config.py:merge_configs` (function)
- `config.py:resolve_environment_variables` (function)
- `config.py:get_default_config_path` (function)
- `config.py:load_config` (function)
- `config.py:save_config` (function)
- `config.py:get_config_value` (function)

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
