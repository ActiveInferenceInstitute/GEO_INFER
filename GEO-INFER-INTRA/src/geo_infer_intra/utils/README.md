# GEO-INFER-INTRA/src/geo_infer_intra/utils

Utils workspace within `GEO-INFER-INTRA`.

## Contents

- `__init__.py`
- `config.py`
- `geospatial_utils.py`
- `module_discovery.py`
- `time_series_utils.py`

## Public Interface

- `config.py:load_config` (function)
- `config.py:get_schema_path` (function)
- `config.py:validate_config` (function)
- `config.py:get_config_value` (function)
- `config.py:merge_configs` (function)
- `config.py:get_default_config_path` (function)
- `config.py:load_default_config` (function)
- `geospatial_utils.py:create_point` (function)
- `geospatial_utils.py:create_bbox` (function)
- `geospatial_utils.py:create_polygon` (function)
- `geospatial_utils.py:create_feature` (function)
- `geospatial_utils.py:create_feature_collection` (function)
- `geospatial_utils.py:is_valid_geojson` (function)
- `geospatial_utils.py:load_geojson_file` (function)
- `geospatial_utils.py:save_geojson_file` (function)
- `geospatial_utils.py:haversine_distance` (function)
- `geospatial_utils.py:create_sample_h3_data` (function)
- `module_discovery.py:collect_test_modules` (function)
- `module_discovery.py:import_module_by_path` (function)
- `module_discovery.py:find_modules_by_name` (function)

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-intra`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
