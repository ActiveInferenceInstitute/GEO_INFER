# GEO-INFER-COG/src/geo_infer_cog/utils

Utils workspace within `GEO-INFER-COG`.

## Contents

- `__init__.py`
- `helpers.py`
- `rng.py`
- `validation.py`

## Public Interface

- `helpers.py:load_cognitive_profile` (function)
- `helpers.py:save_cognitive_profile` (function)
- `helpers.py:load_cognitive_model` (function)
- `helpers.py:save_cognitive_model` (function)
- `helpers.py:create_default_cognitive_config` (function)
- `helpers.py:setup_cognitive_logging` (function)
- `helpers.py:calculate_cognitive_load` (function)
- `helpers.py:format_spatial_data_for_display` (function)
- `helpers.py:create_performance_report` (function)
- `helpers.py:export_cognitive_insights` (function)
- `helpers.py:validate_file_path` (function)
- `helpers.py:create_directory_structure` (function)
- `helpers.py:cleanup_temp_files` (function)
- `rng.py:resolve_rng` (function)
- `rng.py:resolve_optional_rng` (function)
- `rng.py:spawn_rng` (function)
- `rng.py:derive_int_seed` (function)
- `validation.py:validate_spatial_data` (function)
- `validation.py:validate_geometry` (function)
- `validation.py:validate_point_coordinates` (function)

## Module Metadata

- Module: `GEO-INFER-COG`
- Package: `geo_infer_cog`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-COG`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

## Dependencies

- `numpy>=1.20.0`
- `networkx>=2.6`
- `pyyaml>=5.4`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
