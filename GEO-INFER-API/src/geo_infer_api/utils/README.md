# GEO-INFER-API/src/geo_infer_api/utils

Utils workspace within `GEO-INFER-API`.

## Contents

- `__init__.py`
- `geojson_helpers.py`

## Public Interface

- `geojson_helpers.py:validate_polygon_rings` (function)
- `geojson_helpers.py:calculate_polygon_area` (function)
- `geojson_helpers.py:polygon_contains_point` (function)
- `geojson_helpers.py:simplify_polygon` (function)
- `geojson_helpers.py:create_polygon_feature` (function)
- `geojson_helpers.py:create_buffer` (function)
- `geojson_helpers.py:calculate_intersection` (function)
- `geojson_helpers.py:calculate_union` (function)
- `geojson_helpers.py:calculate_distance` (function)

## Module Metadata

- Module: `GEO-INFER-API`
- Package: `geo_infer_api`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-api`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module API`

## Dependencies

- `starlette>=0.27.0`
- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `uvicorn>=0.21.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
