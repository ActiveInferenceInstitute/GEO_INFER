# GEO-INFER-API/examples

Examples workspace within `GEO-INFER-API`.

## Contents

- `python_client.py`
- `curl_examples.sh`
- `javascript_client.js`

## Public Interface

- `python_client.py:list_collections` (function)
- `python_client.py:get_polygon_collection` (function)
- `python_client.py:list_polygon_features` (function)
- `python_client.py:get_polygon_feature` (function)
- `python_client.py:create_polygon_feature` (function)
- `python_client.py:update_polygon_feature` (function)
- `python_client.py:delete_polygon_feature` (function)
- `python_client.py:calculate_polygon_area` (function)
- `python_client.py:simplify_polygon` (function)
- `python_client.py:check_point_in_polygon` (function)
- `python_client.py:main` (function)

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
