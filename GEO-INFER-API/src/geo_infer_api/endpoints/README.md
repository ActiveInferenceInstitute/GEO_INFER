# GEO-INFER-API/src/geo_infer_api/endpoints

Endpoints workspace within `GEO-INFER-API`.

## Contents

- `__init__.py`
- `algorithms_router.py`
- `geojson_router.py`
- `health_router.py`

## Public Interface

- `algorithms_router.py:AlgorithmParameterOut` (class)
- `algorithms_router.py:AlgorithmOut` (class)
- `algorithms_router.py:AlgorithmRunRequest` (class)
- `algorithms_router.py:AlgorithmRunResponse` (class)
- `algorithms_router.py:list_algorithms` (function)
- `algorithms_router.py:get_algorithm` (function)
- `algorithms_router.py:run_algorithm` (function)
- `geojson_router.py:MultiPolygonRequest` (class)
- `geojson_router.py:DistanceRequest` (class)
- `geojson_router.py:list_collections` (function)
- `geojson_router.py:get_polygon_collection` (function)
- `geojson_router.py:list_polygon_features` (function)
- `geojson_router.py:get_polygon_feature` (function)
- `geojson_router.py:create_polygon_feature_endpoint` (function)
- `geojson_router.py:update_polygon_feature` (function)
- `geojson_router.py:delete_polygon_feature` (function)
- `geojson_router.py:calculate_area` (function)
- `geojson_router.py:simplify_polygon_endpoint` (function)
- `geojson_router.py:check_polygon_contains_point` (function)
- `geojson_router.py:create_buffer_endpoint` (function)

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
