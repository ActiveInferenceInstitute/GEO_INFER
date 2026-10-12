# GEO-INFER-API/src/geo_infer_api/models

Models workspace within `GEO-INFER-API`.

## Contents

- `__init__.py`
- `geojson.py`

## Public Interface

- `geojson.py:GeoJSONType` (class)
- `geojson.py:GeometryBase` (class)
- `geojson.py:Point` (class)
- `geojson.py:LineString` (class)
- `geojson.py:Polygon` (class)
- `geojson.py:MultiPoint` (class)
- `geojson.py:MultiLineString` (class)
- `geojson.py:MultiPolygon` (class)
- `geojson.py:Feature` (class)
- `geojson.py:FeatureCollection` (class)
- `geojson.py:PolygonFeature` (class)
- `geojson.py:PolygonFeatureCollection` (class)

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
