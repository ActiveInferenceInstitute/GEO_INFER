# GEO-INFER-LOG/src/geo_infer_log/utils

Utils workspace within `GEO-INFER-LOG`.

## Contents

- `__init__.py`
- `conversion.py`
- `geo.py`
- `optimization.py`
- `visualization.py`

## Public Interface

- `conversion.py:km_to_miles` (function)
- `conversion.py:miles_to_km` (function)
- `conversion.py:meters_to_feet` (function)
- `conversion.py:feet_to_meters` (function)
- `conversion.py:km_per_hour_to_mph` (function)
- `conversion.py:mph_to_km_per_hour` (function)
- `conversion.py:liters_to_gallons` (function)
- `conversion.py:gallons_to_liters` (function)
- `conversion.py:kg_to_pounds` (function)
- `conversion.py:pounds_to_kg` (function)
- `conversion.py:cubic_meters_to_cubic_feet` (function)
- `conversion.py:cubic_feet_to_cubic_meters` (function)
- `conversion.py:celsius_to_fahrenheit` (function)
- `conversion.py:fahrenheit_to_celsius` (function)
- `conversion.py:minutes_to_hours` (function)
- `conversion.py:hours_to_minutes` (function)
- `geo.py:haversine_distance` (function)
- `geo.py:get_bbox` (function)
- `geo.py:coords_to_geojson` (function)
- `geo.py:points_to_gdf` (function)

## Module Metadata

- Module: `GEO-INFER-LOG`
- Package: `geo_infer_log`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-log`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module LOG`

## Dependencies

- `numpy>=1.20.0`
- `starlette>=0.27.0`
- `pandas>=1.3.0`
- `geopandas>=0.13.0`
- `networkx>=2.6.0`
- `pulp>=2.7.0,<3`
- `shapely>=2.0.0`
- `pydantic>=2.0.0`
- `fastapi>=0.100.0`
- `scipy>=1.9.0`
- `matplotlib>=3.5.0`
- `folium>=0.14.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module LOG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
