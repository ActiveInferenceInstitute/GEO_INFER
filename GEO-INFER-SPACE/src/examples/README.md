# GEO-INFER-SPACE/src/examples

Examples workspace within `GEO-INFER-SPACE`.

## Contents

- `california_demo_outputs/`
- `california_multilayer_demo.py`

## Public Interface

- `california_multilayer_demo.py:configure_logging` (function)
- `california_multilayer_demo.py:generate_zoning_geojson` (function)
- `california_multilayer_demo.py:generate_water_geojson` (function)
- `california_multilayer_demo.py:generate_climate_geojson` (function)
- `california_multilayer_demo.py:geojson_to_h3_polygons` (function)
- `california_multilayer_demo.py:cell_to_latlngjson_polygons` (function)
- `california_multilayer_demo.py:add_h3_layer_to_map` (function)
- `california_multilayer_demo.py:add_point_layer_to_map` (function)
- `california_multilayer_demo.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

- `fastapi>=0.100.0`
- `fiona>=1.8.0`
- `geojson-pydantic>=2.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `networkx>=2.6.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.3.0`
- `python-multipart>=0.0.5`
- `pyyaml>=6.0`
- `requests>=2.28.0`
- `rasterio>=1.3.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `shapely>=2.0.0`
- `uvicorn>=0.15.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
