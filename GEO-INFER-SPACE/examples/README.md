# GEO-INFER-SPACE/examples

Examples workspace within `GEO-INFER-SPACE`.

## Contents

- `california_multilayer_demo.py`
- `demo_all_methods.py`
- `h3_advanced_applications.py`
- `h3_comprehensive_examples.py`
- `h3_examples.py`
- `h3_integration_examples.py`
- `multiple_dispatch_demo.py`
- `nested_orchestrator_examples.py`
- `run_all.py`
- `verify_installation.py`

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
- `demo_all_methods.py:run_demo` (function)
- `demo_all_methods.py:main` (function)
- `h3_advanced_applications.py:example_demand_forecasting_ml` (function)
- `h3_advanced_applications.py:example_disaster_response_system` (function)
- `h3_advanced_applications.py:example_performance_optimization` (function)
- `h3_advanced_applications.py:example_integrated_smart_city` (function)
- `h3_advanced_applications.py:main` (function)
- `h3_comprehensive_examples.py:example_1_basic_h3_operations` (function)
- `h3_comprehensive_examples.py:example_2_city_coverage_analysis` (function)
- `h3_comprehensive_examples.py:example_3_transportation_corridor` (function)
- `h3_comprehensive_examples.py:example_4_retail_catchment_analysis` (function)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

- `geo-infer-time>=0.4.0`
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
