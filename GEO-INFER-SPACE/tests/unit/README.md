# GEO-INFER-SPACE/tests/unit

Unit workspace within `GEO-INFER-SPACE`.

## Contents

- `test_algorithm_registry.py`
- `test_analytics_comprehensive.py`
- `test_analytics_context_contract.py`
- `test_analytics_warnings.py`
- `test_api_schemas.py`
- `test_backends_comprehensive.py`
- `test_base_module.py`
- `test_config_loader_packaged_config.py`
- `test_core.py`
- `test_data_integrator.py`
- `test_degradation_warnings.py`
- `test_dispatch_comprehensive.py`
- `test_geolibre_projects.py`
- `test_gis_submodule.py`
- `test_gpu_acceleration.py`
- `test_h3_enhanced.py`
- `test_h3_operations_runtime.py`
- `test_h3_policy.py`
- `test_h3_predicate_logging.py`
- `test_io_modules.py`
- `test_ml_integration_tables.py`
- `test_morans_i_variance_parity.py`
- `test_nested_analytics.py`
- `test_nested_comprehensive.py`
- `test_nested_h3_contract.py`
- `test_nested_messaging_routing.py`
- `test_place_analyzer.py`
- `test_place_analyzer_synthetic.py`
- `test_raster_expression_security.py`
- `test_sparse_transition.py`
- `test_spatial_methods.py`
- `test_spatial_processor.py`
- `test_spatial_statistics.py`
- `test_spatial_utils.py`
- `test_spatiotemporal.py`
- `test_spatiotemporal_alignment.py`
- `test_state_space.py`
- `test_temporal_analytics.py`
- `test_unified_backend.py`
- `test_unified_backend_geojson_seam.py`
- `test_unified_comprehensive.py`
- `test_visualization_engine.py`
- `test_visualization_receipts.py`
- `test_weight_matrix_backend_contract.py`
- `test_whitebox_bridge.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python -m pytest GEO-INFER-SPACE/tests/unit`

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
uv run python -m pytest GEO-INFER-SPACE/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
