# GEO-INFER-PLACE/tests/unit

Unit workspace within `GEO-INFER-PLACE`.

## Contents

- `test_api_clients.py`
- `test_bioregion_visualization.py`
- `test_caching.py`
- `test_cascadia_flowlines.py`
- `test_comprehensive_dashboard.py`
- `test_crescent_city_intel.py`
- `test_dashboard_advanced.py`
- `test_data_sources.py`
- `test_del_norte_analyzers.py`
- `test_del_norte_dashboard_packaged_config.py`
- `test_del_norte_demo_orchestration.py`
- `test_h3_operations.py`
- `test_hydrography_ingestion.py`
- `test_integration_wrappers.py`
- `test_module_bridge.py`
- `test_place_analyzer.py`
- `test_place_interface.py`
- `test_unified_backend.py`
- `test_unified_backend_packaged_config.py`
- `test_visualization_engine.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-PLACE`
- Package: `geo_infer_place`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-place`
- Tests: `uv run python -m pytest GEO-INFER-PLACE/tests/unit`

## Dependencies

- `urllib3>=2.0.6`
- `geopandas>=0.13.0`
- `networkx>=2.6.0`
- `shapely>=2.0.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `folium>=0.14.0`
- `plotly>=5.0.0`
- `matplotlib>=3.5.0`
- `branca>=0.6.0`
- `requests>=2.28.0`
- `geo-infer-space>=0.4.0`
- `geo-infer-time>=0.4.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-PLACE/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
