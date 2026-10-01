# GEO-INFER-LOG/tests/unit

Unit workspace within `GEO-INFER-LOG`.

## Contents

- `test_api_endpoints.py`
- `test_api_error_handling.py`
- `test_api_state.py`
- `test_core.py`
- `test_delivery.py`
- `test_delivery_behavior.py`
- `test_geo_utils.py`
- `test_log_init.py`
- `test_log_lifecycle.py`
- `test_observability.py`
- `test_optimization.py`
- `test_optimization_ortools.py`
- `test_supply_chain.py`
- `test_transport.py`
- `test_transport_behavior.py`
- `test_transport_network_api.py`
- `test_visualization.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-LOG`
- Package: `geo_infer_log`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-log`
- Tests: `uv run python -m pytest GEO-INFER-LOG/tests/unit`

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
uv run python -m pytest GEO-INFER-LOG/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
