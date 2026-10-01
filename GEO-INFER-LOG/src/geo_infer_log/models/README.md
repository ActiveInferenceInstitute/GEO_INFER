# GEO-INFER-LOG/src/geo_infer_log/models

Models workspace within `GEO-INFER-LOG`.

## Contents

- `__init__.py`
- `base.py`
- `schemas.py`

## Public Interface

- `base.py:BaseModel` (class)
- `schemas.py:VehicleType` (class)
- `schemas.py:FuelType` (class)
- `schemas.py:DeliveryStatus` (class)
- `schemas.py:Vehicle` (class)
- `schemas.py:Location` (class)
- `schemas.py:Shipment` (class)
- `schemas.py:Route` (class)
- `schemas.py:RoutingParameters` (class)
- `schemas.py:FacilityLocation` (class)
- `schemas.py:SupplyChainNetwork` (class)

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
