# GEO-INFER-PLACE/src/geo_infer_place/locations/california

California workspace within `GEO-INFER-PLACE`.

## Contents

- `__init__.py`
- `acquisition.py`
- `contracts.py`
- `evaluation.py`
- `geometry.py`
- `inference.py`
- `models.py`
- `monitoring.py`
- `report.py`
- `report_integrity.py`
- `workflow.py`

## Public Interface

- `acquisition.py:SourceEndpoints` (class)
- `acquisition.py:ServiceMetadata` (class)
- `acquisition.py:ObjectIds` (class)
- `acquisition.py:Feature` (class)
- `acquisition.py:Page` (class)
- `acquisition.py:acquire_sources` (function)
- `acquisition.py:check_sources` (function)
- `contracts.py:CaliforniaError` (class)
- `contracts.py:Hazard` (class)
- `contracts.py:Strategy` (class)
- `contracts.py:StrictModel` (class)
- `contracts.py:ExecutionBinding` (class)
- `contracts.py:execution_binding` (function)
- `contracts.py:ObservatoryConfig` (class)
- `contracts.py:CapturedFile` (class)
- `contracts.py:SourceReceipt` (class)
- `contracts.py:last_tuesday` (function)
- `contracts.py:digest` (function)
- `contracts.py:strict_bytes` (function)
- `contracts.py:write_record` (function)

## Module Metadata

- Module: `GEO-INFER-PLACE`
- Package: `geo_infer_place`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-place`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
