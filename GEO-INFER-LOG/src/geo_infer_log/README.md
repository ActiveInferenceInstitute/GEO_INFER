# GEO-INFER-LOG/src/geo_infer_log

Geo Infer Log workspace within `GEO-INFER-LOG`.

## Contents

- `api/`
- `core/`
- `models/`
- `utils/`
- `__init__.py`

## Public Interface

- `__init__.py:LogEntry` (class)
- `__init__.py:SpatialLogContext` (class)
- `__init__.py:PerformanceMetrics` (class)
- `__init__.py:EnhancedLogger` (class)
- `__init__.py:JSONFormatter` (class)
- `__init__.py:LogAnalyzer` (class)
- `__init__.py:get_logger` (function)

## Module Metadata

- Module: `GEO-INFER-LOG`
- Package: `geo_infer_log`
- Version: `0.4.0`
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
