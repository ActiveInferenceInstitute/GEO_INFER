# GEO-INFER-SPACE/src/geo_infer_space/nested/core

Core workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `hierarchy.py`
- `nested_grid.py`

## Public Interface

- `hierarchy.py:RelationshipType` (class)
- `hierarchy.py:HierarchyDirection` (class)
- `hierarchy.py:HierarchicalRelationship` (class)
- `hierarchy.py:HierarchyManager` (class)
- `nested_grid.py:grid_disk` (function)
- `nested_grid.py:grid_distance` (function)
- `nested_grid.py:neighbor_cells` (function)
- `nested_grid.py:NestedCellType` (class)
- `nested_grid.py:NestedSystemState` (class)
- `nested_grid.py:NestedCell` (class)
- `nested_grid.py:NestedSystem` (class)
- `nested_grid.py:NestedH3Grid` (class)

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
