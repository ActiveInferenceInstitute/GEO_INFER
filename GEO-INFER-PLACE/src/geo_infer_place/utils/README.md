# GEO-INFER-PLACE/src/geo_infer_place/utils

Utils workspace within `GEO-INFER-PLACE`.

## Contents

- `__init__.py`
- `caching.py`
- `data_sources.py`
- `h3_operations.py`
- `integration.py`

## Public Interface

- `caching.py:CachedAPIWrapper` (class)
- `data_sources.py:DataSource` (class)
- `data_sources.py:CaliforniaDataSources` (class)
- `h3_operations.py:latlng_to_cell` (function)
- `h3_operations.py:cell_to_latlng` (function)
- `h3_operations.py:cell_to_latlng_boundary` (function)
- `h3_operations.py:geo_to_cells` (function)
- `h3_operations.py:polygon_to_cells` (function)
- `h3_operations.py:grid_disk` (function)
- `h3_operations.py:grid_distance` (function)
- `h3_operations.py:grid_ring` (function)
- `h3_operations.py:cell_area` (function)
- `h3_operations.py:get_resolution` (function)
- `h3_operations.py:is_valid_cell` (function)
- `h3_operations.py:are_neighbor_cells` (function)
- `h3_operations.py:get_base_cell_number` (function)
- `h3_operations.py:cell_to_parent` (function)
- `h3_operations.py:cell_to_children` (function)
- `h3_operations.py:compact_cells` (function)
- `h3_operations.py:uncompact_cells` (function)

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
