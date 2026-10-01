# GEO-INFER-SPACE

H3 v4 spatial indexing and comprehensive geospatial analysis framework with advanced spatial methods and coordinate transformations.

## Contents

- `docs/`
- `examples/`
- `output/`
- `reports/`
- `src/`
- `test_output/`
- `tests/`
- `.gitignore`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-SPACE`
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


## Implemented Nested H3 Contracts

- `geo_infer_space.nested.NestedH3Grid` builds real `h3>=4.5.0,<5`
  hierarchies from seed cells or boundary vertices across ordered resolutions.
- Hierarchy outputs include deterministic `parent_child_map`,
  `child_parent_map`, `same_level_neighbors`, level summaries, validation
  diagnostics, and finite child-to-parent aggregation.
- Validation rejects invalid H3 cells, unordered resolutions, orphan children,
  wrong-resolution children, and parent/child mismatches.

```python
from geo_infer_space.nested import NestedH3Grid

grid = NestedH3Grid("sf_nested")
hierarchy = grid.build_h3_hierarchy_from_cells(
    ["89283082803ffff"],
    resolutions=[7, 8, 9],
)
assert hierarchy["validation"]["is_valid"]
assert hierarchy["validation"]["orphan_count"] == 0
```

## Visualization Contracts

- The visualization engine validates H3 resolution and finite geographic bounds
  at construction, and validates dashboard result/configuration mappings.
- Dashboard configuration honors validated `zoom_start` and `tiles` values.

Nested validation command:

```bash
uv run pytest GEO-INFER-SPACE/tests/unit/test_nested_h3_contract.py -q
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
```

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
