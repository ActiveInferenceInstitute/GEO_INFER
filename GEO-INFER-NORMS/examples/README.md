# GEO-INFER-NORMS/examples

Examples workspace within `GEO-INFER-NORMS`.

## Contents

- `output/`
- `minimal_zoning_example.py`
- `zoning_analysis_example.py`

## Public Interface

- `minimal_zoning_example.py:build_analyzer` (function)
- `minimal_zoning_example.py:main` (function)
- `zoning_analysis_example.py:build_analyzer` (function)
- `zoning_analysis_example.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-NORMS`
- Package: `geo_infer_norms`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-norms`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS`

## Dependencies

- `fastapi>=0.95.0,<1`
- `starlette>=0.27.0`
- `geopandas>=0.13.0,<2`
- `matplotlib>=3.7.0,<4`
- `networkx>=2.6.0,<4`
- `numpy>=1.24.0,<3`
- `pandas>=2.0.0,<3`
- `pydantic>=2.0.0,<3`
- `shapely>=2.0.0,<3`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
