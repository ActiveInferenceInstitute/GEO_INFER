# GEO-INFER-COG/examples

Examples workspace within `GEO-INFER-COG`.

## Contents

- `__init__.py`
- `cognitive_processing_demo.py`
- `cognitive_wayfinding.py`

## Public Interface

- `cognitive_processing_demo.py:main` (function)
- `cognitive_wayfinding.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-COG`
- Package: `geo_infer_cog`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-cog`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

## Dependencies

- `numpy>=1.20.0`
- `networkx>=2.6`
- `pyyaml>=5.4`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
