# GEO-INFER-CLIMATE

Climate modeling, weather analysis, and climate change impact assessment for geospatial systems.

## Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-CLIMATE`
- Package: `geo_infer_climate`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-CLIMATE`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
