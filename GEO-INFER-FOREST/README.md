# GEO-INFER-FOREST

Forest management, carbon sequestration, wildfire risk, and forest ecosystem analysis.

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

- Module: `GEO-INFER-FOREST`
- Package: `geo_infer_forest`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-FOREST`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module FOREST`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module FOREST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
