# GEO-INFER-MARINE

Marine and oceanographic analysis, coastal management, and marine ecosystem monitoring.

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

- Module: `GEO-INFER-MARINE`
- Package: `geo_infer_marine`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-MARINE`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE`

## Dependencies

- `numpy>=1.20.0`
- `xarray>=0.19.0`
- `netcdf4>=1.5.8`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
