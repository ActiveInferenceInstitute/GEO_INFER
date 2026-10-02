# GEO-INFER-ENERGY

Energy systems analysis, renewable energy optimization, and grid management.

## Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `.gitignore`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-ENERGY`
- Package: `geo_infer_energy`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-energy`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ENERGY`

## Dependencies

- `numpy>=1.20.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ENERGY
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
