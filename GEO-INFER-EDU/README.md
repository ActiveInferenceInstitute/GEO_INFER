# GEO-INFER-EDU

Educational technology for geospatial systems including curriculum design, interactive exercises, and learning analytics.

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

- Module: `GEO-INFER-EDU`
- Package: `geo_infer_edu`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-edu`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EDU`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EDU
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
