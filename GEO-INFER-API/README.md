# GEO-INFER-API

Comprehensive API development and integration services enabling interoperability across the GEO-INFER ecosystem and external systems.

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

- Module: `GEO-INFER-API`
- Package: `geo_infer_api`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-api`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module API`

## Dependencies

- `starlette>=0.27.0`
- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `uvicorn>=0.21.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
