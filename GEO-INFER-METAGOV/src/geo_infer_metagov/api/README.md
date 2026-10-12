# GEO-INFER-METAGOV/src/geo_infer_metagov/api

Api workspace within `GEO-INFER-METAGOV`.

## Contents

- `__init__.py`
- `rest_api.py`

## Public Interface

- `rest_api.py:APIVersion` (class)
- `rest_api.py:APIResponse` (class)
- `rest_api.py:GovernanceAPI` (class)
- `rest_api.py:StakeholderAPI` (class)

## Module Metadata

- Module: `GEO-INFER-METAGOV`
- Package: `geo_infer_metagov`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-metagov`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module METAGOV`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module METAGOV
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
