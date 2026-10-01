# GEO-INFER-INTRA/docs/realms

Realms workspace within `GEO-INFER-INTRA`.

## Contents

- `outputs/`
- `realms_api_probe.py`
- `API_TESTING_GUIDE.md`
- `realm_schema.json`
- `realms-geo-infer.md`

## Public Interface

- `realms_api_probe.py:RealmsAPITester` (class)
- `realms_api_probe.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-intra`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
