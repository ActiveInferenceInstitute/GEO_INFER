# GEO-INFER-COMMS/examples

Examples workspace within `GEO-INFER-COMMS`.

## Contents

- `basic_usage.py`

## Public Interface

- `basic_usage.py:message_callback` (function)
- `basic_usage.py:event_callback` (function)
- `basic_usage.py:notification_callback` (function)
- `basic_usage.py:main` (function)
- `basic_usage.py:alternative_usage_example` (function)
- `basic_usage.py:geospatial_operations_example` (function)

## Module Metadata

- Module: `GEO-INFER-COMMS`
- Package: `geo_infer_comms`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-comms`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COMMS`

## Dependencies

- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `uvicorn>=0.23.0`
- `websockets>=12.0`
- `pyjwt>=2.0.0`
- `requests>=2.31.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COMMS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
