# GEO-INFER-COMMS/tests/integration

Integration workspace within `GEO-INFER-COMMS`.

## Contents

- `test_integration.py`
- `test_websocket_broadcast.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-COMMS`
- Package: `geo_infer_comms`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-comms`
- Tests: `uv run python -m pytest GEO-INFER-COMMS/tests/integration`

## Dependencies

- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `uvicorn>=0.23.0`
- `websockets>=12.0`
- `pyjwt>=2.0.0`
- `requests>=2.31.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-COMMS/tests/integration
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
