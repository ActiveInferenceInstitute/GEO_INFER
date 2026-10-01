# GEO-INFER-COMMS/src/geo_infer_comms

Geo Infer Comms workspace within `GEO-INFER-COMMS`.

## Contents

- `api/`
- `core/`
- `integrations/`
- `models/`
- `utils/`
- `__init__.py`

## Public Interface

- `__init__.py:GeospatialCommunicationSystem` (class)
- `__init__.py:get_communication_system` (function)
- `__init__.py:configure_system` (function)
- `__init__.py:send_location_update` (function)
- `__init__.py:create_geospatial_alert` (function)
- `__init__.py:setup_emergency_monitoring` (function)

## Module Metadata

- Module: `GEO-INFER-COMMS`
- Package: `geo_infer_comms`
- Version: `0.3.0`
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
