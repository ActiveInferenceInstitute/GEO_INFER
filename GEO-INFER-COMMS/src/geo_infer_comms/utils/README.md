# GEO-INFER-COMMS/src/geo_infer_comms/utils

Utils workspace within `GEO-INFER-COMMS`.

## Contents

- `__init__.py`
- `validation.py`

## Public Interface

- `validation.py:validate_coordinates` (function)
- `validation.py:validate_crs` (function)
- `validation.py:validate_email` (function)
- `validation.py:validate_phone` (function)
- `validation.py:validate_message_content` (function)
- `validation.py:validate_message_priority` (function)
- `validation.py:validate_message_type` (function)
- `validation.py:validate_user_id` (function)
- `validation.py:validate_channel_id` (function)
- `validation.py:validate_spatial_bounds` (function)
- `validation.py:validate_geojson_feature` (function)
- `validation.py:validate_geojson_geometry` (function)
- `validation.py:validate_notification_type` (function)
- `validation.py:validate_delivery_methods` (function)
- `validation.py:validate_event_type` (function)
- `validation.py:validate_timestamp` (function)
- `validation.py:validate_url` (function)
- `validation.py:validate_file_size` (function)
- `validation.py:validate_message_recipients` (function)
- `validation.py:validate_spatial_filter` (function)

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
