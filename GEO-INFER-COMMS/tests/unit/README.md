# GEO-INFER-COMMS/tests/unit

Unit workspace within `GEO-INFER-COMMS`.

## Contents

- `test_channels.py`
- `test_collaboration_sessions.py`
- `test_core.py`
- `test_email_providers.py`
- `test_event_scheduler.py`
- `test_events_delivery_and_filters.py`
- `test_events_publish_robustness.py`
- `test_message_models.py`
- `test_messaging.py`
- `test_notifications.py`
- `test_package_exports.py`
- `test_protocol_handling.py`
- `test_rest_api.py`
- `test_spatial_models.py`
- `test_spatial_routing.py`
- `test_streaming.py`
- `test_validation.py`
- `test_websocket_auth.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-COMMS`
- Package: `geo_infer_comms`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-comms`
- Tests: `uv run python -m pytest GEO-INFER-COMMS/tests/unit`

## Dependencies

- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `uvicorn>=0.23.0`
- `websockets>=12.0`
- `pyjwt>=2.0.0`
- `requests>=2.31.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-COMMS/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
