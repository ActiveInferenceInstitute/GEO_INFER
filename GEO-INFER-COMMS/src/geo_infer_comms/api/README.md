# GEO-INFER-COMMS/src/geo_infer_comms/api

Api workspace within `GEO-INFER-COMMS`.

## Contents

- `__init__.py`
- `rest_api.py`
- `websocket_api.py`

## Public Interface

- `rest_api.py:CommunicationAPI` (class)
- `rest_api.py:create_api_server` (function)
- `websocket_api.py:WebSocketManager` (class)
- `websocket_api.py:WebSocketConnection` (class)
- `websocket_api.py:WebSocketServer` (class)
- `websocket_api.py:GeospatialWebSocketHandler` (class)
- `websocket_api.py:RealTimeMessageBroadcaster` (class)
- `websocket_api.py:WebSocketAPIManager` (class)

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
