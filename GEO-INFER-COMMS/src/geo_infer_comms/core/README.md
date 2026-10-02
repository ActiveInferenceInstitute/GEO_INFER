# GEO-INFER-COMMS/src/geo_infer_comms/core

Core workspace within `GEO-INFER-COMMS`.

## Contents

- `__init__.py`
- `channels.py`
- `collaboration.py`
- `events.py`
- `messaging.py`
- `notifications.py`
- `spatial_routing.py`
- `streaming.py`

## Public Interface

- `channels.py:ChannelManager` (class)
- `channels.py:ChannelMetrics` (class)
- `channels.py:ChannelPermissionManager` (class)
- `channels.py:ChannelMessageFilter` (class)
- `channels.py:ChannelAnalytics` (class)
- `collaboration.py:CollaborationManager` (class)
- `collaboration.py:CollaborationMetrics` (class)
- `collaboration.py:RealTimeCollaborationEngine` (class)
- `collaboration.py:GeospatialCollaborationCoordinator` (class)
- `collaboration.py:CollaborationNotificationManager` (class)
- `collaboration.py:CollaborationAnalytics` (class)
- `events.py:EventManager` (class)
- `events.py:EventMetrics` (class)
- `events.py:EventProcessor` (class)
- `events.py:DataUpdateProcessor` (class)
- `events.py:SystemAlertProcessor` (class)
- `events.py:UserActionProcessor` (class)
- `events.py:SensorTriggerProcessor` (class)
- `events.py:GeospatialChangeProcessor` (class)
- `events.py:EventFilter` (class)

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
