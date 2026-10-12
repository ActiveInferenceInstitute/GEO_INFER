# GEO-INFER-COMMS/src/geo_infer_comms/models

Models workspace within `GEO-INFER-COMMS`.

## Contents

- `__init__.py`
- `message.py`
- `spatial.py`

## Public Interface

- `message.py:MessagePriority` (class)
- `message.py:MessageType` (class)
- `message.py:MessageStatus` (class)
- `message.py:ChannelType` (class)
- `message.py:ChannelStatus` (class)
- `message.py:NotificationType` (class)
- `message.py:NotificationStatus` (class)
- `message.py:EventType` (class)
- `message.py:CollaborationType` (class)
- `message.py:ParticipantRole` (class)
- `message.py:ParticipantStatus` (class)
- `message.py:MessageMetadata` (class)
- `message.py:MessageRequest` (class)
- `message.py:MessageResponse` (class)
- `message.py:BroadcastRequest` (class)
- `message.py:BroadcastResponse` (class)
- `message.py:ChannelRequest` (class)
- `message.py:ChannelResponse` (class)
- `message.py:SubscriptionRequest` (class)
- `message.py:SubscriptionResponse` (class)

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
