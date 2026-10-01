# GEO-INFER-AGENT/src/geo_infer_agent/api

Api workspace within `GEO-INFER-AGENT`.

## Contents

- `__init__.py`
- `agent_endpoints.py`
- `interface.py`
- `messaging.py`
- `telemetry.py`

## Public Interface

- `agent_endpoints.py:AgentCreate` (class)
- `agent_endpoints.py:AgentAction` (class)
- `agent_endpoints.py:AgentMessage` (class)
- `agent_endpoints.py:AgentResponse` (class)
- `agent_endpoints.py:cors_allow_credentials` (function)
- `agent_endpoints.py:cors_origins` (function)
- `agent_endpoints.py:list_agents` (function)
- `agent_endpoints.py:create_agent` (function)
- `agent_endpoints.py:get_agent` (function)
- `agent_endpoints.py:delete_agent` (function)
- `agent_endpoints.py:start_agent` (function)
- `agent_endpoints.py:stop_agent` (function)
- `agent_endpoints.py:agent_action` (function)
- `agent_endpoints.py:get_agent_state` (function)
- `agent_endpoints.py:send_message` (function)
- `agent_endpoints.py:start_api_server` (function)
- `interface.py:AgentInterface` (class)
- `messaging.py:Message` (class)
- `messaging.py:MessagingService` (class)
- `telemetry.py:MetricType` (class)

## Module Metadata

- Module: `GEO-INFER-AGENT`
- Package: `geo_infer_agent`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-agent`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AGENT`

## Dependencies

- `numpy>=1.23.5`
- `torch>=2.0.0`
- `pyyaml>=6.0`
- `requests>=2.28.2`
- `fastapi>=0.104.0`
- `pydantic>=2.5.0`
- `pandas>=1.3.0`
- `uvicorn>=0.24.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AGENT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
