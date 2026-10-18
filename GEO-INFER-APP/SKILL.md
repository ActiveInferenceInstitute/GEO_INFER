---
name: geo-infer-app
description: Agent UI and management layer for GEO-INFER. Use when building agent control widgets, agent configuration forms, geospatial agent visualization (GeoJSON map features), or managing in-process agent lifecycles (create/start/stop/command) via AgentManager.
prerequisites:
  required: []
  recommended:
    - geo-infer-agent
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-APP

## Instructions

### Core Capabilities

- **Agent management**: create/start/stop/delete/query in-process agents via `AgentManager` / `AgentAPIClient` (JSON-file persistence, status callbacks, metrics counters)
- **Agent widgets**: `AgentWidget` / `WebAgentWidget` — async-refreshed status widgets that render agent state as HTML
- **Configuration**: schema-driven validation of agent config (`AgentConfiguration`, `ConfigFieldType`) and form payloads (`AgentConfigForm`)
- **Visualization**: `AgentVisualization` converts `AgentState` into GeoJSON map features ([lng, lat] ordering) and dashboard data
- **Factory**: `AgentFactory` creates `AgentInterface` implementations by `AgentType` (BDI ships today; other types raise `ValueError`)

### Key Imports

```python
from geo_infer_app import AgentManager, AgentFactory, AgentType
from geo_infer_app.models.interfaces.bdi_interface import BDIAgentInterface
from geo_infer_app.models.agent_visualization import AgentVisualization
from geo_infer_app.components.agent_widget import AgentWidget, WebAgentWidget
from geo_infer_app.components.agent.agent_config_form import AgentConfigForm
from geo_infer_app.models.agent_configuration import AgentConfiguration, ConfigFieldType
```

## Examples

Create and drive an in-process BDI agent through the manager:

```python
import asyncio
from geo_infer_app import AgentManager

async def main():
    manager = AgentManager(config={
        "api_config": {"agents_config_path": "/tmp/agents.json"}
    })
    await manager.initialize()

    agent_id = await manager.create_agent(
        agent_type="bdi",
        name="Spatial Monitor",
        config={"name": "Spatial Monitor", "description": "Monitors a location"},
    )
    await manager.start_agent(agent_id)
    result = await manager.send_command(agent_id, "query", {})
    print(result["result"])  # {"id": ..., "type": "bdi", "status": "running", ...}

    await manager.shutdown()

asyncio.run(main())
```

Render an agent state as a GeoJSON map feature:

```python
from geo_infer_app import AgentFactory, AgentType, AgentVisualization

interface = AgentFactory.create_interface(AgentType.BDI)
agent_id = interface.create_agent(AgentType.BDI, {
    "name": "Explorer",
    "initial_location": {"lat": 40.7128, "lng": -74.0060},
})
state = interface.get_agent_state(agent_id)
feature = AgentVisualization.state_to_map_feature(state)  # GeoJSON Point, [lng, lat]
```

### Non-goals (not part of this module)

- `geo_infer_app.core.dashboard` / map & dashboard components (MapView, DashboardBuilder): design docs only — see `docs/api_schema.yaml` (a design spec; no HTTP server ships).
- Agent interface types beyond BDI (`active_inference`, `reinforcement_learning`, `rule_based`, `hybrid`): `AgentFactory.create_interface` raises `ValueError` for them.

### Agent interfaces

Use `uv run --frozen geo-infer catalog APP` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Agent types for `AgentAPIClient.create_agent` are `AgentType` values: `bdi`, `active_inference`, `reinforcement_learning`, `rule_based`, `hybrid` (`rl` is accepted as an alias of `reinforcement_learning`; unknown types raise `ValueError`).
- `AgentAPIClient` is an in-process registry — there is no HTTP transport.
- HTML form inputs use standard `placeholder` attributes (legitimate usage).
- Map features and dashboard widgets are independent JSON snapshots. Dates and
  other non-JSON native values stringify consistently; nonfinite floating values
  raise `ValueError` before a payload is returned. Prepare missing values in the
  caller when needed rather than sending NaN/Infinity to a browser.

### Integrations

- Integrates with AGENT via an optional `geo_infer_agent` import (graceful fallback when absent)
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module APP`
