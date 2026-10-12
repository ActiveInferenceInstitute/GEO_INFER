# Agent Integration Guide

## Introduction

This guide explains how to integrate GEO-INFER agents with applications built
using GEO-INFER-APP. Everything here runs against the real public API:
`AgentManager` (in-process agent lifecycle) and `BDIAgentInterface` (the BDI
implementation of `AgentInterface` created by `AgentFactory`).

## Quick Start

### 1. Run an In-Process Agent Through the Manager

```python
import asyncio
from geo_infer_app import AgentManager

async def main():
    manager = AgentManager(config={
        "api_config": {"agents_config_path": "agents.json"}
    })
    await manager.initialize()

    agent_id = await manager.create_agent(
        agent_type="bdi",  # AgentType values: bdi, active_inference,
                           # reinforcement_learning (alias "rl"), rule_based, hybrid
        name="air_quality_monitor",
        config={"name": "air_quality_monitor"},
    )
    await manager.start_agent(agent_id)
    ...
    await manager.shutdown()

asyncio.run(main())
```

`AgentManager` keeps agents in an in-process registry persisted to a JSON file
(no HTTP transport). Supported command types: `query`, `update`, `execute`,
`pause`, `resume`, `reset`.

### 2. Drive a BDI Agent Through the Factory Interface

```python
from geo_infer_app import AgentFactory, AgentType

interface = AgentFactory.create_interface(AgentType.BDI)

agent_id = interface.create_agent(AgentType.BDI, {
    "name": "air_quality_monitor",
    "initial_location": {"lat": 40.7128, "lng": -74.0060},
    "desires": ["explore_area", "collect_data"],
})

# Send commands and read state
interface.send_command(agent_id, "deliberate", {})
state = interface.get_agent_state(agent_id)
print(state.status, state.location, state.beliefs)
```

`AgentFactory.create_interface` only supports `AgentType.BDI` today; other
agent types raise `ValueError` (roadmap — see SKILL.md).

### 3. Visualize Agent State

```python
from geo_infer_app import AgentVisualization

# GeoJSON Point feature, coordinates ordered [lng, lat]
feature = AgentVisualization.state_to_map_feature(state)

# Flattened dict for dashboards
dashboard_data = AgentVisualization.state_to_dashboard_data(state)
```

Locations are validated as finite `{"lat": ...}` in [-90, 90] and
`{"lng": ...}` in [-180, 180]; out-of-bounds values raise `ValueError`.

### 4. Build an Agent Configuration Form

```python
from geo_infer_app.components.agent.agent_config_form import AgentConfigForm
from geo_infer_app import AgentConfiguration, AgentType

schema = AgentConfiguration.get_schema(AgentType.BDI)
form = AgentConfigForm(
    schema={"fields": [f.name for f in schema.fields]},
    initial_values=AgentConfiguration.get_default_config(AgentType.BDI),
    on_submit=lambda payload: AgentConfiguration.validate_config(
        AgentType.BDI, payload
    ),
)
errors = form.submit({"learning_rate": 5.0})  # returns validation errors
```

### 5. Embed an Agent Widget in a Web Page

```python
from geo_infer_app.components.agent_widget import WebAgentWidget

widget = WebAgentWidget(
    agent_manager=manager,
    agent_id=agent_id,
    config={"element_id": "spatial-monitor-widget"},
)
await widget.initialize()
html = widget.render()
await widget.shutdown()
```

`WebAgentWidget.render()` emits self-contained HTML; its JavaScript calls
`/api/agents/{id}/start|stop|command` endpoints described (as a design spec
only) in `docs/api_schema.yaml` — no HTTP server ships with this module.

## Monitoring

### Status Callbacks

```python
def on_status(agent_id: str, status: str) -> None:
    print(f"{agent_id} -> {status}")

manager.register_status_callback(agent_id, on_status)
```

### Metrics

```python
metrics = await manager.get_agent_metrics(agent_id)
# {"decision_count": ..., "success_count": ..., "command_count": ...,
#  "success_rate": ..., "uptime_seconds": ..., "status": ...}
```

## Optional GEO-INFER-AGENT Bridge

`geo_infer_agent` is an optional integration: when installed, agent interfaces
delegate to its implementations; otherwise a deterministic local fallback runs.
No hard dependency is declared.

## Best Practices

1. **Always `await manager.shutdown()`** — it persists agents and counters.
2. **Validate configs before create** with `AgentConfiguration.validate_config`.
3. **Register status callbacks** instead of polling.
4. **Treat `docs/api_schema.yaml` as a design spec** — it describes a future
   REST layer, not running code.
