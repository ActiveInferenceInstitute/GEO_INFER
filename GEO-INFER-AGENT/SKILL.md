---
name: geo-infer-agent
description: Multi-agent geospatial systems with Active Inference, BDI, and reinforcement learning. Use when building autonomous agents, implementing perception-action loops, coordinating multi-agent systems, or wiring agent telemetry and messaging.
prerequisites:
  recommended:
    - geo-infer-act
    - geo-infer-space
    - geo-infer-ai
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-AGENT

## Instructions

### Core Capabilities

- **Agent base** (`BaseAgent`): lifecycle (initialize → run → stop), state, message queue, persistence
- **Agent registry**: `AgentRegistry` singleton that creates, runs, and connects agents
- **BDI agents**: Belief-Desire-Intention deliberation with config-driven plan templates
- **Active Inference**: two implementations — a numpy matrix model (`geo_infer_agent.models.active_inference`, the canonical package export) and a torch neural model (`geo_infer_agent.core.active_inference`)
- **Telemetry**: `TelemetryService` singleton with counters/gauges/histograms/timers and health status
- **Messaging**: `MessagingService` pub/sub plus `BaseAgent.send_message` delivery over the registry
- **Rule-based / hybrid / RL agents** in `geo_infer_agent.models`
- **REST API**: FastAPI app at `geo_infer_agent.api.agent_endpoints` + CLI `geo-infer-agent`

### Key Imports

```python
from geo_infer_agent import (            # package root (see __init__.py)
    BaseAgent, AgentState, AgentRegistry,
    BDIAgent, BDIState, Belief, Desire, Plan,
    ActiveInferenceAgent, ActiveInferenceState, GenerativeModel,   # numpy matrix model
    RLAgent, RLState, RuleBasedAgent, HybridAgent,
    MessagingService, Message, TelemetryService,
)
from geo_infer_agent.core.active_inference import (  # torch neural model
    ActiveInferenceAgent as TorchActiveInferenceAgent,
    GenerativeModel as TorchGenerativeModel,
)
from geo_infer_agent.api.agent_endpoints import app          # FastAPI app
from geo_infer_agent.cli import main                          # CLI entrypoint
```

Note: there is no `geo_infer_agent.core.telemetry` and no `GeoAgent` class —
telemetry lives in `geo_infer_agent.api.telemetry.TelemetryService`.

## Examples

### Custom agent on BaseAgent + registry lifecycle

```python
import asyncio
from geo_infer_agent.core.agent_base import BaseAgent
from geo_infer_agent.core.agent_registry import AgentRegistry


class GreeterAgent(BaseAgent):
    async def initialize(self) -> None:
        self.state.update_belief("greeted", False)

    async def perceive(self):
        return {"tick": 1}

    def update_beliefs(self, perception) -> None:
        self.state.update_belief("last_tick", perception["tick"])

    async def decide(self):
        return None if self.state.beliefs.get("greeted") else {"type": "greet"}

    async def act(self, action):
        return {"status": "success", "greeted": True}

    async def shutdown(self) -> None:
        pass


async def main():
    registry = AgentRegistry()
    agent_id = await registry.create_agent(
        agent_type="default", config={"max_runtime": 1}
    )
    await registry.start_agent(agent_id)
    await asyncio.sleep(0.2)
    await registry.stop_agent(agent_id)
    print(registry.get_agent_info(agent_id))

asyncio.run(main())
```

### BDI agent with config-driven plans

```python
import asyncio
from geo_infer_agent import BDIAgent

config = {
    "initial_beliefs": {"region_status": {"value": "unknown", "confidence": 0.5}},
    "initial_desires": [
        {"name": "survey_region", "description": "Survey region", "priority": 0.9}
    ],
    "plans": [
        {
            "name": "survey_plan",
            "desire_name": "survey_region",
            "actions": [{"type": "log", "message": "surveying", "level": "info"}],
        }
    ],
}

async def run_bdi():
    agent = BDIAgent(agent_id="bdi-demo", config=config)
    await agent.initialize()
    try:
        perception = await agent.perceive()
        agent.update_beliefs(perception)
        action = await agent.decide()      # {'type': 'log', ...}
        return await agent.act(action)    # {'success': True, ...}
    finally:
        await agent.shutdown()

result = asyncio.run(run_bdi())
```

Plan templates support `$CONFIG:<key>` placeholders (e.g.
`{"type": "wait", "duration": "$CONFIG:collection_interval"}`), resolved
against the agent config when the plan is instantiated.

### Telemetry snapshots

```python
from geo_infer_agent import TelemetryService

telemetry = TelemetryService()            # process-wide singleton
telemetry.register_counter("agent.steps", "Steps executed", agent_id="a1")
telemetry.register_gauge("agent.cpu", "CPU load", agent_id="a1")
metrics = telemetry.get_metrics("a1")     # {metric_id: {...snapshot...}}
```

### Agent-to-agent messaging

```python
import asyncio
from geo_infer_agent import MessagingService, Message

messaging = MessagingService()

async def demo():
    await messaging.start()
    delivered = await messaging.send_message(
        Message(from_agent_id="a1", to_agent_id="a2", content={"cmd": "ping"})
    )
    await messaging.stop()
    return delivered
```

`BaseAgent.send_message` delivers through the `AgentRegistry`: the message is
placed on the recipient's queue and the method returns True. Sending to an
unregistered agent returns False.

## Guidelines

- In-process messaging goes through `AgentRegistry` (singleton); cross-process
  messaging should subclass/override `send_message` with a real transport.
- `run()` logs and stores crashes in `agent.last_error` without re-raising —
  check `agent.last_error` to distinguish crash vs clean stop.
- Registry task completion releases its running flag and retained task. A
  concurrent restart keeps its own task and flag when an older stop completes.
- Failed message callbacks retain an undelivered message for retry; successful
  callbacks consume it once.
- Async tests use pytest-asyncio function-scoped loops, direct `await`, bounded
  event/task waits, and actual stop/shutdown calls. Do not clear task references
  or close a manual loop as a substitute for awaiting cleanup.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AGENT`

### Integrations

The module ships self-contained (no imports from other GEO-INFER packages).
- **ACT** → shared Active Inference seam (both packages implement the framework)
- **SIM** → multi-agent simulation environments
- **SPACE** → spatial beliefs (`models/bdi/belief.py` spatial queries) are the
  only spatial surface today
