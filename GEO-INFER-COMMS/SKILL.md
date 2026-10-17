---
name: geo-infer-comms
description: Communication systems for geospatial coordination. Use when implementing spatial messaging, multi-channel notifications, event streaming, spatial message routing, or subscriber management for geographic broadcasts.
prerequisites:
  required:
    - geo-infer-api
  recommended:
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-COMMS

## Instructions

### Core Capabilities

- **Messaging**: `MessageBroker` — subscriber registry, priority queue, geo-filtered delivery; optional `recipient_resolver` callback for broadcast targeting
- **Notifications**: `NotificationManager` — in-app/email/SMS/push delivery hooks; the built-in handlers log intended delivery (wire real SMTP/SMS/push transports via `register_delivery_handler`); `AlertSystem` with rule-based triggers and cooldowns
- **Channels**: `ChannelManager` — multi-channel membership, permissions, and geospatial bounds
- **Events**: `EventManager` — event publish/subscribe with spatial indexing; `EventScheduler` for scheduled/recurring publication
- **Spatial routing**: `AdvancedSpatialRouter` — Dijkstra routing over network topology, proximity routing via registered node locations, load balancing
- **Streaming**: `StreamManager` — in-memory geospatial data streams (buffer simulation; no external transport)
- **Collaboration**: `CollaborationManager` — spatial collaboration sessions
- **API**: FastAPI REST server (`CommunicationAPI`) with optional JWT auth; WebSocket API (`WebSocketAPIManager`)

### Key Imports

```python
from geo_infer_comms import GeospatialCommunicationSystem, get_communication_system
from geo_infer_comms.core.messaging import MessageBroker, MessageRouter
from geo_infer_comms.core.notifications import NotificationManager, AlertSystem, AlertRule
from geo_infer_comms.core.channels import ChannelManager, ChannelPermissionManager
from geo_infer_comms.core.events import EventManager, EventScheduler, EventFilter
from geo_infer_comms.core.spatial_routing import (
    AdvancedSpatialRouter, GeospatialLoadBalancer, SpatialRoutingOptimizer,
)
from geo_infer_comms.core.streaming import StreamManager
from geo_infer_comms.api.rest_api import CommunicationAPI, create_api_server
from geo_infer_comms.api.websocket_api import WebSocketAPIManager
```

## Examples

End-to-end with the unified system (recommended):

```python
from geo_infer_comms import GeospatialCommunicationSystem

system = GeospatialCommunicationSystem({"enable_persistence": False})
system.start()
try:
    system.message_broker.subscribe("user_1", print)
    system.send_message(content="Alert: flood detected", recipients=["user_1"])
finally:
    system.stop()
```

Broker with an explicit broadcast recipient resolver (required for
`channel`/`role`/`location_based` target types):

```python
from geo_infer_comms.core.messaging import MessageBroker

broker = MessageBroker(
    recipient_resolver=lambda target_type, criteria: registry.lookup(target_type, criteria)
)
```

Alert rules are dataclass objects, not dicts:

```python
from geo_infer_comms.core.notifications import AlertRule

rule = AlertRule(
    name="High Temperature Alert",
    description="Alert when temperature exceeds threshold",
    conditions={"temperature": {"min": 35.0}},
    alert_title="High Temperature Warning",
    alert_content="Temperature has exceeded safe threshold",
    recipients=["admin@geo-infer.org"],
)
rule_id = system.alert_system.create_alert_rule(rule)
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog COMMS` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Broadcasts with `target_type in {"channel", "role", "location_based"}` require a `recipient_resolver` on the broker; without one they raise `ValueError`.
- REST auth: set `COMMS_JWT_SECRET` to enable HS256 JWT validation (PyJWT); invalid tokens are rejected with 401. Without a secret, a deterministic hash fallback derives the user ID from the token. WebSocket auth follows the same policy.
- Notification delivery handlers for email/SMS/push log intended delivery; register custom handlers via `NotificationManager.register_delivery_handler` for real transports.
- System metrics are nested: `metrics["message_metrics"]["metrics"]["messages_sent"]`, etc.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COMMS`

### Integrations

- **EMERGENCY** → Emergency notification broadcasting
- **IOT** → Sensor alert thresholds
- **PEP** → Stakeholder engagement notifications
- **CIV** → Community geographic broadcasts
- **OPS** → System alert delivery
