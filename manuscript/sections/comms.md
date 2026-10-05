## GEO-INFER-COMMS — Communications Infrastructure for Geospatial Systems

**Purpose.** GEO-INFER-COMMS provides communications infrastructure for geospatial systems, enabling data exchange, messaging, networking, and outreach across distributed applications. Its README positions the module as the coordination fabric: how distributed framework components, human teams, and external stakeholders exchange messages, notifications, alerts, and events.

**Public API.** The `geo_infer_comms` package exports a system facade — `GeospatialCommunicationSystem`, `get_communication_system`, `configure_system` — over four core groups from `core/`: messaging (`MessageBroker`, `MessageRouter`, `MessageFormatter`), notifications (`NotificationManager`, `NotificationMetrics`, `AlertRule`, `AlertSystem`, `EmergencyAlertSystem`), channels (`ChannelManager`, `ChannelPermissionManager`), and events (`EventManager`, `EventScheduler`, `EventFilter`, `EventMetrics`). Supporting types live in `models/message`, `models/spatial`, and `utils/validation`, and an `integrations/` subpackage wires external services.

The tests exercise messaging, notifications, channels, and events. The shared repository runner selects this module with `--module COMMS`.

**Theme role.** Infrastructure: COMMS is the nervous system of the deployment — adjacent to the API service spine but focused on asynchronous, human-and-agent-facing exchange rather than synchronous data interfaces.
