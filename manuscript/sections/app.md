## GEO-INFER-APP — Human-Computer Interaction and Geospatial Applications

**Purpose.** GEO-INFER-APP is the human-computer interaction layer, providing accessible geospatial applications, dashboards, and UI components. Its README places the user-facing surface of the framework here: dashboards that visualize spatial analyses and interfaces through which humans direct and observe the agent stack.

**Public API.** The `geo_infer_app` package exports an agent-centric UI contract: `AgentInterface`, `AgentState`, and `AgentType` (from `models/agent_interface`), with `AgentFactory` constructing configured agents, `AgentVisualization` rendering them, and `AgentConfiguration` managing settings. The client side is served by `AgentAPIClient` and `AgentManager` from `api/agent_api`, which bridge the UI layer to backend services. A `components/` subpackage holds the reusable UI building blocks referenced by the README.

The tests exercise interface construction, factories, and API clients. The shared repository runner selects this module with `--module APP`.

**Theme role.** Infrastructure: APP is the presentation tier of the stack, standing opposite the API service tier and giving the agent and domain-science bands a human-observable window.
