---
name: geo-infer-transport
description: Transportation network analysis and traffic modeling. Use when analyzing road networks, simulating traffic (BPR model), forecasting traffic (EWMA), routing on networks, or analyzing transit coverage and accessibility.
prerequisites:
  recommended:
    - geo-infer-space
    - geo-infer-data
    - geo-infer-log
    - geo-infer-math
    - geo-infer-time
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-TRANSPORT

## Instructions

### Core Capabilities

- **Network analysis** (`TransportNetwork`): build from edge lists, connectivity components/reachability/critical links, centrality, statistics, subgraph extraction
- **Routing** (`RoutingEngine`): Dijkstra / A* / Bellman-Ford, traffic-adjusted weights, OD matrices, alternative routes
- **Traffic analysis** (`TrafficAnalyzer`): BPR congestion modeling, LOS classification, microsimulation, EWMA forecasting with confidence intervals, incident detection
- **Accessibility** (`AccessibilityAnalyzer`): isochrones, service areas, equity analysis
- **Transit planning** (`TransitOptimizer`): frequency optimization, coverage analysis with demographic equity, network design, scenario evaluation

### Key Imports

```python
from geo_infer_transport import (
    TransportNetwork,
    RoutingEngine,
    TrafficAnalyzer,
    AccessibilityAnalyzer,
    TransitOptimizer,
)
```

### Examples

```python
from geo_infer_transport import (
    TrafficAnalyzer,
    TransportNetwork,
    RoutingEngine,
)

# Build a network and route on it
network = TransportNetwork(network_type="road")
network.build_from_edges(
    [
        {"id": "e1", "from": "n1", "to": "n2", "length_m": 1000, "speed_limit": 50},
        {"id": "e2", "from": "n2", "to": "n3", "length_m": 800, "speed_limit": 40},
    ]
)
router = RoutingEngine(network=network, algorithm="dijkstra")
route = router.route({"node_id": "n1"}, {"node_id": "n3"})
print(route.total_distance_m, route.total_time_s)  # check route.route_source

# Traffic simulation and forecasting are methods of TrafficAnalyzer
analyzer = TrafficAnalyzer(model_type="bpr", time_resolution="15min")
demand = {"matrix": [[100, 50], [30, 80]]}
simulation = analyzer.simulate_traffic(
    network=network, demand_matrix=demand, simulation_hours=1, time_step_seconds=60
)
print(f"Completed trips: {simulation['statistics']['completed_trips']}")

forecast = analyzer.forecast_traffic(
    [{"volume": 1000}, {"volume": 1100}, {"volume": 950}], forecast_horizon="1h"
)
print(f"Forecast points: {len(forecast['forecasts'])}")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog TRANSPORT` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `simulate_traffic` uses the real BPR delay function; `forecast_traffic` uses EWMA + trend estimation with widening prediction intervals.
- `forecast_horizon` strings must be a number plus an s/m/h/d unit (e.g. `"30m"`, `"90m"`, `"1h"`, `"1d"`) and must span at least one `time_resolution` interval; anything else raises `ValueError`.
- Routing fallback semantics: `Route.route_source` is `"network"` when computed on the graph, `"estimated_fallback"` when no network is set (haversine estimate). When no path exists on the network, the returned Route has an empty `path` and zero distance/time — check `route_source` before trusting metrics.
- Missing-node routing: if the origin or destination node is not present in the graph, `route()` and `calculate_isochrone()` degrade instead of raising — `route()` returns an empty-path Route (`route_source == "network"`, zero distance/time) and the isochrone degenerates to the origin node only.
- `TrafficAnalyzer` accepts only `model_type="bpr"` (the implemented BPR delay model); any other value raises `ValueError`. `simulate_traffic` raises `ValueError` for `time_step_seconds <= 0`. `route()` takes only `origin`, `destination`, and `optimization` — there are no mode/avoid/via parameters.
- Emissions calculation lives in GEO-INFER-LOG (`geo_infer_log.core.transport.EmissionsCalculator`), not in this module. Install the `log` extra (`pip install geo-infer-transport[log]`) to enable the guarded critical-links integration in `TransportNetwork.analyze_connectivity(method="critical_links")`.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TRANSPORT`

### Integrations

- **LOG** (optional `log` extra): edge-betweenness critical-link identification via `TransportationNetworkAnalyzer`; the standalone EmissionsCalculator also lives there
- **SPACE / DATA / ECON / EMERGENCY / HEALTH**: recommended prerequisites (see frontmatter); no code-level imports exist today
