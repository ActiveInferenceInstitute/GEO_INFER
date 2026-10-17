---
name: geo-infer-emergency
description: Emergency management and disaster response. Use when planning search and rescue, optimizing emergency resource deployment, coordinating evacuation routes, building a common operating picture, or managing multi-agency response logistics.
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-EMERGENCY

## Instructions

### Core Capabilities

- **Incident coordination**: ICS command structure, mutual aid requests, situation reports (`EmergencyCoordinator`)
- **Resource deployment**: travel-time-based allocation optimization, dynamic redeployment, staging management, resource tracking (`ResourceDeployer`)
- **Evacuation planning**: Dijkstra route optimization over a NetworkX road network, phasing, contraflow, shelter management, clearance-time estimation (`EvacuationPlanner`)
- **Situational awareness**: common operating picture layers, sensor integration, threat assessment, confidence-weighted data fusion (weighted average only; other fusion methods raise `ValueError`), dashboards (`SituationalAwareness`)
- **Search and rescue**: mission planning, probability-of-detection, search-pattern generation (parallel, expanding square, sector, grid), team coordination, Bayesian probability updates (`SearchAndRescue`)

### Key Imports

```python
from geo_infer_emergency import (
    EmergencyCoordinator,
    ResourceDeployer,
    EvacuationPlanner,
    SituationalAwareness,
    SearchAndRescue,
)
from geo_infer_emergency.core.geo import haversine_distance_km
```

## Examples

```python
from geo_infer_emergency import SearchAndRescue, EvacuationPlanner

sar = SearchAndRescue()
mission = sar.plan_mission(
    subject={"id": "subject_1", "type": "hiker", "name": "Missing Hiker"},
    last_known_point={"lat": 45.5, "lon": -122.6},
    search_radius=5.0,
)
pattern = sar.generate_pattern(
    area={"center": {"lat": 45.5, "lon": -122.6}, "radius_km": 3.0},
    pattern_type="expanding_square",
)
print(pattern["waypoints"][0], pattern["estimated_distance_km"])

evac = EvacuationPlanner()
estimates = evac.estimate_clearance_time(
    evacuation_plan={"zone": {"population": 10000}, "routes": []},
    scenarios=["best_case", "expected", "worst_case"],
)
print(estimates["expected"]["clearance_hours"])
```

Verified smoke checks for the snippets above:

```bash
uv run --no-sync python -c "from geo_infer_emergency import EmergencyCoordinator, ResourceDeployer, EvacuationPlanner, SituationalAwareness, SearchAndRescue"
uv run --no-sync python -c "from geo_infer_emergency.core.geo import haversine_distance_km; print(round(haversine_distance_km({'lat': 45.5, 'lon': -122.6}, {'lat': 45.6, 'lon': -122.6}), 3))"
uv run --no-sync python examples/multi_hazard_assessment.py
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog EMERGENCY` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Resource-deployment optimization is a greedy nearest-available-resource
  heuristic under a response-time constraint, not a true mixed-integer solver;
  the default `optimization_algorithm` value is `greedy_nearest_resource`.
- `EvacuationPlanner.optimize_routes` requires every origin/destination to be
  a node of the `road_network` NetworkX graph passed at construction.
- `estimate_clearance_time` returns a scenario-keyed dict
  (`{"best_case": {...}, "expected": {...}, ...}`), not a wrapped report.
- `generate_sitrep`'s report-format parameter is `report_format` (not `format`).

### Integrations

- Cross-module integration (TRANSPORT, COMMS, RISK, SPACE) is aspirational —
  no `geo_infer_*` imports exist in this module yet.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EMERGENCY`