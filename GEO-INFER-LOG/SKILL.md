---
name: geo-infer-log
description: Logistics optimization including route planning, fleet management, delivery scheduling, and supply chain modeling. Use when optimizing delivery routes, managing fleets, analyzing supply chain resilience, or computing emissions from transportation.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-data
  recommended:
    - geo-infer-time
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-LOG

## Instructions

### Core Capabilities

- **Delivery**: KMeans clustering, Voronoi tessellation, Haversine service areas
- **Transport**: Dijkstra routing, betweenness centrality, max-flow, emissions
- **Supply chain**: PuLP models solved by SciPy/HiGHS MILP, articulation points,
  EOQ, Monte Carlo. Supply flow and facility selection share the exact in-process
  solver; infeasible constraints raise `ValueError` and solver failures raise
  `RuntimeError` without substituting a heuristic.
- **Fleet management**: Vehicle routing, real-time tracking, ETA calculation
- **Observability**: Enhanced structured logging with spatial context

### Key Imports

```python
from geo_infer_log import LastMileRouter, DeliveryScheduler, ServiceAreaAnalyzer
from geo_infer_log import RouteOptimizer, FleetManager, VehicleRouter
from geo_infer_log import Vehicle, VehicleType, RoutingParameters, TravelTimeEstimator
from geo_infer_log import MultiObjectiveOptimizer, RealTimeTracker
from geo_infer_log import SupplyChainModel, FacilityLocator, InventoryManager
from geo_infer_log import EnhancedLogger, PerformanceMetrics
```

All logistics classes are lazy-loaded via `__getattr__` — zero cost until accessed.

## Examples

```python
from geo_infer_log import FacilityLocator

locator = FacilityLocator()
candidates = [
    {"id": "FAC_A", "location": (2.3522, 48.8566)},
    {"id": "FAC_B", "location": (2.2945, 48.8584)},
]
demand_points = [
    {"id": "D1", "location": (2.30, 48.85), "demand": 40},
    {"id": "D2", "location": (2.31, 48.86), "demand": 25},
]
selected = locator.locate_facilities(
    candidates, demand_points, num_facilities=1, max_distance=50.0
)
coverage = locator.analyze_coverage(selected, demand_points, max_distance=50.0)
```

Network-free routing (no road network required — haversine nearest-neighbor):

```python
from geo_infer_log import RouteOptimizer, FleetManager, Vehicle, VehicleType

fleet = FleetManager()
fleet.add_vehicle(Vehicle(
    id="VAN_001", type=VehicleType.VAN, capacity=500, max_range=200,
    speed=40, cost_per_km=0.8, emissions_per_km=0.25, location=(2.3522, 48.8566),
))
assignment = fleet.assign_delivery(
    "VAN_001",
    delivery_points=[(2.2945, 48.8584), (2.30, 48.85)],
    depot=(2.3522, 48.8566),
)
print(assignment["route"]["distance"])  # km
```

## Guidelines

- Supply-chain optimization requires SciPy 1.9 or newer and does not launch CBC
- Submodules (`api`, `core`, `models`, `utils`) lazy-loaded on attribute access
- `Vehicle`, `VehicleType`, and `RoutingParameters` are canonical Pydantic models
  in `geo_infer_log.models.schemas` (re-exported from `core.routing`)
- FastAPI dependencies are cached singletons — fleet/network/schedule state
  persists across requests
- Logger used for all output — no `print()` statements
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module LOG`

### Integrations

- **ECON** → Logistics cost feeding economic models
- **TRANSPORT** → Route optimization and emissions calculation
- **RISK** → Supply chain risk and disruption modeling
- **SPACE** → Service-area geometry exchange (shapely polygons in (lon, lat))
- **OPS** → Logistics operation monitoring via `EnhancedLogger` / `PerformanceMetrics`
