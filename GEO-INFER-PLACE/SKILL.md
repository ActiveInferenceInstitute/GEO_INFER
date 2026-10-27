---
name: geo-infer-place
description: Place-based analysis with H3 hexagonal indexing. Use when running location-specific analyses (forest health, coastal resilience, fire risk, seismic hazard) for Del Norte County or Cascadia, using the PlaceInterface unified API, H3 v4 operations, or California data clients (NOAA, USGS, CAL FIRE, CDEC).
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-time
  recommended:
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-PLACE

## Instructions

### Core Capabilities

- **Unified place interface**: `PlaceInterface` orchestrates location-specific analyzers, data integration, temporal analysis, and dashboards for a supported location
- **Del Norte County analyzers**: forest health, coastal resilience, fire risk, seismic hazard
- **Cascadia agricultural H3 backend**: `CascadianAgriculturalH3Backend` for agricultural land analysis in the Cascadia bioregion (separate application under `locations/cascadia/`)
- **H3 v4 operations**: wrapped v4 API (`latlng_to_cell`, `grid_disk`, `cell_area`, ...)
- **California data clients**: `CaliforniaAPIManager` with NOAA, USGS, CAL FIRE, CDEC clients (retry + caching)

### Key Imports

```python
from geo_infer_place import PlaceInterface, create_place_interface, get_supported_locations
from geo_infer_place.locations.del_norte_county.forest_health_monitor import ForestHealthMonitor
from geo_infer_place.core import CascadianAgriculturalH3Backend
from geo_infer_place.utils.h3_operations import latlng_to_cell, grid_disk, cell_area
from geo_infer_place.core.api_clients import CaliforniaAPIManager, NOAAClient
```

## Examples

Unified interface (recommended entry point):

```python
from geo_infer_place import create_place_interface

pi = create_place_interface("del_norte")
status = pi.status()
print(status["location_name"], status["available_analyzers"])

results = pi.run_full_analysis()   # runs all analyzers for the location
```

Location analyzer through the interface (constructing analyzers directly
requires config/integrator/processor arguments; the interface wires them):

```python
from geo_infer_place import create_place_interface

pi = create_place_interface("del_norte")
monitor = pi.get_analyzer("forest_health")   # ForestHealthMonitor instance
print(monitor.get_monitoring_status())
```

H3 v4 operations:

```python
from geo_infer_place.utils.h3_operations import latlng_to_cell, grid_disk, cell_area

cell = latlng_to_cell(41.75, -124.2, 8)   # Del Norte County
neighbors = grid_disk(cell, 1)
print(f"cell {cell}, area {cell_area(cell):.2f} km2, {len(neighbors)} neighbors")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog PLACE` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Uses H3 v4 API exclusively (`latlng_to_cell`/`cell_to_latlng`, `[lat, lng]` ordering)
- Supported locations: `del_norte`, `cascadia` (`get_supported_locations()`)
- `locations/cascadia/` is a standalone application (own entry point, `cascadia_main.py`) — `run_full_analysis` does not drive it; run its `cascadia_main.py` directly
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE`

### Integrations

- **SPACE** → H3 tessellation and spatial processing (`unified_backend` builds on `geo_infer_space`)
- **DATA** → data quality management via `PlaceDataManager` (optional, `full` extra)
- **TIME** → required dependency for timestamped trend, anomaly, and forecast operations via `PlaceTemporalAnalyzer`; value-only inputs use sequence statistics
- **CIV/HEALTH/TRANSPORT** → not implemented; no geo-infer-place code imports these modules

## Complete Cascadia boundary

Use `geo_infer_place.core.cascadia_boundary.verify_cascadia_boundary(config_dir)`
to verify the packaged licensed publisher snapshot offline. Explicit capture uses
`python -m geo_infer_place.core.cascadia_boundary FRESH_DIR --capture`, bounded
to 20 MiB and a shared five-minute network deadline. Never replace captured files
in place; retain source/license/epoch/ID/extent evidence and the original raw bytes.
See [capture scope](docs/cascadia_boundary.md); USGS display layers retain narrower
windows, and the boundary is the publisher's watershed delineation.
