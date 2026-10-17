---
name: geo-infer-api
description: GEO-INFER script, CLI and MCP access plus REST geospatial endpoints. Use to discover and invoke public module APIs, compose native scientific objects, or build spatial web services.
prerequisites:
  required:
    - geo-infer-data
  recommended:
    - geo-infer-space
    - geo-infer-time
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-API

## Instructions

### Core Capabilities

- **Agent interfaces**: `AgentSession` script API, `geo-infer` JSON CLI and
  `geo-infer-mcp` stdio tools share public API discovery and execution across
  all 45 modules. Native handles, typed inputs and context entry/exit support
  stateful scientific workflows. Installed wheels include all 46 skill texts.

- **REST endpoints**: FastAPI app with health, GeoJSON feature, and processing-algorithm routes (`/api/v1`)
- **GeoJSON utilities**: RFC 7946 models plus validation, ray-cast containment, buffering, and RDP simplification helpers
- **Middleware**: error handling and request logging wired into `create_app`
- **CORS**: explicit, wildcard-free origin lists enable credentialed CORS (fail-closed)

Out of scope for this module: GraphQL schema, JWT/API-key authentication middleware, rate limiting. There is no auth middleware today; `SECRET_KEY` is a required fail-closed setting only.

### Key Imports

```python
from geo_infer_api.app import create_app
from geo_infer_api import AgentSession
from geo_infer_api.core.config import Settings, get_settings
from geo_infer_api.core.middleware import ErrorHandlerMiddleware, RequestLoggingMiddleware
from geo_infer_api import (
    Feature, FeatureCollection, Polygon, PolygonFeature,
)
from geo_infer_api.utils.geojson_helpers import (
    polygon_contains_point, create_buffer, simplify_polygon,
    calculate_polygon_area, validate_polygon_rings,
)
```

### Examples

Build the FastAPI application (requires `SECRET_KEY` to be set):

```python
import os
os.environ["SECRET_KEY"] = "change-me"

from geo_infer_api.app import create_app
app = create_app()  # serve with: uvicorn geo_infer_api.app:main_app
```

Point-in-polygon and buffering on GeoJSON polygons:

```python
from geo_infer_api.utils.geojson_helpers import polygon_contains_point, create_buffer

square = {"type": "Polygon", "coordinates": [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]}
assert polygon_contains_point(square, (5.0, 5.0)) is True
buffered = create_buffer(square, distance=1.0, unit="kilometers")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog API` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- FastAPI with async handlers for spatial operations
- Response format defaults to GeoJSON (RFC 7946, `[lon, lat]` coordinate order)
- The polygon feature store is process-local memory — data does not survive worker restarts; it is capped at `polygon_store_max_size` (default 10,000) and further POSTs beyond the cap return 409 RESOURCE_CONFLICT
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module API`

### Integrations

- **SEC** → JWT/API-key authentication middleware
- **DATA** → Spatial query endpoint data sources
- **SPACE** → H3 and geographic query parameters (algorithms router proxies `geo_infer_space.core.algorithm_registry` when installed; 503 otherwise)
- **OPS** → API monitoring via `/health`
- **APP** → Frontend consuming API endpoints

### Persistent sessions and performance

Use `geo-infer doctor` to inspect actual runtime and unprobed native status.
Use `geo-infer session` for JSON Lines with persistent handles, or Python
`with AgentSession()` / `async with` for deterministic entered-context cleanup.
MCP exposes `geo_diagnostics` and `geo_close`; close ends execution until restart.
Benchmark the real adapter with `python -m geo_infer_api.agent_benchmarks` or
call `geo_infer_api.agent_benchmarks:benchmark_agent` through any surface.
Read [profiling controls and evidence](../GEO-INFER-INTRA/docs/integration/backend_profiling.md)
before selecting Python 3.15 sampling or interpreting throughput results.
