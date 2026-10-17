---
name: geo-infer-cog
description: Human-centered geospatial cognitive modeling — spatial perception and attention, qualitative spatial reasoning, spatial memory, spatial language processing, cognitive maps, and human-centered decision support. Use when modeling how users perceive, reason about, remember, or decide about geographic space, or when building cognitively optimized geospatial interfaces.
prerequisites:
  recommended:
    - geo-infer-space
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-COG

## Instructions

### Core Capabilities

- **Cognitive processing pipeline**: perception → working memory → reasoning → memory consolidation → decision output
- **Spatial attention**: saliency-driven `AttentionModel` inside the perception stack
- **Qualitative spatial reasoning**: RCC-8 style relation inference over geometries
- **Spatial memory**: working / long-term / episodic memory with consolidation
- **Spatial language**: entity and relation extraction from geographic text
- **Cognitive maps**: landmark/route-based maps with cognitively distorted navigation paths
- **Decision support**: prospect-theory, cognitive-weighted, Bayesian, and multi-criteria frameworks

### Key Imports

```python
from geo_infer_cog import (
    CognitiveProcessingEngine,
    SpatialPerceptionModel,
    SpatialReasoningEngine,
    SpatialMemoryModel,
    SpatialLanguageProcessor,
    SpatialDecisionSupport,
    CognitiveMap,
    SpatialKnowledgeGraph,
    UserCognitiveProfile,
    ProfileManager,
)
```

## Examples

```python
from geo_infer_cog import CognitiveProcessingEngine

engine = CognitiveProcessingEngine()  # deterministic by default (fixed-seed RNG)
result = engine.process_spatial_input(
    spatial_data={
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-122.4, 37.8]},
        "properties": {"name": "observation point"},
    },
    context={"task": "wayfinding"},
)
print(result["decision_result"]["decisions"])
```

```python
from geo_infer_cog import CognitiveMap

cmap = CognitiveMap("city_map", spatial_bounds={"bbox": [-123, 37, -122, 38]})
cmap.add_landmark("pier", {"type": "Point", "coordinates": [0.0, 0.0]},
                  {"name": "Pier"}, saliency=0.9)
cmap.add_route("pier_to_museum", "pier", "museum", segments=[], properties={})
path = cmap.get_navigation_path("pier", "museum")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog COG` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Currently Alpha status; APIs may change between minor versions.
- All engines are deterministic by default: pass `rng=None` and results are reproducible for a fixed seed.
- Passive library logging via the `geo_infer_cog.*` loggers; no prints in library code.
- The REST API (`geo_infer_cog.api.rest_api`) requires the optional `api` extra (`flask`, `flask-cors`).

### Test

`uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

### Integrations

None: this module has no runtime imports of other GEO-INFER modules.