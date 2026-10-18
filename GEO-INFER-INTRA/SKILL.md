---
name: geo-infer-intra
description: Central documentation hub and cross-module integration guides for GEO-INFER. Use when navigating documentation, finding cross-module integration patterns, understanding data flow architecture, or locating tutorials and API references.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-INTRA

## Instructions

### Core Capabilities

- **Documentation hub**: Central `docs/` directory with comprehensive guides
- **Integration guides**: Cross-module data flow patterns and examples
- **Architecture docs**: System design diagrams, module dependency graph
- **API reference**: API navigation and per-module reference index for all 45 modules
- **Tutorials**: Step-by-step workflows spanning multiple modules

### Key Directories

```text
GEO-INFER-INTRA/docs/
├── guides/          # How-to guides for common workflows
├── tutorials/       # Step-by-step multi-module tutorials
├── integration/     # Cross-module integration patterns
├── architecture/    # System design and data flow diagrams
└── api/             # API navigation and per-module reference index
```

### Cross-Module Integration Pattern

```python
# Example: SPACE → MATH → BAYES pipeline
from geo_infer_space.backends.h3 import H3Backend
from geo_infer_math.core.spatial_statistics import MoranI
from geo_infer_bayes.models.spatial_gp import SpatialGP

# 1. Index → 2. Analyze → 3. Model
cells = H3Backend().polygon_to_cells(region, resolution=7)
moran = MoranI(weights)
autocorrelation = moran.compute(values)
model = SpatialGP()
```

## Examples

```python
# Multi-module data flow: DATA → SPACE → MATH → BAYES
from geo_infer_data.connectors.file import FileConnector
from geo_infer_space.backends.h3 import H3Backend
from geo_infer_math.core.spatial_statistics import MoranI
from geo_infer_bayes.models.spatial_gp import SpatialGP

# 1. Load → 2. Index → 3. Analyze → 4. Model
connector = FileConnector(base_path="data")
# In an async workflow: features = await connector.read_geospatial("observations.geojson")
cells = H3Backend().polygon_to_cells(region, resolution=7)
moran = MoranI(weights)
autocorrelation = moran.compute(values)
model = SpatialGP()
```

```python
# Onboarding: discover module capabilities
import importlib
for module_name in ["math", "space", "bayes", "act", "risk"]:
    mod = importlib.import_module(f"geo_infer_{module_name}")
    print(f"geo_infer_{module_name}: {mod.__doc__ or 'No docstring'}")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog INTRA` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Start here when onboarding to GEO-INFER
- Each module's README.md and AGENTS.md provide module-level detail
- Each module's SKILL.md provides quick-reference for Claude Code
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

### Integrations

- **EXAMPLES** → Working code examples referenced from docs
- **All modules** → Central hub linking all module documentation
- **TEST** → Documentation-driven testing patterns

For manuscript generation, use the repository-local producer and the explicit
source commit → `--refresh-config` → config-only commit sequence in
[research artifacts](docs/integration/research_artifacts.md). Preserve the PNG,
SVG, JSON and CSV companions, figure registry, verification record and manifest
together. Run `--check` before sharing generated output; a source fingerprint
alone cannot detect altered delivered files. File counts are not coverage or
scientific validation.
