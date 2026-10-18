# Integration Guide

GEO-INFER integrations are explicit Python imports, validated data/artifact
handoffs, and optional service adapters owned by modules. There is no implicit
framework-wide event bus or hosted INTRA service in this repository.

## Integration workflow

For agent execution across scripts, CLI, MCP and module skills, use the
[shared agent interfaces](agent_interfaces.md). This includes discovery,
runtime signatures, typed scientific inputs and process-local object handles.

For source-stamped PNG/SVG figures, reusable JSON/CSV measurements and complete
publication byte checks, see [research artifacts](research_artifacts.md).

1. Identify the owning module for the input and output.
2. Use public package exports or a documented adapter.
3. Declare the dependency in the consuming module's metadata.
4. Define coordinate system, units, missing-value policy, random seed, and output
   schema at the boundary.
5. Add an integration test and run both module gates.

## Common patterns

### Data to inference

`text
DATA ingestion/validation
  -> SPACE indexing and CRS-aware geometry
  -> TIME alignment when timestamps exist
  -> ACT/BAYES/SPM inference
  -> domain result and validated artifact
`

### Agent and optimization

`text
observations -> AGENT/ANT policy or optimization -> SIM/domain state -> observations
`

Use ACT as the canonical Active Inference contract. ANT accepts explicit
objective functions and bounds; it does not silently fabricate an environment.

### API and application

GEO-INFER-API owns its FastAPI/GeoJSON boundary. GEO-INFER-APP consumes
application-facing interfaces. Keep authentication, serialization, and route
contracts in those modules rather than duplicating them in INTRA docs.

### External GIS / LLM integrations

GEO-INFER deliberately keeps its external integrations optional and
dependency-free by default. The following surfaces were added for interoperability:

- **GeoLibre project emission** (`geo_infer_space.core.geolibre_projects`):
  deterministic, schema-versioned `.geolibre.json` writers (`build_project`,
  `build_h3_grid_project`, `write_project`) so GEO-INFER analysis results open
  in the GeoLibre web/desktop/Jupyter viewer with no JavaScript in this repo.
  Format version `0.1.0`, mirroring opengeos/GeoLibre's documented project
  format. See the runnable example at
  `GEO-INFER-EXAMPLES/examples/getting_started/geolibre_export/`.
- **Cloud-native vector readers** (`geo_infer_data.utils.duckdb_spatial`):
  GeoParquet/FlatGeobuf/Shapefile reads via DuckDB-Spatial when installed,
  transparent GeoPandas/Fiona fallback otherwise (`read_cloud_native_vector`).
  `HAS_DUCKDB` is `False` by default; install the optional DuckDB extra to
  enable the fast path.
- **LLM proxy policy** (`geo_infer_agent.core.llm_proxy`): model allowlist,
  request-size cap, output-token cap, and per-client rate limiting
  (`enforce_llm_proxy_policy`) for server-side LLM serving, mirroring
  GeoLibre's `ai-proxy` shape.
- **WhiteboxTools bridge** (`geo_infer_space.core.whitebox_bridge`): optional
  terrain/hydrology helpers (`flow_accumulation`) gated on `HAS_WHITEBOX` for
  the WATER/FOREST/EMERGENCY domain modules.

`text
GEO-INFER results (H3 grid / GeoJSON layers)
  -> geo_infer_space.core.geolibre_projects.build_h3_grid_project
  -> .geolibre.json project file
  -> GeoLibre web / desktop / Jupyter viewer
`

## Example: spatial data handoff

`python
import numpy as np
from geo_infer_space import latlng_to_cell
from geo_infer_act import FreeEnergyCalculator

cell = latlng_to_cell(37.7749, -122.4194, resolution=9)
free_energy = FreeEnergyCalculator().compute_categorical_free_energy(
    beliefs=np.array([0.7, 0.2, 0.1]),
    observations=np.array([0.8, 0.15, 0.05]),
)
print(cell, free_energy)
`

## Further reading

- [Architecture](../architecture/index.md)
- [Module catalog](../modules/index.md)
- [Module integration guide](../guides/MODULE_INTEGRATION_GUIDE.md)
- [External systems guidance](external_systems.md)
- [Testing guide](../developer_guide/testing_guide.md)
