---
name: geo-infer-marine
description: Marine and ocean analysis for coastal and offshore environments. Use when analyzing ocean currents, marine ecosystems, coastal erosion, sea-level rise, marine protected area planning, or fisheries management.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-data
  recommended:
    - geo-infer-bayes
    - geo-infer-time
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-MARINE

## Instructions

### Core Capabilities

- **Ocean currents**: Ekman transport/pumping, geostrophic flow, mixed-layer depth (`OceanCurrentModeler`)
- **Marine ecosystems**: Coral reef health, fisheries stock, biodiversity indices, blue carbon (`MarineEcosystemModeler`)
- **Coastal analysis**: Coastal vulnerability and erosion analysis (`CoastalAnalyzer`)
- **Sea level**: Sea-level rise projection and inundation assessment (`SeaLevelAnalyzer`)
- **MPA planning**: Marine protected area network design, offshore wind siting (`MarineSpatialPlanner`)
- **Reef thermal stress**: Degree heating weeks and NOAA-style bleaching alert levels (`CoralReefAssessor`)
- **Water quality**: Marine water quality and oceanographic data processing (`MarineWaterQuality`, `OceanographicDataProcessor`)

### Key Imports

```python
from geo_infer_marine import (
    OceanCurrentModeler,
    CoastalAnalyzer,
    MarineEcosystemModeler,
    SeaLevelAnalyzer,
    MarineSpatialPlanner,
    CoralReefAssessor,
)
from geo_infer_marine.core.marine_ecosystems import MarineHabitatType, SpeciesData
```

## Examples

```python
from geo_infer_marine import CoastalAnalyzer, SeaLevelAnalyzer

analyzer = CoastalAnalyzer()
vulnerability = analyzer.assess_coastal_vulnerability(elevation, sea_level, wave_height)
erosion = analyzer.analyze_coastal_erosion(shoreline_data, time_periods=["2000", "2010", "2020"])

sea = SeaLevelAnalyzer()
projection = sea.project_sea_level_rise(historical_sea_level, scenario="rcp85", years=[2050, 2100])
inundation = sea.assess_inundation(elevation, projected_sea_level)
```

See `examples/basic_marine_analysis.py` and `examples/marine_ecosystem_analysis.py` for runnable end-to-end demos.

### Agent interfaces

Use `uv run --frozen geo-infer catalog MARINE` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines


### Integrations

- Integrates with CLIMATE for ocean temperature projections
- Integrates with BIO for marine biodiversity assessment
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE`
