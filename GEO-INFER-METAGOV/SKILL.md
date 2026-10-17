---
name: geo-infer-metagov
description: Meta-governance frameworks for geospatial decision-making. Use when implementing polycentric governance, multi-level institutional analysis, stakeholder engagement, conflict resolution, or adaptive governance scenarios for spatial resource management.
prerequisites:
  required: []
  recommended:
    - geo-infer-space
    - geo-infer-norms
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-METAGOV

## Instructions

### Core Capabilities

- **Polycentric governance**: Multi-center decision structures (Ostrom framework)
- **Multi-level analysis**: Institutional analysis across governance scales
- **Stakeholder management**: Stakeholder mapping, power analysis, engagement tracking
- **Conflict resolution**: Spatial conflict detection, mediation workflows
- **Adaptation**: Adaptive governance, scenario planning, performance monitoring
- **Accountability**: Decision tracking, audit trails, transparency metrics

### Key Imports

```python
from geo_infer_metagov.core.polycentric import PolycentricGovernanceSystem
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator
from geo_infer_metagov.core.conflict_resolution import ConflictResolver
from geo_infer_metagov.core.adaptation import AdaptiveGovernanceSystem
from geo_infer_metagov.core.accountability import AccountabilityFramework
```

## Examples

```python
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator

coordinator = StakeholderGovernanceCoordinator()
analysis = coordinator.analyze_stakeholders(
    governance_domain="watershed",
    spatial_extent={"region": "basin-7"},
    stakeholder_categories=["government", "community", "ngo"],
)
power = analysis["power_dynamics"]
conflicts = analysis["interest_conflicts"]
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog METAGOV` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Keep alpha DAO implementation status in tracked issues, not operational `SKILL.md` claims

### Integrations

- Integrates with NORMS for normative governance checks (self-contained reimplementation in
  `integrations/normative.py` — no hard dependency on `geo_infer_norms`)
- Integrates with SPACE for spatial governance indexing (optional, guarded import in
  `integrations/spatial.py`; degrades gracefully when absent)
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module METAGOV`
  (or directly: `uv run python GEO-INFER-TEST/run_unified_tests.py --module METAGOV`)
