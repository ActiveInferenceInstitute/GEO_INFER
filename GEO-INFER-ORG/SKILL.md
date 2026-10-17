---
name: geo-infer-org
description: Organizational modeling for geospatial entities. Use when modeling organizational structures, governance voting and consensus, team formation from skill pools, or collaboration network analysis for spatial governance.
prerequisites:
  required: []
  recommended:
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../examples/
---

# GEO-INFER-ORG

## Instructions

### Core Capabilities

- **Organizational modeling**: `OrganizationModel` builds a unit hierarchy (`OrgUnit`, `Role`), computes depth/metrics, reporting chains, and budget allocation (`allocate_budget` with proportional/equal/weighted strategies).
- **Governance**: `VotingEngine` tallies simple-majority (plurality), supermajority, unanimous, ranked-choice (IRV with single elimination and deterministic tie-break), weighted, and approval voting. `ConsensusModel` scores multi-round participant ratings and checks convergence.
- **Collaboration networks**: `CollaborationNetwork` computes density, clustering, components, and Brandes betweenness centrality (undirected normalization); `TeamFormation` greedily selects capacity-weighted teams covering required skills and reports skill gaps.

### Key Imports

```python
from geo_infer_org import (
    OrganizationModel, OrgUnit, Role, OrgStructureType,
    VotingEngine, ConsensusModel, VotingMethod, Vote, Proposal,
    CollaborationNetwork, TeamFormation, TeamMember, CollaborationEdge, CollaborationType,
)
```

## Examples

```python
from geo_infer_org import OrganizationModel, OrgUnit, VotingEngine, Proposal, Vote, VotingMethod

model = OrganizationModel()
root = OrgUnit(unit_id="hq", name="Regional Authority")
model.add_unit(root)
model.add_unit(OrgUnit(unit_id="planning", name="Planning", parent_id="hq"))

engine = VotingEngine()
engine.create_proposal(Proposal(
    proposal_id="p1", title="Budget", description="Approve budget",
    proposer_id="hq", options=["yes", "no"],
))
engine.cast_vote("p1", Vote("voter-1", "yes"))
result = engine.tally("p1")
print(result.winner)
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog ORG` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Simple majority is plurality semantics (most votes wins, no strict >50% bar); supermajority requires >= 2/3.
- IRV eliminates exactly one lowest candidate per round; ties break deterministically (lexicographically first candidate id).
- `TeamMember.capacity` (default 1.0) weights candidates in `form_team` selection scoring.
- No spatial jurisdiction-overlap analysis exists in this module; spatial features live in GEO-INFER-SPACE integrations.

### Integrations

- Integrates with METAGOV (GEO-INFER-METAGOV imports `OrganizationModel`).
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ORG`
