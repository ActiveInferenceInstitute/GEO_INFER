---
name: geo-infer-req
description: Requirements engineering and traceability for geospatial projects. Use when managing project requirements, building traceability matrices, analyzing dependency graphs and critical paths, or validating requirement consistency, conflicts, and feasibility.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-REQ

## Instructions

### Core Capabilities

- **Requirements analysis**: dependency graphs, topological ordering, cycle detection, critical path
- **Priority scoring**: weighted composite scores (priority level, dependents, stakeholders, effort)
- **Completeness checks**: descriptions, acceptance criteria, dangling dependencies, type coverage
- **Traceability**: trace links, coverage reports, change impact analysis
- **Validation**: consistency checks, conflict detection, feasibility assessment

Pure-stdlib implementation (dataclasses + enums); no third-party runtime dependencies.

### Key Imports

```python
from geo_infer_req import (
    RequirementsAnalyzer, Requirement, RequirementType,
    RequirementStatus, PriorityLevel, DependencyGraph, CompletenessReport,
    TraceabilityManager, TraceLink, ArtifactType, TraceMatrixEntry,
    CoverageReport, ImpactReport,
    RequirementValidator, RequirementSpec, ValidationIssue,
    ValidationSeverity, ConflictType, ConflictDetectionResult,
    ConsistencyReport, FeasibilityAssessment,
)
```

## Examples

### Requirements analysis

```python
from geo_infer_req import RequirementsAnalyzer, Requirement, RequirementType, PriorityLevel

analyzer = RequirementsAnalyzer()
analyzer.add_requirements([
    Requirement("R001", "H3 indexing",
                "The system shall support H3 v4 cell indexing",
                RequirementType.FUNCTIONAL, PriorityLevel.HIGH),
    Requirement("R002", "Query latency",
                "Spatial queries shall return within 200 ms at p95",
                RequirementType.PERFORMANCE,
                dependencies=["R001"]),
])

graph = analyzer.build_dependency_graph()
graph.topological_order   # ['R001', 'R002']
graph.critical_path       # ['R001', 'R002']
graph.cycles              # [] (cycles are reported as closed node loops)

scores = analyzer.compute_priority_scores()   # {req_id: composite score}
report = analyzer.check_completeness()        # missing descriptions/criteria, orphans
```

### Traceability

```python
from geo_infer_req import TraceabilityManager, TraceLink, ArtifactType

tm = TraceabilityManager()
tm.register_requirement("R001")
tm.add_trace_link(TraceLink("R001", "test_h3_backend.py", ArtifactType.TEST_CASE))
tm.verify_link("R001", "test_h3_backend.py")      # True; marks the link verified

coverage = tm.analyze_coverage()                  # coverage_ratio, untraced, by type
impact = tm.analyze_impact("R001")                # affected requirements + artifacts
```

### Validation

```python
from geo_infer_req import RequirementValidator, RequirementSpec

validator = RequirementValidator()
validator.add_spec(RequirementSpec(
    "R001", "H3 indexing", "The system shall support H3 v4 cell indexing",
    priority=4, effort_estimate=10.0,
    tags=["spatial"], resources_required=["backend_dev"],
))
validator.set_resource_capacity({"backend_dev": 12.0})  # person-days

consistency = validator.check_consistency()  # errors / warnings / info items
conflicts = validator.detect_conflicts()     # overlapping constraints, resource conflicts
feasibility = validator.assess_feasibility(available_effort=20.0)
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog REQ` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `find_dependency_cycles` in `geo_infer_req.core.validation` is the single
  canonical cycle detector; `DependencyGraph.cycles` and
  `ConsistencyReport` both report through it.
- Cycle errors never block loading — inspect `graph.cycles` and break the
  reported back-edge.

### Integrations

No cross-module imports are implemented; the module is self-contained
(Governance category). Integrations with NORMS/SEC/SPACE would be new work.

- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module REQ`