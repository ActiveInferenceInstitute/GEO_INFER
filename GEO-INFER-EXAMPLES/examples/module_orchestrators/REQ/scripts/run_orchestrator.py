#!/usr/bin/env python3
"""GEO-INFER-REQ module orchestrator.

Runs one documented end-to-end REQ operation on synthetic data: register five
requirements for an H3 analytics service, build the dependency graph and
priority scores, trace each requirement to design/test artifacts, analyze the
impact of changing the indexing requirement, and validate consistency,
conflicts and feasibility against a resource budget. All work goes through
the real ``geo_infer_req`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_req import (
        ArtifactType,
        PriorityLevel,
        Requirement,
        RequirementsAnalyzer,
        RequirementSpec,
        RequirementType,
        RequirementValidator,
        TraceabilityManager,
        TraceLink,
    )

    requirements = [
        Requirement(
            "R001",
            "H3 indexing",
            "The system shall index observations with H3 v4 cells",
            RequirementType.FUNCTIONAL,
            PriorityLevel.CRITICAL,
            stakeholders=["analysts", "platform"],
            acceptance_criteria=["latlng_to_cell round-trips at resolution 9"],
            effort_estimate=8.0,
        ),
        Requirement(
            "R002",
            "Query latency",
            "Spatial queries shall return within 200 ms at p95",
            RequirementType.PERFORMANCE,
            PriorityLevel.HIGH,
            dependencies=["R001"],
            stakeholders=["analysts"],
            acceptance_criteria=["p95 latency below 200 ms on 1e6 cells"],
            effort_estimate=5.0,
        ),
        Requirement(
            "R003",
            "Access control",
            "Only authenticated users shall read restricted layers",
            RequirementType.SECURITY,
            PriorityLevel.HIGH,
            stakeholders=["security"],
            acceptance_criteria=["unauthenticated reads return 401"],
            effort_estimate=4.0,
        ),
        Requirement(
            "R004",
            "GeoJSON export",
            "The system shall export cell aggregates as GeoJSON",
            RequirementType.INTERFACE,
            PriorityLevel.MEDIUM,
            dependencies=["R001", "R003"],
            stakeholders=["analysts"],
            acceptance_criteria=["exported polygons use [lng, lat] order"],
            effort_estimate=3.0,
        ),
        Requirement(
            "R005",
            "Aggregation dashboard",
            "The system shall render daily cell aggregates",
            RequirementType.FUNCTIONAL,
            PriorityLevel.LOW,
            dependencies=["R002", "R004"],
            effort_estimate=6.0,
        ),
    ]

    analyzer = RequirementsAnalyzer()
    analyzer.add_requirements(requirements)
    graph = analyzer.build_dependency_graph()
    scores = analyzer.compute_priority_scores()
    completeness = analyzer.check_completeness()

    trace = TraceabilityManager()
    trace.register_requirements([req.req_id for req in requirements])
    trace.add_trace_links(
        [
            TraceLink("R001", "design_h3_index.md", ArtifactType.DESIGN_DOCUMENT),
            TraceLink("R001", "test_h3_index.py", ArtifactType.TEST_CASE),
            TraceLink("R002", "bench_query_latency.py", ArtifactType.TEST_CASE),
            TraceLink("R003", "auth_middleware.py", ArtifactType.SOURCE_CODE),
            TraceLink("R004", "geojson_export.py", ArtifactType.SOURCE_CODE),
        ]
    )
    trace.verify_link("R001", "test_h3_index.py")
    coverage = trace.analyze_coverage()
    impact = trace.analyze_impact("R001")

    validator = RequirementValidator()
    validator.add_specs(
        [
            RequirementSpec(
                req.req_id,
                req.title,
                req.description,
                priority=req.priority.value,
                effort_estimate=req.effort_estimate or 0.0,
                dependencies=list(req.dependencies),
                resources_required=["backend_dev"],
            )
            for req in requirements
        ]
    )
    validator.set_resource_capacity({"backend_dev": 20.0})
    consistency = validator.check_consistency()
    conflicts = validator.detect_conflicts()
    feasibility = validator.assess_feasibility(available_effort=24.0)

    return {
        "dependency_graph": {
            "topological_order": graph.topological_order,
            "critical_path": graph.critical_path,
            "depth": graph.depth,
            "cycles": graph.cycles,
        },
        "priority_scores": {k: round(v, 4) for k, v in scores.items()},
        "completeness": {
            "score": completeness.completeness_score,
            "missing_acceptance_criteria": completeness.missing_acceptance_criteria,
        },
        "traceability": {
            "coverage_ratio": coverage.coverage_ratio,
            "untraced_requirements": coverage.untraced_requirements,
            "unverified_links": len(trace.get_unverified_links()),
        },
        "impact_of_R001": {
            "affected_count": impact.affected_count,
            "indirectly_affected_requirements": (
                impact.indirectly_affected_requirements
            ),
            "impact_severity": impact.impact_severity,
        },
        "validation": {
            "is_consistent": consistency.is_consistent,
            "consistency_score": consistency.consistency_score,
            "total_conflicts": conflicts.total_conflicts,
            "overall_feasibility": feasibility.overall_feasibility,
            "resource_utilization": feasibility.resource_utilization,
            "bottleneck_requirements": feasibility.bottleneck_requirements,
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("REQ", _operation))
