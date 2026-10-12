#!/usr/bin/env python3

"""
Basic requirements engineering example using GEO-INFER-REQ.

This example demonstrates:
- Requirements analysis with dependency graphs and critical paths
- Traceability links, coverage, and change impact analysis
- Requirement validation: consistency, conflicts, and feasibility
"""

from geo_infer_req import (
    RequirementsAnalyzer,
    Requirement,
    RequirementType,
    PriorityLevel,
    TraceabilityManager,
    TraceLink,
    ArtifactType,
    RequirementValidator,
    RequirementSpec,
)


def main() -> None:
    """Run the basic requirements engineering example."""
    print("=" * 60)
    print("GEO-INFER-REQ: Basic Requirements Engineering Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Requirements analysis
    # ------------------------------------------------------------------
    print("\nStep 1: Requirements analysis")
    analyzer = RequirementsAnalyzer()
    analyzer.add_requirements(
        [
            Requirement(
                "R001",
                "H3 cell indexing",
                "The system shall support H3 v4 cell indexing for spatial data",
                RequirementType.FUNCTIONAL,
                PriorityLevel.HIGH,
                stakeholders=["platform_team"],
                acceptance_criteria=[
                    "H3 v4 cells encode/decode correctly",
                    "API documented",
                ],
            ),
            Requirement(
                "R002",
                "Query latency",
                "Spatial queries shall return within 200 ms at p95",
                RequirementType.PERFORMANCE,
                PriorityLevel.MEDIUM,
                dependencies=["R001"],
                acceptance_criteria=["p95 latency <= 200 ms under load test"],
            ),
            Requirement(
                "R003",
                "Access control",
                "All spatial query endpoints shall require authenticated access",
                RequirementType.SECURITY,
                PriorityLevel.CRITICAL,
                stakeholders=["security_team", "platform_team"],
                acceptance_criteria=["Unauthenticated requests rejected with 401"],
            ),
        ]
    )

    graph = analyzer.build_dependency_graph()
    print(f"  Nodes: {graph.nodes}")
    print(f"  Edges: {graph.edges}")
    print(f"  Topological order: {graph.topological_order}")
    print(f"  Critical path: {graph.critical_path} (depth {graph.depth})")
    print(f"  Cycles: {graph.cycles}")

    scores = analyzer.compute_priority_scores()
    print(f"  Priority scores: {scores}")

    completeness = analyzer.check_completeness()
    print(f"  Completeness: {completeness.completeness_score:.0%}")
    print(f"  Missing descriptions: {completeness.missing_descriptions}")

    # ------------------------------------------------------------------
    # Step 2: Traceability
    # ------------------------------------------------------------------
    print("\nStep 2: Traceability")
    tm = TraceabilityManager()
    tm.register_requirements(["R001", "R002", "R003"])
    tm.add_trace_links(
        [
            TraceLink("R001", "src/h3_backend.py", ArtifactType.SOURCE_CODE),
            TraceLink(
                "R001",
                "tests/test_h3_backend.py",
                ArtifactType.TEST_CASE,
                verified=True,
            ),
            TraceLink("R002", "tests/test_query_latency.py", ArtifactType.TEST_CASE),
        ]
    )
    # R003 has no links yet -> untraced

    coverage = tm.analyze_coverage()
    print(f"  Coverage ratio: {coverage.coverage_ratio:.0%}")
    print(f"  Untraced: {coverage.untraced_requirements}")

    impact = tm.analyze_impact("R001")
    print(f"  Changing R001 affects {impact.affected_count} elements:")
    print(f"    indirect requirements: {impact.indirectly_affected_requirements}")

    # ------------------------------------------------------------------
    # Step 3: Validation
    # ------------------------------------------------------------------
    print("\nStep 3: Validation")
    validator = RequirementValidator()
    validator.add_specs(
        [
            RequirementSpec(
                "R001",
                "H3 cell indexing",
                "The system shall support H3 v4 cell indexing for spatial data",
                priority=4,
                effort_estimate=10.0,
                tags=["spatial", "backend"],
                resources_required=["backend_dev"],
            ),
            RequirementSpec(
                "R002",
                "Query latency",
                "Spatial queries shall return within 200 ms at p95",
                priority=2,
                effort_estimate=5.0,
                dependencies=["R001"],
                tags=["spatial", "performance"],
                resources_required=["backend_dev"],
            ),
        ]
    )
    validator.set_resource_capacity({"backend_dev": 12.0})  # person-days

    consistency = validator.check_consistency()
    print(f"  Consistent: {consistency.is_consistent}")
    print(
        f"  Issues: {consistency.total_issues} "
        f"(errors: {len(consistency.errors)}, warnings: {len(consistency.warnings)}, "
        f"info: {len(consistency.info_items)})"
    )

    conflicts = validator.detect_conflicts()
    print(f"  Conflicts: {conflicts.total_conflicts}")
    for conflict in conflicts.conflicts:
        print(f"    [{conflict.severity.value}] {conflict.description}")

    feasibility = validator.assess_feasibility(available_effort=20.0)
    print(f"  Feasibility: {feasibility.overall_feasibility}")
    print(f"  Resource utilization: {feasibility.resource_utilization}")

    print("\n" + "=" * 60)
    print("Requirements engineering example complete!")


if __name__ == "__main__":
    main()
