"""
Comprehensive example: Full governance system with all METAGOV capabilities.

This example demonstrates:
- Multi-level governance design
- Stakeholder coordination
- Institutional design
- Conflict resolution
- Performance evaluation
- Scenario planning
- Accountability and transparency
- Adaptive governance
"""

from geo_infer_metagov.core.multi_level import MultiLevelGovernanceFramework
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator
from geo_infer_metagov.core.institutional import InstitutionalDesigner
from geo_infer_metagov.core.accountability import AccountabilityFramework
from geo_infer_metagov.core.conflict_resolution import (
    ConflictResolver,
    ConflictResolutionMethod,
)
from geo_infer_metagov.core.performance import PerformanceEvaluator
from geo_infer_metagov.core.scenarios import ScenarioPlanner
from geo_infer_metagov.core.adaptation import AdaptiveGovernanceSystem


def main():
    """Demonstrate comprehensive governance system."""

    print("=" * 80)
    print("GEO-INFER-METAGOV: Comprehensive Governance System Example")
    print("=" * 80)

    # 1. Design governance structure
    print("\n1. DESIGNING MULTI-LEVEL GOVERNANCE STRUCTURE")
    print("-" * 80)

    mlg_framework = MultiLevelGovernanceFramework(
        governance_levels=["local", "regional", "national"],
        coordination_mechanisms=[
            "vertical_alignment",
            "horizontal_integration",
            "subsidiarity",
        ],
        domain_coverage=["environmental", "economic", "social", "infrastructure"],
    )

    governance_structure = mlg_framework.design_governance_structure(
        spatial_scope={
            "name": "Coastal Region",
            "area_km2": 15000,
            "jurisdictions": 12,
            "population": 2500000,
        },
        stakeholder_groups=[
            {"name": "Local Governments"},
            {"name": "Regional Authority"},
            {"name": "National Agency"},
            {"name": "Environmental NGOs"},
            {"name": "Business Associations"},
            {"name": "Community Groups"},
            {"name": "Indigenous Communities"},
            {"name": "Research Institutions"},
        ],
        decision_domains=[
            "coastal_protection",
            "fisheries_management",
            "tourism_development",
            "environmental_conservation",
            "infrastructure_planning",
            "disaster_preparedness",
        ],
        time_horizons=[1, 5, 10, 20],
    )

    print(f"✓ Governance structure: {len(governance_structure.entities)} entities")
    print(f"✓ Decision domains: {len(governance_structure.decision_domains)}")

    # 2. Stakeholder analysis
    print("\n2. STAKEHOLDER ANALYSIS & COORDINATION")
    print("-" * 80)

    stakeholder_coordinator = StakeholderGovernanceCoordinator(
        stakeholder_engagement_level="co-production",
        governance_approach="collaborative",
        equity_focus=True,
    )

    stakeholder_analysis = stakeholder_coordinator.analyze_stakeholders(
        governance_domain="coastal_governance",
        spatial_extent=governance_structure.spatial_scope,
        stakeholder_categories=[
            "government",
            "community",
            "business",
            "ngo",
            "indigenous",
            "academic",
        ],
    )

    print(f"✓ Stakeholder groups: {len(stakeholder_analysis['stakeholder_groups'])}")
    print(
        f"✓ Power balance: {stakeholder_analysis['power_dynamics']['power_balance_assessment']}"
    )
    print(
        f"✓ Collaboration potential: {stakeholder_analysis['collaboration_potential']:.1%}"
    )

    # 3. Conflict resolution
    print("\n3. CONFLICT RESOLUTION SYSTEM")
    print("-" * 80)

    conflict_resolver = ConflictResolver()

    # Simulate a conflict
    conflict = {
        "conflict_id": "fisheries_tourism_conflict",
        "type": "resource_allocation",
        "parties": ["fisheries_association", "tourism_industry"],
        "stakeholder_interests": {
            "fisheries_association": {"priority": 0.9, "resources": 0.7},
            "tourism_industry": {"priority": 0.8, "resources": 0.9},
        },
        "resource_constraints": {"budget": 5000000, "area_km2": 500},
    }

    resolution = conflict_resolver.resolve_conflict(
        conflict=conflict,
        stakeholders=[
            {
                "id": "fisheries_association",
                "name": "Fisheries Association",
                "decision_power": 0.6,
                "interest_level": 0.9,
            },
            {
                "id": "tourism_industry",
                "name": "Tourism Industry",
                "decision_power": 0.4,
                "interest_level": 0.8,
            },
        ],
        method=ConflictResolutionMethod.MEDIATION,
    )

    print(f"✓ Conflict resolution: {resolution.resolved}")
    print(f"✓ Resolution quality: {resolution.resolution_quality:.2f}")
    print(f"✓ Method: {resolution.resolution_method.value}")

    # 4. Performance evaluation
    print("\n4. PERFORMANCE EVALUATION")
    print("-" * 80)

    performance_evaluator = PerformanceEvaluator()

    performance_metrics = performance_evaluator.evaluate_governance_performance(
        governance_structure=governance_structure.__dict__,
        performance_data={
            "outcome_achievement": 0.75,
            "stakeholder_engagement": 0.70,
            "process_efficiency": 0.65,
            "compliance_rate": 0.80,
        },
    )

    print(
        f"✓ Overall performance: {performance_metrics.overall_score:.2f} ({performance_metrics.performance_rating})"
    )
    print("✓ Top dimensions:")
    sorted_dims = sorted(
        performance_metrics.dimension_scores.items(), key=lambda x: x[1], reverse=True
    )[:3]
    for dim, score in sorted_dims:
        print(f"  - {dim}: {score:.2f}")

    # 5. Scenario planning
    print("\n5. SCENARIO PLANNING")
    print("-" * 80)

    scenario_planner = ScenarioPlanner()

    scenarios = scenario_planner.generate_scenarios(
        governance_structure=governance_structure.__dict__,
        scenario_types=["optimistic", "pessimistic", "status_quo", "disruptive"],
        time_horizon=10,
    )

    scenario_analysis = scenario_planner.analyze_scenarios(
        governance_structure=governance_structure.__dict__, scenarios=scenarios
    )

    print(f"✓ Scenarios generated: {len(scenarios)}")
    print(
        f"✓ Base case performance: {scenario_analysis.base_case.get('scenario_performance', 0.5):.2f}"
    )
    print(
        f"✓ Critical factors: {len(scenario_analysis.sensitivity_analysis.get('critical_factors', []))}"
    )

    # 6. Accountability and transparency
    print("\n6. ACCOUNTABILITY & TRANSPARENCY")
    print("-" * 80)

    accountability_framework = AccountabilityFramework(
        accountability_model="multi_directional",
        transparency_level="full_disclosure",
        public_participation=True,
    )

    accountability_mechanisms = accountability_framework.establish_accountability(
        governing_bodies=[entity.__dict__ for entity in governance_structure.entities],
        stakeholder_groups=[
            sg.__dict__ for sg in stakeholder_analysis["stakeholder_groups"]
        ],
        accountability_directions=[
            "upward_to_public",
            "downward_to_communities",
            "horizontal_to_peers",
        ],
        enforcement_capacity="strong",
    )

    transparency_system = accountability_framework.implement_transparency(
        information_types=[
            "decisions",
            "budgets",
            "performance_reports",
            "stakeholder_input",
        ],
        disclosure_frequency="quarterly",
        accessibility_requirements=["digital", "multiple_languages"],
        documentation_standards="comprehensive",
    )

    print(
        f"✓ Accountability mechanisms: {len(accountability_mechanisms.audit_mechanisms)}"
    )
    print(f"✓ Transparency score: {transparency_system.transparency_score:.2f}")

    # 7. Adaptive governance
    print("\n7. ADAPTIVE GOVERNANCE SYSTEM")
    print("-" * 80)

    adaptive_system = AdaptiveGovernanceSystem()
    # Monitor performance
    performance_results = adaptive_system.monitor_performance(
        governance_indicators=[
            "effectiveness",
            "efficiency",
            "stakeholder_satisfaction",
        ],
        data_sources=["governance_structure", "performance_metrics"],
        evaluation_periods="annual",
    )

    # Adapt governance
    adaptation_result = adaptive_system.adapt_governance(
        performance_results=performance_results,
        learning_outcomes={
            "lessons_learned": ["improve_coordination", "enhance_participation"]
        },
        scenario_changes=[],
        adaptation_pathways=[
            {
                "name": "Enhanced Coordination",
                "expected_impact": 0.15,
                "feasibility": 0.8,
            },
            {
                "name": "Stakeholder Engagement",
                "expected_impact": 0.12,
                "feasibility": 0.7,
            },
        ],
    )

    print(
        f"✓ Performance indicators monitored: {len(performance_results.get('indicators', []))}"
    )
    print(f"✓ Adaptations made: {adaptation_result.get('adaptations_made', 0)}")
    print(
        f"✓ Predicted improvement: {adaptation_result.get('predicted_improvement', 0):.2f}"
    )

    # 8. Institutional design
    print("\n8. INSTITUTIONAL DESIGN")
    print("-" * 80)

    institutional_designer = InstitutionalDesigner(framework="iad")

    ostrom_design = institutional_designer.apply_ostrom_principles(
        principle_set=[
            "clear_boundaries",
            "congruence",
            "collective_choice_arrangements",
            "monitoring",
            "conflict_resolution",
            "nested_enterprises",
        ],
        resource_system=governance_structure.spatial_scope,
        governance_context={"scale": "regional", "complexity": "high"},
    )

    print(f"✓ Ostrom principles applied: {len(ostrom_design['governance_design'])}")
    print(f"✓ Design coherence: {ostrom_design['design_coherence']:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GOVERNANCE SYSTEM SUMMARY")
    print("=" * 80)
    print(f"✓ Governance Levels: {len(governance_structure.governance_levels)}")
    print(f"✓ Stakeholder Groups: {len(stakeholder_analysis['stakeholder_groups'])}")
    print(f"✓ Decision Domains: {len(governance_structure.decision_domains)}")
    print(f"✓ Performance Rating: {performance_metrics.performance_rating}")
    print(f"✓ Scenarios Analyzed: {len(scenarios)}")
    print(f"✓ Conflict Resolution: {resolution.resolved}")
    print("✓ Accountability: Multi-directional")
    print("✓ Transparency: Full disclosure")
    print("✓ Adaptive Mechanisms: Active")
    print("\n✓ Comprehensive governance system operational!")
    print("=" * 80)


if __name__ == "__main__":
    main()
