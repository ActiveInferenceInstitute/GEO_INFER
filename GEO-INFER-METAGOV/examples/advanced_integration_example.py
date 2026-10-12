"""
Advanced example: Multi-level governance with organizational and normative integration.

This example demonstrates how to integrate GEO-INFER-METAGOV with other modules
to create a complete governance system including organizational structures,
governance rules, and accountability frameworks.
"""

from geo_infer_metagov.core.multi_level import MultiLevelGovernanceFramework
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator
from geo_infer_metagov.core.institutional import InstitutionalDesigner
from geo_infer_metagov.core.accountability import AccountabilityFramework


def main():
    """Demonstrate integrated governance system for urban climate adaptation."""

    print("=" * 80)
    print(
        "GEO-INFER-METAGOV: Integrated Governance System for Urban Climate Adaptation"
    )
    print("=" * 80)

    # 1. Design multi-level governance structure
    print("\n1. DESIGNING MULTI-LEVEL GOVERNANCE STRUCTURE")
    print("-" * 80)

    mlg_framework = MultiLevelGovernanceFramework(
        governance_levels=["local", "regional", "national"],
        coordination_mechanisms=[
            "vertical_alignment",
            "horizontal_integration",
            "subsidiarity",
        ],
        domain_coverage=[
            "climate_adaptation",
            "environmental",
            "civic",
            "infrastructure",
        ],
    )

    governance_structure = mlg_framework.design_governance_structure(
        spatial_scope={
            "name": "Metropolitan Area",
            "area_km2": 1200,
            "jurisdictions": 8,
            "population": 4000000,
        },
        stakeholder_groups=[
            {"name": "City Government"},
            {"name": "District Municipalities"},
            {"name": "Climate Action Committee"},
            {"name": "Community Organizations"},
            {"name": "Environmental NGOs"},
            {"name": "Business Community"},
            {"name": "Utility Providers"},
            {"name": "Academic Institutions"},
        ],
        decision_domains=[
            "urban_heat_mitigation",
            "flood_risk_management",
            "green_infrastructure",
            "transport_electrification",
            "building_efficiency",
            "renewable_energy",
            "water_security",
            "air_quality",
        ],
        time_horizons=[1, 5, 10, 20, 50],
    )

    print(
        f"✓ Created governance structure with {len(governance_structure.entities)} levels"
    )
    print(f"✓ Decision domains: {len(governance_structure.decision_domains)}")
    print("✓ Governance entities:")
    for entity in governance_structure.entities:
        print(
            f"  - {entity.name} ({entity.governance_level.value}): {len(entity.responsibilities)} responsibilities"
        )

    # 2. Analyze stakeholders and establish platforms
    print("\n2. STAKEHOLDER ENGAGEMENT & PLATFORM ESTABLISHMENT")
    print("-" * 80)

    stakeholder_coordinator = StakeholderGovernanceCoordinator(
        stakeholder_engagement_level="co-production",
        governance_approach="collaborative",
        equity_focus=True,
    )

    stakeholder_analysis = stakeholder_coordinator.analyze_stakeholders(
        governance_domain="urban_climate_adaptation",
        spatial_extent=governance_structure.spatial_scope,
        stakeholder_categories=[
            "government",
            "community",
            "business",
            "ngo",
            "academic",
        ],
    )

    print(
        f"✓ Identified {len(stakeholder_analysis['stakeholder_groups'])} stakeholder groups"
    )
    print(
        f"✓ Collaboration potential: {stakeholder_analysis['collaboration_potential']:.1%}"
    )
    print(
        f"✓ Power distribution: {stakeholder_analysis['power_dynamics']['power_balance_assessment']}"
    )

    # Establish governance platform
    governance_platform = stakeholder_coordinator.establish_governance_platform(
        participants=stakeholder_analysis["stakeholder_groups"],
        governance_mechanisms=[
            "climate_action_committee",
            "participatory_budgeting",
            "consensus_workshops",
            "digital_engagement",
        ],
        decision_domains=governance_structure.decision_domains,
        conflict_resolution_capacity=True,
    )

    print(
        f"✓ Governance platform established with {len(governance_platform.stakeholders)} stakeholders"
    )

    # 3. Design institutional framework
    print("\n3. INSTITUTIONAL DESIGN & RULES")
    print("-" * 80)

    institutional_designer = InstitutionalDesigner(framework="iad")

    institutional_rules = [
        {
            "name": "Climate Action Charter",
            "type": "boundary",
            "description": "Defines boundaries and membership in climate governance",
        },
        {
            "name": "Decision Authority Matrix",
            "type": "position",
            "description": "Specifies roles and decision-making authority at each level",
        },
        {
            "name": "Participatory Decision Rules",
            "type": "choice",
            "description": "How decisions are made including consensus requirements",
        },
        {
            "name": "Performance Reporting",
            "type": "information",
            "description": "What information is collected and shared",
        },
    ]

    institutional_analysis = institutional_designer.analyze_institutions(
        current_institutions=institutional_rules,
        stakeholder_groups=[
            {"name": sg.name, "category": sg.category}
            for sg in stakeholder_analysis["stakeholder_groups"]
        ],
        resource_system=governance_structure.spatial_scope,
        decision_outcomes=[
            {
                "effectiveness": 0.7,
                "stakeholders": ["Government Group", "Community Group"],
            },
            {"effectiveness": 0.6, "stakeholders": ["Business Group", "Ngo Group"]},
            {"effectiveness": 0.8, "stakeholders": ["Academic Group"]},
        ],
    )

    print("✓ Institutional analysis complete")
    print(f"✓ Recommendations: {len(institutional_analysis.recommendations)}")
    for i, rec in enumerate(institutional_analysis.recommendations[:3], 1):
        print(f"  {i}. {rec}")

    # Apply Ostrom's principles
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
        governance_context={
            "scale": "metropolitan",
            "complexity": "high",
            "urgency": "climate_action",
        },
    )

    print(
        f"✓ Applied {len(ostrom_design['governance_design'])} Ostrom design principles"
    )
    print(f"✓ Design coherence: {ostrom_design['design_coherence']:.2f}")

    # 4. Establish accountability and transparency
    print("\n4. ACCOUNTABILITY & TRANSPARENCY FRAMEWORK")
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
            "climate_action_plans",
            "budget_allocations",
            "emission_data",
            "progress_reports",
            "decisions_and_rationale",
        ],
        disclosure_frequency="quarterly",
        accessibility_requirements=[
            "multiple_languages",
            "digital_and_traditional",
            "simplified_versions",
        ],
        documentation_standards="comprehensive",
    )

    print(
        f"✓ Accountability mechanisms: {len(accountability_mechanisms.audit_mechanisms)}"
    )
    print(
        f"✓ Transparency system: {len(transparency_system.public_access_mechanisms)} access mechanisms"
    )
    print("✓ Public participation mechanisms:")
    participation = accountability_framework.enable_participation(
        participation_forms=[
            "information_access",
            "consultation",
            "co-management",
            "co-production",
        ],
        barriers_to_remove=["language", "digital_access", "time_constraints"],
        capacity_building="supported",
    )
    for form in participation["participation_forms"]:
        print(f"  - {form}")

    # 5. Coordinate governance levels
    print("\n5. VERTICAL GOVERNANCE COORDINATION")
    print("-" * 80)

    climate_action_policy = {
        "id": "climate_action_plan_2025",
        "name": "Comprehensive Climate Action Plan",
        "objectives": [
            "50% emission reduction by 2030",
            "Climate resilience",
            "Equity",
        ],
        "priority": "critical",
        "budget_required": 1500000000,
        "stakeholders_involved": 8,
    }

    coordination_result = mlg_framework.coordinate_vertical_levels(
        governance_structure=governance_structure, policy_proposal=climate_action_policy
    )

    approvals = sum(
        1
        for a in coordination_result["level_approvals"].values()
        if a.get("approval_status") == "approved"
    )
    print(
        f"✓ Policy coordination across {len(coordination_result['level_approvals'])} levels"
    )
    print(f"✓ Approvals: {approvals}/{len(coordination_result['level_approvals'])}")
    print(
        f"✓ Cross-level conflicts: {len(coordination_result['cross_level_conflicts'])}"
    )
    print(
        f"✓ Implementation approach: {coordination_result['coordinated_implementation'].get('recommended_approach')}"
    )

    # 6. Apply subsidiarity
    print("\n6. SUBSIDIARITY PRINCIPLE APPLICATION")
    print("-" * 80)

    decision_domains_to_analyze = [
        "urban_heat_mitigation",
        "green_infrastructure",
        "flood_risk_management",
    ]

    for domain in decision_domains_to_analyze:
        subsidiarity = mlg_framework.apply_subsidiarity_principle(
            governance_structure=governance_structure, decision_domain=domain
        )
        print(f"\n✓ {domain.replace('_', ' ').title()}:")
        print(f"  - Subsidiary level: {subsidiarity['subsidiary_level']}")
        print(f"  - Escalation needed: {subsidiarity['escalation_needed']}")
        if subsidiarity["coordination_requirements"]:
            print(
                f"  - Coordination: {', '.join(subsidiarity['coordination_requirements'][:2])}"
            )

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATED GOVERNANCE SYSTEM SUMMARY")
    print("=" * 80)
    print(
        f"✓ Governance Model: Multi-level with {len(governance_structure.governance_levels)} levels"
    )
    print(f"✓ Stakeholders: {len(stakeholder_analysis['stakeholder_groups'])} groups")
    print(f"✓ Decision Domains: {len(governance_structure.decision_domains)}")
    print(
        f"✓ Institutional Framework: IAD with {len(ostrom_design['governance_design'])} Ostrom principles"
    )
    print(f"✓ Governance Platform: {governance_platform.platform_id}")
    print("✓ Accountability Directions: Multi-directional")
    print("✓ Transparency Level: Full disclosure with quarterly reporting")
    print("✓ Subsidiarity: Applied across all decision domains")
    print("\n✓ System ready for climate action implementation!")
    print("=" * 80)


if __name__ == "__main__":
    main()
