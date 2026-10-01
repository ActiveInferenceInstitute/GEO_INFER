"""
Basic example: Multi-level governance design for watershed management.

This example demonstrates how to use GEO-INFER-METAGOV to design a
multi-level governance structure for managing a shared watershed resource.
"""

from geo_infer_metagov.core.multi_level import MultiLevelGovernanceFramework
from geo_infer_metagov.core.stakeholder import StakeholderGovernanceCoordinator
from geo_infer_metagov.core.institutional import InstitutionalDesigner


def main():
    """Demonstrate watershed governance design."""

    print("=" * 70)
    print("GEO-INFER-METAGOV: Watershed Governance Design Example")
    print("=" * 70)

    # 1. Create multi-level governance framework
    print("\n1. Creating Multi-Level Governance Framework...")
    mlg_framework = MultiLevelGovernanceFramework(
        governance_levels=["local", "watershed", "regional"],
        coordination_mechanisms=["vertical_alignment", "horizontal_integration"],
        domain_coverage=["water_quality", "allocation", "flood_management"],
    )
    print("   ✓ Framework created with 3 governance levels")

    # 2. Design governance structure
    print("\n2. Designing Governance Structure...")
    spatial_scope = {
        "name": "Sacramento River Watershed",
        "area_km2": 70000,
        "jurisdictions": 6,
    }

    stakeholder_groups = [
        {"name": "Local Communities"},
        {"name": "Environmental Agencies"},
        {"name": "Water Users"},
        {"name": "Agricultural Interests"},
        {"name": "Conservation NGOs"},
        {"name": "Business Community"},
    ]

    decision_domains = [
        "water_allocation",
        "water_quality_standards",
        "environmental_flows",
        "flood_management",
        "agricultural_use",
        "groundwater_management",
    ]

    governance_structure = mlg_framework.design_governance_structure(
        spatial_scope=spatial_scope,
        stakeholder_groups=stakeholder_groups,
        decision_domains=decision_domains,
        time_horizons=[1, 5, 10, 20],
    )

    print(
        f"   ✓ Governance structure designed with {len(governance_structure.entities)} entities"
    )
    print(f"   ✓ Structure ID: {governance_structure.governance_id}")

    # 3. Analyze stakeholders
    print("\n3. Analyzing Stakeholders...")
    stakeholder_coordinator = StakeholderGovernanceCoordinator(
        stakeholder_engagement_level="co-production",
        governance_approach="collaborative",
        equity_focus=True,
    )

    stakeholder_analysis = stakeholder_coordinator.analyze_stakeholders(
        governance_domain="watershed_management",
        spatial_extent=spatial_scope,
        stakeholder_categories=[
            "government",
            "community",
            "ngo",
            "business",
            "indigenous",
        ],
    )

    print(
        f"   ✓ Identified {len(stakeholder_analysis['stakeholder_groups'])} stakeholder groups"
    )
    print(
        f"   ✓ Collaboration potential: {stakeholder_analysis['collaboration_potential']:.2%}"
    )
    print(
        f"   ✓ Power balance: {stakeholder_analysis['power_dynamics']['power_balance_assessment']}"
    )

    # 4. Design institutional framework
    print("\n4. Applying Institutional Design Principles...")
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
        resource_system=spatial_scope,
        governance_context={"scale": "watershed", "complexity": "high"},
    )

    print(f"   ✓ Applied {len(ostrom_design['governance_design'])} Ostrom principles")
    print(f"   ✓ Design coherence score: {ostrom_design['design_coherence']:.2f}")

    # 5. Establish governance platform
    print("\n5. Establishing Multi-Stakeholder Governance Platform...")
    governance_platform = stakeholder_coordinator.establish_governance_platform(
        participants=stakeholder_analysis["stakeholder_groups"],
        governance_mechanisms=[
            "participatory_workshops",
            "consensus_building",
            "shared_decision_making",
        ],
        decision_domains=decision_domains,
        conflict_resolution_capacity=True,
    )

    print(
        f"   ✓ Platform established with {len(governance_platform.stakeholders)} stakeholders"
    )
    print(f"   ✓ Platform ID: {governance_platform.platform_id}")

    # 6. Coordinate vertical governance
    print("\n6. Coordinating Vertical Governance Levels...")
    policy_proposal = {
        "id": "water_allocation_policy_2025",
        "name": "Water Allocation Framework",
        "target_stakeholders": ["all"],
        "urgency": "high",
    }

    coordination = mlg_framework.coordinate_vertical_levels(
        governance_structure=governance_structure, policy_proposal=policy_proposal
    )

    approvals = sum(
        1
        for a in coordination["level_approvals"].values()
        if a.get("approval_status") == "approved"
    )
    print(
        f"   ✓ Coordination evaluated across {len(coordination['level_approvals'])} levels"
    )
    print(f"   ✓ Approvals: {approvals}/{len(coordination['level_approvals'])}")
    print(f"   ✓ Conflicts identified: {len(coordination['cross_level_conflicts'])}")

    # 7. Apply subsidiarity principle
    print("\n7. Applying Subsidiarity Principle...")
    subsidiarity = mlg_framework.apply_subsidiarity_principle(
        governance_structure=governance_structure,
        decision_domain="water_quality_standards",
    )

    print(f"   ✓ Subsidiary level: {subsidiarity['subsidiary_level']}")
    print(f"   ✓ Escalation needed: {subsidiarity['escalation_needed']}")
    print(
        f"   ✓ Coordination requirements: {', '.join(subsidiarity['coordination_requirements'])}"
    )

    # Summary
    print("\n" + "=" * 70)
    print("GOVERNANCE DESIGN SUMMARY")
    print("=" * 70)
    print(
        f"✓ Governance Model: Multi-level with {len(governance_structure.governance_levels)} levels"
    )
    print(f"✓ Stakeholder Groups: {len(stakeholder_analysis['stakeholder_groups'])}")
    print(f"✓ Decision Domains: {len(decision_domains)}")
    print("✓ Institutional Framework: IAD with Ostrom's 8 principles")
    print(f"✓ Governance Platform: {governance_platform.platform_id}")
    coord_mech = ", ".join(
        [
            m.value if hasattr(m, "value") else str(m)
            for m in governance_structure.coordination_mechanisms
        ]
    )
    print(f"✓ Coordination Mechanisms: {coord_mech}")
    print("\n✓ Governance structure ready for implementation!")
    print("=" * 70)


if __name__ == "__main__":
    main()
