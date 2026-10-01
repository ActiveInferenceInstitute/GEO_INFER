#!/usr/bin/env python3
"""GEO-INFER-ORG module orchestrator.

Runs one documented end-to-end ORG operation on synthetic data: model a
synthetic org hierarchy, compute structural metrics and a proportional budget
allocation, tally a governance proposal, and analyze a collaboration network
with an optimized team formation. All work goes through the real
``geo_infer_org`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_org import (
        CollaborationEdge,
        CollaborationNetwork,
        CollaborationType,
        OrgStructureType,
        OrgUnit,
        OrganizationModel,
        Proposal,
        Role,
        RoleLevel,
        TeamFormation,
        TeamMember,
        Vote,
        VotingEngine,
        VotingMethod,
    )

    # Synthetic org: a coastal resource institute with a two-level hierarchy.
    org = OrganizationModel(structure_type=OrgStructureType.HIERARCHICAL)
    units = [
        OrgUnit(unit_id="root", name="Institute", member_count=48, budget=480_000.0),
        OrgUnit(
            unit_id="research",
            name="Research Division",
            parent_id="root",
            member_count=22,
            budget=260_000.0,
        ),
        OrgUnit(
            unit_id="outreach",
            name="Outreach Division",
            parent_id="root",
            member_count=14,
            budget=120_000.0,
        ),
        OrgUnit(
            unit_id="ops",
            name="Operations Division",
            parent_id="root",
            member_count=12,
            budget=100_000.0,
        ),
        OrgUnit(
            unit_id="lab_eco",
            name="Ecology Lab",
            parent_id="research",
            member_count=10,
            budget=90_000.0,
        ),
        OrgUnit(
            unit_id="lab_geo",
            name="Geospatial Lab",
            parent_id="research",
            member_count=12,
            budget=170_000.0,
        ),
    ]
    for unit in units:
        org.add_unit(unit)
    roles = [
        Role(role_id="r1", title="Director", level=RoleLevel.EXECUTIVE, unit_id="root"),
        Role(
            role_id="r2",
            title="Division Lead",
            level=RoleLevel.DIRECTOR,
            unit_id="research",
            reports_to="r1",
        ),
        Role(
            role_id="r3",
            title="Division Lead",
            level=RoleLevel.DIRECTOR,
            unit_id="outreach",
            reports_to="r1",
        ),
        Role(
            role_id="r4",
            title="Lab Manager",
            level=RoleLevel.MANAGER,
            unit_id="lab_eco",
            reports_to="r2",
        ),
        Role(
            role_id="r5",
            title="Lab Manager",
            level=RoleLevel.MANAGER,
            unit_id="lab_geo",
            reports_to="r2",
        ),
        Role(
            role_id="r6",
            title="Research Scientist",
            level=RoleLevel.INDIVIDUAL,
            unit_id="lab_eco",
            reports_to="r4",
        ),
        Role(
            role_id="r7",
            title="GIS Analyst",
            level=RoleLevel.INDIVIDUAL,
            unit_id="lab_geo",
            reports_to="r5",
        ),
        Role(
            role_id="r8",
            title="Coordinator",
            level=RoleLevel.INDIVIDUAL,
            unit_id="outreach",
            reports_to="r3",
        ),
    ]
    for role in roles:
        org.add_role(role)
    metrics = org.compute_metrics()
    budget_allocation = org.allocate_budget(
        total_budget=1_500_000.0, strategy="proportional"
    )
    reporting_chain = [role.title for role in org.find_reporting_chain("r7")]

    # Governance: simple-majority vote on a synthetic field-station proposal.
    engine = VotingEngine()
    proposal = Proposal(
        proposal_id="prop-2026-01",
        title="Fund the Klamath estuary field station",
        description="Allocate reserve funds for a shared field station.",
        proposer_id="r2",
        options=["approve", "reject", "defer"],
        voting_method=VotingMethod.SIMPLE_MAJORITY,
        eligible_voters=7,
        quorum_fraction=0.5,
    )
    engine.create_proposal(proposal)
    choices = ["approve", "approve", "approve", "reject", "approve", "defer", "approve"]
    for i, choice in enumerate(choices):
        engine.cast_vote(
            "prop-2026-01", Vote(voter_id=f"member-{i:02d}", choice=choice)
        )
    result = engine.tally("prop-2026-01")

    # Collaboration network: cross-lab project interactions.
    network = CollaborationNetwork()
    people = [f"staff-{i:02d}" for i in range(12)]
    for person in people:
        network.add_node(person)
    for i in range(18):
        source = people[i % 12]
        target = people[(i * 5 + 3) % 12]
        if source != target:
            network.add_edge(
                CollaborationEdge(
                    source_id=source,
                    target_id=target,
                    collaboration_type=CollaborationType.KNOWLEDGE_SHARE
                    if i % 3 == 0
                    else CollaborationType.TASK_COORDINATION,
                    strength=0.4 + (i % 6) / 10.0,
                )
            )
    net_metrics = network.compute_metrics()
    centrality = network.compute_betweenness_centrality()

    # Team formation: cover a synthetic monitoring-project skill set.
    formation = TeamFormation()
    formation.add_members(
        [
            TeamMember(
                member_id="staff-00",
                name="Ada Nunez",
                skills=["field_survey", "gis"],
                unit_id="lab_geo",
            ),
            TeamMember(
                member_id="staff-01",
                name="Bo Chen",
                skills=["gis", "python"],
                unit_id="lab_geo",
            ),
            TeamMember(
                member_id="staff-02",
                name="Cy Okafor",
                skills=["hydrology", "field_survey"],
                unit_id="lab_eco",
            ),
            TeamMember(
                member_id="staff-03",
                name="Di Rao",
                skills=["statistics", "python"],
                unit_id="lab_eco",
            ),
            TeamMember(
                member_id="staff-04",
                name="Eli Marsh",
                skills=["community_engagement"],
                unit_id="outreach",
            ),
            TeamMember(
                member_id="staff-05",
                name="Fay Lund",
                skills=["statistics", "gis"],
                unit_id="lab_eco",
            ),
        ]
    )
    team = formation.form_team(
        required_skills=[
            "field_survey",
            "gis",
            "python",
            "statistics",
            "community_engagement",
        ],
        max_size=5,
    )

    return {
        "operation": "org_model_governance_and_collaboration",
        "org_metrics": {
            "total_units": metrics.total_units,
            "total_roles": metrics.total_roles,
            "max_depth": metrics.max_depth,
            "avg_span_of_control": metrics.avg_span_of_control,
            "centralization_score": metrics.centralization_score,
            "hierarchy_ratio": metrics.hierarchy_ratio,
        },
        "budget_allocation": budget_allocation,
        "reporting_chain_r7": reporting_chain,
        "governance": {
            "proposal_id": result.proposal_id,
            "winner": result.winner,
            "vote_counts": result.vote_counts,
            "total_votes": result.total_votes,
            "quorum_met": result.quorum_met,
        },
        "collaboration_network": {
            "node_count": net_metrics.node_count,
            "edge_count": net_metrics.edge_count,
            "density": net_metrics.density,
            "avg_degree": net_metrics.avg_degree,
            "clustering_coefficient": net_metrics.clustering_coefficient,
            "connected_components": net_metrics.connected_components,
            "top_central_node": (
                max(centrality.items(), key=lambda kv: kv[1]) if centrality else None
            ),
        },
        "team_formation": {
            "team_members": team.team_members,
            "skill_coverage": team.skill_coverage,
            "team_diversity": team.team_diversity,
            "overall_score": team.overall_score,
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ORG", _operation))
