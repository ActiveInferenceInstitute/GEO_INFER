"""Polycentric governance systems with multiple overlapping authorities."""

from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolycentricDesign:
    """Polycentric governance structure."""

    design_id: str
    governing_bodies: list[dict[str, Any]]
    jurisdictional_overlaps: dict[str, list[str]]
    spatial_scales: list[str]
    functional_domains: list[str]
    feedback_mechanisms: dict[str, Any]
    redundancy_assessment: dict[str, float]


class PolycentricGovernanceSystem:
    """Design and coordinate polycentric governance with multiple overlapping authorities."""

    def __init__(
        self,
        governance_model: str = "polycentric",
        coordination_mechanism: str = "network_based",
        redundancy_level: str = "adaptive",
    ) -> None:
        self.governance_model = governance_model
        self.coordination_mechanism = coordination_mechanism
        self.redundancy_level = redundancy_level
        self.polycentric_designs: dict[str, PolycentricDesign] = {}

    def design_polycentric_structure(
        self,
        governing_bodies: list[dict[str, Any]],
        jurisdictional_overlaps: dict[str, list[str]],
        spatial_scales: list[str],
        functional_domains: list[str],
        feedback_mechanisms: dict[str, Any],
    ) -> PolycentricDesign:
        """Design polycentric governance structure."""
        design_id = f"polycentric_{len(self.polycentric_designs)}"

        redundancy = self._assess_redundancy(
            governing_bodies=governing_bodies,
            jurisdictional_overlaps=jurisdictional_overlaps,
        )

        design = PolycentricDesign(
            design_id=design_id,
            governing_bodies=governing_bodies,
            jurisdictional_overlaps=jurisdictional_overlaps,
            spatial_scales=spatial_scales,
            functional_domains=functional_domains,
            feedback_mechanisms=feedback_mechanisms,
            redundancy_assessment=redundancy,
        )

        self.polycentric_designs[design_id] = design
        logger.info(f"Polycentric design created: {design_id}")
        return design

    def _assess_redundancy(
        self,
        governing_bodies: list[dict[str, Any]],
        jurisdictional_overlaps: dict[str, list[str]],
    ) -> dict[str, float]:
        """Assess redundancy in governance."""
        total_bodies = len(governing_bodies)
        overlap_count = sum(len(v) for v in jurisdictional_overlaps.values())
        redundancy_ratio = overlap_count / total_bodies if total_bodies > 0 else 0

        return {
            "redundancy_ratio": min(redundancy_ratio, 1.0),
            "resilience_level": 0.7 + (redundancy_ratio * 0.2),
            "efficiency_impact": 0.7 - (redundancy_ratio * 0.15),
        }

    def analyze_authority_relationships(
        self,
        authorities: list[dict[str, Any]],
        relationships: list[str],
        effectiveness_measures: list[str],
    ) -> dict[str, Any]:
        """
        Analyze relationships between authorities using network analysis.

        Calculates:
        - Coordination network density
        - Authority overlap analysis
        - Redundancy and resilience metrics
        - Coordination effectiveness

        References:
        - Ostrom, E. (2010). Beyond Markets and States: Polycentric Governance
        - Carlisle, K., & Gruby, R. L. (2019). Polycentric systems of governance
        """
        if not authorities:
            return {
                "authority_count": 0,
                "relationship_types": relationships,
                "effectiveness_measures": effectiveness_measures,
                "coordination_index": 0.0,
            }

        # Build relationship network
        network_edges = []
        for i, auth1 in enumerate(authorities):
            for j, auth2 in enumerate(authorities[i + 1 :], start=i + 1):
                # Check if authorities have overlapping jurisdictions or domains
                auth1_domains = set(auth1.get("domains", []))
                auth2_domains = set(auth2.get("domains", []))
                overlap = len(auth1_domains & auth2_domains)

                if overlap > 0 or any(
                    rel in relationships for rel in ["coordination", "cooperation"]
                ):
                    network_edges.append(
                        {
                            "source": auth1.get("id", f"auth_{i}"),
                            "target": auth2.get("id", f"auth_{j}"),
                            "weight": overlap
                            / max(1, len(auth1_domains | auth2_domains)),
                            "type": "coordination" if overlap > 0 else "cooperation",
                        }
                    )

        # Calculate network density
        n = len(authorities)
        max_possible_edges = n * (n - 1) / 2 if n > 1 else 1
        network_density = (
            len(network_edges) / max_possible_edges if max_possible_edges > 0 else 0.0
        )

        # Calculate coordination index
        # Based on network density, relationship types, and overlap
        relationship_score = len(relationships) / max(
            1, len(["coordination", "cooperation", "competition", "hierarchy"])
        )
        overlap_score = (
            sum(edge["weight"] for edge in network_edges) / max(1, len(network_edges))
            if network_edges
            else 0.0
        )

        coordination_index = (
            network_density * 0.4 + relationship_score * 0.3 + overlap_score * 0.3
        )

        # Analyze authority overlap
        overlap_analysis = self._analyze_authority_overlaps(authorities)

        # Calculate redundancy metrics
        redundancy_metrics = self._calculate_redundancy_metrics(
            authorities, network_edges
        )

        # Assess coordination failure risk
        failure_risk = self._assess_coordination_failure_risk(
            authorities, network_edges, relationships
        )

        return {
            "authority_count": len(authorities),
            "relationship_types": relationships,
            "effectiveness_measures": effectiveness_measures,
            "coordination_index": coordination_index,
            "network_density": network_density,
            "network_edges": len(network_edges),
            "overlap_analysis": overlap_analysis,
            "redundancy_metrics": redundancy_metrics,
            "coordination_failure_risk": failure_risk,
            "resilience_score": 1.0 - failure_risk,
        }

    def _analyze_authority_overlaps(
        self, authorities: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze jurisdictional and functional overlaps between authorities."""
        overlap_matrix_out: dict[str, dict[str, Any]] = {}
        overlaps: dict[str, Any] = {
            "jurisdictional_overlaps": 0,
            "functional_overlaps": 0,
            "total_overlap_pairs": 0,
            "overlap_matrix": overlap_matrix_out,
        }

        for i, auth1 in enumerate(authorities):
            auth1_id = auth1.get("id", f"auth_{i}")
            overlap_matrix_out[auth1_id] = {}

            for j, auth2 in enumerate(authorities):
                if i == j:
                    continue

                auth2_id = auth2.get("id", f"auth_{j}")

                # Check jurisdictional overlap
                auth1_jurisdiction = set(auth1.get("jurisdiction", []))
                auth2_jurisdiction = set(auth2.get("jurisdiction", []))
                jurisdictional_overlap = len(auth1_jurisdiction & auth2_jurisdiction)

                # Check functional overlap
                auth1_domains = set(auth1.get("domains", []))
                auth2_domains = set(auth2.get("domains", []))
                functional_overlap = len(auth1_domains & auth2_domains)

                if jurisdictional_overlap > 0:
                    overlaps["jurisdictional_overlaps"] += 1
                if functional_overlap > 0:
                    overlaps["functional_overlaps"] += 1
                if jurisdictional_overlap > 0 or functional_overlap > 0:
                    overlaps["total_overlap_pairs"] += 1

                overlap_matrix_out[auth1_id][auth2_id] = {
                    "jurisdictional": jurisdictional_overlap,
                    "functional": functional_overlap,
                }

        return overlaps

    def _calculate_redundancy_metrics(
        self, authorities: list[dict[str, Any]], network_edges: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate redundancy and resilience metrics."""
        if not authorities:
            return {
                "redundancy_ratio": 0.0,
                "resilience_level": 0.0,
                "efficiency_impact": 1.0,
            }

        # Calculate functional redundancy (multiple authorities covering same domain)
        all_domains = set()
        domain_coverage = {}

        for auth in authorities:
            domains = set(auth.get("domains", []))
            all_domains.update(domains)
            for domain in domains:
                if domain not in domain_coverage:
                    domain_coverage[domain] = 0
                domain_coverage[domain] += 1

        # Redundancy ratio: proportion of domains covered by multiple authorities
        redundant_domains = sum(1 for count in domain_coverage.values() if count > 1)
        redundancy_ratio = redundant_domains / len(all_domains) if all_domains else 0.0

        # Resilience increases with redundancy (up to a point)
        resilience_level = min(1.0, 0.5 + redundancy_ratio * 0.4)

        # Efficiency decreases with excessive redundancy
        efficiency_impact = max(0.3, 1.0 - redundancy_ratio * 0.4)

        return {
            "redundancy_ratio": redundancy_ratio,
            "resilience_level": resilience_level,
            "efficiency_impact": efficiency_impact,
            "redundant_domains": redundant_domains,
            "total_domains": len(all_domains),
        }

    def _assess_coordination_failure_risk(
        self,
        authorities: list[dict[str, Any]],
        network_edges: list[dict[str, Any]],
        relationships: list[str],
    ) -> float:
        """Assess risk of coordination failure."""
        if not authorities:
            return 1.0  # High risk if no authorities

        risk_factors = []

        # Factor 1: Low network density (poor connectivity)
        n = len(authorities)
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = len(network_edges) / max_edges if max_edges > 0 else 0.0
        if density < 0.3:
            risk_factors.append(0.3)  # High risk from low connectivity

        # Factor 2: Competition without coordination
        if "competition" in relationships and "coordination" not in relationships:
            risk_factors.append(0.25)

        # Factor 3: Too many authorities (coordination complexity)
        if n > 10:
            risk_factors.append(0.2)

        # Factor 4: Lack of hierarchical relationships
        if "hierarchy" not in relationships and n > 3:
            risk_factors.append(0.15)

        # Factor 5: Authority capacity issues
        low_capacity_count = sum(
            1 for auth in authorities if auth.get("capacity", 1.0) < 0.5
        )
        if low_capacity_count > len(authorities) / 2:
            risk_factors.append(0.1)

        # Calculate overall risk
        total_risk = sum(risk_factors) if risk_factors else 0.0
        return min(1.0, total_risk)
