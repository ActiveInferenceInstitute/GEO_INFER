"""Stakeholder governance coordination for multi-stakeholder systems."""

from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Stakeholder:
    """Represents a governance stakeholder."""

    stakeholder_id: str
    name: str
    category: str  # government, community, business, ngo, indigenous, etc.
    interests: list[str]
    influence_level: float  # 0-1 scale
    dependence_on_resource: float  # 0-1 scale
    decision_power: float  # 0-1 scale


@dataclass
class GovernancePlatform:
    """Multi-stakeholder governance platform."""

    platform_id: str
    stakeholders: list[Stakeholder]
    governance_mechanisms: list[str]
    decision_domains: list[str]
    conflict_resolution_capacity: bool
    participation_level: str  # information, consultation, co-management, co-production
    decision_process: dict[str, Any] = field(default_factory=dict)


class StakeholderGovernanceCoordinator:
    """
    Coordinates governance across diverse stakeholder groups with different interests.

    Manages multi-stakeholder platforms, participatory governance design, conflict
    identification and resolution, power dynamics analysis, and inclusive decision-making.

    References:
    - Ansell, C., & Gash, A. (2008). Collaborative Governance in Theory and Practice
    - Pahl-Wostl, C., et al. (2007). The Importance of Social Learning
    """

    def __init__(
        self,
        stakeholder_engagement_level: str = "consultation",
        governance_approach: str = "collaborative",
        equity_focus: bool = True,
    ):
        """
        Initialize stakeholder governance coordinator.

        Parameters:
        -----------
        stakeholder_engagement_level : str
            Level of stakeholder engagement (information, consultation, co-management, co-production)
        governance_approach : str
            Governance approach (hierarchical, collaborative, participatory)
        equity_focus : bool
            Whether to focus on equity outcomes
        """
        self.engagement_level = stakeholder_engagement_level
        self.governance_approach = governance_approach
        self.equity_focus = equity_focus
        self.governance_platforms: dict[str, GovernancePlatform] = {}

    def analyze_stakeholders(
        self,
        governance_domain: str,
        spatial_extent: dict[str, Any],
        stakeholder_categories: list[str],
    ) -> dict[str, Any]:
        """
        Identify and analyze stakeholders.

        Parameters:
        -----------
        governance_domain : str
            Domain of governance (watershed, protected area, city, etc.)
        spatial_extent : Dict[str, Any]
            Geographic extent of governance
        stakeholder_categories : List[str]
            Categories of stakeholders to include

        Returns:
        --------
        Dict[str, Any]
            Comprehensive stakeholder analysis
        """
        stakeholder_groups_out: list[Stakeholder] = []
        power_dynamics_out: dict[str, Any] = {}
        interest_conflicts_out: list[dict[str, Any]] = []
        stakeholder_analysis: dict[str, Any] = {
            "governance_domain": governance_domain,
            "spatial_extent": spatial_extent,
            "stakeholder_groups": stakeholder_groups_out,
            "power_dynamics": power_dynamics_out,
            "interest_conflicts": interest_conflicts_out,
            "collaboration_potential": 0.0,
        }

        # Create stakeholders for each category
        stakeholder_idx = 0
        for category in stakeholder_categories:
            stakeholder = Stakeholder(
                stakeholder_id=f"stakeholder_{stakeholder_idx}",
                name=f"{category.replace('_', ' ').title()} Group",
                category=category,
                interests=self._identify_interests(category),
                influence_level=self._estimate_influence(category),
                dependence_on_resource=self._estimate_dependence(category),
                decision_power=self._estimate_decision_power(category),
            )
            stakeholder_groups_out.append(stakeholder)
            stakeholder_idx += 1

        # Analyze power dynamics
        power_dynamics_out = self._analyze_power_dynamics(stakeholder_groups_out)
        stakeholder_analysis["power_dynamics"] = power_dynamics_out

        # Identify interest conflicts
        interest_conflicts_out = self._identify_conflicts(stakeholder_groups_out)
        stakeholder_analysis["interest_conflicts"] = interest_conflicts_out

        # Assess collaboration potential
        stakeholder_analysis["collaboration_potential"] = (
            self._assess_collaboration_potential(stakeholder_groups_out)
        )

        return stakeholder_analysis

    def _identify_interests(self, stakeholder_category: str) -> list[str]:
        """Identify typical interests for stakeholder category."""
        interests_map = {
            "government": ["regulation", "equity", "sustainability", "public_benefit"],
            "community": ["livelihood", "resource_access", "local_control", "equity"],
            "business": ["profit", "market_access", "efficiency", "growth"],
            "ngo": ["conservation", "social_justice", "sustainability", "equity"],
            "indigenous": [
                "traditional_rights",
                "cultural_preservation",
                "land_rights",
                "autonomy",
            ],
            "private_sector": [
                "efficiency",
                "return_on_investment",
                "market_opportunity",
                "scale",
            ],
        }
        return interests_map.get(
            stakeholder_category.lower(), ["resource_use", "benefit_sharing"]
        )

    def _estimate_influence(self, stakeholder_category: str) -> float:
        """Estimate influence level by category."""
        influence_map = {
            "government": 0.9,
            "business": 0.7,
            "ngo": 0.5,
            "community": 0.4,
            "indigenous": 0.3,
            "private_sector": 0.6,
        }
        return influence_map.get(stakeholder_category.lower(), 0.5)

    def _estimate_dependence(self, stakeholder_category: str) -> float:
        """Estimate dependence on resource."""
        dependence_map = {
            "community": 0.9,
            "indigenous": 0.8,
            "business": 0.6,
            "government": 0.3,
            "ngo": 0.2,
            "private_sector": 0.5,
        }
        return dependence_map.get(stakeholder_category.lower(), 0.5)

    def _estimate_decision_power(self, stakeholder_category: str) -> float:
        """Estimate decision-making power."""
        power_map = {
            "government": 0.85,
            "business": 0.6,
            "ngo": 0.4,
            "community": 0.35,
            "indigenous": 0.3,
            "private_sector": 0.55,
        }
        return power_map.get(stakeholder_category.lower(), 0.4)

    def _analyze_power_dynamics(
        self, stakeholders: list[Stakeholder]
    ) -> dict[str, Any]:
        """
        Analyze power dynamics among stakeholders using network analysis concepts.

        Calculates:
        - Power concentration (Herfindahl index)
        - Power distribution metrics
        - Influence network characteristics
        - Equity indicators

        References:
        - Freeman, L. C. (1979). Centrality in Social Networks
        - Bonacich, P. (1987). Power and Centrality: A Family of Measures
        """
        if not stakeholders:
            return {
                "total_influence": 0.0,
                "total_power": 0.0,
                "power_concentration": 0.0,
                "power_disparity": 0.0,
                "power_balance_assessment": "no_data",
                "herfindahl_index": 0.0,
                "gini_coefficient": 0.0,
            }

        total_influence = sum(s.influence_level for s in stakeholders)
        total_power = sum(s.decision_power for s in stakeholders)

        # Calculate Herfindahl index for power concentration
        # H = sum(p_i^2) where p_i is the proportion of power held by stakeholder i
        power_values = [s.decision_power for s in stakeholders]
        if total_power > 0:
            power_proportions = [p / total_power for p in power_values]
            herfindahl_index = sum(p**2 for p in power_proportions)
        else:
            herfindahl_index = 0.0

        # Calculate Gini coefficient for inequality
        sorted_powers = sorted(power_values)
        n = len(sorted_powers)
        if n > 1 and total_power > 0:
            # Gini = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n+1)/n
            gini_numerator = sum(
                (i + 1) * power for i, power in enumerate(sorted_powers)
            )
            gini_coefficient = (2 * gini_numerator) / (n * total_power) - (n + 1) / n
            gini_coefficient = max(0.0, min(1.0, gini_coefficient))
        else:
            gini_coefficient = 0.0

        # Power concentration metrics
        max_power = max(power_values) if power_values else 0
        min_power = min(power_values) if power_values else 0
        power_disparity = max_power - min_power

        # Power balance assessment
        if herfindahl_index > 0.6:
            balance_assessment = "highly_concentrated"
        elif herfindahl_index > 0.4:
            balance_assessment = "moderately_concentrated"
        elif herfindahl_index > 0.25:
            balance_assessment = "relatively_balanced"
        else:
            balance_assessment = "well_distributed"

        # Calculate power concentration ratio
        power_concentration_ratio = max_power / total_power if total_power > 0 else 0

        return {
            "total_influence": total_influence,
            "total_power": total_power,
            "power_concentration": power_concentration_ratio,
            "power_disparity": power_disparity,
            "power_balance_assessment": balance_assessment,
            "herfindahl_index": herfindahl_index,
            "gini_coefficient": gini_coefficient,
            "num_stakeholders": len(stakeholders),
            "power_statistics": {
                "mean": total_power / len(stakeholders) if stakeholders else 0,
                "median": sorted_powers[n // 2] if sorted_powers else 0,
                "std_dev": self._calculate_std_dev(power_values) if power_values else 0,
            },
        }

    def _calculate_std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return float(variance**0.5)

    def _identify_conflicts(
        self, stakeholders: list[Stakeholder]
    ) -> list[dict[str, Any]]:
        """Identify potential conflicts between stakeholders."""
        conflicts = []

        for i, s1 in enumerate(stakeholders):
            for s2 in stakeholders[i + 1 :]:
                # Find conflicting interests
                conflicting_interests = set(s1.interests) & set(s2.interests)
                if not conflicting_interests:  # Different interests = conflict
                    conflicts.append(
                        {
                            "stakeholder_1": s1.name,
                            "stakeholder_2": s2.name,
                            "conflict_type": "interest_divergence",
                            "severity": "medium",
                        }
                    )

        return conflicts

    def _assess_collaboration_potential(self, stakeholders: list[Stakeholder]) -> float:
        """Assess potential for collaboration."""
        if not stakeholders:
            return 0.0

        # Calculate based on overlapping interests and power distribution
        total_interests_count = len(
            set().union(*[set(s.interests) for s in stakeholders])
        )
        overlapping_count = sum(
            len(set(stakeholders[i].interests) & set(stakeholders[j].interests))
            for i in range(len(stakeholders))
            for j in range(i + 1, len(stakeholders))
        )

        collaboration_score = (
            0.5
            + (
                overlapping_count / total_interests_count
                if total_interests_count > 0
                else 0
            )
            * 0.3
        )
        return min(collaboration_score, 0.95)

    def establish_governance_platform(
        self,
        participants: list[Any],
        governance_mechanisms: list[str],
        decision_domains: list[str],
        conflict_resolution_capacity: bool,
    ) -> GovernancePlatform:
        """
        Establish multi-stakeholder governance platform.

        Parameters:
        -----------
        participants : List[Any]
            Stakeholder participants
        governance_mechanisms : List[str]
            Governance mechanisms to use
        decision_domains : List[str]
            Decision domains for platform
        conflict_resolution_capacity : bool
            Whether platform has conflict resolution capacity

        Returns:
        --------
        GovernancePlatform
            Established governance platform
        """
        platform_id = f"platform_{len(self.governance_platforms)}"

        # Convert participants to Stakeholder objects if needed
        stakeholders = []
        for i, participant in enumerate(participants):
            if isinstance(participant, Stakeholder):
                stakeholders.append(participant)
            else:
                stakeholders.append(
                    Stakeholder(
                        stakeholder_id=f"stakeholder_{i}",
                        name=participant.get("name", f"Stakeholder {i}"),
                        category=participant.get("category", "other"),
                        interests=participant.get("interests", []),
                        influence_level=participant.get("influence", 0.5),
                        dependence_on_resource=participant.get("dependence", 0.5),
                        decision_power=participant.get("power", 0.5),
                    )
                )

        platform = GovernancePlatform(
            platform_id=platform_id,
            stakeholders=stakeholders,
            governance_mechanisms=governance_mechanisms,
            decision_domains=decision_domains,
            conflict_resolution_capacity=conflict_resolution_capacity,
            participation_level=self.engagement_level,
        )

        self.governance_platforms[platform_id] = platform
        logger.info(f"Governance platform established: {platform_id}")

        return platform

    def design_participatory_process(
        self,
        stakeholder_groups: list[Any],
        decision_type: str,
        equity_principles: list[str],
        transparency_requirements: bool,
    ) -> dict[str, Any]:
        """
        Design inclusive decision-making process.

        Parameters:
        -----------
        stakeholder_groups : List[Any]
            Stakeholder groups to include
        decision_type : str
            Type of decision (collective_choice, resource_allocation, etc.)
        equity_principles : List[str]
            Equity principles to apply
        transparency_requirements : bool
            Whether full transparency is required

        Returns:
        --------
        Dict[str, Any]
            Designed participatory process
        """
        return {
            "process_design": {
                "stakeholder_groups": len(stakeholder_groups),
                "decision_type": decision_type,
                "phases": [
                    "information_sharing",
                    "consultation",
                    "deliberation",
                    "decision",
                    "implementation",
                ],
                "participation_requirements": self._design_participation_requirements(
                    equity_principles
                ),
                "decision_rule": "consensus_oriented",
                "transparency": transparency_requirements,
            },
            "equity_mechanisms": {
                "principles": equity_principles,
                "voice_mechanisms": self._design_voice_mechanisms(),
                "representation_mechanisms": self._design_representation_mechanisms(),
                "influence_mechanisms": self._design_influence_mechanisms(),
                "distribution_mechanisms": self._design_distribution_mechanisms(),
            },
            "timeline": {
                "information_phase": "2 weeks",
                "consultation_phase": "4 weeks",
                "deliberation_phase": "6 weeks",
                "decision_phase": "2 weeks",
                "implementation": "ongoing",
            },
        }

    def _design_participation_requirements(
        self, equity_principles: list[str]
    ) -> dict[str, str]:
        """Design participation requirements."""
        return {
            "minimum_representation": "all_stakeholder_categories",
            "frequency": "monthly_meetings",
            "accessibility": "multiple_formats",
            "capacity_building": "as_needed",
            "equity_focus": "yes" if self.equity_focus else "no",
        }

    def _design_voice_mechanisms(self) -> list[str]:
        """Design mechanisms for stakeholder voice."""
        return [
            "direct_participation",
            "representative_bodies",
            "advisory_committees",
            "public_forums",
        ]

    def _design_representation_mechanisms(self) -> list[str]:
        """Design representation mechanisms."""
        return [
            "proportional_representation",
            "categorical_representation",
            "rotational_roles",
        ]

    def _design_influence_mechanisms(self) -> list[str]:
        """Design mechanisms for stakeholder influence."""
        return [
            "consensus_decision_making",
            "voting_with_thresholds",
            "deliberative_processes",
        ]

    def _design_distribution_mechanisms(self) -> list[str]:
        """Design benefit and cost distribution mechanisms."""
        return [
            "needs_based_distribution",
            "contribution_based_distribution",
            "equal_distribution",
        ]
