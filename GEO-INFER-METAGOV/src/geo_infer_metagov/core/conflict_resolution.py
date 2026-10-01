"""Conflict resolution system for governance conflicts."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ConflictResolutionMethod(Enum):
    """Conflict resolution methods."""

    NEGOTIATION = "negotiation"
    MEDIATION = "mediation"
    ARBITRATION = "arbitration"
    CONSENSUS_BUILDING = "consensus_building"
    VOTING = "voting"
    ESCALATION = "escalation"


@dataclass
class ConflictResolution:
    """Result of conflict resolution attempt."""

    conflict_id: str
    resolution_method: ConflictResolutionMethod
    resolved: bool
    resolution_agreement: dict[str, Any] | None = None
    stakeholder_acceptance: dict[str, float] = field(default_factory=dict)
    resolution_quality: float = 0.5
    time_taken: float | None = None  # in days
    costs: float = 0.0


class ConflictResolver:
    """
    Resolve governance conflicts using various resolution methods.

    Implements:
    - Nash bargaining solution
    - Alternating offers negotiation
    - Mediation procedures
    - Arbitration mechanisms
    - Consensus-building algorithms

    References:
    - Nash, J. F. (1950). The Bargaining Problem
    - Rubinstein, A. (1982). Perfect Equilibrium in a Bargaining Model
    - Raiffa, H. (1982). The Art and Science of Negotiation
    """

    def __init__(self) -> None:
        """Initialize conflict resolver."""
        self.resolution_history: list[ConflictResolution] = []

    def resolve_conflict(
        self,
        conflict: dict[str, Any],
        stakeholders: list[dict[str, Any]],
        method: ConflictResolutionMethod | None = None,
    ) -> ConflictResolution:
        """
        Resolve a conflict using specified or appropriate method.

        Parameters:
        -----------
        conflict : Dict[str, Any]
            Conflict description and characteristics
        stakeholders : List[Dict[str, Any]]
            Stakeholders involved in conflict
        method : Optional[ConflictResolutionMethod]
            Preferred resolution method (auto-selected if None)

        Returns:
        --------
        ConflictResolution
            Resolution outcome
        """
        conflict_id = conflict.get("id", f"conflict_{len(self.resolution_history)}")

        # Select appropriate method if not specified
        if method is None:
            method = self._select_resolution_method(conflict, stakeholders)

        # Resolve using selected method
        if method == ConflictResolutionMethod.NEGOTIATION:
            resolution = self._negotiate(conflict, stakeholders)
        elif method == ConflictResolutionMethod.MEDIATION:
            resolution = self._mediate(conflict, stakeholders)
        elif method == ConflictResolutionMethod.ARBITRATION:
            resolution = self._arbitrate(conflict, stakeholders)
        elif method == ConflictResolutionMethod.CONSENSUS_BUILDING:
            resolution = self._build_consensus(conflict, stakeholders)
        else:
            resolution = self._negotiate(conflict, stakeholders)  # Default

        resolution.conflict_id = conflict_id
        resolution.resolution_method = method

        self.resolution_history.append(resolution)
        logger.info(
            f"Conflict {conflict_id} resolved using {method.value}: {resolution.resolved}"
        )

        return resolution

    def _select_resolution_method(
        self, conflict: dict[str, Any], stakeholders: list[dict[str, Any]]
    ) -> ConflictResolutionMethod:
        """Select appropriate resolution method based on conflict characteristics."""
        conflict_type = conflict.get("type", "unknown")
        severity = conflict.get("severity", "medium")
        num_stakeholders = len(stakeholders)

        # High severity conflicts may need arbitration
        if severity == "high" and num_stakeholders > 2:
            return ConflictResolutionMethod.ARBITRATION

        # Resource conflicts often benefit from negotiation
        if "resource" in conflict_type.lower():
            return ConflictResolutionMethod.NEGOTIATION

        # Interest conflicts may need mediation
        if "interest" in conflict_type.lower():
            return ConflictResolutionMethod.MEDIATION

        # Large groups benefit from consensus building
        if num_stakeholders > 4:
            return ConflictResolutionMethod.CONSENSUS_BUILDING

        # Default to negotiation
        return ConflictResolutionMethod.NEGOTIATION

    def _negotiate(
        self, conflict: dict[str, Any], stakeholders: list[dict[str, Any]]
    ) -> ConflictResolution:
        """
        Resolve conflict through negotiation (Nash bargaining approach).

        Implements Nash bargaining solution where parties maximize
        the product of their utility gains.
        """
        if len(stakeholders) < 2:
            return ConflictResolution(
                conflict_id="",
                resolution_method=ConflictResolutionMethod.NEGOTIATION,
                resolved=False,
                resolution_quality=0.0,
            )

        # Extract stakeholder utilities and BATNAs (Best Alternative To Negotiated Agreement)
        stakeholder_data = []
        for i, stakeholder in enumerate(stakeholders):
            power = stakeholder.get("decision_power", 0.5)
            interest = stakeholder.get("interest_level", 0.5)
            batna = stakeholder.get("batna", 0.3)  # Default BATNA

            stakeholder_data.append(
                {
                    "id": stakeholder.get("id", f"s_{i}"),
                    "power": power,
                    "interest": interest,
                    "batna": batna,
                    "utility": power * interest,
                }
            )

        # Calculate Nash bargaining solution
        # Maximize product of (utility_i - batna_i) for all i
        utility_gains = [
            max(0, data["utility"] - data["batna"]) for data in stakeholder_data
        ]

        # Nash product
        nash_product = 1.0
        for gain in utility_gains:
            if gain > 0:
                nash_product *= gain

        # Check if solution exists (all parties can gain)
        all_can_gain = all(gain > 0 for gain in utility_gains)

        if all_can_gain and nash_product > 0:
            # Calculate agreement terms (simplified)
            total_power = sum(data["power"] for data in stakeholder_data)
            agreement = {}
            acceptance = {}

            for data in stakeholder_data:
                # Allocation proportional to power and interest
                allocation = (
                    (data["power"] * data["interest"]) / total_power
                    if total_power > 0
                    else 1.0 / len(stakeholder_data)
                )
                agreement[data["id"]] = {
                    "allocation": allocation,
                    "utility_gain": utility_gains[stakeholder_data.index(data)],
                }
                # Acceptance based on utility gain relative to BATNA
                acceptance[data["id"]] = min(
                    1.0,
                    utility_gains[stakeholder_data.index(data)]
                    / max(0.1, data["batna"]),
                )

            # Resolution quality based on Nash product and acceptance
            avg_acceptance = (
                sum(acceptance.values()) / len(acceptance) if acceptance else 0.0
            )
            resolution_quality = min(
                1.0, math.sqrt(nash_product) * 0.5 + avg_acceptance * 0.5
            )

            return ConflictResolution(
                conflict_id="",
                resolution_method=ConflictResolutionMethod.NEGOTIATION,
                resolved=True,
                resolution_agreement=agreement,
                stakeholder_acceptance=acceptance,
                resolution_quality=resolution_quality,
                time_taken=7.0,  # Estimated days
                costs=1000.0 * len(stakeholders),
            )
        else:
            # No mutually beneficial solution found
            return ConflictResolution(
                conflict_id="",
                resolution_method=ConflictResolutionMethod.NEGOTIATION,
                resolved=False,
                resolution_quality=0.3,
                time_taken=14.0,
                costs=2000.0 * len(stakeholders),
            )

    def _mediate(
        self, conflict: dict[str, Any], stakeholders: list[dict[str, Any]]
    ) -> ConflictResolution:
        """Resolve conflict through mediation."""
        # Mediation involves a neutral third party facilitating discussion
        # Simplified implementation

        # Calculate potential for agreement
        interests = [s.get("interests", []) for s in stakeholders]
        common_interests = set(interests[0]) if interests else set()
        for interest_set in interests[1:]:
            common_interests &= set(interest_set)

        # Mediation success depends on common ground
        common_ground_ratio = len(common_interests) / max(
            1, sum(len(i) for i in interests)
        )

        if common_ground_ratio > 0.2:  # Some common ground exists
            agreement = {
                "common_interests": list(common_interests),
                "mediation_outcome": "partial_agreement",
            }
            acceptance = {
                s.get("id", f"s_{i}"): 0.6 + common_ground_ratio * 0.3
                for i, s in enumerate(stakeholders)
            }

            return ConflictResolution(
                conflict_id="",
                resolution_method=ConflictResolutionMethod.MEDIATION,
                resolved=common_ground_ratio > 0.3,
                resolution_agreement=agreement,
                stakeholder_acceptance=acceptance,
                resolution_quality=0.5 + common_ground_ratio * 0.4,
                time_taken=14.0,
                costs=1500.0 * len(stakeholders),
            )
        else:
            return ConflictResolution(
                conflict_id="",
                resolution_method=ConflictResolutionMethod.MEDIATION,
                resolved=False,
                resolution_quality=0.3,
                time_taken=21.0,
                costs=2000.0 * len(stakeholders),
            )

    def _arbitrate(
        self, conflict: dict[str, Any], stakeholders: list[dict[str, Any]]
    ) -> ConflictResolution:
        """Resolve conflict through arbitration."""
        # Arbitration involves a third party making a binding decision
        # Decision based on rules, evidence, and fairness

        # Calculate fairness of potential outcomes
        powers = [s.get("decision_power", 0.5) for s in stakeholders]

        # Arbitrator decision (simplified - in practice would use actual rules/evidence)
        # Tends to favor more balanced outcomes
        agreement = {}
        acceptance = {}

        for i, stakeholder in enumerate(stakeholders):
            # Allocation based on fairness and power balance
            power_ratio = (
                powers[i] / sum(powers) if sum(powers) > 0 else 1.0 / len(stakeholders)
            )
            # Adjust toward average (arbitrator seeks balance)
            balanced_allocation = 0.7 * power_ratio + 0.3 * (1.0 / len(stakeholders))

            agreement[stakeholder.get("id", f"s_{i}")] = {
                "allocation": balanced_allocation,
                "arbitration_decision": "binding",
            }

            # Acceptance depends on how close to their power level
            acceptance[stakeholder.get("id", f"s_{i}")] = 1.0 - abs(
                balanced_allocation - power_ratio
            )

        avg_acceptance = (
            sum(acceptance.values()) / len(acceptance) if acceptance else 0.5
        )

        return ConflictResolution(
            conflict_id="",
            resolution_method=ConflictResolutionMethod.ARBITRATION,
            resolved=True,  # Arbitration is binding
            resolution_agreement=agreement,
            stakeholder_acceptance=acceptance,
            resolution_quality=0.6 + avg_acceptance * 0.3,
            time_taken=30.0,  # Longer process
            costs=3000.0 * len(stakeholders),
        )

    def _build_consensus(
        self, conflict: dict[str, Any], stakeholders: list[dict[str, Any]]
    ) -> ConflictResolution:
        """Build consensus among stakeholders."""
        # Consensus building through iterative discussion and compromise

        # Initial positions
        initial_positions = {
            s.get("id", f"s_{i}"): s.get("position", 0.5)
            for i, s in enumerate(stakeholders)
        }

        # Iterative convergence toward consensus
        positions = initial_positions.copy()
        iterations = 0
        max_iterations = 10
        convergence_threshold = 0.1

        while iterations < max_iterations:
            # Calculate average position
            avg_position = (
                sum(positions.values()) / len(positions) if positions else 0.5
            )

            # Check convergence
            max_deviation = (
                max(abs(p - avg_position) for p in positions.values())
                if positions
                else 1.0
            )
            if max_deviation < convergence_threshold:
                break

            # Move positions toward average (compromise)
            for key in positions:
                positions[key] = 0.7 * positions[key] + 0.3 * avg_position

            iterations += 1

        # Consensus reached if positions converged
        consensus_reached = max_deviation < convergence_threshold

        agreement = {
            "consensus_position": avg_position,
            "iterations": iterations,
            "convergence": consensus_reached,
        }

        acceptance = {}
        for key, final_pos in positions.items():
            initial_pos = initial_positions.get(key, 0.5)
            # Acceptance based on how much they had to move
            movement = abs(final_pos - initial_pos)
            acceptance[key] = 1.0 - min(
                1.0, movement * 2
            )  # Less movement = higher acceptance

        return ConflictResolution(
            conflict_id="",
            resolution_method=ConflictResolutionMethod.CONSENSUS_BUILDING,
            resolved=consensus_reached,
            resolution_agreement=agreement,
            stakeholder_acceptance=acceptance,
            resolution_quality=0.7 if consensus_reached else 0.4,
            time_taken=21.0 + iterations * 2,
            costs=1200.0 * len(stakeholders) * (iterations + 1),
        )
