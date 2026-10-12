"""Performance evaluation system for governance structures."""

from dataclasses import dataclass, field
from typing import Any
from geo_infer_metagov.utils.helpers import entity_field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceDimension(Enum):
    """Performance dimensions for governance evaluation."""

    EFFECTIVENESS = "effectiveness"
    EFFICIENCY = "efficiency"
    EQUITY = "equity"
    SUSTAINABILITY = "sustainability"
    PARTICIPATION = "participation"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    LEGITIMACY = "legitimacy"
    ADAPTABILITY = "adaptability"
    RESILIENCE = "resilience"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for governance."""

    evaluation_id: str
    governance_structure_id: str
    evaluation_date: datetime
    dimension_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    performance_rating: str = "fair"
    trends: dict[str, str] = field(default_factory=dict)
    benchmarks: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for governance systems.

    Evaluates governance performance across multiple dimensions:
    - Effectiveness: Achievement of governance objectives
    - Efficiency: Resource utilization and process efficiency
    - Equity: Fair distribution of costs and benefits
    - Sustainability: Long-term viability
    - Participation: Stakeholder engagement
    - Transparency: Information disclosure
    - Accountability: Responsibility and enforcement
    - Legitimacy: Acceptance and trust
    - Adaptability: Ability to respond to change
    - Resilience: Capacity to withstand shocks

    References:
    - UNDP (2009). Governance Indicators: A Users' Guide
    - Kaufmann, D., et al. (2010). The Worldwide Governance Indicators
    """

    def __init__(self) -> None:
        """Initialize performance evaluator."""
        self.evaluations: dict[str, PerformanceMetrics] = {}
        self.benchmarks: dict[str, float] = {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.2,
        }

    def evaluate_governance_performance(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None = None,
    ) -> PerformanceMetrics:
        """
        Evaluate comprehensive governance performance.

        Parameters:
        -----------
        governance_structure : Dict[str, Any]
            Governance structure to evaluate
        performance_data : Optional[Dict[str, Any]]
            Performance data (if available)

        Returns:
        --------
        PerformanceMetrics
            Comprehensive performance evaluation
        """
        evaluation_id = f"eval_{len(self.evaluations)}"
        structure_id = governance_structure.get("governance_id", "unknown")

        # Evaluate each performance dimension
        dimension_scores = {}

        dimension_scores["effectiveness"] = self._evaluate_effectiveness(
            governance_structure, performance_data
        )
        dimension_scores["efficiency"] = self._evaluate_efficiency(
            governance_structure, performance_data
        )
        dimension_scores["equity"] = self._evaluate_equity(
            governance_structure, performance_data
        )
        dimension_scores["sustainability"] = self._evaluate_sustainability(
            governance_structure, performance_data
        )
        dimension_scores["participation"] = self._evaluate_participation(
            governance_structure, performance_data
        )
        dimension_scores["transparency"] = self._evaluate_transparency(
            governance_structure, performance_data
        )
        dimension_scores["accountability"] = self._evaluate_accountability(
            governance_structure, performance_data
        )
        dimension_scores["legitimacy"] = self._evaluate_legitimacy(
            governance_structure, performance_data
        )
        dimension_scores["adaptability"] = self._evaluate_adaptability(
            governance_structure, performance_data
        )
        dimension_scores["resilience"] = self._evaluate_resilience(
            governance_structure, performance_data
        )

        # Calculate overall score
        weights = {
            "effectiveness": 0.15,
            "efficiency": 0.12,
            "equity": 0.15,
            "sustainability": 0.12,
            "participation": 0.10,
            "transparency": 0.08,
            "accountability": 0.10,
            "legitimacy": 0.08,
            "adaptability": 0.05,
            "resilience": 0.05,
        }

        overall_score = sum(
            weights.get(dim, 0.1) * score for dim, score in dimension_scores.items()
        )

        # Determine performance rating
        if overall_score >= self.benchmarks["excellent"]:
            rating = "excellent"
        elif overall_score >= self.benchmarks["good"]:
            rating = "good"
        elif overall_score >= self.benchmarks["fair"]:
            rating = "fair"
        else:
            rating = "poor"

        # Identify trends
        trends = self._identify_trends(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            dimension_scores, overall_score
        )

        metrics = PerformanceMetrics(
            evaluation_id=evaluation_id,
            governance_structure_id=structure_id,
            evaluation_date=datetime.now(),
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            performance_rating=rating,
            trends=trends,
            benchmarks=self.benchmarks,
            recommendations=recommendations,
        )

        self.evaluations[evaluation_id] = metrics
        logger.info(
            f"Performance evaluation completed: {evaluation_id} (score: {overall_score:.2f})"
        )

        return metrics

    def _evaluate_effectiveness(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance effectiveness."""
        # Base score from structure quality
        entities = governance_structure.get("entities", [])
        num_entities = len(entities)
        num_domains = len(governance_structure.get("decision_domains", []))

        # Effectiveness indicators
        structure_score = min(1.0, (num_entities * 0.1 + num_domains * 0.1))

        # If performance data available, use actual outcomes
        if performance_data:
            outcome_achievement = performance_data.get("outcome_achievement", 0.5)
            objective_fulfillment = performance_data.get("objective_fulfillment", 0.5)
            return float(
                (outcome_achievement * 0.6 + objective_fulfillment * 0.4) * 0.7
                + structure_score * 0.3
            )

        return structure_score

    def _evaluate_efficiency(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance efficiency."""
        entities = governance_structure.get("entities", [])
        if not entities:
            return 0.5

        # Calculate resource utilization
        total_budget = sum(
            entity_field(entity_field(e, "resources", {}), "budget", 0)
            for e in entities
        )
        avg_capacity = sum(entity_field(e, "capacity", 0.5) for e in entities) / len(
            entities
        )

        # Efficiency = capacity utilization / resource investment
        efficiency = avg_capacity * min(1.0, 1000000 / max(1, total_budget))

        if performance_data:
            process_efficiency = performance_data.get("process_efficiency", 0.5)
            resource_efficiency = performance_data.get("resource_efficiency", 0.5)
            return float(
                (process_efficiency * 0.5 + resource_efficiency * 0.5) * 0.7
                + efficiency * 0.3
            )

        return float(efficiency)

    def _evaluate_equity(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance equity."""
        stakeholder_groups = governance_structure.get("stakeholder_groups", [])
        entities = governance_structure.get("entities", [])

        # Check stakeholder representation
        if stakeholder_groups and entities:
            # Count stakeholders per entity
            stakeholders_per_entity = sum(
                len(entity_field(e, "stakeholders", [])) for e in entities
            ) / len(entities)
            representation_score = min(
                1.0, stakeholders_per_entity / len(stakeholder_groups)
            )
        else:
            representation_score = 0.5

        if performance_data:
            benefit_distribution = performance_data.get(
                "benefit_distribution_equity", 0.5
            )
            access_equity = performance_data.get("access_equity", 0.5)
            return float(
                (benefit_distribution * 0.5 + access_equity * 0.5) * 0.7
                + representation_score * 0.3
            )

        return representation_score

    def _evaluate_sustainability(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance sustainability."""
        # Check for adaptive mechanisms
        has_adaptive_mechanisms = any(
            "adaptive" in str(mech).lower() or "learning" in str(mech).lower()
            for mech in governance_structure.get("coordination_mechanisms", [])
        )

        sustainability_score = 0.5
        if has_adaptive_mechanisms:
            sustainability_score = 0.7

        if performance_data:
            resource_sustainability = performance_data.get(
                "resource_sustainability", 0.5
            )
            long_term_viability = performance_data.get("long_term_viability", 0.5)
            return float(
                (resource_sustainability * 0.5 + long_term_viability * 0.5) * 0.7
                + sustainability_score * 0.3
            )

        return sustainability_score

    def _evaluate_participation(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate stakeholder participation."""
        stakeholder_groups = governance_structure.get("stakeholder_groups", [])
        entities = governance_structure.get("entities", [])

        # Participation score based on stakeholder involvement
        if stakeholder_groups and entities:
            total_stakeholder_connections = sum(
                len(entity_field(e, "stakeholders", [])) for e in entities
            )
            participation_score = min(
                1.0,
                total_stakeholder_connections
                / (len(stakeholder_groups) * len(entities)),
            )
        else:
            participation_score = 0.5

        if performance_data:
            engagement_level = performance_data.get("stakeholder_engagement", 0.5)
            participation_rate = performance_data.get("participation_rate", 0.5)
            return float(
                (engagement_level * 0.5 + participation_rate * 0.5) * 0.7
                + participation_score * 0.3
            )

        return participation_score

    def _evaluate_transparency(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance transparency."""
        # Check for information flows
        information_flows = governance_structure.get("information_flows", {})
        transparency_score = min(1.0, len(information_flows) * 0.2)

        if performance_data:
            disclosure_rate = performance_data.get("disclosure_rate", 0.5)
            information_accessibility = performance_data.get(
                "information_accessibility", 0.5
            )
            return float(
                (disclosure_rate * 0.5 + information_accessibility * 0.5) * 0.7
                + transparency_score * 0.3
            )

        return transparency_score

    def _evaluate_accountability(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance accountability."""
        # Check for reporting relationships
        reporting_relationships = governance_structure.get(
            "reporting_relationships", {}
        )
        accountability_score = min(1.0, len(reporting_relationships) * 0.3)

        if performance_data:
            audit_frequency = performance_data.get("audit_frequency", 0.5)
            compliance_rate_p = performance_data.get("compliance_rate", 0.5)
            return float(
                (audit_frequency * 0.5 + compliance_rate_p * 0.5) * 0.7
                + accountability_score * 0.3
            )

        return accountability_score

    def _evaluate_legitimacy(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance legitimacy."""
        # Legitimacy based on stakeholder acceptance
        stakeholder_groups = governance_structure.get("stakeholder_groups", [])
        legitimacy_score = 0.6 if stakeholder_groups else 0.4

        if performance_data:
            acceptance_rate = performance_data.get("stakeholder_acceptance", 0.5)
            trust_level = performance_data.get("trust_level", 0.5)
            return float(
                (acceptance_rate * 0.5 + trust_level * 0.5) * 0.7
                + legitimacy_score * 0.3
            )

        return float(legitimacy_score)

    def _evaluate_adaptability(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance adaptability."""
        # Check for adaptive mechanisms
        has_learning_mechanisms = any(
            "adaptive" in str(m).lower() or "learning" in str(m).lower()
            for m in governance_structure.get("coordination_mechanisms", [])
        )

        adaptability_score = 0.7 if has_learning_mechanisms else 0.4

        if performance_data:
            adaptation_capacity = performance_data.get("adaptation_capacity", 0.5)
            learning_rate = performance_data.get("learning_rate", 0.5)
            return float(
                (adaptation_capacity * 0.5 + learning_rate * 0.5) * 0.7
                + adaptability_score * 0.3
            )

        return float(adaptability_score)

    def _evaluate_resilience(
        self,
        governance_structure: dict[str, Any],
        performance_data: dict[str, Any] | None,
    ) -> float:
        """Evaluate governance resilience."""
        entities = governance_structure.get("entities", [])
        if not entities:
            return 0.5

        # Resilience based on redundancy and capacity
        avg_capacity = sum(entity_field(e, "capacity", 0.5) for e in entities) / len(
            entities
        )
        num_entities = len(entities)
        redundancy_score = min(
            1.0, num_entities / 5.0
        )  # More entities = more redundancy

        resilience_score = avg_capacity * 0.6 + redundancy_score * 0.4

        if performance_data:
            shock_resistance = performance_data.get("shock_resistance", 0.5)
            recovery_capacity = performance_data.get("recovery_capacity", 0.5)
            return float(
                (shock_resistance * 0.5 + recovery_capacity * 0.5) * 0.7
                + resilience_score * 0.3
            )

        return float(resilience_score)

    def _identify_trends(self, dimension_scores: dict[str, float]) -> dict[str, str]:
        """Identify performance trends."""
        trends = {}

        for dimension, score in dimension_scores.items():
            if score >= 0.75:
                trends[dimension] = "improving"
            elif score >= 0.6:
                trends[dimension] = "stable"
            elif score >= 0.4:
                trends[dimension] = "declining"
            else:
                trends[dimension] = "critical"

        return trends

    def _generate_performance_recommendations(
        self, dimension_scores: dict[str, float], overall_score: float
    ) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Identify low-performing dimensions
        low_dimensions = [dim for dim, score in dimension_scores.items() if score < 0.6]

        for dimension in low_dimensions:
            if dimension == "effectiveness":
                recommendations.append(
                    "Improve decision-making processes and outcome tracking"
                )
            elif dimension == "efficiency":
                recommendations.append(
                    "Optimize resource allocation and streamline processes"
                )
            elif dimension == "equity":
                recommendations.append(
                    "Enhance stakeholder representation and benefit distribution"
                )
            elif dimension == "sustainability":
                recommendations.append(
                    "Strengthen long-term planning and resource management"
                )
            elif dimension == "participation":
                recommendations.append(
                    "Increase stakeholder engagement and participation mechanisms"
                )
            elif dimension == "transparency":
                recommendations.append(
                    "Improve information disclosure and accessibility"
                )
            elif dimension == "accountability":
                recommendations.append(
                    "Strengthen audit mechanisms and compliance checking"
                )
            elif dimension == "legitimacy":
                recommendations.append("Build stakeholder trust and acceptance")
            elif dimension == "adaptability":
                recommendations.append(
                    "Implement adaptive management and learning mechanisms"
                )
            elif dimension == "resilience":
                recommendations.append(
                    "Build redundancy and capacity for shock resistance"
                )

        if overall_score < 0.6:
            recommendations.append("Consider comprehensive governance reform")

        return recommendations

    def benchmark_against_standards(
        self,
        performance_metrics: PerformanceMetrics,
        standards: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Benchmark performance against governance standards.

        Parameters:
        -----------
        performance_metrics : PerformanceMetrics
            Performance metrics to benchmark
        standards : Optional[Dict[str, float]]
            Custom standards (uses defaults if None)

        Returns:
        --------
        Dict[str, Any]
            Benchmarking results
        """
        if standards is None:
            standards = {"excellent": 0.8, "good": 0.6, "fair": 0.4, "poor": 0.2}

        overall_score = performance_metrics.overall_score

        # Determine benchmark level
        if overall_score >= standards.get("excellent", 0.8):
            benchmark_level = "excellent"
        elif overall_score >= standards.get("good", 0.6):
            benchmark_level = "good"
        elif overall_score >= standards.get("fair", 0.4):
            benchmark_level = "fair"
        else:
            benchmark_level = "poor"

        # Calculate gap to next level
        if benchmark_level == "poor":
            gap_to_next = standards.get("fair", 0.4) - overall_score
            next_level = "fair"
        elif benchmark_level == "fair":
            gap_to_next = standards.get("good", 0.6) - overall_score
            next_level = "good"
        elif benchmark_level == "good":
            gap_to_next = standards.get("excellent", 0.8) - overall_score
            next_level = "excellent"
        else:
            gap_to_next = 0.0
            next_level = "excellent"

        return {
            "benchmark_level": benchmark_level,
            "overall_score": overall_score,
            "gap_to_next_level": gap_to_next,
            "next_level": next_level,
            "standards_used": standards,
            "dimension_benchmarks": {
                dim: "excellent"
                if score >= 0.8
                else "good"
                if score >= 0.6
                else "fair"
                if score >= 0.4
                else "poor"
                for dim, score in performance_metrics.dimension_scores.items()
            },
        }

    def compare_performances(
        self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics
    ) -> dict[str, Any]:
        """
        Compare two performance evaluations.

        Parameters:
        -----------
        metrics1 : PerformanceMetrics
            First performance metrics
        metrics2 : PerformanceMetrics
            Second performance metrics

        Returns:
        --------
        Dict[str, Any]
            Comparison results
        """
        dimension_differences_out: dict[str, float] = {}
        improved_dimensions_out: list[str] = []
        declined_dimensions_out: list[str] = []
        stable_dimensions_out: list[str] = []
        comparison: dict[str, Any] = {
            "overall_difference": metrics2.overall_score - metrics1.overall_score,
            "dimension_differences": dimension_differences_out,
            "improved_dimensions": improved_dimensions_out,
            "declined_dimensions": declined_dimensions_out,
            "stable_dimensions": stable_dimensions_out,
        }

        # Compare dimensions
        all_dimensions = set(metrics1.dimension_scores.keys()) | set(
            metrics2.dimension_scores.keys()
        )

        for dimension in all_dimensions:
            score1 = metrics1.dimension_scores.get(dimension, 0.0)
            score2 = metrics2.dimension_scores.get(dimension, 0.0)
            difference = score2 - score1

            dimension_differences_out[dimension] = difference

            if difference > 0.1:
                improved_dimensions_out.append(dimension)
            elif difference < -0.1:
                declined_dimensions_out.append(dimension)
            else:
                stable_dimensions_out.append(dimension)

        return comparison
