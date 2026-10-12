"""Adaptive governance systems with learning and evolution."""

from dataclasses import dataclass
from typing import Any
import hashlib
import logging
import random

logger = logging.getLogger(__name__)


def _seeded_noise(indicator: str, span: float = 0.1) -> float:
    """Deterministic pseudo-random noise for an indicator.

    Seeds an isolated ``random.Random`` from the SHA-256 digest of the
    indicator name — stable across processes and Python versions —
    replacing the unseeded ``random`` use that made results
    non-reproducible.
    """
    seed = int(hashlib.sha256(indicator.encode("utf-8")).hexdigest(), 16) % 2**32
    return random.Random(seed).uniform(-span, span)


@dataclass
class AdaptiveManagementCycle:
    """Adaptive management cycle for governance."""

    cycle_id: str
    governance_domain: str
    decision_frequency: str
    learning_mechanisms: list[str]
    stakeholder_participation: str
    monitoring_plan: dict[str, Any]
    evaluation_schedule: dict[str, Any]


class AdaptiveGovernanceSystem:
    """Enable governance systems to learn and adapt."""

    def __init__(
        self,
        learning_approach: str = "adaptive_management",
        timeframe: str = "multi_year_cycles",
        feedback_mechanisms: str = "real_time",
    ) -> None:
        self.learning_approach = learning_approach
        self.timeframe = timeframe
        self.feedback_mechanisms = feedback_mechanisms
        self.adaptive_cycles: dict[str, AdaptiveManagementCycle] = {}

    def establish_adaptive_cycle(
        self,
        governance_domain: str,
        decision_frequency: str,
        learning_mechanisms: list[str],
        stakeholder_participation: str,
    ) -> AdaptiveManagementCycle:
        """Establish adaptive management cycle."""
        cycle_id = f"cycle_{len(self.adaptive_cycles)}"

        cycle = AdaptiveManagementCycle(
            cycle_id=cycle_id,
            governance_domain=governance_domain,
            decision_frequency=decision_frequency,
            learning_mechanisms=learning_mechanisms,
            stakeholder_participation=stakeholder_participation,
            monitoring_plan=self._design_monitoring_plan(governance_domain),
            evaluation_schedule=self._design_evaluation_schedule(decision_frequency),
        )

        self.adaptive_cycles[cycle_id] = cycle
        logger.info(f"Adaptive cycle established: {cycle_id}")
        return cycle

    def _design_monitoring_plan(self, governance_domain: str) -> dict[str, Any]:
        """Design monitoring plan."""
        return {
            "domain": governance_domain,
            "indicators": ["effectiveness", "equity", "sustainability"],
            "frequency": "monthly",
            "data_sources": ["administrative", "stakeholder_feedback", "scientific"],
        }

    def _design_evaluation_schedule(self, decision_frequency: str) -> dict[str, Any]:
        """Design evaluation schedule."""
        return {
            "frequency": decision_frequency,
            "evaluation_periods": ["quarterly", "annual", "multi_year"],
            "review_process": "participatory",
        }

    def monitor_performance(
        self,
        governance_indicators: list[str],
        data_sources: list[str],
        evaluation_periods: str,
    ) -> dict[str, Any]:
        """
        Monitor governance performance using simulated indicator data.

        Simulated basis, explicitly: no real data sources are queried.
        - Performance scores are hardcoded base scores per indicator type,
          adjusted by data-source coverage (an indicator scores higher when
          a declared data source name contains the indicator name, capped
          at two matching sources) and seeded noise derived from the
          SHA-256 digest of the indicator name. Identical inputs always
          produce identical outputs.
        - ``data_quality`` completeness, reliability, and timeliness are
          fixed illustrative constants.

        Tracks:
        - Indicator values over time
        - Trends and patterns
        - Data quality and completeness
        - Performance gaps

        Returns a dict flagged ``"simulated": True`` — suitable for
        deterministic tests and demos, not for real governance decisions.
        """

        performance_scores = {}
        performance_trends = {}
        data_quality = {}

        for indicator in governance_indicators:
            # Simulate performance score based on indicator type
            base_scores = {
                "effectiveness": 0.75,
                "efficiency": 0.70,
                "equity": 0.65,
                "sustainability": 0.72,
                "participation": 0.68,
                "transparency": 0.73,
                "accountability": 0.71,
                "legitimacy": 0.69,
            }

            # Find matching base score
            score = 0.7  # Default
            for key, value in base_scores.items():
                if key in indicator.lower():
                    score = value
                    break

            # Data-source coverage feeds scoring: an indicator covered by a
            # declared source (source name contains the indicator name)
            # scores higher, capped at two matching sources.
            source_count = len([s for s in data_sources if indicator in s.lower()])
            coverage_adjustment = 0.05 * min(source_count, 2)

            # Deterministic seeded variation (replaces unseeded randomness)
            score = max(
                0.0,
                min(1.0, score + coverage_adjustment + _seeded_noise(indicator)),
            )
            performance_scores[indicator] = score

            # Determine trend (improving, stable, declining)
            if score > 0.75:
                trend = "improving"
            elif score > 0.6:
                trend = "stable"
            else:
                trend = "declining"
            performance_trends[indicator] = trend

            # Assess data quality (simulated illustrative values)
            data_quality[indicator] = {
                "completeness": 0.85,  # Percentage of data available
                "reliability": 0.80,  # Data reliability score
                "timeliness": 0.75,  # How current the data is
                "source_count": source_count,
            }

        # Calculate overall performance
        overall_performance = (
            sum(performance_scores.values()) / len(performance_scores)
            if performance_scores
            else 0.0
        )

        # Identify performance gaps
        performance_gaps = {
            ind: 1.0 - score for ind, score in performance_scores.items() if score < 0.7
        }

        return {
            "simulated": True,
            "indicators": governance_indicators,
            "data_sources": data_sources,
            "evaluation_periods": evaluation_periods,
            "performance_scores": performance_scores,
            "overall_performance": overall_performance,
            "performance_trends": performance_trends,
            "data_quality": data_quality,
            "performance_gaps": performance_gaps,
            "monitoring_timestamp": self._get_current_timestamp(),
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for monitoring."""
        from datetime import datetime

        return datetime.now().isoformat()

    def adapt_governance(
        self,
        performance_results: dict[str, Any],
        learning_outcomes: dict[str, Any],
        scenario_changes: list[dict[str, Any]],
        adaptation_pathways: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Adapt governance based on learning and performance analysis.

        Uses learning algorithms to:
        - Identify adaptation needs
        - Select optimal adaptation pathways
        - Predict adaptation outcomes
        - Plan implementation

        References:
        - Armitage, D., et al. (2007). Adaptive co-management
        - Folke, C., et al. (2005). Adaptive governance
        """
        if not adaptation_pathways:
            return {
                "adaptations_made": 0,
                "pathways_selected": [],
                "implementation_timeline": "none",
                "stakeholder_support": 0.0,
                "adaptation_quality": 0.0,
            }

        # Analyze performance to identify adaptation priorities
        performance_gaps = performance_results.get("performance_gaps", {})

        # Calculate adaptation urgency
        overall_performance = performance_results.get("overall_performance", 0.5)
        urgency_score = (
            1.0 - overall_performance
        )  # Higher urgency for lower performance

        # Evaluate adaptation pathways
        pathway_evaluations = []
        for pathway in adaptation_pathways:
            # Calculate pathway suitability
            pathway_impact = pathway.get("expected_impact", 0.5)
            pathway_feasibility = pathway.get("feasibility", 0.5)
            pathway_cost = pathway.get("cost", 0.5)

            # Match pathway to performance gaps
            pathway_domains = set(pathway.get("target_domains", []))
            gap_domains = set(performance_gaps.keys())
            domain_match = len(pathway_domains & gap_domains) / max(1, len(gap_domains))

            # Calculate pathway score
            pathway_score = (
                pathway_impact * 0.4
                + pathway_feasibility * 0.3
                + domain_match * 0.2
                + (1.0 - pathway_cost) * 0.1
            )

            pathway_evaluations.append(
                {
                    "pathway": pathway,
                    "score": pathway_score,
                    "domain_match": domain_match,
                }
            )

        # Select top pathways
        pathway_evaluations.sort(key=lambda x: x["score"], reverse=True)
        selected_pathways = pathway_evaluations[: min(3, len(pathway_evaluations))]

        # Calculate adaptation quality
        if selected_pathways:
            avg_pathway_score = sum(p["score"] for p in selected_pathways) / len(
                selected_pathways
            )
            adaptation_quality = avg_pathway_score * (1.0 + urgency_score * 0.2)
        else:
            adaptation_quality = 0.0

        # Estimate stakeholder support based on learning outcomes
        lessons_learned = learning_outcomes.get("lessons", [])
        positive_lessons = sum(
            1
            for lesson in lessons_learned
            if any(word in lesson.lower() for word in ["success", "improve", "benefit"])
        )
        stakeholder_support = (
            0.5 + (positive_lessons / max(1, len(lessons_learned))) * 0.4
        )

        # Determine implementation timeline
        num_adaptations = len(selected_pathways)
        if urgency_score > 0.7:
            timeline = "immediate"
            timeline_weeks = 4
        elif urgency_score > 0.4:
            timeline = "phased"
            timeline_weeks = 8 + num_adaptations * 2
        else:
            timeline = "gradual"
            timeline_weeks = 16 + num_adaptations * 4

        return {
            "adaptations_made": len(selected_pathways),
            "pathways_selected": [
                p["pathway"].get("name", "pathway") for p in selected_pathways
            ],
            "pathway_details": [
                {
                    "name": p["pathway"].get("name", "pathway"),
                    "score": p["score"],
                    "expected_impact": p["pathway"].get("expected_impact", 0.5),
                    "feasibility": p["pathway"].get("feasibility", 0.5),
                }
                for p in selected_pathways
            ],
            "implementation_timeline": timeline,
            "timeline_weeks": timeline_weeks,
            "stakeholder_support": min(1.0, stakeholder_support),
            "adaptation_quality": min(1.0, adaptation_quality),
            "urgency_score": urgency_score,
            "predicted_improvement": sum(
                p["pathway"].get("expected_impact", 0.5) for p in selected_pathways
            )
            / max(1, len(selected_pathways)),
        }
