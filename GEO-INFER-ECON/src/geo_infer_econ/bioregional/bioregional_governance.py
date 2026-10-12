"""
Bioregional Governance Models Module

This module provides bioregional governance modeling, community resource
management, adaptive management systems, stakeholder engagement, and
cooperative economics capabilities.
"""

from typing import Any
import numpy as np
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class BioregionalGovernanceModels:
    """
    Bioregional governance modeling using multi-criteria decision analysis.

    Evaluates governance structures across dimensions: participation,
    transparency, accountability, effectiveness, and ecological alignment.
    """

    # Governance dimensions and their default weights
    DEFAULT_DIMENSIONS = {
        "participation": 0.20,
        "transparency": 0.15,
        "accountability": 0.15,
        "effectiveness": 0.20,
        "ecological_alignment": 0.20,
        "equity": 0.10,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize bioregional governance models."""
        self.config = config or {}
        self.dimensions = self.config.get("dimensions", self.DEFAULT_DIMENSIONS)
        logger.info(
            "BioregionalGovernanceModels initialized with %d dimensions",
            len(self.dimensions),
        )

    def model_governance(self, governance_data: dict[str, Any]) -> dict[str, Any]:
        """Model bioregional governance systems.

        Args:
            governance_data: Dict with:
                - region_id: identifier for the bioregion
                - scores: dict mapping dimension -> score (0-100)
                - institutions: list of governance institutions
                - population: population count
                - area_km2: area in square kilometers

        Returns:
            Dict with governance index, dimension scores, recommendations.
        """
        region_id = governance_data.get("region_id", "unknown")
        scores = governance_data.get("scores", {})
        population = governance_data.get("population", 0)
        area_km2 = governance_data.get("area_km2", 0.0)

        logger.info("Modeling governance for region: %s", region_id)

        # Calculate weighted governance index
        weighted_scores = {}
        total_index = 0.0
        for dim, weight in self.dimensions.items():
            score = float(scores.get(dim, 50.0))  # Default to 50/100
            weighted_scores[dim] = {
                "raw_score": score,
                "weight": weight,
                "weighted_score": round(score * weight, 2),
            }
            total_index += score * weight

        # Identify weak dimensions (below 40)
        weak_dims = [d for d, s in scores.items() if float(s) < 40.0]

        # Generate recommendations
        recommendations = []
        if "participation" in weak_dims:
            recommendations.append(
                "Establish community assemblies and participatory budgeting"
            )
        if "transparency" in weak_dims:
            recommendations.append(
                "Implement open-data portals and public meeting records"
            )
        if "ecological_alignment" in weak_dims:
            recommendations.append(
                "Integrate bioregional boundaries into governance structures"
            )
        if "accountability" in weak_dims:
            recommendations.append(
                "Create independent oversight bodies with enforcement powers"
            )
        if "equity" in weak_dims:
            recommendations.append(
                "Deploy equity impact assessments for all major decisions"
            )

        result = {
            "region_id": region_id,
            "governance_index": round(total_index, 2),
            "governance_grade": self._grade(total_index),
            "dimension_scores": weighted_scores,
            "weak_dimensions": weak_dims,
            "recommendations": recommendations,
            "density_ratio": round(population / max(area_km2, 1), 2),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Region %s governance index: %.2f (%s)",
            region_id,
            total_index,
            result["governance_grade"],
        )
        return result

    @staticmethod
    def _grade(index: float) -> str:
        if index >= 80:
            return "A"
        elif index >= 65:
            return "B"
        elif index >= 50:
            return "C"
        elif index >= 35:
            return "D"
        return "F"


class CommunityResourceManagement:
    """Community resource management using common-pool resource theory (Ostrom)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.sustainability_threshold = self.config.get("sustainability_threshold", 0.7)
        logger.info("CommunityResourceManagement initialized")

    def manage_resources(self, resource_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze and recommend community resource management strategies.

        Args:
            resource_data: Dict with:
                - resources: list of {name, stock, regeneration_rate, extraction_rate}
                - community_size: number of households
                - governance_rules: list of existing rules

        Returns:
            Dict with sustainability assessment, depletion risks, recommendations.
        """
        resources = resource_data.get("resources", [])
        community_size = resource_data.get("community_size", 100)
        rules = resource_data.get("governance_rules", [])

        logger.info(
            "Analyzing %d resources for community of %d", len(resources), community_size
        )

        assessments = []
        at_risk = []
        for res in resources:
            name = res.get("name", "unnamed")
            stock = float(res.get("stock", 100))
            regen = float(res.get("regeneration_rate", 0.05))
            extract = float(res.get("extraction_rate", 0.03))

            # Sustainability ratio: regeneration / extraction
            sustainability = regen / max(extract, 1e-6)
            years_to_depletion = (
                stock / max(extract * stock - regen * stock, 1e-6)
                if extract > regen
                else float("inf")
            )

            assessment = {
                "name": name,
                "current_stock": stock,
                "regeneration_rate": regen,
                "extraction_rate": extract,
                "sustainability_ratio": round(sustainability, 3),
                "sustainable": sustainability >= self.sustainability_threshold,
                "years_to_depletion": round(years_to_depletion, 1)
                if years_to_depletion != float("inf")
                else None,
            }
            assessments.append(assessment)
            if not assessment["sustainable"]:
                at_risk.append(name)

        # Evaluate Ostrom's design principles
        ostrom_score = self._evaluate_ostrom_principles(rules)

        return {
            "resource_assessments": assessments,
            "at_risk_resources": at_risk,
            "overall_sustainability": len(at_risk) == 0,
            "ostrom_principles_score": ostrom_score,
            "per_household_allocation": round(
                sum(r.get("stock", 0) for r in resources) / max(community_size, 1), 2
            ),
            "recommendations": self._generate_crm_recommendations(
                at_risk, ostrom_score
            ),
        }

    def _evaluate_ostrom_principles(self, rules: list[str]) -> dict[str, bool]:
        """Evaluate against Ostrom's 8 design principles for commons governance."""
        principles = {
            "clearly_defined_boundaries": any(
                "boundary" in r.lower() or "membership" in r.lower() for r in rules
            ),
            "proportional_equivalence": any(
                "proportional" in r.lower() or "fair share" in r.lower() for r in rules
            ),
            "collective_choice": any(
                "voting" in r.lower()
                or "assembly" in r.lower()
                or "consensus" in r.lower()
                for r in rules
            ),
            "monitoring": any(
                "monitor" in r.lower() or "patrol" in r.lower() for r in rules
            ),
            "graduated_sanctions": any(
                "sanction" in r.lower() or "penalty" in r.lower() or "fine" in r.lower()
                for r in rules
            ),
            "conflict_resolution": any(
                "conflict" in r.lower()
                or "mediat" in r.lower()
                or "dispute" in r.lower()
                for r in rules
            ),
            "minimal_rights_recognition": any(
                "right" in r.lower() or "autonomy" in r.lower() for r in rules
            ),
            "nested_enterprises": any(
                "federat" in r.lower()
                or "nested" in r.lower()
                or "multi-level" in r.lower()
                for r in rules
            ),
        }
        return principles

    @staticmethod
    def _generate_crm_recommendations(
        at_risk: list[str], ostrom: dict[str, bool]
    ) -> list[str]:
        recs = []
        if at_risk:
            recs.append(f"Reduce extraction rates for: {', '.join(at_risk)}")
        missing = [k.replace("_", " ") for k, v in ostrom.items() if not v]
        if missing:
            recs.append(f"Strengthen governance by addressing: {', '.join(missing)}")
        return recs


class AdaptiveManagementSystems:
    """Adaptive management systems using plan-do-check-act cycles."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.cycle_length_months = self.config.get("cycle_length_months", 6)
        logger.info(
            "AdaptiveManagementSystems initialized (cycle=%d months)",
            self.cycle_length_months,
        )

    def design_adaptive_system(self, system_params: dict[str, Any]) -> dict[str, Any]:
        """Design an adaptive management system.

        Args:
            system_params: Dict with:
                - objectives: list of management objectives
                - indicators: list of {name, target, current, unit}
                - uncertainty_level: 'low', 'medium', 'high'
                - budget: annual budget

        Returns:
            Dict with management plan, monitoring schedule, decision triggers.
        """
        objectives = system_params.get("objectives", [])
        indicators = system_params.get("indicators", [])
        uncertainty = system_params.get("uncertainty_level", "medium")
        budget = float(system_params.get("budget", 0))

        logger.info(
            "Designing adaptive system: %d objectives, %d indicators, uncertainty=%s",
            len(objectives),
            len(indicators),
            uncertainty,
        )

        # Calculate gap analysis for each indicator
        indicator_gaps = []
        for ind in indicators:
            target = float(ind.get("target", 0))
            current = float(ind.get("current", 0))
            gap = target - current
            gap_pct = (gap / max(abs(target), 1e-6)) * 100
            indicator_gaps.append(
                {
                    "name": ind.get("name", "unnamed"),
                    "target": target,
                    "current": current,
                    "gap": round(gap, 3),
                    "gap_percent": round(gap_pct, 1),
                    "unit": ind.get("unit", ""),
                    "on_track": gap_pct <= 10,
                }
            )

        # Determine monitoring frequency based on uncertainty
        freq_map = {"low": 12, "medium": 6, "high": 3}
        monitoring_freq_months = freq_map.get(uncertainty, 6)

        # Budget allocation: 70% implementation, 20% monitoring, 10% evaluation
        budget_allocation = {
            "implementation": round(budget * 0.70, 2),
            "monitoring": round(budget * 0.20, 2),
            "evaluation": round(budget * 0.10, 2),
        }

        # Decision triggers
        triggers = []
        for ig in indicator_gaps:
            if not ig["on_track"]:
                triggers.append(
                    {
                        "indicator": ig["name"],
                        "trigger_condition": f"{ig['name']} gap exceeds {abs(ig['gap_percent']):.0f}%",
                        "action": "Review and adjust management strategy",
                        "priority": "high" if abs(ig["gap_percent"]) > 25 else "medium",
                    }
                )

        return {
            "management_cycle_months": self.cycle_length_months,
            "monitoring_frequency_months": monitoring_freq_months,
            "objectives": objectives,
            "indicator_gaps": indicator_gaps,
            "budget_allocation": budget_allocation,
            "decision_triggers": triggers,
            "uncertainty_level": uncertainty,
            "adaptive_capacity_score": self._compute_adaptive_capacity(
                len(indicator_gaps),
                sum(1 for ig in indicator_gaps if ig["on_track"]),
                uncertainty,
            ),
        }

    @staticmethod
    def _compute_adaptive_capacity(
        total: int, on_track: int, uncertainty: str
    ) -> float:
        if total == 0:
            return 0.0
        track_ratio = on_track / total
        uncertainty_penalty = {"low": 0, "medium": 0.1, "high": 0.25}.get(
            uncertainty, 0.1
        )
        return round(max(0, min(1, track_ratio - uncertainty_penalty)), 3)


class StakeholderEngagement:
    """Stakeholder engagement analysis using power-interest matrix."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("StakeholderEngagement initialized")

    def engage_stakeholders(self, stakeholder_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze stakeholders and recommend engagement strategies.

        Args:
            stakeholder_data: Dict with:
                - stakeholders: list of {name, power (0-100), interest (0-100), influence, sector}
                - decision_context: description of the decision
                - timeline_months: engagement timeline

        Returns:
            Dict with stakeholder map, quadrant classification, engagement plan.
        """
        stakeholders = stakeholder_data.get("stakeholders", [])
        context = stakeholder_data.get("decision_context", "")
        timeline = stakeholder_data.get("timeline_months", 6)

        logger.info(
            "Analyzing %d stakeholders for: %s", len(stakeholders), context[:50]
        )

        classified = []
        quadrant_counts = {
            "manage_closely": 0,
            "keep_satisfied": 0,
            "keep_informed": 0,
            "monitor": 0,
        }

        for sh in stakeholders:
            power = float(sh.get("power", 50))
            interest = float(sh.get("interest", 50))

            if power >= 50 and interest >= 50:
                quadrant = "manage_closely"
                strategy = "Active engagement, co-design, regular consultation"
            elif power >= 50 and interest < 50:
                quadrant = "keep_satisfied"
                strategy = "Address concerns early, periodic briefings"
            elif power < 50 and interest >= 50:
                quadrant = "keep_informed"
                strategy = "Regular updates, participation opportunities"
            else:
                quadrant = "monitor"
                strategy = "Periodic communication, public information"

            quadrant_counts[quadrant] += 1
            classified.append(
                {
                    "name": sh.get("name", "unnamed"),
                    "power": power,
                    "interest": interest,
                    "sector": sh.get("sector", "general"),
                    "quadrant": quadrant,
                    "engagement_strategy": strategy,
                }
            )

        # Calculate engagement intensity score
        engagement_intensity = (
            quadrant_counts["manage_closely"] * 4
            + quadrant_counts["keep_satisfied"] * 3
            + quadrant_counts["keep_informed"] * 2
            + quadrant_counts["monitor"] * 1
        ) / max(len(stakeholders), 1)

        return {
            "decision_context": context,
            "stakeholder_map": classified,
            "quadrant_distribution": quadrant_counts,
            "total_stakeholders": len(stakeholders),
            "engagement_intensity": round(engagement_intensity, 2),
            "recommended_timeline_months": max(timeline, int(engagement_intensity * 2)),
            "engagement_phases": self._design_phases(timeline, quadrant_counts),
        }

    @staticmethod
    def _design_phases(timeline: int, counts: dict[str, int]) -> list[dict[str, Any]]:
        phase_len = max(1, timeline // 3)
        return [
            {
                "phase": "Scoping",
                "duration_months": phase_len,
                "focus": "Identify needs, build relationships with key stakeholders",
            },
            {
                "phase": "Co-design",
                "duration_months": phase_len,
                "focus": f"Active engagement with {counts['manage_closely']} key stakeholders",
            },
            {
                "phase": "Implementation",
                "duration_months": phase_len,
                "focus": "Execute plan, maintain communication channels",
            },
        ]


class CooperativeEconomics:
    """Cooperative economics models for mutual-aid and solidarity economy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("CooperativeEconomics initialized")

    def model_cooperative(self, cooperative_data: dict[str, Any]) -> dict[str, Any]:
        """Model cooperative economic systems.

        Args:
            cooperative_data: Dict with:
                - members: list of {id, contribution, skills}
                - revenue: annual revenue
                - costs: annual costs
                - surplus_distribution: 'equal', 'proportional', 'need_based'
                - cooperative_type: 'worker', 'consumer', 'producer', 'multi_stakeholder'

        Returns:
            Dict with financial analysis, surplus distribution, viability metrics.
        """
        members = cooperative_data.get("members", [])
        revenue = float(cooperative_data.get("revenue", 0))
        costs = float(cooperative_data.get("costs", 0))
        dist_method = cooperative_data.get("surplus_distribution", "equal")
        coop_type = cooperative_data.get("cooperative_type", "worker")

        logger.info(
            "Modeling %s cooperative: %d members, revenue=%.2f, costs=%.2f",
            coop_type,
            len(members),
            revenue,
            costs,
        )

        surplus = revenue - costs
        n_members = max(len(members), 1)

        # Calculate surplus distribution
        distributions = []
        if dist_method == "equal":
            share = surplus / n_members
            for m in members:
                distributions.append(
                    {"member_id": m.get("id"), "share": round(share, 2)}
                )
        elif dist_method == "proportional":
            total_contrib = sum(float(m.get("contribution", 1)) for m in members)
            for m in members:
                contrib = float(m.get("contribution", 1))
                share = surplus * (contrib / max(total_contrib, 1))
                distributions.append(
                    {"member_id": m.get("id"), "share": round(share, 2)}
                )
        elif dist_method == "need_based":
            # Simple need-based: inverse of contribution (those contributing less get more)
            contribs = [float(m.get("contribution", 1)) for m in members]
            inv_contribs = [1 / max(c, 0.01) for c in contribs]
            total_inv = sum(inv_contribs)
            for m, inv_c in zip(members, inv_contribs):
                share = surplus * (inv_c / max(total_inv, 1))
                distributions.append(
                    {"member_id": m.get("id"), "share": round(share, 2)}
                )

        # Viability metrics
        per_member_revenue = revenue / n_members
        cost_ratio = costs / max(revenue, 1)
        gini = (
            self._gini_coefficient([d["share"] for d in distributions])
            if distributions
            else 0
        )

        return {
            "cooperative_type": coop_type,
            "member_count": n_members,
            "financial_summary": {
                "revenue": revenue,
                "costs": costs,
                "surplus": round(surplus, 2),
                "surplus_margin": round(surplus / max(revenue, 1) * 100, 1),
            },
            "distribution_method": dist_method,
            "member_distributions": distributions,
            "viability_metrics": {
                "per_member_revenue": round(per_member_revenue, 2),
                "cost_ratio": round(cost_ratio, 3),
                "equality_index": round(1 - gini, 3),  # 1 = perfect equality
                "viable": surplus > 0 and per_member_revenue > costs * 0.3,
            },
        }

    @staticmethod
    def _gini_coefficient(values: list[float]) -> float:
        """Compute Gini coefficient of a distribution."""
        if not values or len(values) < 2:
            return 0.0
        arr = np.array(sorted(values), dtype=float)
        n = len(arr)
        index = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * arr) - (n + 1) * np.sum(arr))
            / (n * np.sum(arr) + 1e-10)
        )
