"""
Policy Analysis Engine - Framework for economic policy impact assessment.
"""

from typing import Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime


class PolicyType(Enum):
    """Types of economic policies."""

    FISCAL = "fiscal"
    MONETARY = "monetary"
    TRADE = "trade"
    REGULATORY = "regulatory"
    INFRASTRUCTURE = "infrastructure"
    ENVIRONMENTAL = "environmental"


@dataclass
class PolicyScenario:
    """Definition of a policy scenario for analysis."""

    name: str
    description: str
    policy_type: PolicyType
    parameters: dict[str, Any]
    spatial_scope: str | None = None  # 'national', 'regional', 'local'
    temporal_scope: dict[str, Any] | None = None
    implementation_date: datetime | None = None


@dataclass
class PolicyImpact:
    """Container for policy impact results."""

    scenario_name: str
    gdp_impact: dict[str, float]  # Regional GDP changes
    employment_impact: dict[str, float]  # Employment changes
    welfare_impact: dict[str, float]  # Welfare changes
    distributional_impact: dict[str, Any]  # Distributional effects
    spatial_spillovers: dict[str, float] | None = None
    confidence_intervals: dict[str, tuple[float, float]] | None = None


@dataclass
class PolicyComparison:
    """Comparison of multiple policy scenarios."""

    baseline_scenario: str
    alternative_scenarios: list[str]
    comparison_metrics: dict[str, dict[str, float]]
    ranking: list[tuple[str, float]]  # (scenario_name, score)
    recommendations: list[str]


class PolicyAnalysisEngine:
    """
    Comprehensive framework for economic policy impact assessment.

    Provides capabilities for:
    - Policy scenario definition and modeling
    - Impact assessment across multiple dimensions
    - Comparative policy analysis
    - Spatial policy spillover analysis
    - Dynamic policy evaluation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the Policy Analysis Engine.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.scenarios: dict[str, Any] = {}
        self.baseline_data: dict[str, Any] = {}
        self.impact_cache: dict[str, Any] = {}

    def add_baseline_data(
        self, data_type: str, data: pd.DataFrame | dict[str, Any]
    ) -> None:
        """
        Add baseline economic data for policy analysis.

        Args:
            data_type: Type of baseline data ('gdp', 'employment', 'demographics', etc.)
            data: Baseline data
        """
        self.baseline_data[data_type] = data
        self.logger.info(f"Added baseline data: {data_type}")

    def define_scenario(self, scenario: PolicyScenario) -> None:
        """
        Define a policy scenario for analysis.

        Args:
            scenario: Policy scenario definition
        """
        self.scenarios[scenario.name] = scenario
        self.logger.info(f"Defined policy scenario: {scenario.name}")

    def _require_baseline(self, data_type: str) -> dict[str, Any]:
        """
        Return required baseline data or raise if missing/empty.

        Args:
            data_type: Baseline data key ('gdp', 'emissions', etc.)

        Returns:
            The stored baseline data

        Raises:
            ValueError: If no baseline data of the requested type exists
        """
        data = self.baseline_data.get(data_type, {})
        if not data:
            raise ValueError(
                f"Baseline {data_type.upper()} data required for policy analysis"
            )
        return data

    def assess_fiscal_policy(self, scenario: PolicyScenario) -> PolicyImpact:
        """
        Assess the impact of fiscal policy changes.

        Args:
            scenario: Fiscal policy scenario

        Returns:
            Policy impact assessment
        """
        if scenario.policy_type != PolicyType.FISCAL:
            raise ValueError("Scenario must be of fiscal policy type")

        params = scenario.parameters

        # Extract fiscal policy parameters
        tax_change = params.get("tax_rate_change", 0)
        spending_change = params.get("government_spending_change", 0)
        transfer_change = params.get("transfer_payments_change", 0)

        # Simple fiscal multiplier model
        # In practice, this would use sophisticated macroeconomic models

        # Get baseline GDP data
        baseline_gdp = self._require_baseline("gdp")

        # Calculate multipliers
        spending_multiplier = params.get("spending_multiplier", 1.5)
        tax_multiplier = params.get("tax_multiplier", -0.8)
        transfer_multiplier = params.get("transfer_multiplier", 0.6)

        # Calculate regional impacts
        gdp_impact = {}
        employment_impact = {}
        welfare_impact = {}

        for region, baseline_value in baseline_gdp.items():
            # GDP impact
            gdp_change = (
                spending_change * spending_multiplier
                + tax_change * tax_multiplier
                + transfer_change * transfer_multiplier
            )
            gdp_impact[region] = gdp_change * baseline_value / 100

            # Employment impact (Okun's law approximation: du = -beta*dGDP, so
            # employment moves in the same direction as GDP regardless of the
            # sign convention of the supplied coefficient)
            okun_coefficient = params.get("okun_coefficient", -2.0)
            if okun_coefficient == 0:
                raise ValueError("okun_coefficient must be non-zero")
            employment_change = gdp_change / abs(okun_coefficient)
            employment_impact[region] = employment_change

            # Welfare impact (simplified)
            welfare_change = gdp_change * 0.7  # Rough approximation
            welfare_impact[region] = welfare_change

        # Distributional impacts
        distributional_impact = self._calculate_distributional_effects(
            tax_change, spending_change, transfer_change, params
        )

        return PolicyImpact(
            scenario_name=scenario.name,
            gdp_impact=gdp_impact,
            employment_impact=employment_impact,
            welfare_impact=welfare_impact,
            distributional_impact=distributional_impact,
        )

    def assess_infrastructure_policy(self, scenario: PolicyScenario) -> PolicyImpact:
        """
        Assess the impact of infrastructure investment policies.

        Args:
            scenario: Infrastructure policy scenario

        Returns:
            Policy impact assessment
        """
        if scenario.policy_type != PolicyType.INFRASTRUCTURE:
            raise ValueError("Scenario must be of infrastructure policy type")

        params = scenario.parameters

        # Infrastructure investment parameters
        investment_amount = params.get("investment_amount", 0)
        infrastructure_type = params.get("type", "transport")
        regional_allocation = params.get("regional_allocation", {})

        # Infrastructure impact modeling
        baseline_gdp = self._require_baseline("gdp")

        # Different multipliers for different infrastructure types
        multipliers = {
            "transport": {"short_term": 1.2, "long_term": 2.1},
            "digital": {"short_term": 0.8, "long_term": 2.8},
            "energy": {"short_term": 1.0, "long_term": 1.9},
            "water": {"short_term": 1.1, "long_term": 1.6},
        }

        infrastructure_multiplier = multipliers.get(
            infrastructure_type, {"short_term": 1.0, "long_term": 1.5}
        )

        gdp_impact = {}
        employment_impact = {}
        welfare_impact = {}

        for region, baseline_value in baseline_gdp.items():
            # Regional investment share
            regional_share = regional_allocation.get(region, 1.0 / len(baseline_gdp))
            regional_investment = investment_amount * regional_share

            # Calculate impacts
            short_term_impact = (
                regional_investment / baseline_value
            ) * infrastructure_multiplier["short_term"]
            long_term_impact = (
                regional_investment / baseline_value
            ) * infrastructure_multiplier["long_term"]

            # Weighted average impact (time preference)
            time_preference = params.get(
                "time_preference", 0.7
            )  # Weight for short-term
            gdp_change = (
                time_preference * short_term_impact
                + (1 - time_preference) * long_term_impact
            )

            gdp_impact[region] = gdp_change * baseline_value

            # Employment impacts
            construction_jobs = regional_investment / params.get("cost_per_job", 100000)
            permanent_jobs = construction_jobs * params.get("permanent_job_ratio", 0.1)
            employment_impact[region] = construction_jobs + permanent_jobs

            # Welfare impacts (including accessibility improvements)
            accessibility_improvement = params.get("accessibility_improvement", 0.05)
            welfare_impact[region] = (
                gdp_impact[region] + baseline_value * accessibility_improvement
            )

        # Spatial spillovers for infrastructure
        spatial_spillovers = self._calculate_infrastructure_spillovers(
            regional_allocation, infrastructure_type, params
        )

        distributional_impact = {
            "income_quintile_effects": self._infrastructure_distributional_effects(
                infrastructure_type, investment_amount
            )
        }

        return PolicyImpact(
            scenario_name=scenario.name,
            gdp_impact=gdp_impact,
            employment_impact=employment_impact,
            welfare_impact=welfare_impact,
            distributional_impact=distributional_impact,
            spatial_spillovers=spatial_spillovers,
        )

    def assess_environmental_policy(self, scenario: PolicyScenario) -> PolicyImpact:
        """
        Assess the impact of environmental policies.

        Args:
            scenario: Environmental policy scenario

        Returns:
            Policy impact assessment
        """
        if scenario.policy_type != PolicyType.ENVIRONMENTAL:
            raise ValueError("Scenario must be of environmental policy type")

        params = scenario.parameters

        # Environmental policy parameters
        carbon_tax = params.get("carbon_tax", 0)
        green_subsidies = params.get("green_subsidies", 0)

        baseline_gdp = self._require_baseline("gdp")
        baseline_emissions = self.baseline_data.get("emissions", {})

        gdp_impact = {}
        employment_impact = {}
        welfare_impact = {}

        for region in baseline_gdp.keys():
            # Carbon tax impacts
            regional_emissions = baseline_emissions.get(region, 0)
            carbon_cost = carbon_tax * regional_emissions

            # Short-term GDP cost
            carbon_intensity = params.get(
                "carbon_intensity", 0.5
            )  # Tons CO2 per $1000 GDP
            gdp_cost = carbon_cost * carbon_intensity

            # Green investment boost
            green_investment = green_subsidies * params.get(
                "regional_green_share", {}
            ).get(region, 1.0)
            green_multiplier = params.get("green_multiplier", 1.3)
            gdp_boost = green_investment * green_multiplier

            # Net GDP impact
            gdp_impact[region] = gdp_boost - gdp_cost

            # Employment impacts
            # Job losses in carbon-intensive sectors
            carbon_job_loss = carbon_cost * params.get("jobs_per_carbon_cost", 0.05)
            # Job gains in green sectors
            green_job_gain = green_investment * params.get(
                "green_jobs_per_investment", 0.08
            )

            employment_impact[region] = green_job_gain - carbon_job_loss

            # Welfare impacts (including environmental benefits)
            environmental_benefit = regional_emissions * params.get(
                "environmental_value_per_ton", 50
            )
            health_benefit = environmental_benefit * params.get(
                "health_benefit_ratio", 0.3
            )

            welfare_impact[region] = (
                gdp_impact[region] + environmental_benefit + health_benefit
            )

        # Environmental justice considerations
        distributional_impact = {
            "environmental_burden_by_income": self._environmental_distributional_effects(
                carbon_tax, green_subsidies, params
            )
        }

        return PolicyImpact(
            scenario_name=scenario.name,
            gdp_impact=gdp_impact,
            employment_impact=employment_impact,
            welfare_impact=welfare_impact,
            distributional_impact=distributional_impact,
        )

    def compare_scenarios(
        self, scenario_names: list[str], weights: dict[str, float] | None = None
    ) -> PolicyComparison:
        """
        Compare multiple policy scenarios.

        Args:
            scenario_names: List of scenario names to compare
            weights: Optional weights for different impact dimensions

        Returns:
            Policy comparison results
        """
        if not scenario_names:
            raise ValueError("At least one scenario required for comparison")

        # Default weights
        if weights is None:
            weights = {
                "gdp": 0.3,
                "employment": 0.3,
                "welfare": 0.2,
                "distributional": 0.2,
            }

        # Collect impact results for all scenarios
        scenario_impacts = {}
        for name in scenario_names:
            if name not in self.scenarios:
                raise ValueError(f"Scenario '{name}' not defined")

            scenario = self.scenarios[name]

            # Assess impact based on policy type
            if scenario.policy_type == PolicyType.FISCAL:
                impact = self.assess_fiscal_policy(scenario)
            elif scenario.policy_type == PolicyType.INFRASTRUCTURE:
                impact = self.assess_infrastructure_policy(scenario)
            elif scenario.policy_type == PolicyType.ENVIRONMENTAL:
                impact = self.assess_environmental_policy(scenario)
            else:
                # Generic assessment
                impact = self._generic_policy_assessment(scenario)

            scenario_impacts[name] = impact

        # Calculate comparison metrics
        comparison_metrics = {}
        scenario_scores = {}

        for name, impact in scenario_impacts.items():
            metrics = {}

            # Aggregate impacts
            total_gdp_impact = sum(impact.gdp_impact.values())
            total_employment_impact = sum(impact.employment_impact.values())
            total_welfare_impact = sum(impact.welfare_impact.values())

            # Distributional score (lower inequality is better)
            distributional_score = self._calculate_distributional_score(
                impact.distributional_impact
            )

            metrics["total_gdp_impact"] = total_gdp_impact
            metrics["total_employment_impact"] = total_employment_impact
            metrics["total_welfare_impact"] = total_welfare_impact
            metrics["distributional_score"] = distributional_score

            comparison_metrics[name] = metrics

            # Overall score
            score = (
                weights["gdp"] * total_gdp_impact
                + weights["employment"] * total_employment_impact
                + weights["welfare"] * total_welfare_impact
                + weights["distributional"] * distributional_score
            )

            scenario_scores[name] = score

        # Rank scenarios
        ranking = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate recommendations
        recommendations = self._generate_policy_recommendations(
            scenario_impacts, ranking
        )

        return PolicyComparison(
            baseline_scenario=scenario_names[0],
            alternative_scenarios=scenario_names[1:],
            comparison_metrics=comparison_metrics,
            ranking=ranking,
            recommendations=recommendations,
        )

    def _calculate_distributional_effects(
        self,
        tax_change: float,
        spending_change: float,
        transfer_change: float,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate distributional effects of fiscal policy."""
        income_quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]

        # Tax incidence by income quintile
        tax_incidence = params.get(
            "tax_incidence",
            {"Q1": 0.05, "Q2": 0.10, "Q3": 0.20, "Q4": 0.25, "Q5": 0.40},
        )

        # Spending benefit distribution
        spending_distribution = params.get(
            "spending_distribution",
            {"Q1": 0.30, "Q2": 0.25, "Q3": 0.20, "Q4": 0.15, "Q5": 0.10},
        )

        # Transfer distribution
        transfer_distribution = params.get(
            "transfer_distribution",
            {"Q1": 0.40, "Q2": 0.30, "Q3": 0.20, "Q4": 0.07, "Q5": 0.03},
        )

        distributional_effects = {}
        for quintile in income_quintiles:
            tax_effect = tax_change * tax_incidence[quintile]
            spending_effect = spending_change * spending_distribution[quintile]
            transfer_effect = transfer_change * transfer_distribution[quintile]

            net_effect = spending_effect + transfer_effect - tax_effect
            distributional_effects[quintile] = net_effect

        return {"income_quintile_effects": distributional_effects}

    def _calculate_infrastructure_spillovers(
        self,
        regional_allocation: dict[str, float],
        infrastructure_type: str,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate spatial spillovers from infrastructure investment."""
        spillover_rates = {
            "transport": 0.15,
            "digital": 0.25,
            "energy": 0.10,
            "water": 0.05,
        }

        spillover_rate = spillover_rates.get(infrastructure_type, 0.10)
        spillovers = {}

        for region, investment_share in regional_allocation.items():
            # Simplified spillover calculation
            spillover_effect = investment_share * spillover_rate
            spillovers[f"{region}_spillover"] = spillover_effect

        return spillovers

    def _infrastructure_distributional_effects(
        self, infrastructure_type: str, investment_amount: float
    ) -> dict[str, float]:
        """Calculate distributional effects of infrastructure investment."""
        # Different infrastructure types benefit different income groups differently
        benefit_patterns = {
            "transport": {"Q1": 0.15, "Q2": 0.20, "Q3": 0.25, "Q4": 0.22, "Q5": 0.18},
            "digital": {"Q1": 0.10, "Q2": 0.15, "Q3": 0.20, "Q4": 0.25, "Q5": 0.30},
            "energy": {"Q1": 0.25, "Q2": 0.22, "Q3": 0.20, "Q4": 0.18, "Q5": 0.15},
            "water": {"Q1": 0.30, "Q2": 0.25, "Q3": 0.20, "Q4": 0.15, "Q5": 0.10},
        }

        pattern = benefit_patterns.get(
            infrastructure_type,
            {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        )

        effects = {}
        for quintile, share in pattern.items():
            effects[quintile] = investment_amount * share

        return effects

    def _environmental_distributional_effects(
        self, carbon_tax: float, green_subsidies: float, params: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate distributional effects of environmental policy."""
        # Carbon tax is typically regressive
        carbon_tax_burden = {"Q1": 0.25, "Q2": 0.22, "Q3": 0.20, "Q4": 0.18, "Q5": 0.15}

        # Green subsidies can be progressive if well-targeted
        green_subsidy_benefit = {
            "Q1": 0.15,
            "Q2": 0.18,
            "Q3": 0.20,
            "Q4": 0.22,
            "Q5": 0.25,
        }

        effects = {}
        for quintile in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            tax_burden = carbon_tax * carbon_tax_burden[quintile]
            subsidy_benefit = green_subsidies * green_subsidy_benefit[quintile]
            net_effect = subsidy_benefit - tax_burden
            effects[quintile] = net_effect

        return effects

    def _generic_policy_assessment(self, scenario: PolicyScenario) -> PolicyImpact:
        """Generic policy impact assessment for unspecified policy types."""
        # Baseline for generic assessment logic
        return PolicyImpact(
            scenario_name=scenario.name,
            gdp_impact={"national": 0.0},
            employment_impact={"national": 0.0},
            welfare_impact={"national": 0.0},
            distributional_impact={
                "note": "Generic assessment - requires specific modeling"
            },
        )

    def _calculate_distributional_score(
        self, distributional_impact: dict[str, Any]
    ) -> float:
        """Calculate a single score representing distributional effects."""
        if "income_quintile_effects" in distributional_impact:
            effects = distributional_impact["income_quintile_effects"]
            # Calculate Gini-like coefficient - lower values indicate more equitable distribution
            values = list(effects.values())
            if len(values) > 1:
                mean_val = np.mean(values)
                return -np.std(values) / abs(mean_val) if mean_val != 0 else 0
        return 0.0

    def _generate_policy_recommendations(
        self,
        scenario_impacts: dict[str, PolicyImpact],
        ranking: list[tuple[str, float]],
    ) -> list[str]:
        """Generate policy recommendations based on analysis results."""
        recommendations = []

        if ranking:
            best_scenario = ranking[0][0]
            recommendations.append(
                f"Recommended policy: {best_scenario} (highest overall score)"
            )

            best_impact = scenario_impacts[best_scenario]
            if sum(best_impact.gdp_impact.values()) > 0:
                recommendations.append(
                    "Policy shows positive GDP impacts across regions"
                )
            if sum(best_impact.employment_impact.values()) > 0:
                recommendations.append("Policy expected to create net employment gains")

        # Cross-cutting recommendations
        recommendations.append(
            "Consider implementation sequencing and transition support"
        )
        recommendations.append(
            "Monitor distributional effects and provide targeted assistance if needed"
        )
        recommendations.append(
            "Evaluate spatial spillovers and coordinate across jurisdictions"
        )

        return recommendations
