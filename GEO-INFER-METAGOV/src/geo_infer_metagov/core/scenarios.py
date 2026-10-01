"""Scenario planning for governance systems."""

from dataclasses import dataclass
from typing import Any
import logging

from geo_infer_metagov.utils.helpers import entity_field

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Governance scenario definition."""

    scenario_id: str
    name: str
    description: str
    assumptions: dict[str, Any]
    modifications: dict[str, Any]
    probability: float = 0.5
    time_horizon: int = 5  # years


@dataclass
class ScenarioAnalysis:
    """Results of scenario analysis."""

    analysis_id: str
    base_case: dict[str, Any]
    scenarios: list[dict[str, Any]]
    sensitivity_analysis: dict[str, Any]
    recommendations: list[str]


class ScenarioPlanner:
    """
    Scenario planning for governance systems.

    Provides:
    - Scenario generation
    - Scenario evaluation
    - Scenario comparison
    - Sensitivity analysis
    - Scenario-based decision support

    References:
    - Schoemaker, P. J. (1995). Scenario Planning: A Tool for Strategic Thinking
    - Van der Heijden, K. (2005). Scenarios: The Art of Strategic Conversation
    """

    def __init__(self) -> None:
        """Initialize scenario planner."""
        self.scenarios: dict[str, Scenario] = {}
        self.analyses: dict[str, ScenarioAnalysis] = {}

    def generate_scenarios(
        self,
        governance_structure: dict[str, Any],
        scenario_types: list[str],
        time_horizon: int = 5,
    ) -> list[Scenario]:
        """
        Generate scenarios for governance planning.

        Parameters:
        -----------
        governance_structure : Dict[str, Any]
            Current governance structure
        scenario_types : List[str]
            Types of scenarios to generate ('optimistic', 'pessimistic', 'status_quo', 'disruptive')
        time_horizon : int
            Planning horizon in years

        Returns:
        --------
        List[Scenario]
            Generated scenarios
        """
        scenarios = []

        for scenario_type in scenario_types:
            scenario = self._create_scenario(
                governance_structure, scenario_type, time_horizon
            )
            scenarios.append(scenario)
            self.scenarios[scenario.scenario_id] = scenario

        logger.info(f"Generated {len(scenarios)} scenarios")
        return scenarios

    def _create_scenario(
        self,
        governance_structure: dict[str, Any],
        scenario_type: str,
        time_horizon: int,
    ) -> Scenario:
        """Create a specific scenario."""
        scenario_id = f"scenario_{scenario_type}_{len(self.scenarios)}"

        # Define scenario characteristics based on type
        scenario_definitions: dict[str, dict[str, Any]] = {
            "optimistic": {
                "name": "Optimistic Future",
                "description": "Best-case scenario with favorable conditions",
                "assumptions": {
                    "stakeholder_cooperation": "high",
                    "resource_availability": "high",
                    "external_support": "strong",
                    "technological_advancement": "rapid",
                },
                "modifications": {
                    "stakeholder_engagement": 0.9,
                    "resource_budget": 1.2,  # 20% increase
                    "capacity": 1.1,
                    "coordination_efficiency": 0.9,
                },
                "probability": 0.2,
            },
            "pessimistic": {
                "name": "Challenging Future",
                "description": "Worst-case scenario with adverse conditions",
                "assumptions": {
                    "stakeholder_cooperation": "low",
                    "resource_availability": "low",
                    "external_support": "weak",
                    "environmental_stress": "high",
                },
                "modifications": {
                    "stakeholder_engagement": 0.4,
                    "resource_budget": 0.7,  # 30% decrease
                    "capacity": 0.8,
                    "coordination_efficiency": 0.5,
                },
                "probability": 0.2,
            },
            "status_quo": {
                "name": "Status Quo",
                "description": "Continuation of current trends",
                "assumptions": {
                    "stakeholder_cooperation": "moderate",
                    "resource_availability": "stable",
                    "external_support": "moderate",
                    "change_rate": "gradual",
                },
                "modifications": {
                    "stakeholder_engagement": 0.6,
                    "resource_budget": 1.0,
                    "capacity": 1.0,
                    "coordination_efficiency": 0.7,
                },
                "probability": 0.4,
            },
            "disruptive": {
                "name": "Disruptive Change",
                "description": "Significant disruption requiring adaptation",
                "assumptions": {
                    "technological_disruption": "high",
                    "regulatory_change": "major",
                    "stakeholder_shifts": "significant",
                    "environmental_crisis": "moderate",
                },
                "modifications": {
                    "stakeholder_engagement": 0.5,
                    "resource_budget": 0.9,
                    "capacity": 0.7,
                    "coordination_efficiency": 0.6,
                    "adaptation_required": True,
                },
                "probability": 0.2,
            },
        }

        scenario_def = scenario_definitions.get(
            scenario_type.lower(), scenario_definitions["status_quo"]
        )

        return Scenario(
            scenario_id=scenario_id,
            name=scenario_def["name"],
            description=scenario_def["description"],
            assumptions=scenario_def["assumptions"],
            modifications=scenario_def["modifications"],
            probability=scenario_def["probability"],
            time_horizon=time_horizon,
        )

    def evaluate_scenario(
        self, scenario: Scenario, governance_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate governance structure under a scenario.

        Parameters:
        -----------
        scenario : Scenario
            Scenario to evaluate
        governance_structure : Dict[str, Any]
            Governance structure to evaluate

        Returns:
        --------
        Dict[str, Any]
            Scenario evaluation results
        """
        # Apply scenario modifications to structure
        modified_structure = self._apply_scenario_modifications(
            governance_structure, scenario.modifications
        )

        # Evaluate modified structure
        evaluation = self._evaluate_structure_under_scenario(
            modified_structure, scenario
        )

        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "evaluation": evaluation,
            "modified_structure": modified_structure,
            "scenario_probability": scenario.probability,
            "time_horizon": scenario.time_horizon,
        }

    def _apply_scenario_modifications(
        self, structure: dict[str, Any], modifications: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply scenario modifications to governance structure."""
        modified = structure.copy()

        # Modify entities (dict payloads or GovernanceEntity dataclass instances)
        if "entities" in modified:
            for entity in modified["entities"]:
                if "capacity" in modifications:
                    if isinstance(entity, dict):
                        entity["capacity"] = (
                            entity.get("capacity", 1.0) * modifications["capacity"]
                        )
                    else:
                        entity.capacity = (
                            entity_field(entity, "capacity", 1.0)
                            * modifications["capacity"]
                        )
                if "resource_budget" in modifications:
                    if isinstance(entity, dict):
                        if "resources" not in entity:
                            entity["resources"] = {}
                        entity["resources"]["budget"] = (
                            entity.get("resources", {}).get("budget", 1000000)
                            * modifications["resource_budget"]
                        )
                    else:
                        resources = dict(entity_field(entity, "resources", {}))
                        resources["budget"] = (
                            resources.get("budget", 1000000)
                            * modifications["resource_budget"]
                        )
                        entity.resources = resources

        # Modify stakeholder engagement
        if "stakeholder_engagement" in modifications:
            modified["stakeholder_engagement_level"] = modifications[
                "stakeholder_engagement"
            ]

        # Modify coordination efficiency
        if "coordination_efficiency" in modifications:
            modified["coordination_efficiency"] = modifications[
                "coordination_efficiency"
            ]

        return modified

    def _evaluate_structure_under_scenario(
        self, structure: dict[str, Any], scenario: Scenario
    ) -> dict[str, Any]:
        """Evaluate structure performance under scenario."""
        entities = structure.get("entities", [])

        # Calculate scenario-adjusted metrics
        avg_capacity = (
            sum(entity_field(e, "capacity", 0.5) for e in entities) / len(entities)
            if entities
            else 0.5
        )
        total_budget = sum(
            entity_field(entity_field(e, "resources", {}), "budget", 0)
            for e in entities
        )

        # Scenario impact on performance
        capacity_factor = avg_capacity
        resource_factor = min(1.0, total_budget / 1000000)
        coordination_factor = structure.get("coordination_efficiency", 0.7)

        # Calculate scenario performance score
        scenario_score = (
            capacity_factor * 0.4 + resource_factor * 0.3 + coordination_factor * 0.3
        )

        return {
            "scenario_performance": scenario_score,
            "capacity_score": capacity_factor,
            "resource_score": resource_factor,
            "coordination_score": coordination_factor,
            "viability": "viable"
            if scenario_score >= 0.6
            else "challenged"
            if scenario_score >= 0.4
            else "non_viable",
            "adaptation_needed": scenario_score < 0.6,
        }

    def analyze_scenarios(
        self, governance_structure: dict[str, Any], scenarios: list[Scenario]
    ) -> ScenarioAnalysis:
        """
        Analyze multiple scenarios for governance structure.

        Parameters:
        -----------
        governance_structure : Dict[str, Any]
            Governance structure to analyze
        scenarios : List[Scenario]
            Scenarios to analyze

        Returns:
        --------
        ScenarioAnalysis
            Comprehensive scenario analysis
        """
        analysis_id = f"analysis_{len(self.analyses)}"

        # Evaluate base case
        base_case = self._evaluate_structure_under_scenario(
            governance_structure,
            Scenario(
                scenario_id="base",
                name="Base Case",
                description="Current state",
                assumptions={},
                modifications={},
                probability=1.0,
            ),
        )

        # Evaluate each scenario
        scenario_evaluations = []
        for scenario in scenarios:
            evaluation = self.evaluate_scenario(scenario, governance_structure)
            scenario_evaluations.append(evaluation)

        # Perform sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(
            governance_structure, scenarios
        )

        # Generate recommendations
        recommendations = self._generate_scenario_recommendations(
            base_case, scenario_evaluations
        )

        analysis = ScenarioAnalysis(
            analysis_id=analysis_id,
            base_case=base_case,
            scenarios=scenario_evaluations,
            sensitivity_analysis=sensitivity,
            recommendations=recommendations,
        )

        self.analyses[analysis_id] = analysis
        logger.info(f"Scenario analysis completed: {analysis_id}")

        return analysis

    def _perform_sensitivity_analysis(
        self, governance_structure: dict[str, Any], scenarios: list[Scenario]
    ) -> dict[str, Any]:
        """Perform sensitivity analysis on key variables."""
        sensitivity: dict[str, Any] = {
            "key_variables": [],
            "sensitivity_scores": {},
            "critical_factors": [],
        }

        # Identify key variables from scenarios
        all_modifications = {}
        for scenario in scenarios:
            all_modifications.update(scenario.modifications)

        key_variables = list(all_modifications.keys())
        sensitivity["key_variables"] = key_variables

        # Calculate sensitivity for each variable
        for variable in key_variables:
            # Test impact of variable changes
            test_modifications = {variable: 0.8}  # 20% reduction

            test_structure = self._apply_scenario_modifications(
                governance_structure.copy(), test_modifications
            )
            test_evaluation = self._evaluate_structure_under_scenario(
                test_structure, scenarios[0]
            )

            base_evaluation = self._evaluate_structure_under_scenario(
                governance_structure, scenarios[0]
            )

            sensitivity_score = abs(
                test_evaluation["scenario_performance"]
                - base_evaluation["scenario_performance"]
            )

            sensitivity["sensitivity_scores"][variable] = sensitivity_score

            if sensitivity_score > 0.15:  # High sensitivity threshold
                sensitivity["critical_factors"].append(variable)

        return sensitivity

    def _generate_scenario_recommendations(
        self, base_case: dict[str, Any], scenario_evaluations: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on scenario analysis."""
        recommendations = []

        # Check for scenarios with poor performance
        poor_scenarios = [
            eval
            for eval in scenario_evaluations
            if eval["evaluation"].get("scenario_performance", 0.5) < 0.5
        ]

        if poor_scenarios:
            recommendations.append("Develop contingency plans for adverse scenarios")
            recommendations.append("Build resilience to handle challenging conditions")

        # Check for adaptation needs
        adaptation_needed = any(
            eval["evaluation"].get("adaptation_needed", False)
            for eval in scenario_evaluations
        )

        if adaptation_needed:
            recommendations.append("Implement adaptive governance mechanisms")
            recommendations.append("Establish learning and feedback systems")

        # Check base case performance
        if base_case.get("scenario_performance", 0.5) < 0.6:
            recommendations.append(
                "Improve current governance structure before scenario planning"
            )

        return recommendations

    def compare_scenarios(
        self, scenario_evaluations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compare multiple scenario evaluations.

        Parameters:
        -----------
        scenario_evaluations : List[Dict[str, Any]]
            Scenario evaluation results

        Returns:
        --------
        Dict[str, Any]
            Scenario comparison results
        """
        if not scenario_evaluations:
            return {"compared": False, "reason": "No scenarios to compare"}

        # Extract performance scores
        performances = {
            eval["scenario_name"]: eval["evaluation"].get("scenario_performance", 0.5)
            for eval in scenario_evaluations
        }

        probabilities = {
            eval["scenario_name"]: eval.get("scenario_probability", 0.5)
            for eval in scenario_evaluations
        }

        # Calculate expected performance (weighted by probability)
        expected_performance = sum(
            performances[name] * probabilities[name] for name in performances.keys()
        )

        # Find best and worst scenarios
        best_scenario = max(performances.items(), key=lambda x: x[1])
        worst_scenario = min(performances.items(), key=lambda x: x[1])

        # Calculate performance range
        performance_range = best_scenario[1] - worst_scenario[1]

        return {
            "compared": True,
            "scenario_count": len(scenario_evaluations),
            "performances": performances,
            "probabilities": probabilities,
            "expected_performance": expected_performance,
            "best_scenario": {
                "name": best_scenario[0],
                "performance": best_scenario[1],
            },
            "worst_scenario": {
                "name": worst_scenario[0],
                "performance": worst_scenario[1],
            },
            "performance_range": performance_range,
            "recommendation": self._recommend_scenario_strategy(
                expected_performance, performance_range
            ),
        }

    def _recommend_scenario_strategy(
        self, expected_performance: float, performance_range: float
    ) -> str:
        """Recommend strategy based on scenario analysis."""
        if expected_performance >= 0.7 and performance_range < 0.3:
            return "stable_optimization"  # Good performance, low uncertainty
        elif expected_performance >= 0.6 and performance_range > 0.4:
            return "robust_preparation"  # Moderate performance, high uncertainty
        elif expected_performance < 0.5:
            return "transformation_required"  # Poor expected performance
        else:
            return "adaptive_management"  # Moderate performance, moderate uncertainty
