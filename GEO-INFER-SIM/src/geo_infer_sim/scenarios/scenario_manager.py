"""
Scenario management for GEO-INFER-SIM.

This module provides scenario definition, management, and comparison
capabilities for simulation experiments.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, UTC
import uuid
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Represents a simulation scenario."""

    scenario_id: str
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    initial_conditions: dict[str, Any] = field(default_factory=dict)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class ScenarioManager:
    """
    Manager for simulation scenarios.

    Provides scenario definition, storage, comparison, and batch execution
    capabilities for simulation experiments.
    """

    def __init__(self) -> None:
        """Initialize the scenario manager."""
        self.scenarios: dict[str, Scenario] = {}
        self.scenario_results: dict[str, dict[str, Any]] = {}

    def create_scenario(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        initial_conditions: dict[str, Any] | None = None,
        interventions: list[dict[str, Any]] | None = None,
        scenario_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Scenario:
        """
        Create a new simulation scenario.

        Args:
            name: Scenario name
            description: Scenario description
            parameters: Simulation parameters
            initial_conditions: Initial conditions
            interventions: List of interventions to apply
            scenario_id: Optional scenario ID (if None, generates one)
            metadata: Optional scenario metadata

        Returns:
            Created Scenario object
        """
        scenario_id = scenario_id or str(uuid.uuid4())

        scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            parameters=parameters or {},
            initial_conditions=initial_conditions or {},
            interventions=interventions or [],
            metadata=metadata or {},
        )

        self.scenarios[scenario_id] = scenario
        logger.info(f"Created scenario: {name} (ID: {scenario_id})")

        return scenario

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        """
        Get a scenario by ID.

        Args:
            scenario_id: Scenario identifier

        Returns:
            Scenario object or None if not found
        """
        return self.scenarios.get(scenario_id)

    def list_scenarios(self) -> list[Scenario]:
        """
        List all scenarios.

        Returns:
            List of all scenarios
        """
        return list(self.scenarios.values())

    def update_scenario(
        self,
        scenario_id: str,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        initial_conditions: dict[str, Any] | None = None,
        interventions: list[dict[str, Any]] | None = None,
    ) -> bool:
        """
        Update an existing scenario.

        Args:
            scenario_id: Scenario identifier
            name: Optional new name
            description: Optional new description
            parameters: Optional updated parameters
            initial_conditions: Optional updated initial conditions
            interventions: Optional updated interventions

        Returns:
            True if scenario was updated, False if not found
        """
        if scenario_id not in self.scenarios:
            return False

        scenario = self.scenarios[scenario_id]

        if name is not None:
            scenario.name = name
        if description is not None:
            scenario.description = description
        if parameters is not None:
            scenario.parameters.update(parameters)
        if initial_conditions is not None:
            scenario.initial_conditions.update(initial_conditions)
        if interventions is not None:
            scenario.interventions = interventions

        logger.info(f"Updated scenario: {scenario_id}")
        return True

    def delete_scenario(self, scenario_id: str) -> bool:
        """
        Delete a scenario.

        Args:
            scenario_id: Scenario identifier

        Returns:
            True if scenario was deleted, False if not found
        """
        if scenario_id not in self.scenarios:
            return False

        del self.scenarios[scenario_id]

        if scenario_id in self.scenario_results:
            del self.scenario_results[scenario_id]

        logger.info(f"Deleted scenario: {scenario_id}")
        return True

    def save_scenario_result(self, scenario_id: str, result: dict[str, Any]) -> None:
        """
        Save results for a scenario.

        Args:
            scenario_id: Scenario identifier
            result: Simulation results
        """
        self.scenario_results[scenario_id] = result
        logger.info(f"Saved results for scenario: {scenario_id}")

    def get_scenario_result(self, scenario_id: str) -> dict[str, Any] | None:
        """
        Get results for a scenario.

        Args:
            scenario_id: Scenario identifier

        Returns:
            Results dictionary or None if not found
        """
        return self.scenario_results.get(scenario_id)

    def compare_scenarios(
        self, scenario_ids: list[str], metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Compare multiple scenarios.

        Args:
            scenario_ids: List of scenario IDs to compare
            metrics: Optional list of metrics to compare

        Returns:
            Comparison results dictionary
        """
        comparison: dict[str, Any] = {
            "scenarios": [],
            "metrics": {},
            "summary": {},
        }

        for scenario_id in scenario_ids:
            if scenario_id not in self.scenarios:
                continue

            scenario = self.scenarios[scenario_id]
            result = self.scenario_results.get(scenario_id, {})

            scenarios_list: list[dict[str, Any]] = comparison["scenarios"]
            scenarios_list.append(
                {
                    "scenario_id": scenario_id,
                    "name": scenario.name,
                    "parameters": scenario.parameters,
                    "result_available": scenario_id in self.scenario_results,
                }
            )

            # Extract metrics if results available
            if result and metrics:
                for metric in metrics:
                    metrics_dict: dict[str, list[dict[str, Any]]] = comparison[
                        "metrics"
                    ]
                    if metric not in metrics_dict:
                        metrics_dict[metric] = []

                    metric_value = result.get("metrics", {}).get(metric)
                    metrics_dict[metric].append(
                        {
                            "scenario_id": scenario_id,
                            "value": metric_value,
                        }
                    )

        # Calculate summary statistics
        if comparison["metrics"]:
            summary: dict[str, dict[str, Any]] = comparison["summary"]
            for metric_name, metric_values in comparison["metrics"].items():
                values = [m["value"] for m in metric_values if m["value"] is not None]
                if values:
                    summary[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "count": len(values),
                    }

        return comparison

    def run_scenarios(
        self,
        scenario_ids: list[str],
        simulation_func: Any,
        parallel: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Run multiple scenarios using a simulation function.

        Args:
            scenario_ids: List of scenario IDs to run
            simulation_func: Function that takes a Scenario and returns results
            parallel: Whether to run scenarios in parallel

        Returns:
            Dictionary mapping scenario IDs to results
        """
        results = {}

        if parallel:
            # Thread-pool execution: scenarios run concurrently in one process.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_scenario = {
                    executor.submit(
                        simulation_func, self.scenarios[scenario_id]
                    ): scenario_id
                    for scenario_id in scenario_ids
                    if scenario_id in self.scenarios
                }

                for future in concurrent.futures.as_completed(future_to_scenario):
                    scenario_id = future_to_scenario[future]
                    try:
                        result = future.result()
                        results[scenario_id] = result
                        self.save_scenario_result(scenario_id, result)
                    except Exception as e:
                        logger.exception(f"Scenario {scenario_id} failed")
                        results[scenario_id] = {"error": str(e)}
        else:
            # Sequential execution
            for scenario_id in scenario_ids:
                if scenario_id not in self.scenarios:
                    continue

                try:
                    scenario = self.scenarios[scenario_id]
                    result = simulation_func(scenario)
                    results[scenario_id] = result
                    self.save_scenario_result(scenario_id, result)
                except Exception as e:
                    logger.exception(f"Scenario {scenario_id} failed")
                    results[scenario_id] = {"error": str(e)}

        return results

    def analyze_results(self, scenario_ids: list[str] | None = None) -> dict[str, Any]:
        """
        Analyze results across scenarios.

        Args:
            scenario_ids: Optional list of scenario IDs to analyze (all if None)

        Returns:
            Analysis results dictionary
        """
        if scenario_ids is None:
            scenario_ids = list(self.scenario_results.keys())

        analysis: dict[str, Any] = {
            "scenarios_analyzed": len(scenario_ids),
            "successful_runs": 0,
            "failed_runs": 0,
            "metrics_summary": {},
            "best_scenarios": {},
        }

        all_metrics: set = set()
        for scenario_id in scenario_ids:
            result = self.scenario_results.get(scenario_id)
            if result and "error" not in result:
                successful_runs: int = analysis["successful_runs"]
                analysis["successful_runs"] = successful_runs + 1
                if "metrics" in result:
                    all_metrics.update(result["metrics"].keys())
            else:
                failed_runs: int = analysis["failed_runs"]
                analysis["failed_runs"] = failed_runs + 1

        # Aggregate metrics
        for metric_name in all_metrics:
            metric_values = []
            metric_scenarios = []
            for scenario_id in scenario_ids:
                result = self.scenario_results.get(scenario_id, {})
                if "metrics" in result and metric_name in result["metrics"]:
                    metric_values.append(result["metrics"][metric_name])
                    metric_scenarios.append(scenario_id)

            if metric_values:
                metrics_summary: dict[str, dict[str, Any]] = analysis["metrics_summary"]
                metrics_summary[metric_name] = {
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "min": float(np.min(metric_values)),
                    "max": float(np.max(metric_values)),
                    "count": len(metric_values),
                }

                # Find best scenario for each metric
                best_idx = int(np.argmax(metric_values)) if metric_values else None
                if best_idx is not None:
                    best_scenario_id = metric_scenarios[best_idx]
                    best_scenario = self.scenarios.get(best_scenario_id)
                    best_scenarios: dict[str, dict[str, Any]] = analysis[
                        "best_scenarios"
                    ]
                    best_scenarios[metric_name] = {
                        "scenario_id": best_scenario_id,
                        "value": float(metric_values[best_idx]),
                        "scenario_name": best_scenario.name
                        if best_scenario is not None
                        else "Unknown",
                    }

        return analysis
