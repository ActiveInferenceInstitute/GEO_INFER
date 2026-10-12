"""
Performance Metrics and Evaluation Framework for GEO-INFER-ANT

This module provides comprehensive performance metrics and evaluation tools for
swarm intelligence systems, including efficiency measures, robustness analysis,
scalability assessment, and comparative benchmarking.

Key Features:
- Multi-dimensional performance evaluation
- Robustness analysis under various failure scenarios
- Scalability testing across different system sizes
- Statistical validation and hypothesis testing
- Integration with benchmarking frameworks
- Performance visualization and reporting
"""

import numpy as np
import logging
from scipy import stats
from typing import Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfiguration:
    """Configuration for performance evaluation."""

    evaluation_criteria: list[str] = field(
        default_factory=lambda: [
            "efficiency",
            "robustness",
            "adaptability",
            "scalability",
        ]
    )
    benchmark_datasets: list[str] = field(default_factory=list)
    statistical_analysis: list[str] = field(
        default_factory=lambda: [
            "hypothesis_testing",
            "effect_size",
            "confidence_intervals",
        ]
    )
    performance_thresholds: dict[str, float] = field(default_factory=dict)
    comparison_baselines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_criteria = [
            "efficiency",
            "robustness",
            "adaptability",
            "scalability",
            "accuracy",
            "speed",
        ]
        for criterion in self.evaluation_criteria:
            if criterion not in valid_criteria:
                raise ValueError(f"Invalid evaluation criterion: {criterion}")


class SwarmPerformanceMetrics:
    """
    Comprehensive performance metrics and evaluation system for swarm intelligence.

    This class provides sophisticated evaluation tools for assessing swarm system
    performance across multiple dimensions, including efficiency, robustness,
    adaptability, and scalability.

    Integration Points:
    - GEO-INFER-MATH: Statistical analysis and hypothesis testing
    - GEO-INFER-TIME: Performance tracking over time
    - External benchmarking frameworks
    """

    def __init__(
        self,
        evaluation_criteria: list[str] | None = None,
        benchmark_datasets: list[str] | None = None,
        statistical_analysis: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize performance metrics system.

        Args:
            evaluation_criteria: Criteria for performance evaluation
            benchmark_datasets: Standard datasets for benchmarking
            statistical_analysis: Statistical methods to apply
            **kwargs: Additional configuration parameters
        """
        self.config = PerformanceConfiguration(
            evaluation_criteria=evaluation_criteria
            or ["efficiency", "robustness", "adaptability"],
            benchmark_datasets=benchmark_datasets or [],
            statistical_analysis=statistical_analysis
            or ["hypothesis_testing", "confidence_intervals"],
        )

        # Performance data storage
        self.performance_history: list[dict[str, Any]] = []
        self.baseline_metrics: dict[str, Any] = {}
        self.comparative_results: dict[str, Any] = {}

        # Statistical analysis state
        self.statistical_tests: dict[str, Any] = {}
        self.confidence_intervals: dict[str, Any] = {}

        logger.info(
            f"SwarmPerformanceMetrics initialized with {len(self.config.evaluation_criteria)} criteria"
        )

    def evaluate_performance(
        self,
        swarm_behavior: dict[str, Any],
        task_objectives: dict[str, Any],
        environmental_conditions: dict[str, Any] | None = None,
        comparison_baselines: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate comprehensive swarm system performance.

        Args:
            swarm_behavior: Observed swarm behavior data
            task_objectives: Required task performance objectives
            environmental_conditions: Environmental context for evaluation
            comparison_baselines: Alternative algorithms for comparison

        Returns:
            Comprehensive performance assessment
        """
        logger.info("Evaluating swarm system performance")

        assessment: dict[str, Any] = {
            "evaluation_time": datetime.now(),
            "performance_scores": {},
            "efficiency_metrics": {},
            "robustness_metrics": {},
            "adaptability_metrics": {},
            "scalability_metrics": {},
            "statistical_analysis": {},
            "recommendations": [],
        }

        try:
            # Skip evaluation if no data provided
            if not swarm_behavior and not task_objectives:
                return assessment

            # Evaluate each performance criterion
            for criterion in self.config.evaluation_criteria:
                if criterion == "efficiency":
                    efficiency = self._evaluate_efficiency(
                        swarm_behavior, task_objectives
                    )
                    assessment["efficiency_metrics"] = efficiency
                    assessment["performance_scores"]["efficiency"] = efficiency.get(
                        "overall_score", 0.0
                    )

                elif criterion == "robustness":
                    robustness = self._evaluate_robustness(
                        swarm_behavior, environmental_conditions
                    )
                    assessment["robustness_metrics"] = robustness
                    assessment["performance_scores"]["robustness"] = robustness.get(
                        "overall_score", 0.0
                    )

                elif criterion == "adaptability":
                    adaptability = self._evaluate_adaptability(
                        swarm_behavior, environmental_conditions
                    )
                    assessment["adaptability_metrics"] = adaptability
                    assessment["performance_scores"]["adaptability"] = adaptability.get(
                        "overall_score", 0.0
                    )

                elif criterion == "scalability":
                    scalability = self._evaluate_scalability(swarm_behavior)
                    assessment["scalability_metrics"] = scalability
                    assessment["performance_scores"]["scalability"] = scalability.get(
                        "overall_score", 0.0
                    )

            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(assessment)
            assessment["statistical_analysis"] = statistical_analysis

            # Generate recommendations
            recommendations = self._generate_performance_recommendations(assessment)
            assessment["recommendations"] = recommendations

            # Store performance history
            self.performance_history.append(assessment)

        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            assessment["error"] = str(e)

        logger.info("Performance evaluation completed")
        return assessment

    def _evaluate_efficiency(
        self, swarm_behavior: dict[str, Any], task_objectives: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate swarm system efficiency."""
        efficiency: dict[str, Any] = {
            "overall_score": 0.0,
            "task_completion_rate": 0.0,
            "resource_utilization": 0.0,
            "energy_efficiency": 0.0,
            "communication_efficiency": 0.0,
            "computational_efficiency": 0.0,
        }

        try:
            # Task completion rate
            if "task_completion" in swarm_behavior:
                completed_tasks = swarm_behavior["task_completion"].get("completed", 0)
                total_tasks = swarm_behavior["task_completion"].get("total", 1)
                efficiency["task_completion_rate"] = completed_tasks / total_tasks

            # Resource utilization
            if "resource_usage" in swarm_behavior:
                resources = swarm_behavior["resource_usage"]
                efficiency["resource_utilization"] = resources.get(
                    "utilization_rate", 0.0
                )

            # Energy efficiency
            if "energy_metrics" in swarm_behavior:
                energy_data = swarm_behavior["energy_metrics"]
                total_energy = energy_data.get("total_consumed", 1.0)
                useful_energy = energy_data.get("useful_energy", 0.0)
                efficiency["energy_efficiency"] = (
                    useful_energy / total_energy if total_energy > 0 else 0.0
                )

            # Communication efficiency
            if "communication_metrics" in swarm_behavior:
                comm_data = swarm_behavior["communication_metrics"]
                successful_comms = comm_data.get("successful", 0)
                total_comms = comm_data.get("total", 1)
                efficiency["communication_efficiency"] = successful_comms / total_comms

            # Computational efficiency
            if "computation_metrics" in swarm_behavior:
                comp_data = swarm_behavior["computation_metrics"]
                efficiency["computational_efficiency"] = comp_data.get(
                    "efficiency_score", 0.0
                )

            # Calculate overall efficiency score
            weights = {
                "task_completion": 0.3,
                "resource_utilization": 0.2,
                "energy_efficiency": 0.2,
                "communication_efficiency": 0.15,
                "computational_efficiency": 0.15,
            }

            overall_score = 0.0
            for metric, weight in weights.items():
                if metric in efficiency:
                    overall_score += efficiency[metric] * weight

            efficiency["overall_score"] = min(1.0, overall_score)

        except Exception as e:
            logger.warning(f"Efficiency evaluation failed: {e}")
            efficiency["error"] = str(e)

        return efficiency

    def _evaluate_robustness(
        self,
        swarm_behavior: dict[str, Any],
        environmental_conditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate swarm system robustness."""
        robustness: dict[str, Any] = {
            "overall_score": 0.0,
            "failure_recovery": 0.0,
            "environmental_resilience": 0.0,
            "communication_robustness": 0.0,
            "task_robustness": 0.0,
        }

        try:
            # Failure recovery analysis
            if "failure_scenarios" in swarm_behavior:
                failure_data = swarm_behavior["failure_scenarios"]
                recovered_failures = failure_data.get("recovered", 0)
                total_failures = failure_data.get("total", 1)
                robustness["failure_recovery"] = recovered_failures / total_failures

            # Environmental resilience
            if environmental_conditions:
                env_volatility = environmental_conditions.get("volatility", 0.0)
                noise_level = environmental_conditions.get("noise_level", 0.0)
                robustness["environmental_resilience"] = 1.0 - min(
                    1.0, (env_volatility + noise_level) / 2.0
                )

            # Communication robustness
            if "communication_failures" in swarm_behavior:
                comm_failures = swarm_behavior["communication_failures"]
                robustness["communication_robustness"] = 1.0 - comm_failures.get(
                    "failure_rate", 0.0
                )

            # Task robustness (ability to complete tasks despite disruptions)
            if "task_disruptions" in swarm_behavior:
                disruptions = swarm_behavior["task_disruptions"]
                robustness["task_robustness"] = disruptions.get("completion_rate", 0.0)

            # Calculate overall robustness score
            weights = {
                "failure_recovery": 0.3,
                "environmental_resilience": 0.25,
                "communication_robustness": 0.25,
                "task_robustness": 0.2,
            }

            overall_score = 0.0
            for metric, weight in weights.items():
                if metric in robustness:
                    overall_score += robustness[metric] * weight

            robustness["overall_score"] = min(1.0, overall_score)

        except Exception as e:
            logger.warning(f"Robustness evaluation failed: {e}")
            robustness["error"] = str(e)

        return robustness

    def _evaluate_adaptability(
        self,
        swarm_behavior: dict[str, Any],
        environmental_conditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate swarm system adaptability."""
        adaptability: dict[str, Any] = {
            "overall_score": 0.0,
            "learning_rate": 0.0,
            "parameter_adaptation": 0.0,
            "environmental_adaptation": 0.0,
            "behavioral_flexibility": 0.0,
        }

        try:
            # Learning rate analysis
            if "learning_metrics" in swarm_behavior:
                learning_data = swarm_behavior["learning_metrics"]
                adaptability["learning_rate"] = learning_data.get(
                    "improvement_rate", 0.0
                )

            # Parameter adaptation
            if "parameter_changes" in swarm_behavior:
                param_data = swarm_behavior["parameter_changes"]
                adaptability["parameter_adaptation"] = param_data.get(
                    "adaptation_effectiveness", 0.0
                )

            # Environmental adaptation
            if environmental_conditions:
                env_changes = environmental_conditions.get("change_frequency", 0.0)
                adaptability["environmental_adaptation"] = 1.0 - min(1.0, env_changes)

            # Behavioral flexibility
            if "behavioral_changes" in swarm_behavior:
                behavioral_data = swarm_behavior["behavioral_changes"]
                adaptability["behavioral_flexibility"] = behavioral_data.get(
                    "flexibility_score", 0.0
                )

            # Calculate overall adaptability score
            weights = {
                "learning_rate": 0.3,
                "parameter_adaptation": 0.25,
                "environmental_adaptation": 0.25,
                "behavioral_flexibility": 0.2,
            }

            overall_score = 0.0
            for metric, weight in weights.items():
                if metric in adaptability:
                    overall_score += adaptability[metric] * weight

            adaptability["overall_score"] = min(1.0, overall_score)

        except Exception as e:
            logger.warning(f"Adaptability evaluation failed: {e}")
            adaptability["error"] = str(e)

        return adaptability

    def _evaluate_scalability(self, swarm_behavior: dict[str, Any]) -> dict[str, Any]:
        """Evaluate swarm system scalability."""
        scalability: dict[str, Any] = {
            "overall_score": 0.0,
            "size_scalability": 0.0,
            "performance_scaling": 0.0,
            "communication_scaling": 0.0,
            "memory_scaling": 0.0,
        }

        try:
            # Size scalability (how performance changes with swarm size)
            if "size_performance" in swarm_behavior:
                size_data = swarm_behavior["size_performance"]
                scalability["size_scalability"] = size_data.get(
                    "scaling_efficiency", 0.0
                )

            # Performance scaling (how well performance scales with problem size)
            if "problem_scaling" in swarm_behavior:
                problem_data = swarm_behavior["problem_scaling"]
                scalability["performance_scaling"] = problem_data.get(
                    "scaling_efficiency", 0.0
                )

            # Communication scaling
            if "communication_scaling" in swarm_behavior:
                comm_data = swarm_behavior["communication_scaling"]
                scalability["communication_scaling"] = comm_data.get(
                    "scaling_efficiency", 0.0
                )

            # Memory scaling
            if "memory_scaling" in swarm_behavior:
                memory_data = swarm_behavior["memory_scaling"]
                scalability["memory_scaling"] = memory_data.get(
                    "scaling_efficiency", 0.0
                )

            # Calculate overall scalability score
            weights = {
                "size_scalability": 0.3,
                "performance_scaling": 0.3,
                "communication_scaling": 0.2,
                "memory_scaling": 0.2,
            }

            overall_score = 0.0
            for metric, weight in weights.items():
                if metric in scalability:
                    overall_score += scalability[metric] * weight

            scalability["overall_score"] = min(1.0, overall_score)

        except Exception as e:
            logger.warning(f"Scalability evaluation failed: {e}")
            scalability["error"] = str(e)

        return scalability

    def _perform_statistical_analysis(
        self, assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform statistical analysis on performance metrics."""
        analysis: dict[str, Any] = {
            "hypothesis_tests": {},
            "confidence_intervals": {},
            "effect_sizes": {},
            "statistical_significance": {},
        }

        try:
            performance_scores = assessment.get("performance_scores", {})

            for metric, score in performance_scores.items():
                if isinstance(score, (int, float, np.number)):
                    samples = np.asarray([score], dtype=float)
                else:
                    samples = np.asarray(score, dtype=float).reshape(-1)
                samples = samples[np.isfinite(samples)]
                if samples.size == 0:
                    continue
                mean_score = float(np.mean(samples))
                if samples.size > 1:
                    standard_error = float(stats.sem(samples))
                    critical_value = float(stats.t.ppf(0.975, samples.size - 1))
                    confidence_interval = (
                        mean_score - critical_value * standard_error,
                        mean_score + critical_value * standard_error,
                    )
                    test = stats.ttest_1samp(samples, popmean=0.5)
                    p_value = float(test.pvalue)
                    significant = bool(p_value < 0.05)
                else:
                    confidence_interval = (mean_score, mean_score)
                    p_value = None
                    significant = False
                analysis["confidence_intervals"][metric] = confidence_interval
                analysis["statistical_significance"][metric] = significant
                analysis["hypothesis_tests"][metric] = {
                    "sample_size": int(samples.size),
                    "null_value": 0.5,
                    "p_value": p_value,
                    "significant": significant,
                }

            if len(performance_scores) > 1:
                scalar_scores = [
                    float(np.mean(np.asarray(value, dtype=float)))
                    for value in performance_scores.values()
                ]
                analysis["hypothesis_tests"]["performance_consistency"] = {
                    "test_statistic": float(np.std(scalar_scores)),
                    "sample_size": len(scalar_scores),
                }

        except Exception as e:
            logger.warning(f"Statistical analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def _generate_performance_recommendations(
        self, assessment: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate performance improvement recommendations."""
        recommendations = []

        try:
            performance_scores = assessment.get("performance_scores", {})

            # Efficiency recommendations
            efficiency_score = performance_scores.get("efficiency", 0.0)
            if efficiency_score < 0.7:
                recommendations.append(
                    {
                        "category": "efficiency",
                        "priority": "high" if efficiency_score < 0.5 else "medium",
                        "description": f"Improve efficiency from {efficiency_score:.2f} to >0.7",
                        "actions": [
                            "optimize_resource_allocation",
                            "reduce_communication_overhead",
                            "improve_task_coordination",
                        ],
                    }
                )

            # Robustness recommendations
            robustness_score = performance_scores.get("robustness", 0.0)
            if robustness_score < 0.8:
                recommendations.append(
                    {
                        "category": "robustness",
                        "priority": "high" if robustness_score < 0.6 else "medium",
                        "description": f"Improve robustness from {robustness_score:.2f} to >0.8",
                        "actions": [
                            "add_redundancy_mechanisms",
                            "improve_failure_recovery",
                            "enhance_environmental_resilience",
                        ],
                    }
                )

            # Adaptability recommendations
            adaptability_score = performance_scores.get("adaptability", 0.0)
            if adaptability_score < 0.6:
                recommendations.append(
                    {
                        "category": "adaptability",
                        "priority": "medium",
                        "description": f"Improve adaptability from {adaptability_score:.2f} to >0.6",
                        "actions": [
                            "enhance_learning_algorithms",
                            "implement_adaptive_parameters",
                            "add_environmental_sensing",
                        ],
                    }
                )

            # Scalability recommendations
            scalability_score = performance_scores.get("scalability", 0.0)
            if scalability_score < 0.7:
                recommendations.append(
                    {
                        "category": "scalability",
                        "priority": "medium",
                        "description": f"Improve scalability from {scalability_score:.2f} to >0.7",
                        "actions": [
                            "optimize_parallel_processing",
                            "implement_spatial_partitioning",
                            "add_load_balancing",
                        ],
                    }
                )

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")

        return recommendations

    def analyze_robustness(
        self,
        failure_scenarios: list[str],
        recovery_mechanisms: list[str],
        performance_degradation: dict[str, float],
    ) -> dict[str, Any]:
        """
        Analyze system robustness under various failure scenarios.

        Args:
            failure_scenarios: Types of failures to test
            recovery_mechanisms: Recovery strategies to evaluate
            performance_degradation: Acceptable performance degradation limits

        Returns:
            Robustness analysis results
        """
        logger.info(
            f"Analyzing robustness for {len(failure_scenarios)} failure scenarios"
        )

        robustness_analysis: dict[str, Any] = {
            "analysis_time": datetime.now(),
            "failure_scenarios": failure_scenarios,
            "recovery_mechanisms": recovery_mechanisms,
            "scenario_results": {},
            "overall_robustness": 0.0,
            "critical_failures": [],
            "recovery_effectiveness": {},
        }

        try:
            # Analyze each failure scenario
            for scenario in failure_scenarios:
                scenario_result = self._analyze_single_failure_scenario(
                    scenario, recovery_mechanisms, performance_degradation
                )
                robustness_analysis["scenario_results"][scenario] = scenario_result

                # Check if failure is critical
                degradation = scenario_result.get("performance_degradation", 1.0)
                if degradation > performance_degradation.get("critical_threshold", 0.8):
                    robustness_analysis["critical_failures"].append(scenario)

            # Analyze recovery effectiveness
            for mechanism in recovery_mechanisms:
                effectiveness = self._evaluate_recovery_mechanism(
                    mechanism, robustness_analysis["scenario_results"]
                )
                robustness_analysis["recovery_effectiveness"][mechanism] = effectiveness

            # Calculate overall robustness
            scenario_scores = [
                result.get("robustness_score", 0.0)
                for result in robustness_analysis["scenario_results"].values()
            ]
            robustness_analysis["overall_robustness"] = (
                np.mean(scenario_scores) if scenario_scores else 0.0
            )

        except Exception as e:
            logger.error(f"Robustness analysis failed: {e}")
            robustness_analysis["error"] = str(e)

        logger.info(
            f"Robustness analysis completed: overall score = {robustness_analysis['overall_robustness']}"
        )
        return robustness_analysis

    def _analyze_single_failure_scenario(
        self,
        scenario: str,
        recovery_mechanisms: list[str],
        performance_degradation: dict[str, float],
    ) -> dict[str, Any]:
        """Analyze a single failure scenario."""
        scenario_results = {
            "scenario": scenario,
            "failure_impact": 0.0,
            "recovery_time": 0.0,
            "performance_degradation": 0.0,
            "robustness_score": 1.0,
        }

        # Default impact priors can be overridden by measured scenario impact.
        failure_impacts = {
            "agent_loss": 0.2,
            "communication_failure": 0.3,
            "environmental_change": 0.1,
            "resource_shortage": 0.25,
            "coordination_breakdown": 0.4,
        }

        base_impact = performance_degradation.get(
            scenario, failure_impacts.get(scenario, 0.3)
        )
        base_impact = float(np.clip(base_impact, 0.0, 1.0))
        scenario_results["failure_impact"] = base_impact

        # Recovery durations are model parameters and can be supplied through
        # the recovery mechanism names used by the caller.
        recovery_times = {
            "redundancy": 10.0,
            "adaptation": 30.0,
            "reorganization": 60.0,
            "restart": 120.0,
        }

        if recovery_mechanisms:
            fastest_recovery = min(
                [recovery_times.get(mech, 60.0) for mech in recovery_mechanisms]
            )
            scenario_results["recovery_time"] = fastest_recovery

        # Each available recovery mechanism reduces the modeled impact by the
        # observed degradation limit, without allowing negative degradation.
        degradation = base_impact * (1.0 - min(0.9, 0.1 * len(recovery_mechanisms)))
        scenario_results["performance_degradation"] = min(1.0, degradation)

        # Calculate robustness score
        scenario_results["robustness_score"] = 1.0 - degradation

        return scenario_results

    def _evaluate_recovery_mechanism(
        self, mechanism: str, scenario_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate effectiveness of a recovery mechanism."""
        effectiveness: dict[str, Any] = {
            "mechanism": mechanism,
            "avg_recovery_time": 0.0,
            "success_rate": 0.0,
            "performance_restoration": 0.0,
            "overall_effectiveness": 0.0,
        }

        try:
            recovery_times = []
            success_rates = []
            performance_restoration = []

            for result in scenario_results.values():
                if mechanism in [
                    "redundancy",
                    "adaptation",
                    "reorganization",
                    "restart",
                ]:  # All mechanisms apply
                    recovery_times.append(result.get("recovery_time", 60.0))
                    success_rates.append(
                        1.0 - result.get("performance_degradation", 0.5)
                    )
                    performance_restoration.append(result.get("robustness_score", 0.5))

            if recovery_times:
                effectiveness["avg_recovery_time"] = np.mean(recovery_times)
                effectiveness["success_rate"] = np.mean(success_rates)
                effectiveness["performance_restoration"] = np.mean(
                    performance_restoration
                )

                # Overall effectiveness combines factors (all terms capped at their weight)
                weights = {
                    "recovery_time": 0.3,
                    "success_rate": 0.4,
                    "performance_restoration": 0.3,
                }
                effectiveness["overall_effectiveness"] = min(
                    1.0,
                    (
                        min(1.0, 60.0 / effectiveness["avg_recovery_time"])
                        * weights["recovery_time"]  # Faster is better
                        + effectiveness["success_rate"] * weights["success_rate"]
                        + effectiveness["performance_restoration"]
                        * weights["performance_restoration"]
                    ),
                )

        except Exception as e:
            logger.warning(f"Recovery mechanism evaluation failed: {e}")
            effectiveness["error"] = str(e)

        return effectiveness

    def assess_scalability(
        self,
        swarm_sizes: list[int],
        problem_complexity_levels: list[str],
        computational_resources: dict[str, Any],
        performance_requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Assess system scalability across different configurations.

        Args:
            swarm_sizes: Different swarm sizes to test
            problem_complexity_levels: Different problem complexity levels
            computational_resources: Available computational resources
            performance_requirements: Required performance constraints

        Returns:
            Scalability assessment results
        """
        logger.info(
            f"Assessing scalability for {len(swarm_sizes)} swarm sizes and {len(problem_complexity_levels)} complexity levels"
        )

        scalability_assessment: dict[str, Any] = {
            "assessment_time": datetime.now(),
            "test_configurations": [],
            "scaling_results": {},
            "performance_bottlenecks": [],
            "scaling_recommendations": [],
            "max_scalable_size": 0,
            "overall_scalability_score": 0.0,
        }

        try:
            # Test each configuration
            for swarm_size in swarm_sizes:
                for complexity in problem_complexity_levels:
                    config_result = self._test_scaling_configuration(
                        swarm_size,
                        complexity,
                        computational_resources,
                        performance_requirements,
                    )
                    scalability_assessment["test_configurations"].append(
                        {
                            "swarm_size": swarm_size,
                            "complexity": complexity,
                            "result": config_result,
                        }
                    )

                    # Store results by size and complexity
                    if swarm_size not in scalability_assessment["scaling_results"]:
                        scalability_assessment["scaling_results"][swarm_size] = {}
                    scalability_assessment["scaling_results"][swarm_size][
                        complexity
                    ] = config_result

            # Analyze scaling trends
            scaling_analysis = self._analyze_scaling_trends(
                scalability_assessment["scaling_results"]
            )
            scalability_assessment.update(scaling_analysis)

            # Identify bottlenecks
            bottlenecks = self._identify_performance_bottlenecks(
                scalability_assessment["scaling_results"]
            )
            scalability_assessment["performance_bottlenecks"] = bottlenecks

            # Generate recommendations
            recommendations = self._generate_scaling_recommendations(
                scalability_assessment["scaling_results"], performance_requirements
            )
            scalability_assessment["scaling_recommendations"] = recommendations

        except Exception as e:
            logger.error(f"Scalability assessment failed: {e}")
            scalability_assessment["error"] = str(e)

        logger.info(
            f"Scalability assessment completed: max size = {scalability_assessment['max_scalable_size']}"
        )
        return scalability_assessment

    def _test_scaling_configuration(
        self,
        swarm_size: int,
        complexity: str,
        computational_resources: dict[str, Any],
        performance_requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Test a specific scaling configuration."""
        config_result: dict[str, Any] = {
            "swarm_size": swarm_size,
            "complexity": complexity,
            "performance_score": 0.0,
            "resource_utilization": {},
            "meets_requirements": False,
            "bottlenecks": [],
        }

        base_performance = float(
            performance_requirements.get("baseline_performance", 1.0)
        )
        if base_performance <= 0:
            raise ValueError("baseline_performance must be positive")

        # Size scaling follows the measured exponent when provided.
        size_exponent = float(performance_requirements.get("size_exponent", 0.1))
        size_factor = 1.0 / (1.0 + size_exponent * np.log(max(1, swarm_size)))

        # Complexity scaling
        complexity_factors = {"simple": 1.0, "moderate": 0.8, "complex": 0.6}
        complexity_factor = complexity_factors.get(complexity, 0.7)

        config_result["performance_score"] = (
            base_performance * size_factor * complexity_factor
        )

        agents_per_core = float(
            computational_resources.get("agents_per_cpu_core", 100.0)
        )
        agents_per_gb = float(computational_resources.get("agents_per_gb", 64.0))
        agents_per_bandwidth = float(
            computational_resources.get("agents_per_bandwidth_unit", 1.0)
        )
        cpu_cores = max(float(computational_resources.get("cpu_cores", 1.0)), 1e-9)
        memory_gb = max(float(computational_resources.get("memory_gb", 1.0)), 1e-9)
        bandwidth = max(
            float(computational_resources.get("network_bandwidth", swarm_size)), 1e-9
        )
        config_result["resource_utilization"] = {
            "cpu": swarm_size / (cpu_cores * agents_per_core),
            "memory": swarm_size / (memory_gb * agents_per_gb),
            "network": swarm_size / (bandwidth * agents_per_bandwidth),
        }

        # Check if requirements are met
        meets_cpu = config_result["resource_utilization"]["cpu"] <= 1.0
        meets_memory = config_result["resource_utilization"]["memory"] <= 1.0
        meets_network = config_result["resource_utilization"]["network"] <= 1.0
        meets_performance = config_result[
            "performance_score"
        ] >= performance_requirements.get("min_performance", 0.7)

        config_result["meets_requirements"] = bool(
            meets_cpu and meets_memory and meets_network and meets_performance
        )

        # Identify bottlenecks
        if not meets_cpu:
            config_result["bottlenecks"].append("cpu")
        if not meets_memory:
            config_result["bottlenecks"].append("memory")
        if not meets_network:
            config_result["bottlenecks"].append("network")
        if not meets_performance:
            config_result["bottlenecks"].append("performance")

        return config_result

    def _analyze_scaling_trends(
        self, scaling_results: dict[int, Any]
    ) -> dict[str, Any]:
        """Analyze trends in scaling performance."""
        analysis: dict[str, Any] = {
            "max_scalable_size": 0,
            "scaling_efficiency": 0.0,
            "performance_trends": {},
            "resource_trends": {},
        }

        try:
            # Find the largest size that still meets the configured requirements.
            max_size = 0
            for size in sorted(scaling_results.keys()):
                complexities = scaling_results[size]

                # Check if any complexity level meets requirements
                meets_requirements = any(
                    result.get("meets_requirements", False)
                    for result in complexities.values()
                )

                if meets_requirements:
                    max_size = int(size)
                else:
                    break  # Stop when requirements are no longer met

            analysis["max_scalable_size"] = max_size

            # Analyze scaling efficiency (how performance changes with size)
            if len(scaling_results) > 1:
                sizes = [int(s) for s in sorted(scaling_results.keys())]
                avg_performance = []

                for size in sizes:
                    complexities = scaling_results[size]
                    avg_perf = np.mean(
                        [
                            result.get("performance_score", 0.0)
                            for result in complexities.values()
                        ]
                    )
                    avg_performance.append(avg_perf)

                # Calculate scaling efficiency (negative of performance degradation rate)
                if len(avg_performance) > 1:
                    performance_change = avg_performance[-1] - avg_performance[0]
                    size_change = sizes[-1] - sizes[0]
                    if size_change > 0:
                        scaling_rate = performance_change / size_change
                        analysis["scaling_efficiency"] = 1.0 - abs(
                            scaling_rate
                        )  # Closer to 0 change is better

            # Performance trends by complexity
            for complexity in ["simple", "moderate", "complex"]:
                performances = []
                for size in scaling_results:
                    result = scaling_results[size].get(complexity)
                    if result:
                        performances.append(result.get("performance_score", 0.0))

                if performances:
                    analysis["performance_trends"][complexity] = {
                        "avg_performance": np.mean(performances),
                        "performance_range": (min(performances), max(performances)),
                        "performance_trend": (
                            "stable" if np.std(performances) < 0.1 else "variable"
                        ),
                    }

        except Exception as e:
            logger.warning(f"Scaling trend analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def _identify_performance_bottlenecks(
        self, scaling_results: dict[int, Any]
    ) -> list[dict[str, Any]]:
        """Identify performance bottlenecks in scaling results."""
        bottlenecks = []

        try:
            # Analyze resource utilization patterns
            for size in scaling_results:
                complexities = scaling_results[size]

                for complexity, result in complexities.items():
                    utilization = result.get("resource_utilization", {})

                    # Check each resource type
                    for resource, usage in utilization.items():
                        if usage > 0.9:  # High utilization threshold
                            bottlenecks.append(
                                {
                                    "swarm_size": size,
                                    "complexity": complexity,
                                    "resource": resource,
                                    "utilization": usage,
                                    "severity": "high" if usage > 0.95 else "medium",
                                }
                            )

        except Exception as e:
            logger.warning(f"Bottleneck identification failed: {e}")

        return bottlenecks

    def _generate_scaling_recommendations(
        self, scaling_results: dict[int, Any], performance_requirements: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations for improving scalability."""
        recommendations = []

        try:
            # Check if scaling requirements are met
            max_size = max(scaling_results.keys()) if scaling_results else 0
            required_size = performance_requirements.get("max_swarm_size", 1000)

            if max_size < required_size:
                recommendations.append(
                    {
                        "type": "scaling_limitation",
                        "priority": "high",
                        "description": f"Current max size ({max_size}) < required size ({required_size})",
                        "actions": [
                            "optimize_algorithms",
                            "add_parallel_processing",
                            "implement_spatial_partitioning",
                        ],
                    }
                )

            # Check resource utilization
            high_utilization_resources = set()
            for size in scaling_results:
                for complexity in scaling_results[size]:
                    result = scaling_results[size][complexity]
                    utilization = result.get("resource_utilization", {})

                    for resource, usage in utilization.items():
                        if usage > 0.8:
                            high_utilization_resources.add(resource)

            if high_utilization_resources:
                recommendations.append(
                    {
                        "type": "resource_optimization",
                        "priority": "medium",
                        "description": f"High utilization detected for: {list(high_utilization_resources)}",
                        "actions": [
                            "scale_computational_resources",
                            "optimize_memory_usage",
                            "implement_load_balancing",
                        ],
                    }
                )

        except Exception as e:
            logger.warning(f"Scaling recommendation generation failed: {e}")

        return recommendations

    def generate_performance_report(
        self,
        assessment_results: dict[str, Any],
        report_format: str = "comprehensive",
        include_visualizations: bool = True,
        comparative_analysis: bool = True,
    ) -> dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            assessment_results: Performance assessment results
            report_format: Format of the report ('summary', 'comprehensive', 'detailed')
            include_visualizations: Whether to include visualization data
            comparative_analysis: Whether to include comparative analysis

        Returns:
            Formatted performance report
        """
        logger.info(f"Generating {report_format} performance report")

        report: dict[str, Any] = {
            "report_type": report_format,
            "generation_time": datetime.now(),
            "summary": {},
            "detailed_metrics": {},
            "visualization_data": {},
            "comparative_analysis": {},
            "recommendations": [],
        }

        try:
            # Generate summary
            performance_scores = assessment_results.get("performance_scores", {})
            report["summary"] = {
                "overall_performance": (
                    np.mean(list(performance_scores.values()))
                    if performance_scores
                    else 0.0
                ),
                "best_metric": (
                    max(performance_scores.items(), key=lambda x: x[1])
                    if performance_scores
                    else None
                ),
                "worst_metric": (
                    min(performance_scores.items(), key=lambda x: x[1])
                    if performance_scores
                    else None
                ),
                "performance_balance": (
                    np.std(list(performance_scores.values()))
                    if performance_scores
                    else 0.0
                ),
            }

            # Include detailed metrics based on format
            if report_format in ["comprehensive", "detailed"]:
                report["detailed_metrics"] = assessment_results

            # Include visualization data
            if include_visualizations:
                viz_data = self._generate_visualization_data(assessment_results)
                report["visualization_data"] = viz_data

            # Include comparative analysis
            if comparative_analysis and self.baseline_metrics:
                comparison = self._perform_comparative_analysis(assessment_results)
                report["comparative_analysis"] = comparison

            # Include recommendations
            recommendations = assessment_results.get("recommendations", [])
            report["recommendations"] = recommendations

        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            report["error"] = str(e)

        logger.info(
            f"Performance report generated: {len(report['recommendations'])} recommendations"
        )
        return report

    def _generate_visualization_data(
        self, assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data for performance visualizations."""
        viz_data: dict[str, Any] = {
            "performance_radar": {},
            "timeline_charts": {},
            "comparison_bars": {},
            "trend_lines": {},
        }

        try:
            performance_scores = assessment.get("performance_scores", {})

            # Radar chart data
            viz_data["performance_radar"] = {
                "metrics": list(performance_scores.keys()),
                "scores": list(performance_scores.values()),
                "ideal_scores": [0.9]
                * len(performance_scores),  # Ideal performance levels
            }

            # Timeline data (if historical data available)
            if self.performance_history:
                timeline_metrics = defaultdict(list)
                for record in self.performance_history[-10:]:  # Last 10 records
                    for metric, score in record.get("performance_scores", {}).items():
                        timeline_metrics[metric].append(score)

                viz_data["timeline_charts"] = dict(timeline_metrics)

        except Exception as e:
            logger.warning(f"Visualization data generation failed: {e}")
            viz_data["error"] = str(e)

        return viz_data

    def _perform_comparative_analysis(
        self, current_assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform comparative analysis with baseline metrics."""
        comparison = {
            "baselines": self.baseline_metrics,
            "current_performance": current_assessment.get("performance_scores", {}),
            "improvements": {},
            "regressions": {},
            "overall_comparison": {},
        }

        try:
            current_scores = comparison["current_performance"]

            for metric, current_score in current_scores.items():
                if metric in self.baseline_metrics:
                    baseline_score = self.baseline_metrics[metric]
                    improvement = current_score - baseline_score

                    if improvement > 0.05:  # Significant improvement
                        comparison["improvements"][metric] = improvement
                    elif improvement < -0.05:  # Significant regression
                        comparison["regressions"][metric] = abs(improvement)

            # Overall comparison
            if current_scores and self.baseline_metrics:
                current_avg = np.mean(list(current_scores.values()))
                baseline_avg = np.mean(list(self.baseline_metrics.values()))

                comparison["overall_comparison"] = {
                    "current_average": current_avg,
                    "baseline_average": baseline_avg,
                    "overall_improvement": current_avg - baseline_avg,
                    "improvement_percentage": (
                        ((current_avg - baseline_avg) / baseline_avg) * 100
                        if baseline_avg > 0
                        else 0
                    ),
                }

        except Exception as e:
            logger.warning(f"Comparative analysis failed: {e}")
            comparison["error"] = str(e)

        return comparison

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of all performance evaluations."""
        summary: dict[str, Any] = {
            "total_evaluations": len(self.performance_history),
            "evaluation_criteria": self.config.evaluation_criteria,
            "performance_trends": {},
            "best_performance": {},
            "worst_performance": {},
            "average_performance": {},
        }

        try:
            if not self.performance_history:
                return summary

            # Calculate performance trends
            all_scores = defaultdict(list)
            for evaluation in self.performance_history:
                scores = evaluation.get("performance_scores", {})
                for metric, score in scores.items():
                    all_scores[metric].append(score)

            # Calculate trends for each metric
            for metric, scores in all_scores.items():
                if len(scores) > 1:
                    trend = np.polyfit(range(len(scores)), scores, 1)[0]  # Linear trend
                    summary["performance_trends"][metric] = trend
                else:
                    summary["performance_trends"][metric] = 0.0

            # Find best and worst performances
            recent_evaluation = self.performance_history[-1]
            recent_scores = recent_evaluation.get("performance_scores", {})

            summary["best_performance"] = (
                max(recent_scores.items(), key=lambda x: x[1])
                if recent_scores
                else None
            )
            summary["worst_performance"] = (
                min(recent_scores.items(), key=lambda x: x[1])
                if recent_scores
                else None
            )

            # Calculate averages
            for metric, scores in all_scores.items():
                summary["average_performance"][metric] = np.mean(scores)

        except Exception as e:
            logger.warning(f"Performance summary generation failed: {e}")
            summary["error"] = str(e)

        return summary
