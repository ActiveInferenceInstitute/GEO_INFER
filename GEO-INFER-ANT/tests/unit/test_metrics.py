#!/usr/bin/env python3
"""
Unit Tests for GEO-INFER-ANT Performance Metrics

This module contains comprehensive unit tests for the performance metrics and
evaluation framework in the GEO-INFER-ANT system, including efficiency analysis,
robustness testing, scalability assessment, and statistical validation.

Tests cover:
- Performance evaluation across multiple criteria
- Robustness analysis under failure scenarios
- Scalability testing with different system sizes
- Statistical analysis and hypothesis testing
- Report generation and visualization
- Error handling and edge cases
"""

import pytest

# Import modules to test
from geo_infer_ant.analysis.metrics import SwarmPerformanceMetrics


class TestPerformanceMetrics:
    """Test cases for SwarmPerformanceMetrics class."""

    def test_metrics_initialization(self):
        """Test performance metrics system initialization."""
        metrics = SwarmPerformanceMetrics(
            evaluation_criteria=["efficiency", "robustness", "adaptability"],
            statistical_analysis=["hypothesis_testing", "confidence_intervals"],
        )

        assert len(metrics.config.evaluation_criteria) == 3
        assert "efficiency" in metrics.config.evaluation_criteria
        assert len(metrics.config.statistical_analysis) == 2
        assert len(metrics.performance_history) == 0

    def test_efficiency_evaluation(self):
        """Test efficiency performance evaluation."""
        metrics = SwarmPerformanceMetrics()

        # Create sample swarm behavior data
        swarm_behavior = {
            "task_completion": {"completed": 80, "total": 100},
            "resource_usage": {"utilization_rate": 0.75},
            "energy_metrics": {"total_consumed": 1000, "useful_energy": 800},
            "communication_metrics": {"successful": 95, "total": 100},
            "computation_metrics": {"efficiency_score": 0.9},
        }

        task_objectives = {"completion_target": 0.8, "efficiency_target": 0.7}

        efficiency = metrics._evaluate_efficiency(swarm_behavior, task_objectives)

        assert "overall_score" in efficiency
        assert 0 <= efficiency["overall_score"] <= 1
        assert efficiency["task_completion_rate"] == 0.8
        assert efficiency["energy_efficiency"] == 0.8

    def test_robustness_evaluation(self):
        """Test robustness performance evaluation."""
        metrics = SwarmPerformanceMetrics()

        swarm_behavior = {
            "failure_scenarios": {"recovered": 18, "total": 20},
            "communication_failures": {"failure_rate": 0.1},
            "task_disruptions": {"completion_rate": 0.85},
        }

        environmental_conditions = {"volatility": 0.2, "noise_level": 0.1}

        robustness = metrics._evaluate_robustness(
            swarm_behavior, environmental_conditions
        )

        assert "overall_score" in robustness
        assert 0 <= robustness["overall_score"] <= 1
        assert robustness["failure_recovery"] == 0.9
        assert robustness["environmental_resilience"] == 0.85

    def test_adaptability_evaluation(self):
        """Test adaptability performance evaluation."""
        metrics = SwarmPerformanceMetrics()

        swarm_behavior = {
            "learning_metrics": {"improvement_rate": 0.15},
            "parameter_changes": {"adaptation_effectiveness": 0.8},
            "behavioral_changes": {"flexibility_score": 0.7},
        }

        environmental_conditions = {"change_frequency": 0.3}

        adaptability = metrics._evaluate_adaptability(
            swarm_behavior, environmental_conditions
        )

        assert "overall_score" in adaptability
        assert 0 <= adaptability["overall_score"] <= 1
        assert adaptability["learning_rate"] == 0.15
        assert adaptability["environmental_adaptation"] == 0.7

    def test_scalability_evaluation(self):
        """Test scalability performance evaluation."""
        metrics = SwarmPerformanceMetrics()

        swarm_behavior = {
            "size_performance": {"scaling_efficiency": 0.8},
            "problem_scaling": {"scaling_efficiency": 0.75},
            "communication_scaling": {"scaling_efficiency": 0.9},
            "memory_scaling": {"scaling_efficiency": 0.85},
        }

        scalability = metrics._evaluate_scalability(swarm_behavior)

        assert "overall_score" in scalability
        assert 0 <= scalability["overall_score"] <= 1
        assert scalability["size_scalability"] == 0.8

    def test_comprehensive_performance_evaluation(self):
        """Test comprehensive performance evaluation."""
        metrics = SwarmPerformanceMetrics(
            evaluation_criteria=[
                "efficiency",
                "robustness",
                "adaptability",
                "scalability",
            ]
        )

        # Create comprehensive test data
        swarm_behavior = {
            "task_completion": {"completed": 90, "total": 100},
            "resource_usage": {"utilization_rate": 0.8},
            "energy_metrics": {"total_consumed": 1200, "useful_energy": 1000},
            "communication_metrics": {"successful": 98, "total": 100},
            "computation_metrics": {"efficiency_score": 0.85},
            "failure_scenarios": {"recovered": 15, "total": 20},
            "learning_metrics": {"improvement_rate": 0.2},
            "size_performance": {"scaling_efficiency": 0.9},
        }

        task_objectives = {"completion_target": 0.85, "efficiency_target": 0.8}
        environmental_conditions = {"volatility": 0.15, "change_frequency": 0.2}

        assessment = metrics.evaluate_performance(
            swarm_behavior=swarm_behavior,
            task_objectives=task_objectives,
            environmental_conditions=environmental_conditions,
        )

        assert "evaluation_time" in assessment
        assert "performance_scores" in assessment
        assert len(assessment["performance_scores"]) == 4  # All criteria evaluated
        assert "recommendations" in assessment
        assert len(metrics.performance_history) == 1


class TestRobustnessAnalysis:
    """Test cases for robustness analysis."""

    def test_robustness_analysis(self):
        """Test comprehensive robustness analysis."""
        metrics = SwarmPerformanceMetrics()

        failure_scenarios = [
            "agent_loss",
            "communication_failure",
            "environmental_change",
        ]
        recovery_mechanisms = ["redundancy", "adaptation", "reorganization"]
        performance_degradation = {
            "critical_threshold": 0.8,
            "acceptable_threshold": 0.5,
        }

        analysis = metrics.analyze_robustness(
            failure_scenarios=failure_scenarios,
            recovery_mechanisms=recovery_mechanisms,
            performance_degradation=performance_degradation,
        )

        assert "analysis_time" in analysis
        assert "overall_robustness" in analysis
        assert len(analysis["scenario_results"]) == len(failure_scenarios)
        assert len(analysis["recovery_effectiveness"]) == len(recovery_mechanisms)

        # Check scenario results
        for scenario in failure_scenarios:
            assert scenario in analysis["scenario_results"]
            scenario_result = analysis["scenario_results"][scenario]
            assert "robustness_score" in scenario_result
            assert 0 <= scenario_result["robustness_score"] <= 1

    def test_single_failure_scenario_analysis(self):
        """Test analysis of individual failure scenarios."""
        metrics = SwarmPerformanceMetrics()

        scenario = "communication_failure"
        recovery_mechanisms = ["redundancy", "restart"]
        performance_degradation = {"critical_threshold": 0.8}

        result = metrics._analyze_single_failure_scenario(
            scenario, recovery_mechanisms, performance_degradation
        )

        assert result["scenario"] == scenario
        assert "failure_impact" in result
        assert "recovery_time" in result
        assert "robustness_score" in result
        assert 0 <= result["robustness_score"] <= 1

    def test_recovery_mechanism_evaluation(self):
        """Test evaluation of recovery mechanisms."""
        metrics = SwarmPerformanceMetrics()

        mechanism = "redundancy"
        scenario_results = {
            "agent_loss": {
                "recovery_time": 15,
                "performance_degradation": 0.2,
                "robustness_score": 0.8,
            },
            "communication_failure": {
                "recovery_time": 10,
                "performance_degradation": 0.1,
                "robustness_score": 0.9,
            },
        }

        effectiveness = metrics._evaluate_recovery_mechanism(
            mechanism, scenario_results
        )

        assert effectiveness["mechanism"] == mechanism
        assert "avg_recovery_time" in effectiveness
        assert "success_rate" in effectiveness
        assert "overall_effectiveness" in effectiveness
        assert 0 <= effectiveness["overall_effectiveness"] <= 1


class TestScalabilityAssessment:
    """Test cases for scalability assessment."""

    def test_scalability_assessment(self):
        """Test comprehensive scalability assessment."""
        metrics = SwarmPerformanceMetrics()

        swarm_sizes = [50, 100, 200, 500]
        problem_complexities = ["simple", "moderate", "complex"]
        computational_resources = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "network_bandwidth": 1000,
        }
        performance_requirements = {"min_performance": 0.7, "max_swarm_size": 1000}

        assessment = metrics.assess_scalability(
            swarm_sizes=swarm_sizes,
            problem_complexity_levels=problem_complexities,
            computational_resources=computational_resources,
            performance_requirements=performance_requirements,
        )

        assert "assessment_time" in assessment
        assert "test_configurations" in assessment
        assert len(assessment["test_configurations"]) == len(swarm_sizes) * len(
            problem_complexities
        )
        assert "scaling_results" in assessment
        assert "max_scalable_size" in assessment
        assert "overall_scalability_score" in assessment

    def test_scaling_configuration_testing(self):
        """Test testing of individual scaling configurations."""
        metrics = SwarmPerformanceMetrics()

        config_result = metrics._test_scaling_configuration(
            swarm_size=100,
            complexity="moderate",
            computational_resources={"cpu_cores": 4, "memory_gb": 8},
            performance_requirements={"min_performance": 0.7, "max_latency": 1.0},
        )

        assert config_result["swarm_size"] == 100
        assert config_result["complexity"] == "moderate"
        assert "performance_score" in config_result
        assert "resource_utilization" in config_result
        assert "meets_requirements" in config_result
        assert isinstance(config_result["meets_requirements"], bool)

    def test_scaling_trend_analysis(self):
        """Test analysis of scaling trends."""
        metrics = SwarmPerformanceMetrics()

        # Create measured scaling-result fixtures
        scaling_results = {
            50: {
                "simple": {"performance_score": 0.9, "meets_requirements": True},
                "moderate": {"performance_score": 0.85, "meets_requirements": True},
            },
            100: {
                "simple": {"performance_score": 0.85, "meets_requirements": True},
                "moderate": {"performance_score": 0.8, "meets_requirements": True},
            },
            200: {
                "simple": {"performance_score": 0.8, "meets_requirements": True},
                "moderate": {"performance_score": 0.7, "meets_requirements": False},
            },
        }

        trends = metrics._analyze_scaling_trends(scaling_results)

        assert "max_scalable_size" in trends
        assert "scaling_efficiency" in trends
        assert "performance_trends" in trends
        assert (
            trends["max_scalable_size"] == 200
        )  # Should find max size that meets requirements

    def test_bottleneck_identification(self):
        """Test identification of performance bottlenecks."""
        metrics = SwarmPerformanceMetrics()

        scaling_results = {
            100: {
                "simple": {
                    "resource_utilization": {"cpu": 0.95, "memory": 0.6, "network": 0.7}
                }
            },
            200: {
                "simple": {
                    "resource_utilization": {"cpu": 0.98, "memory": 0.8, "network": 0.9}
                }
            },
        }

        bottlenecks = metrics._identify_performance_bottlenecks(scaling_results)

        # Should identify CPU as bottleneck
        cpu_bottlenecks = [b for b in bottlenecks if b["resource"] == "cpu"]
        assert len(cpu_bottlenecks) > 0


class TestStatisticalAnalysis:
    """Test cases for statistical analysis."""

    def test_statistical_analysis(self):
        """Test statistical analysis of performance metrics."""
        metrics = SwarmPerformanceMetrics()

        assessment = {
            "performance_scores": {
                "efficiency": 0.8,
                "robustness": 0.75,
                "adaptability": 0.9,
                "scalability": 0.7,
            }
        }

        analysis = metrics._perform_statistical_analysis(assessment)

        assert "confidence_intervals" in analysis
        assert "statistical_significance" in analysis
        assert len(analysis["confidence_intervals"]) == 4  # One for each metric

    def test_recommendation_generation(self):
        """Test performance recommendation generation."""
        metrics = SwarmPerformanceMetrics()

        assessment = {
            "performance_scores": {
                "efficiency": 0.6,  # Below threshold
                "robustness": 0.9,  # Good
                "adaptability": 0.5,  # Below threshold
                "scalability": 0.8,  # Good
            }
        }

        recommendations = metrics._generate_performance_recommendations(assessment)

        # Should generate recommendations for low-performing metrics
        efficiency_recs = [r for r in recommendations if r["category"] == "efficiency"]
        adaptability_recs = [
            r for r in recommendations if r["category"] == "adaptability"
        ]

        assert len(efficiency_recs) > 0
        assert len(adaptability_recs) > 0


class TestReportGeneration:
    """Test cases for performance report generation."""

    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        metrics = SwarmPerformanceMetrics()

        # Create sample assessment
        assessment = {
            "performance_scores": {
                "efficiency": 0.8,
                "robustness": 0.75,
                "adaptability": 0.6,
                "scalability": 0.9,
            },
            "recommendations": [
                {
                    "type": "efficiency",
                    "priority": "medium",
                    "description": "Improve efficiency",
                },
                {
                    "type": "adaptability",
                    "priority": "high",
                    "description": "Enhance learning",
                },
            ],
        }

        report = metrics.generate_performance_report(
            assessment_results=assessment,
            report_format="comprehensive",
            include_visualizations=True,
            comparative_analysis=True,
        )

        assert "report_type" in report
        assert "generation_time" in report
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "visualization_data" in report
        assert "recommendations" in report

        # Check summary
        summary = report["summary"]
        assert "overall_performance" in summary
        assert summary["overall_performance"] > 0

    def test_visualization_data_generation(self):
        """Test visualization data generation."""
        metrics = SwarmPerformanceMetrics()

        assessment = {
            "performance_scores": {
                "efficiency": 0.8,
                "robustness": 0.75,
                "adaptability": 0.6,
            }
        }

        viz_data = metrics._generate_visualization_data(assessment)

        assert "performance_radar" in viz_data
        assert "metrics" in viz_data["performance_radar"]
        assert "scores" in viz_data["performance_radar"]
        assert len(viz_data["performance_radar"]["metrics"]) == 3


class TestPerformanceHistory:
    """Test cases for performance history management."""

    def test_performance_history_tracking(self):
        """Test performance history tracking."""
        metrics = SwarmPerformanceMetrics()

        # Run multiple evaluations
        for i in range(3):
            swarm_behavior = {
                "task_completion": {"completed": 70 + i * 10, "total": 100},
                "energy_metrics": {
                    "total_consumed": 1000,
                    "useful_energy": 700 + i * 50,
                },
            }

            _assessment = metrics.evaluate_performance(
                swarm_behavior=swarm_behavior,
                task_objectives={"completion_target": 0.8},
            )

        # Check history
        assert len(metrics.performance_history) == 3
        assert all(
            "performance_scores" in evaluation
            for evaluation in metrics.performance_history
        )

    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        metrics = SwarmPerformanceMetrics()

        # Run multiple evaluations with different scores
        for i in range(5):
            swarm_behavior = {
                "task_completion": {"completed": 60 + i * 8, "total": 100},
                "resource_usage": {"utilization_rate": 0.5 + i * 0.1},
            }

            _assessment = metrics.evaluate_performance(
                swarm_behavior=swarm_behavior,
                task_objectives={"completion_target": 0.8},
            )

        summary = metrics.get_performance_summary()

        assert "total_evaluations" in summary
        assert summary["total_evaluations"] == 5
        assert "performance_trends" in summary
        assert "average_performance" in summary

        # Should have trend data for evaluated metrics
        assert len(summary["performance_trends"]) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_performance_evaluation(self):
        """Test performance evaluation with empty data."""
        metrics = SwarmPerformanceMetrics()

        empty_behavior = {}
        empty_objectives = {}

        assessment = metrics.evaluate_performance(
            swarm_behavior=empty_behavior, task_objectives=empty_objectives
        )

        # Should handle gracefully
        assert "performance_scores" in assessment
        assert len(assessment["performance_scores"]) == 0

    def test_scalability_with_single_size(self):
        """Test scalability assessment with single swarm size."""
        metrics = SwarmPerformanceMetrics()

        assessment = metrics.assess_scalability(
            swarm_sizes=[100],
            problem_complexity_levels=["simple"],
            computational_resources={"cpu_cores": 4},
            performance_requirements={"min_performance": 0.7},
        )

        assert "test_configurations" in assessment
        assert len(assessment["test_configurations"]) == 1
        assert "max_scalable_size" in assessment

    def test_robustness_with_no_failures(self):
        """Test robustness analysis with no failure scenarios."""
        metrics = SwarmPerformanceMetrics()

        analysis = metrics.analyze_robustness(
            failure_scenarios=[],
            recovery_mechanisms=["redundancy"],
            performance_degradation={"critical_threshold": 0.8},
        )

        assert "scenario_results" in analysis
        assert len(analysis["scenario_results"]) == 0
        assert analysis["overall_robustness"] == 0.0


if __name__ == "__main__":
    # Run metrics tests
    pytest.main([__file__, "-v", "--tb=short"])
