#!/usr/bin/env python3
"""
Unit Tests for GEO-INFER-ANT Analysis Tools

This module contains comprehensive unit tests for all analysis and evaluation tools
in the GEO-INFER-ANT framework, including pattern recognition, emergence detection,
and performance metrics.

Tests cover:
- Spatial pattern analysis and recognition
- Temporal pattern analysis and synchronization
- Interaction network analysis and metrics
- Information theory measures (mutual information, transfer entropy)
- Complexity analysis (fractal dimension, Lyapunov exponents)
- Emergent phenomenon detection and interpretation
- Performance metrics and evaluation
- Error handling and edge cases
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta

# Import modules to test
from geo_infer_ant.analysis.patterns import SwarmPatternAnalyzer
from geo_infer_ant.core.stigmergy import PheromoneSystem
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy


class TestSwarmPatternAnalyzer:
    """Test cases for SwarmPatternAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test pattern analyzer initialization."""
        analyzer = SwarmPatternAnalyzer(
            analysis_types=["spatial_patterns", "interaction_networks"],
            statistical_methods=["cluster_analysis", "network_analysis"],
            visualization_tools=["trajectory_plots", "interaction_graphs"],
        )

        assert len(analyzer.config.analysis_types) == 2
        assert "spatial_patterns" in analyzer.config.analysis_types
        assert len(analyzer.config.statistical_methods) == 2
        assert len(analyzer.config.visualization_tools) == 2

    def test_spatial_pattern_analysis(self):
        """Test spatial pattern analysis in agent trajectories."""
        analyzer = SwarmPatternAnalyzer()

        # Generate sample trajectories with clustering pattern
        n_agents = 20
        n_steps = 30
        trajectories = []

        # Create 3 clusters
        cluster_centers = [np.array([-8, -8]), np.array([0, 0]), np.array([8, 8])]

        for agent in range(n_agents):
            cluster_idx = agent % 3
            center = cluster_centers[cluster_idx]

            # Generate trajectory around cluster center
            agent_trajectory = []
            current_pos = center + np.random.normal(0, 1, 2)

            for _ in range(n_steps):
                # Movement with attraction to center
                attraction = -0.1 * (current_pos - center)
                noise = np.random.normal(0, 0.5, 2)
                current_pos += attraction + noise
                agent_trajectory.append(current_pos.copy())

            trajectories.append(np.array(agent_trajectory))

        # Analyze patterns
        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=trajectories,
            pattern_types=["clustering", "flocking"],
            spatial_scale=1000.0,
        )

        assert analysis["analysis_type"] == "spatial_patterns"
        assert "patterns_detected" in analysis
        assert "statistical_measures" in analysis
        assert "interpretation" in analysis

        # Should detect clustering
        clustering_result = analysis["patterns_detected"].get("clustering", {})
        assert clustering_result.get("n_clusters", 0) > 1

    def test_flocking_pattern_detection(self):
        """Test flocking pattern detection."""
        analyzer = SwarmPatternAnalyzer()

        # Generate flocking trajectories
        n_agents = 10
        n_steps = 20
        trajectories = []

        # Initial positions in a group
        initial_positions = np.random.normal(0, 2, (n_agents, 2))

        for agent in range(n_agents):
            trajectory = [initial_positions[agent]]

            for _ in range(n_steps):
                # Calculate average velocity of neighbors
                neighbors = [
                    initial_positions[i] for i in range(n_agents) if i != agent
                ]
                avg_position = np.mean(neighbors, axis=0)

                # Move toward average position (flocking behavior)
                direction = avg_position - trajectory[-1]
                velocity = 0.1 * direction / (np.linalg.norm(direction) + 1e-6)

                # Add some noise
                noise = np.random.normal(0, 0.2, 2)
                new_position = trajectory[-1] + velocity + noise
                trajectory.append(new_position)

            trajectories.append(np.array(trajectory))

        # Analyze flocking patterns
        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=trajectories, pattern_types=["flocking"]
        )

        flocking_result = analysis["patterns_detected"].get("flocking", {})
        assert "flocking_measures" in flocking_result
        assert "flocking_detected" in flocking_result

    def test_flocking_separation_matches_pairwise_bruteforce(self):
        """Vectorized separation must equal the per-pair reference computation."""
        analyzer = SwarmPatternAnalyzer()

        rng = np.random.default_rng(7)
        trajectories = rng.normal(0.0, 3.0, size=(12, 5, 2))

        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=trajectories, pattern_types=["flocking"]
        )
        measures = analysis["patterns_detected"]["flocking"]["flocking_measures"]

        bruteforce = []
        for step in range(trajectories.shape[1]):
            positions = trajectories[:, step, :]
            min_distances = [
                min(
                    np.linalg.norm(positions[i] - positions[j])
                    for j in range(len(positions))
                    if j != i
                )
                for i in range(len(positions))
            ]
            bruteforce.append(np.mean(min_distances))

        assert measures["separation"] == float(np.mean(bruteforce))
        assert measures["separation"] > 0.0

    def test_flocking_analysis_single_agent_reports_degenerate_input(self):
        """Fewer than two agents is reported as a degenerate flocking input."""
        analyzer = SwarmPatternAnalyzer()

        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=np.zeros((1, 4, 2)), pattern_types=["flocking"]
        )

        flocking_result = analysis["patterns_detected"]["flocking"]
        assert flocking_result == {
            "status": "flocking_analysis_failed",
            "error": "flocking analysis requires at least 2 agents; got 1",
        }

    def test_migration_pattern_detection(self):
        """Test migration pattern detection."""
        analyzer = SwarmPatternAnalyzer()

        # Generate migration trajectories
        n_agents = 15
        n_steps = 25
        trajectories = []

        # Start and end positions for migration
        start_positions = np.random.normal(-10, 2, (n_agents, 2))
        end_positions = np.random.normal(10, 2, (n_agents, 2))

        for agent in range(n_agents):
            trajectory = []

            for step in range(n_steps):
                # Linear interpolation with noise (migration path)
                progress = step / (n_steps - 1)
                ideal_position = (1 - progress) * start_positions[
                    agent
                ] + progress * end_positions[agent]

                noise = np.random.normal(0, 0.5, 2)
                position = ideal_position + noise
                trajectory.append(position)

            trajectories.append(np.array(trajectory))

        # Analyze migration patterns
        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=trajectories, pattern_types=["migration"]
        )

        migration_result = analysis["patterns_detected"].get("migration", {})
        assert "migration_measures" in migration_result
        assert "migration_detected" in migration_result

        # Should detect migration
        assert migration_result["migration_detected"] is True

    def test_interaction_network_analysis(self):
        """Test interaction network analysis."""
        analyzer = SwarmPatternAnalyzer()

        # Generate communication data
        communication_data = []
        n_agents = 10

        for _ in range(50):  # 50 communication events
            comm = {
                "from": f"agent_{np.random.randint(0, n_agents)}",
                "to": f"agent_{np.random.randint(0, n_agents)}",
                "type": np.random.choice(["status_update", "alert", "coordination"]),
                "timestamp": datetime.now()
                - timedelta(minutes=np.random.randint(0, 60)),
                "location": np.random.uniform(-10, 10, 2),
            }
            communication_data.append(comm)

        # Generate proximity data
        n_steps = 20
        proximity_data = np.random.uniform(0, 20, (n_steps, n_agents, n_agents))

        # Analyze interactions
        analysis = analyzer.analyze_interactions(
            communication_data=communication_data,
            proximity_data=proximity_data,
            network_metrics=["centrality", "clustering"],
        )

        assert analysis["analysis_type"] == "interaction_networks"
        assert "network_structure" in analysis
        assert "communication_patterns" in analysis
        assert "network_metrics" in analysis

        # Check network properties
        network_props = analysis["network_structure"].get("network_properties", {})
        assert "density" in network_props
        assert "clustering_coefficient" in network_props

    def test_emergence_detection(self):
        """Test emergent phenomenon detection."""
        analyzer = SwarmPatternAnalyzer()

        # Generate individual behaviors
        individual_behaviors = []
        n_agents = 20

        for i in range(n_agents):
            behavior = {
                "agent_id": f"agent_{i}",
                "action_type": np.random.choice(["forage", "rest", "communicate"]),
                "position": np.random.uniform(-10, 10, 2),
                "timestamp": datetime.now()
                - timedelta(minutes=np.random.randint(0, 60)),
            }
            individual_behaviors.append(behavior)

        # Generate collective outcomes
        collective_outcomes = {
            "clustering_observed": True,
            "coordination_efficiency": 0.7,
            "information_flow": 0.8,
        }

        # Detect emergence
        emergence_analysis = analyzer.detect_emergence(
            individual_behaviors=individual_behaviors,
            collective_outcomes=collective_outcomes,
            information_measures=["mutual_information"],
            complexity_measures=["fractal_dimension"],
        )

        assert emergence_analysis["analysis_type"] == "emergent_phenomena"
        assert "emergence_detected" in emergence_analysis
        assert "information_theory" in emergence_analysis
        assert "complexity_analysis" in emergence_analysis
        assert "emergence_interpretation" in emergence_analysis

    def test_mutual_information_calculation(self):
        """Test mutual information calculation."""
        analyzer = SwarmPatternAnalyzer()

        # Generate correlated individual and collective behaviors
        individual_behaviors = []
        collective_outcomes = {}

        # Create correlation: when agents forage collectively, clustering emerges
        for i in range(20):
            foraging_action = np.random.choice(["forage", "rest"], p=[0.7, 0.3])
            behavior = {
                "agent_id": f"agent_{i}",
                "action_type": foraging_action,
                "timestamp": datetime.now(),
            }
            individual_behaviors.append(behavior)

        collective_outcomes = {"clustering_observed": True}

        mi_result = analyzer._calculate_mutual_information(
            individual_behaviors, collective_outcomes
        )

        assert "mutual_information_score" in mi_result
        assert 0 <= mi_result["mutual_information_score"] <= 1

    def test_fractal_dimension_calculation(self):
        """Test fractal dimension calculation."""
        analyzer = SwarmPatternAnalyzer()

        # Generate spatial behavior data
        individual_behaviors = []
        for i in range(30):
            behavior = {
                "agent_id": f"agent_{i}",
                "position": np.random.uniform(-10, 10, 2),
                "timestamp": datetime.now(),
            }
            individual_behaviors.append(behavior)

        fd_result = analyzer._calculate_fractal_dimension(individual_behaviors)

        assert "fractal_dimension" in fd_result
        assert fd_result["fractal_dimension"] > 0

    def test_lyapunov_exponent_calculation(self):
        """Test Lyapunov exponent calculation."""
        analyzer = SwarmPatternAnalyzer()

        # Generate temporal behavior sequences
        individual_behaviors = []
        base_time = datetime.now()

        for i in range(20):
            for t in range(10):  # 10 time steps per agent
                behavior = {
                    "agent_id": f"agent_{i}",
                    "action_type": np.random.choice(["forage", "rest", "move"]),
                    "timestamp": base_time + timedelta(minutes=t),
                }
                individual_behaviors.append(behavior)

        le_result = analyzer._calculate_lyapunov_exponents(individual_behaviors)

        assert "max_lyapunov_exponent" in le_result
        assert "chaos_detected" in le_result

    def test_analysis_caching(self):
        """Test analysis result caching."""
        analyzer = SwarmPatternAnalyzer()

        # Generate test data
        trajectories = [np.random.uniform(-10, 10, (20, 2)) for _ in range(10)]

        # Run analysis twice
        analysis1 = analyzer.analyze_spatial_patterns(trajectories)
        analysis2 = analyzer.analyze_spatial_patterns(trajectories)

        # Should cache results
        assert len(analyzer.pattern_cache) > 0
        assert len(analyzer.analysis_history) == 2

        # Results should be consistent
        assert analysis1["analysis_type"] == analysis2["analysis_type"]

    def test_analysis_summary(self):
        """Test analysis summary generation."""
        analyzer = SwarmPatternAnalyzer()

        # Run multiple analyses
        trajectories = [np.random.uniform(-10, 10, (20, 2)) for _ in range(5)]
        analyzer.analyze_spatial_patterns(trajectories)
        analyzer.analyze_interactions(communication_data=[])

        summary = analyzer.get_analysis_summary()

        assert "total_analyses" in summary
        assert "analysis_types_performed" in summary
        assert summary["total_analyses"] == 2
        assert len(summary["analysis_types_performed"]) > 0


class TestIntegrationWithCoreComponents:
    """Test integration with core swarm components."""

    def test_analyzer_with_pheromone_system(self):
        """Test pattern analyzer integration with pheromone system."""
        pheromone_system = PheromoneSystem(pheromone_types=["trail", "food"])

        _analyzer = SwarmPatternAnalyzer()

        async def integration_test():
            # Add pheromone deposits
            for i in range(20):
                await pheromone_system.deposit_pheromone(
                    agent_id=f"agent_{i}",
                    pheromone_type="trail",
                    location=np.random.uniform(-10, 10, 2),
                    intensity=np.random.uniform(0.5, 2.0),
                )

            # The field is populated through the public spatial deposit API.
            assert pheromone_system.pheromone_types is not None

        asyncio.run(integration_test())

    def test_analyzer_with_digital_stigmergy(self):
        """Test pattern analyzer integration with digital stigmergy."""
        digital_stigmergy = DigitalStigmergy(information_types=["sensor_data"])

        _analyzer = SwarmPatternAnalyzer()

        async def integration_test():
            # Add digital traces
            for i in range(15):
                await digital_stigmergy.contribute_information(
                    agent_id=f"agent_{i}",
                    information_type="sensor_data",
                    content={"temperature": 20 + i},
                    location=np.random.uniform(-5, 5, 2),
                )

            # Extract patterns from digital stigmergy
            patterns = await digital_stigmergy.extract_patterns(
                pattern_types=["clusters", "flows"]
            )

            assert "status" in patterns

        asyncio.run(integration_test())


class TestAnalysisPerformance:
    """Test analysis performance and scalability."""

    def test_large_trajectory_analysis(self):
        """Test analysis performance with large trajectory datasets."""
        analyzer = SwarmPatternAnalyzer()

        # Generate large trajectory dataset
        n_agents = 100
        n_steps = 100
        large_trajectories = []

        for _ in range(n_agents):
            trajectory = []
            for _ in range(n_steps):
                position = np.random.uniform(-50, 50, 2)
                trajectory.append(position)
            large_trajectories.append(np.array(trajectory))

        import time

        start_time = time.time()

        analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=large_trajectories,
            pattern_types=["clustering", "migration"],
        )

        analysis_time = time.time() - start_time

        # Should complete in reasonable time
        assert analysis_time < 30.0  # 30 seconds for large dataset
        assert analysis["analysis_type"] == "spatial_patterns"

    def test_emergence_analysis_performance(self):
        """Test emergence analysis performance with many behaviors."""
        analyzer = SwarmPatternAnalyzer()

        # Generate large behavior dataset
        n_agents = 50
        n_behaviors_per_agent = 20
        individual_behaviors = []

        for agent in range(n_agents):
            for behavior in range(n_behaviors_per_agent):
                behavior_data = {
                    "agent_id": f"agent_{agent}",
                    "action_type": np.random.choice(
                        ["forage", "rest", "communicate", "move"]
                    ),
                    "position": np.random.uniform(-20, 20, 2),
                    "timestamp": datetime.now() - timedelta(minutes=behavior),
                }
                individual_behaviors.append(behavior_data)

        import time

        start_time = time.time()

        emergence_analysis = analyzer.detect_emergence(
            individual_behaviors=individual_behaviors,
            collective_outcomes={"system_efficiency": 0.8},
            information_measures=["mutual_information"],
            complexity_measures=["fractal_dimension"],
        )

        analysis_time = time.time() - start_time

        # Should complete in reasonable time
        assert analysis_time < 15.0  # 15 seconds for complex analysis
        assert "emergence_detected" in emergence_analysis


class TestAnalysisErrorHandling:
    """Test error handling in analysis tools."""

    def test_analysis_with_empty_data(self):
        """Test analysis with empty datasets."""
        analyzer = SwarmPatternAnalyzer()

        # Test with empty trajectories
        empty_analysis = analyzer.analyze_spatial_patterns([])
        assert (
            "error" in empty_analysis
            or empty_analysis["analysis_type"] == "spatial_patterns"
        )

        # Test with insufficient data
        minimal_trajectories = [np.random.uniform(-5, 5, (2, 2))]  # Very small dataset
        minimal_analysis = analyzer.analyze_spatial_patterns(minimal_trajectories)
        assert minimal_analysis["analysis_type"] == "spatial_patterns"

    def test_emergence_with_insufficient_data(self):
        """Test emergence detection with insufficient data."""
        analyzer = SwarmPatternAnalyzer()

        # Test with very few behaviors
        minimal_behaviors = [
            {
                "agent_id": "agent_1",
                "action_type": "forage",
                "timestamp": datetime.now(),
            },
            {"agent_id": "agent_2", "action_type": "rest", "timestamp": datetime.now()},
        ]

        minimal_outcomes = {"efficiency": 0.5}

        emergence_result = analyzer.detect_emergence(
            individual_behaviors=minimal_behaviors, collective_outcomes=minimal_outcomes
        )

        assert "emergence_detected" in emergence_result
        # Should handle gracefully without errors

    def test_network_analysis_with_empty_data(self):
        """Test network analysis with empty communication data."""
        analyzer = SwarmPatternAnalyzer()

        # Test with empty communication data
        analysis = analyzer.analyze_interactions(communication_data=[])

        assert analysis["analysis_type"] == "interaction_networks"
        assert "communication_patterns" in analysis
        assert "network_structure" in analysis


class TestAnalysisValidation:
    """Test analysis result validation and quality."""

    def test_spatial_analysis_validation(self):
        """Test validation of spatial analysis results."""
        analyzer = SwarmPatternAnalyzer()

        # Generate valid trajectory data
        trajectories = [np.random.uniform(-10, 10, (20, 2)) for _ in range(10)]

        analysis = analyzer.analyze_spatial_patterns(trajectories)

        # Validate result structure
        required_keys = [
            "analysis_type",
            "analysis_time",
            "patterns_detected",
            "interpretation",
        ]
        for key in required_keys:
            assert key in analysis

        # Validate pattern detection structure
        patterns = analysis["patterns_detected"]
        for pattern_result in patterns.values():
            assert isinstance(pattern_result, dict)
            if "status" in pattern_result:
                assert pattern_result["status"] in [
                    "success",
                    "failed",
                    "insufficient_data",
                ]

    def test_emergence_analysis_validation(self):
        """Test validation of emergence analysis results."""
        analyzer = SwarmPatternAnalyzer()

        # Generate test data
        behaviors = [
            {"agent_id": f"agent_{i}", "action_type": "forage"} for i in range(10)
        ]
        outcomes = {"coordination": 0.7}

        emergence = analyzer.detect_emergence(behaviors, outcomes)

        # Validate result structure
        required_keys = [
            "analysis_type",
            "emergence_detected",
            "information_theory",
            "emergence_interpretation",
        ]
        for key in required_keys:
            assert key in emergence

        # Validate interpretation structure
        interpretation = emergence["emergence_interpretation"]
        assert "emergence_level" in interpretation
        assert interpretation["emergence_level"] in [
            "detected",
            "not_detected",
            "insufficient_data",
        ]


if __name__ == "__main__":
    # Run analysis tests
    pytest.main([__file__, "-v", "--tb=short"])
