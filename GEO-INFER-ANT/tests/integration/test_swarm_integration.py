#!/usr/bin/env python3
"""
Integration Tests for GEO-INFER-ANT Swarm Systems

This module contains integration tests that verify the interaction between
different components of the GEO-INFER-ANT framework, ensuring that the
complete system works together as expected.

Tests cover:
- End-to-end swarm simulations
- Cross-component data flow
- Integration with other GEO-INFER modules
- Performance under realistic conditions
- System-level emergent behaviors
"""

import os
import pytest
import numpy as np
import asyncio
from datetime import datetime
import json
import tempfile

# Import modules to test
from geo_infer_ant.core import (
    SwarmAgent,
    SensoryInput,
    AgentPopulation,
    PheromoneSystem,
    DigitalStigmergy,
)
from geo_infer_ant.algorithms import (
    AntColonyOptimization,
    ParticleSwarmOptimization,  # noqa: F401
)  # noqa: F401
from geo_infer_ant.applications import EnvironmentalMonitoringSwarm
from geo_infer_ant.analysis import SwarmPatternAnalyzer
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_math.core.optimization import Optimizer


class TestEndToEndSimulation:
    """Test complete end-to-end swarm simulations."""

    def test_basic_swarm_simulation(self):
        """Test basic swarm simulation workflow."""

        async def simulation_test():
            # Create population
            population = AgentPopulation(
                population_size=20, spatial_distribution="clustered"
            )

            # Initialize environment
            _environment = population.initialize_environment(
                spatial_bounds={
                    "min_lat": -5,
                    "max_lat": 5,
                    "min_lng": -5,
                    "max_lng": 5,
                }
            )

            # Configure behaviors
            population.set_behavioral_rules(
                foraging_rules={"target_preference": "nearest"},
                communication_rules={"frequency": "adaptive"},
            )

            # Run simulation
            results = await population.run_simulation(
                time_steps=50,
                data_collection=["trajectories", "interactions", "emergent_patterns"],
            )

            # Verify results
            assert results.time_steps == 50
            assert results.population_size == 20
            assert len(results.trajectories) > 0
            assert results.simulation_time > 0

            # Check for emergent patterns
            assert "emergent_patterns" in results.performance_metrics
            assert "summary" in results.performance_metrics

        asyncio.run(simulation_test())

    def test_pheromone_guided_behavior(self):
        """Test swarm behavior guided by pheromone trails."""

        async def pheromone_test():
            # Initialize pheromone system
            pheromone_system = PheromoneSystem(
                pheromone_types=["trail", "food"],
                bounds={"min_lat": -10, "max_lat": 10, "min_lng": -10, "max_lng": 10},
            )

            # Create agents
            agents = []
            for i in range(10):
                agent = SwarmAgent(f"agent_{i}", np.random.uniform(-10, 10, 2))
                agents.append(agent)

            # Simulate pheromone-guided movement
            for agent in agents:
                # Sense pheromones
                sensed = await pheromone_system.sense_pheromones(
                    location=agent.position, sensory_range=5.0
                )

                # Make decision based on pheromones
                if sensed.get("trail", 0) > 0.1:
                    decision = agent.make_decision(
                        SensoryInput(stigmergic_signals={"trail_detected": True}),
                        motivations={"follow_trail": 0.8},
                    )
                    assert decision.action_type in [
                        "follow_pheromone",
                        "move_toward_resource",
                    ]

        asyncio.run(pheromone_test())

    def test_digital_stigmergy_coordination(self):
        """Test digital stigmergy coordination."""

        async def digital_test():
            # Initialize digital stigmergy
            digital_stigmergy = DigitalStigmergy(
                information_types=["task_status", "resource_discovery"]
            )

            # Simulate agent coordination
            agents = [f"agent_{i}" for i in range(5)]

            for agent in agents:
                # Contribute task status
                await digital_stigmergy.contribute_information(
                    agent_id=agent,
                    information_type="task_status",
                    content={"task": "foraging", "status": "active"},
                    location=np.random.uniform(-5, 5, 2),
                )

                # Query for coordination
                coordination_info = await digital_stigmergy.query_stigmergy(
                    agent_id=agent,
                    query_type="task_coordination",
                    credibility_threshold=0.5,
                )

                assert len(coordination_info) >= 0  # May be empty but should not error

        asyncio.run(digital_test())

    def test_optimization_integration(self):
        """Test integration between optimization algorithms and swarm behavior."""
        # Test ACO for path optimization
        aco = AntColonyOptimization(number_of_ants=20, max_iterations=30)

        # Create sample problem
        n_cities = 8
        city_positions = np.random.uniform(-10, 10, (n_cities, 2))

        # Calculate distance matrix
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                distance_matrix[i, j] = np.linalg.norm(
                    city_positions[i] - city_positions[j]
                )

        # Solve optimization problem
        aco.initialize_problem(city_positions.tolist(), distance_matrix)
        result = aco.solve()

        assert result.best_fitness < float("inf")
        assert len(result.best_solution) > 0
        assert result.convergence_achieved or result.iterations_completed > 0

    def test_environmental_monitoring_integration(self):
        """Test environmental monitoring application integration."""

        async def monitoring_test():
            # Create monitoring swarm
            monitoring_swarm = EnvironmentalMonitoringSwarm(
                swarm_size=50,
                monitoring_objectives=["air_quality"],
                adaptive_sampling=True,
            )

            # Deploy agents
            deployment = await monitoring_swarm.deploy_agents(
                environmental_priorities={"air_quality": 0.8}
            )

            assert len(deployment["agents"]) == 50
            assert deployment["coverage_achieved"] > 0

            # Generate sample sensor data
            sensor_readings = []
            for i in range(100):
                reading = {
                    "agent_id": f"agent_{np.random.randint(0, 50)}",
                    "sensor_type": "pm25_sensor",
                    "value": np.random.normal(25, 10),
                    "location": np.random.uniform(-10, 10, 2),
                    "timestamp": datetime.now(),
                    "quality_score": np.random.uniform(0.7, 1.0),
                }
                sensor_readings.append(reading)

            # Process collective intelligence
            assessment = await monitoring_swarm.process_collective_intelligence(
                individual_measurements=sensor_readings, anomaly_detection="statistical"
            )

            assert "data_summary" in assessment
            assert "anomaly_detection" in assessment
            assert len(assessment["recommendations"]) >= 0

        asyncio.run(monitoring_test())


class TestCrossModuleIntegration:
    """Test integration with other GEO-INFER modules."""

    def test_spatial_integration(self):
        """Test integration with GEO-INFER-SPACE."""
        # Test spatial indexing with swarm data
        indexer = SpatialIndexingInterface(backend="h3")

        # Test coordinate conversion
        cell_id = indexer.latlng_to_cell(37.7749, -122.4194, 8)
        assert cell_id is not None

        # Test with swarm positions
        positions = np.random.uniform(-10, 10, (20, 2))
        for pos in positions:
            cell = indexer.latlng_to_cell(pos[0], pos[1], 8)
            assert cell is not None

    def test_act_integration(self):
        """Test integration with GEO-INFER-ACT."""
        # Test Active Inference model creation
        model = ActiveInferenceModel(
            model_type="spatial_temporal", preferences={"forage": 0.8, "rest": 0.6}
        )

        assert model.model_type == "spatial_temporal"
        assert model.preferences is not None

    def test_math_integration(self):
        """Test integration with GEO-INFER-MATH."""
        # Test mathematical optimization integration
        assert issubclass(Optimizer, object)


class TestPerformanceIntegration:
    """Test performance under realistic conditions."""

    def test_large_scale_simulation(self):
        """Test large-scale swarm simulation performance."""

        async def large_scale_test():
            # Create larger population
            population = AgentPopulation(population_size=200)

            import time

            start_time = time.time()

            # Initialize and run
            _agents = population.create_agents()
            _environment = population.initialize_environment()

            creation_time = time.time() - start_time
            assert creation_time < 30.0  # Should create 200 agents in < 30 seconds

            # Run medium-length simulation
            start_time = time.time()
            results = await population.run_simulation(
                time_steps=100, data_collection=["trajectories"]
            )

            simulation_time = time.time() - start_time
            assert simulation_time < 60.0  # Should complete in < 60 seconds

            # Verify results quality
            assert results.time_steps == 100
            assert len(results.trajectories) > 0

        asyncio.run(large_scale_test())

    def test_memory_efficiency(self):
        """Test memory efficiency of large simulations."""
        import psutil
        import os

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and run simulation
        population = AgentPopulation(population_size=100)
        _agents = population.create_agents()
        _environment = population.initialize_environment()

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 200  # Should use less than 200MB additional memory


class TestEmergentBehavior:
    """Test emergent behavior detection and analysis."""

    def test_emergence_detection(self):
        """Test detection of emergent behaviors."""
        # Create pattern analyzer
        analyzer = SwarmPatternAnalyzer()

        # Generate sample trajectories that show clustering
        n_agents = 30
        n_steps = 50
        trajectories = []

        # Create 3 clusters
        cluster_centers = [np.array([-8, -8]), np.array([0, 0]), np.array([8, 8])]

        for agent in range(n_agents):
            cluster_idx = agent % 3
            center = cluster_centers[cluster_idx]

            # Generate trajectory around cluster center
            agent_trajectory = []
            for step in range(n_steps):
                # Movement with attraction to center
                noise = np.random.normal(0, 0.5, 2)
                attraction = -0.1 * (np.random.uniform(-10, 10, 2) - center)
                position = center + noise + attraction
                agent_trajectory.append(position)

            trajectories.append(np.array(agent_trajectory))

        # Analyze patterns
        spatial_analysis = analyzer.analyze_spatial_patterns(
            agent_trajectories=trajectories, pattern_types=["clustering", "flocking"]
        )

        # Should detect clustering
        clustering_result = spatial_analysis["patterns_detected"].get("clustering", {})
        assert clustering_result.get("n_clusters", 0) > 1

        # Test emergence detection
        individual_behaviors = [
            {"agent_id": f"agent_{i}", "position": trajectories[i][0]}
            for i in range(n_agents)
        ]
        collective_outcomes = {
            "clustering_detected": clustering_result.get("n_clusters", 0) > 1
        }

        emergence_analysis = analyzer.detect_emergence(
            individual_behaviors=individual_behaviors,
            collective_outcomes=collective_outcomes,
        )

        # Results should be valid
        assert "emergence_detected" in emergence_analysis
        assert "emergence_measures" in emergence_analysis


class TestDataPersistence:
    """Test data persistence and serialization."""

    def test_simulation_results_serialization(self):
        """Test serialization of simulation results."""
        population = AgentPopulation(population_size=10)

        async def serialization_test():
            results = await population.run_simulation(
                time_steps=20, data_collection=["trajectories", "interactions"]
            )

            # Test serialization
            results_dict = results.to_dict()

            assert "trajectories" in results_dict
            assert "interactions" in results_dict
            assert "simulation_time" in results_dict
            assert "time_steps" in results_dict

            # Test JSON serialization
            json_str = json.dumps(results_dict, default=str)
            assert len(json_str) > 0

            # Test deserialization
            restored_dict = json.loads(json_str)
            assert restored_dict["time_steps"] == 20

        asyncio.run(serialization_test())

    def test_pheromone_system_persistence(self):
        """Test pheromone system persistence."""
        pheromone_system = PheromoneSystem()

        async def persistence_test():
            # Add pheromones
            for i in range(5):
                await pheromone_system.deposit_pheromone(
                    agent_id=f"agent_{i}",
                    pheromone_type="trail",
                    location=np.random.uniform(-5, 5, 2),
                    intensity=1.0,
                )

            # Save state
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                success = pheromone_system.save_pheromone_fields(f.name)
                assert success

                # Load state
                loaded_success = pheromone_system.load_pheromone_fields(f.name)
                assert loaded_success

                # Clean up
                os.unlink(f.name)

        asyncio.run(persistence_test())


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
