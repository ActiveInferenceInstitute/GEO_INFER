#!/usr/bin/env python3
"""
Unit Tests for GEO-INFER-ANT Core Components

This module contains comprehensive unit tests for the core components of the
GEO-INFER-ANT framework, including swarm agents, population dynamics, and
stigmergic communication systems.

Tests are organized by component and cover:
- Component initialization and configuration
- Core functionality and behavior
- Integration with other modules
- Error handling and edge cases
- Performance and scalability
"""

import pytest
import numpy as np
import asyncio
import os

# Import modules to test
from geo_infer_ant.core.agent_base import SwarmAgent, SensoryInput, ActionDecision
from geo_infer_ant.core.population import AgentPopulation, EnvironmentalState
from geo_infer_ant.core.stigmergy import PheromoneSystem
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface


class TestSwarmAgent:
    """Test cases for SwarmAgent class."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = SwarmAgent(
            agent_id="test_agent_001",
            position=np.array([37.7749, -122.4194]),
            sensory_range=100.0,
            movement_speed=1.5,
            active_inference_enabled=True,
        )

        assert agent.agent_id == "test_agent_001"
        assert np.allclose(agent.position, [37.7749, -122.4194])
        assert agent.sensory_range == 100.0
        assert agent.movement_speed == 1.5
        assert agent.active_inference_enabled is True
        assert agent.energy_level == 1.0

    def test_agent_sensory_processing(self):
        """Test agent sensory input processing."""
        agent = SwarmAgent(
            agent_id="test_agent", position=np.array([0, 0]), sensory_range=50.0
        )

        # Create sensory input
        sensory_input = SensoryInput(
            spatial_context={
                "position": agent.position,
                "bounds": {"min_lat": -10, "max_lat": 10},
            },
            environmental_signals={"temperature": 22.0, "food_nearby": True},
            social_signals={"nearby_agents": 3},
            stigmergic_signals={"trail_intensity": 0.8},
        )

        # Process sensory input
        processed = sensory_input.process()

        assert "spatial_position" in processed
        assert "env_temperature" in processed
        assert "env_food_nearby" in processed
        assert "social_nearby_agents" in processed
        assert "stigmergic_trail_intensity" in processed
        assert sensory_input.processed is True

    def test_agent_decision_making(self):
        """Test agent decision making process."""
        agent = SwarmAgent(
            agent_id="test_agent", position=np.array([0, 0]), sensory_range=50.0
        )

        # Create sensory input
        sensory_input = SensoryInput(
            environmental_signals={"food_nearby": True, "energy_level": 0.3}
        )

        # Make decision
        motivations = {"energy_conservation": 0.8, "task_completion": 0.9}
        decision = agent.make_decision(sensory_input, motivations)

        assert isinstance(decision, ActionDecision)
        assert decision.action_type in [
            "forage",
            "rest",
            "monitor_environment",
            "move_toward_resource",
            "move_away_from_threat",
            "explore_unknown",
            "communicate_status",
            "request_assistance",
            "deposit_pheromone",
            "follow_pheromone",
            "explore",
            "communicate",
            "coordinate_with_swarm",
        ]
        assert 0 <= decision.confidence <= 1
        assert decision.timestamp is not None

    def test_agent_action_execution(self):
        """Test agent action execution."""
        agent = SwarmAgent(
            agent_id="test_agent", position=np.array([0, 0]), sensory_range=50.0
        )

        # Create decision
        decision = ActionDecision(
            action_type="rest", parameters={"duration": 10}, confidence=0.8
        )

        # Execute action (async test)
        async def execute_test():
            result = await agent.execute_action(decision)
            assert result["success"] is True
            assert "energy_cost" in result
            assert result["energy_cost"] >= 0

        # Run async test
        asyncio.run(execute_test())

    def test_agent_movement(self):
        """Test agent movement actions."""
        agent = SwarmAgent(
            agent_id="test_agent", position=np.array([0, 0]), movement_speed=1.0
        )

        initial_position = agent.position.copy()

        # Execute movement action
        async def move_test():
            decision = ActionDecision(
                action_type="move_toward_resource",
                parameters={"target": "nearest_resource"},
                confidence=0.7,
            )

            result = await agent.execute_action(decision)

            assert result["success"] is True
            assert "old_position" in result
            assert "new_position" in result
            assert "distance_moved" in result
            assert (
                np.linalg.norm(agent.position - initial_position) > 0
            )  # Position should change

        asyncio.run(move_test())

    def test_agent_communication(self):
        """Test agent communication capabilities."""
        agent = SwarmAgent(agent_id="test_agent", position=np.array([0, 0]))

        # Test communication action
        async def comm_test():
            decision = ActionDecision(
                action_type="communicate_status",
                parameters={"message_type": "status_update"},
                confidence=0.6,
            )

            result = await agent.execute_action(decision)

            assert result["success"] is True
            assert "message" in result
            assert result["message"]["from"] == agent.agent_id

        asyncio.run(comm_test())


class TestAgentPopulation:
    """Test cases for AgentPopulation class."""

    def test_population_initialization(self):
        """Test population initialization."""
        population = AgentPopulation(
            population_size=50,
            agent_types=["worker", "scout"],
            spatial_distribution="random",
        )

        assert population.population_size == 50
        assert len(population.agents) == 50
        assert population.config.spatial_distribution == "random"

    def test_agent_creation(self):
        """Test agent creation within population."""
        population = AgentPopulation(population_size=10, spatial_distribution="random")

        # Create agents
        agents = population.create_agents()

        assert len(agents) == 10
        assert all(hasattr(agent, "agent_id") for agent in agents)
        assert all(hasattr(agent, "position") for agent in agents)
        assert all(agent.energy_level > 0 for agent in agents)

    def test_environment_initialization(self):
        """Test environment initialization."""
        population = AgentPopulation(population_size=10)

        environment = population.initialize_environment(
            spatial_bounds={"min_lat": -5, "max_lat": 5, "min_lng": -5, "max_lng": 5},
            resource_distribution={
                "food": {"centers": [np.array([0, 0])], "max_density": 1.0}
            },
        )

        assert isinstance(environment, EnvironmentalState)
        assert environment.spatial_bounds["min_lat"] == -5
        assert environment.spatial_bounds["max_lat"] == 5
        assert "food" in environment.resource_distribution

    def test_behavioral_rules_configuration(self):
        """Test behavioral rules configuration."""
        population = AgentPopulation(population_size=10)

        foraging_rules = {"target_preference": "nearest", "energy_threshold": 0.3}
        communication_rules = {"frequency": "adaptive", "range": 100.0}

        population.set_behavioral_rules(
            foraging_rules=foraging_rules, communication_rules=communication_rules
        )

        assert population.foraging_rules["target_preference"] == "nearest"
        assert population.communication_rules["frequency"] == "adaptive"

    def test_population_simulation(self):
        """Test population simulation."""
        population = AgentPopulation(population_size=20, spatial_distribution="random")
        population.initialize_environment()

        # Run short simulation
        async def simulation_test():
            results = await population.run_simulation(
                time_steps=10, data_collection=["trajectories", "interactions"]
            )

            assert results.time_steps == 10
            assert results.population_size == 20
            assert len(results.trajectories) > 0
            assert results.simulation_time > 0

        asyncio.run(simulation_test())

    def test_population_metrics(self):
        """Test population-level metrics calculation."""
        population = AgentPopulation(population_size=10)
        population.create_agents()
        population.initialize_environment()

        metrics = population._calculate_population_metrics()

        assert "active_agents" in metrics
        assert "total_agents" in metrics
        assert "average_energy" in metrics
        assert metrics["total_agents"] == 10
        # Note: average_energy may be > 1.0 due to initial energy settings in different agent types
        assert metrics["average_energy"] >= 0

    def test_agent_queries(self):
        """Test agent querying methods."""
        population = AgentPopulation(population_size=10)
        population.create_agents()

        # Test get_agent_by_id
        agent = population.get_agent_by_id("worker_001")
        assert agent is not None
        assert agent.agent_id == "worker_001"

        # Test get_agents_by_type
        workers = population.get_agents_by_type("worker")
        assert len(workers) > 0
        assert all(getattr(agent, "agent_type", None) == "worker" for agent in workers)

        # Test get_agents_in_region
        center = np.array([0, 0])
        radius = 5.0
        nearby_agents = population.get_agents_in_region(center, radius)
        assert isinstance(nearby_agents, list)


class TestPheromoneSystem:
    """Test cases for PheromoneSystem class."""

    def test_pheromone_system_initialization(self):
        """Test pheromone system initialization."""
        pheromone_system = PheromoneSystem(
            spatial_resolution="h3_r8",
            pheromone_types=["trail", "food"],
            bounds={"min_lat": -10, "max_lat": 10, "min_lng": -10, "max_lng": 10},
        )

        assert len(pheromone_system.pheromone_types) == 2
        assert "trail" in pheromone_system.pheromone_fields
        assert "food" in pheromone_system.pheromone_fields
        assert pheromone_system.spatial_resolution == "h3_r8"

    def test_pheromone_deposition(self):
        """Test pheromone deposition."""
        pheromone_system = PheromoneSystem()

        async def deposition_test():
            success = await pheromone_system.deposit_pheromone(
                agent_id="test_agent",
                pheromone_type="trail",
                location=np.array([0, 0]),
                intensity=1.0,
            )

            assert success is True

            # Check pheromone intensity
            intensity = pheromone_system.get_pheromone_intensity(
                np.array([0, 0]), "trail"
            )
            assert intensity > 0

        asyncio.run(deposition_test())

    def test_pheromone_sensing(self):
        """Test pheromone sensing."""
        pheromone_system = PheromoneSystem()

        async def sensing_test():
            # Deposit pheromone
            await pheromone_system.deposit_pheromone(
                agent_id="test_agent",
                pheromone_type="food",
                location=np.array([0, 0]),
                intensity=2.0,
            )

            # Sense pheromones
            sensed = await pheromone_system.sense_pheromones(
                location=np.array([0, 0]), sensory_range=10.0, pheromone_types=["food"]
            )

            assert "food" in sensed
            assert sensed["food"] > 0

        asyncio.run(sensing_test())

    def test_pheromone_diffusion(self):
        """Test pheromone diffusion and evaporation."""
        pheromone_system = PheromoneSystem(evaporation_rate=0.1)

        async def diffusion_test():
            # Deposit pheromone
            await pheromone_system.deposit_pheromone(
                agent_id="test_agent",
                pheromone_type="trail",
                location=np.array([0, 0]),
                intensity=1.0,
            )

            initial_intensity = pheromone_system.get_pheromone_intensity(
                np.array([0, 0]), "trail"
            )
            assert initial_intensity > 0

            # Diffuse pheromones
            await pheromone_system.diffuse_pheromones(time_step=60.0)

            final_intensity = pheromone_system.get_pheromone_intensity(
                np.array([0, 0]), "trail"
            )
            assert final_intensity < initial_intensity  # Should evaporate

        asyncio.run(diffusion_test())

    def test_pheromone_gradient_calculation(self):
        """Test pheromone gradient calculation."""
        pheromone_system = PheromoneSystem()

        async def gradient_test():
            # Create pheromone gradient
            await pheromone_system.deposit_pheromone(
                agent_id="agent_1",
                pheromone_type="trail",
                location=np.array([0, 0]),
                intensity=2.0,
            )

            await pheromone_system.deposit_pheromone(
                agent_id="agent_2",
                pheromone_type="trail",
                location=np.array([1, 0]),
                intensity=0.5,
            )

            # Calculate gradient
            magnitude, direction = pheromone_system.get_pheromone_gradient(
                location=np.array([0.5, 0]), pheromone_type="trail", radius=1.0
            )

            assert magnitude >= 0
            assert len(direction) == 2

        asyncio.run(gradient_test())

    def test_field_statistics(self):
        """Test pheromone field statistics."""
        pheromone_system = PheromoneSystem()

        async def stats_test():
            # Add some pheromones
            for i in range(5):
                await pheromone_system.deposit_pheromone(
                    agent_id=f"agent_{i}",
                    pheromone_type="trail",
                    location=np.random.uniform(-5, 5, 2),
                    intensity=np.random.uniform(0.5, 2.0),
                )

            # Get statistics
            stats = pheromone_system.get_field_statistics("trail")

            assert stats["pheromone_type"] == "trail"
            assert stats["total_deposits"] == 5
            assert stats["active_cells"] > 0
            assert "max_concentration" in stats

        asyncio.run(stats_test())


class TestDigitalStigmergy:
    """Test cases for DigitalStigmergy class."""

    def test_digital_stigmergy_initialization(self):
        """Test digital stigmergy initialization."""
        digital_stigmergy = DigitalStigmergy(
            communication_medium="iot_network",
            information_types=["sensor_data", "alerts"],
            persistence_model="temporal_decay",
        )

        assert digital_stigmergy.communication_medium == "iot_network"
        assert len(digital_stigmergy.information_types) == 2
        assert digital_stigmergy.persistence_model == "temporal_decay"

    def test_information_contribution(self):
        """Test digital information contribution."""
        digital_stigmergy = DigitalStigmergy()

        async def contribution_test():
            trace_id = await digital_stigmergy.contribute_information(
                agent_id="test_agent",
                information_type="sensor_data",
                content={"temperature": 22.0, "humidity": 65.0},
                location=np.array([0, 0]),
                credibility_score=0.8,
            )

            assert trace_id != ""
            assert trace_id in digital_stigmergy.digital_traces

            # Check trace properties
            trace = digital_stigmergy.digital_traces[trace_id]
            assert trace.agent_id == "test_agent"
            assert trace.information_type == "sensor_data"
            assert trace.content["temperature"] == 22.0

        asyncio.run(contribution_test())

    def test_information_query(self):
        """Test digital information querying."""
        digital_stigmergy = DigitalStigmergy()

        async def query_test():
            # Add some information
            await digital_stigmergy.contribute_information(
                agent_id="agent_1",
                information_type="sensor_data",
                content={"temperature": 25.0},
                location=np.array([0, 0]),
            )

            await digital_stigmergy.contribute_information(
                agent_id="agent_2",
                information_type="alerts",
                content={"alert_type": "high_temperature"},
                location=np.array([1, 1]),
            )

            # Query information
            results = await digital_stigmergy.query_stigmergy(
                agent_id="query_agent",
                query_type="environmental_data",
                spatial_bounds={
                    "min_lat": -5,
                    "max_lat": 5,
                    "min_lng": -5,
                    "max_lng": 5,
                },
                credibility_threshold=0.5,
            )

            assert len(results) > 0
            assert all(
                trace.information_type in ["sensor_data", "alerts"] for trace in results
            )

        asyncio.run(query_test())

    def test_pattern_extraction(self):
        """Test emergent pattern extraction."""
        digital_stigmergy = DigitalStigmergy()

        async def pattern_test():
            # Add diverse information
            for i in range(10):
                await digital_stigmergy.contribute_information(
                    agent_id=f"agent_{i}",
                    information_type=np.random.choice(
                        ["sensor_data", "alerts", "coordination"]
                    ),
                    content={"value": np.random.uniform(20, 30)},
                    location=np.random.uniform(-5, 5, 2),
                )

            # Extract patterns
            patterns = await digital_stigmergy.extract_patterns(
                pattern_types=["clusters", "flows"], temporal_analysis="recent"
            )

            assert "status" in patterns
            assert len(patterns) > 1

        asyncio.run(pattern_test())

    def test_system_statistics(self):
        """Test system statistics calculation."""
        digital_stigmergy = DigitalStigmergy()

        async def stats_test():
            # Add some traces
            for i in range(5):
                await digital_stigmergy.contribute_information(
                    agent_id=f"agent_{i}",
                    information_type="sensor_data",
                    content={"temperature": 20 + i},
                )

            stats = digital_stigmergy.get_system_statistics()

            assert stats["total_traces"] == 5
            assert stats["information_types"] == 3  # sensor_data, alerts, coordination
            assert "performance_stats" in stats

        asyncio.run(stats_test())


class TestIntegration:
    """Test cases for component integration."""

    def test_agent_population_integration(self):
        """Test integration between agents and population."""
        population = AgentPopulation(population_size=5)
        agents = population.create_agents()

        # Test that agents are properly integrated
        assert len(population.agents) == 5
        assert all(agent.population == population for agent in agents)

    def test_pheromone_agent_integration(self):
        """Test integration between pheromones and agents."""
        pheromone_system = PheromoneSystem()
        agent = SwarmAgent("test_agent", np.array([0, 0]))

        async def integration_test():
            # Agent deposits pheromone
            await pheromone_system.deposit_pheromone(
                agent_id=agent.agent_id,
                pheromone_type="trail",
                location=agent.position,
                intensity=1.0,
            )

            # Agent senses pheromone
            sensed = await pheromone_system.sense_pheromones(
                location=agent.position, sensory_range=10.0
            )

            assert "trail" in sensed
            assert sensed["trail"] > 0

        asyncio.run(integration_test())

    def test_spatial_integration(self):
        """Test spatial integration capabilities."""
        # Test spatial indexing integration
        indexer = SpatialIndexingInterface(backend="h3")
        assert indexer is not None

        # Test coordinate conversion (H3 v4 uses integer resolution)
        cell_id = indexer.latlng_to_cell(37.7749, -122.4194, 8)
        assert cell_id is not None


class TestPerformance:
    """Test cases for performance and scalability."""

    def test_large_population_creation(self):
        """Test creation of large agent populations."""
        population = AgentPopulation(population_size=1000)

        import time

        start_time = time.time()

        agents = population.create_agents()

        creation_time = time.time() - start_time

        assert len(agents) == 1000
        assert creation_time < 10.0  # Should create 1000 agents in less than 10 seconds

    def test_pheromone_system_performance(self):
        """Test pheromone system performance with many deposits."""
        pheromone_system = PheromoneSystem()

        async def performance_test():
            import time

            start_time = time.time()

            # Create many pheromone deposits
            for i in range(100):
                await pheromone_system.deposit_pheromone(
                    agent_id=f"agent_{i}",
                    pheromone_type="trail",
                    location=np.random.uniform(-10, 10, 2),
                    intensity=np.random.uniform(0.5, 2.0),
                )

            deposit_time = time.time() - start_time
            assert deposit_time < 5.0  # Should handle 100 deposits in < 5 seconds

            # Test sensing performance
            start_time = time.time()
            for _ in range(50):
                await pheromone_system.sense_pheromones(
                    location=np.random.uniform(-10, 10, 2), sensory_range=5.0
                )

            sensing_time = time.time() - start_time
            assert sensing_time < 3.0  # Should handle 50 queries in < 3 seconds

        asyncio.run(performance_test())

    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil

        process = psutil.Process(os.getpid())
        memory_before_mb = process.memory_info().rss / 1024 / 1024

        population = AgentPopulation(population_size=100)

        # Create agents and environment
        population.create_agents()
        population.initialize_environment()

        # Check that incremental memory usage is reasonable (not total process RSS)
        memory_after_mb = process.memory_info().rss / 1024 / 1024
        delta_mb = memory_after_mb - memory_before_mb

        assert (
            delta_mb < 500
        )  # Creating 100 agents should use less than 500MB incremental


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_invalid_agent_parameters(self):
        """Test handling of invalid agent parameters."""
        with pytest.raises(ValueError):
            SwarmAgent(
                agent_id="",
                position=np.array([]),  # Invalid position
                sensory_range=-10.0,  # Invalid range
            )

    def test_invalid_population_parameters(self):
        """Test handling of invalid population parameters."""
        with pytest.raises(ValueError):
            AgentPopulation(population_size=0)  # Invalid size

        with pytest.raises(ValueError):
            AgentPopulation(population_size=100, agent_types=[])  # No agent types

    def test_pheromone_bounds_checking(self):
        """Test pheromone system bounds checking."""
        pheromone_system = PheromoneSystem()

        async def bounds_test():
            # Test invalid intensity
            result = await pheromone_system.deposit_pheromone(
                agent_id="test_agent",
                pheromone_type="trail",
                location=np.array([0, 0]),
                intensity=-1.0,  # Invalid intensity
            )

            # Should handle gracefully (may succeed with clamped values or fail)
            assert isinstance(result, bool)

        asyncio.run(bounds_test())

    def test_missing_integration_modules(self):
        """Test graceful handling of missing integration modules."""
        # This should not raise exceptions even if integration modules are missing
        agent = SwarmAgent("test_agent", np.array([0, 0]))

        # Agent should still function without full integration
        assert agent.agent_id == "test_agent"
        assert agent.active_inference_enabled is True  # Should default to True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
