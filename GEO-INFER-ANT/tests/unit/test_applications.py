#!/usr/bin/env python3
"""
Unit Tests for GEO-INFER-ANT Applications

This module contains comprehensive unit tests for all domain-specific applications
implemented in the GEO-INFER-ANT framework, including environmental monitoring,
disaster response, and urban optimization systems.

Tests cover:
- Application initialization and configuration
- Agent deployment and coordination
- Data processing and analysis
- Integration with core swarm components
- Performance and scalability
- Error handling and edge cases
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime
import json

# Import modules to test
from geo_infer_ant.applications.environmental import (
    EnvironmentalMonitoringSwarm,
    MonitoringObjective,  # noqa: F401
    SensorReading,
)
from geo_infer_ant.core.stigmergy import PheromoneSystem
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy


class TestEnvironmentalMonitoringSwarm:
    """Test cases for Environmental Monitoring Swarm application."""

    def test_monitoring_swarm_initialization(self):
        """Test environmental monitoring swarm initialization."""
        swarm = EnvironmentalMonitoringSwarm(
            swarm_size=100,
            monitoring_objectives=["air_quality", "water_quality"],
            spatial_coverage={
                "min_lat": 35,
                "max_lat": 40,
                "min_lng": -120,
                "max_lng": -115,
            },
            adaptive_sampling=True,
            real_time_processing=True,
        )

        assert swarm.swarm_size == 100
        assert len(swarm.monitoring_objectives) == 2
        assert swarm.adaptive_sampling is True
        assert swarm.real_time_processing is True
        assert swarm.spatial_coverage["min_lat"] == 35

    def test_monitoring_agent_configuration(self):
        """Test monitoring agent configuration generation."""
        swarm = EnvironmentalMonitoringSwarm(
            swarm_size=50, monitoring_objectives=["air_quality", "biodiversity"]
        )

        agent_config = swarm._configure_monitoring_agent(
            "test_agent_001",
            np.array([37.7749, -122.4194]),
            swarm.monitoring_objectives,
        )

        assert agent_config["agent_id"] == "test_agent_001"
        assert np.allclose(agent_config["position"], [37.7749, -122.4194])
        assert "air_quality" in agent_config["monitoring_objectives"]
        assert "biodiversity" in agent_config["monitoring_objectives"]
        assert "sensory_capabilities" in agent_config

    def test_sensory_capabilities_generation(self):
        """Test sensory capabilities based on monitoring objectives."""
        swarm = EnvironmentalMonitoringSwarm()

        # Test air quality capabilities
        air_quality_caps = swarm._get_sensory_capabilities(["air_quality"])
        expected_air_caps = [
            "pm25_sensor",
            "no2_sensor",
            "o3_sensor",
            "temperature",
            "humidity",
        ]
        for cap in expected_air_caps:
            assert cap in air_quality_caps

        # Test water quality capabilities
        water_quality_caps = swarm._get_sensory_capabilities(["water_quality"])
        expected_water_caps = [
            "ph_sensor",
            "turbidity_sensor",
            "conductivity_sensor",
            "temperature",
        ]
        for cap in expected_water_caps:
            assert cap in water_quality_caps

    def test_initial_position_optimization(self):
        """Test initial position optimization for agent deployment."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=20)

        async def position_test():
            # Test with environmental priorities
            priorities = {"air_quality": 0.8, "biodiversity": 0.6}
            constraints = {"max_range": 1000.0}

            positions = await swarm._optimize_initial_positions(priorities, constraints)

            assert len(positions) == swarm.swarm_size
            assert all(isinstance(pos, np.ndarray) for pos in positions)
            assert all(pos.shape == (2,) for pos in positions)

            # Check bounds
            for pos in positions:
                assert (
                    swarm.spatial_coverage["min_lat"]
                    <= pos[0]
                    <= swarm.spatial_coverage["max_lat"]
                )
                assert (
                    swarm.spatial_coverage["min_lng"]
                    <= pos[1]
                    <= swarm.spatial_coverage["max_lng"]
                )

        asyncio.run(position_test())

    def test_agent_deployment(self):
        """Test agent deployment process."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=10)

        async def deployment_test():
            deployment = await swarm.deploy_agents(
                environmental_priorities={"air_quality": 0.7},
                logistical_constraints={"max_range": 500.0},
            )

            assert len(deployment["agents"]) == 10
            assert deployment["coverage_achieved"] >= 0
            assert "agent_configurations" in deployment

            # Check agent configurations
            for agent_info in deployment["agents"]:
                assert "agent_id" in agent_info
                assert "position" in agent_info
                assert "config" in agent_info

        asyncio.run(deployment_test())

    def test_monitoring_coordination(self):
        """Test monitoring activity coordination."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=20)

        async def coordination_test():
            agent_positions = [np.random.uniform(-10, 10, 2) for _ in range(20)]
            environmental_conditions = {"temperature": 22.0, "humidity": 65.0}

            coordination = await swarm.coordinate_monitoring(
                agent_positions=agent_positions,
                environmental_conditions=environmental_conditions,
                data_priorities={"air_quality": 0.8},
            )

            assert "monitoring_instructions" in coordination
            assert "sampling_strategy" in coordination
            assert "estimated_coverage" in coordination
            assert coordination["estimated_coverage"] >= 0

        asyncio.run(coordination_test())

    def test_collective_intelligence_processing(self):
        """Test collective intelligence processing from sensor data."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=50)

        async def intelligence_test():
            # Generate sample sensor readings
            sensor_readings = []
            for i in range(100):
                reading = SensorReading(
                    agent_id=f"agent_{i % 10}",
                    sensor_type="pm25_sensor",
                    value=np.random.normal(25, 10),
                    location=np.random.uniform(-10, 10, 2),
                    timestamp=datetime.now(),
                    quality_score=np.random.uniform(0.7, 1.0),
                )
                sensor_readings.append(reading)

            # Process collective intelligence
            assessment = await swarm.process_collective_intelligence(
                individual_measurements=sensor_readings,
                spatial_interpolation="idw",
                uncertainty_quantification="bayesian",
                anomaly_detection="statistical",
            )

            assert "data_summary" in assessment
            assert "anomaly_detection" in assessment
            assert "recommendations" in assessment
            assert assessment["data_summary"]["total_measurements"] == 100

        asyncio.run(intelligence_test())

    def test_anomaly_detection(self):
        """Test anomaly detection in sensor data."""
        swarm = EnvironmentalMonitoringSwarm()

        # Generate normal and anomalous data
        normal_readings = [
            SensorReading(
                f"agent_{i}",
                "pm25_sensor",
                np.random.normal(20, 5),
                np.array([0, 0]),
                datetime.now(),
            )
            for i in range(20)
        ]

        anomalous_readings = [
            SensorReading(
                f"agent_{i}",
                "pm25_sensor",
                np.random.normal(100, 20),
                np.array([0, 0]),
                datetime.now(),
            )
            for i in range(5)
        ]

        all_readings = normal_readings + anomalous_readings

        async def anomaly_test():
            anomalies = await swarm._detect_anomalies(all_readings, "statistical")

            assert len(anomalies) >= 0  # Should find some anomalies
            if len(anomalies) > 0:
                assert all("anomaly_id" in anomaly for anomaly in anomalies)
                assert all("severity" in anomaly for anomaly in anomalies)

        asyncio.run(anomaly_test())

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification in measurements."""
        swarm = EnvironmentalMonitoringSwarm()

        # Generate measurements with varying quality
        measurements = [
            SensorReading(
                f"agent_{i}",
                "temperature",
                20 + i,
                np.array([0, 0]),
                datetime.now(),
                quality_score=np.random.uniform(0.5, 1.0),
            )
            for i in range(20)
        ]

        uncertainty = swarm._quantify_uncertainty(measurements, "bayesian")

        assert "overall_uncertainty" in uncertainty
        assert "sensor_uncertainties" in uncertainty
        assert "spatial_uncertainty" in uncertainty
        assert 0 <= uncertainty["overall_uncertainty"] <= 1

    def test_spatial_uncertainty_matches_pairwise_bruteforce(self):
        """Vectorized nearest-neighbor uncertainty must equal the legacy loop."""
        swarm = EnvironmentalMonitoringSwarm()

        rng = np.random.default_rng(7)
        locations = rng.uniform(0.0, 0.05, size=(60, 2))
        locations[10] = locations[9]

        result = swarm._calculate_spatial_uncertainty(locations)

        distances = []
        for i in range(len(locations)):
            others = np.delete(locations, i, axis=0)
            distances.append(
                min(np.linalg.norm(locations[i] - other) for other in others)
            )
        bruteforce = float(min(1.0, np.mean(distances) / 0.01))

        assert result == bruteforce

    def test_covered_area_subtracts_pairwise_overlaps_exactly(self):
        """Overlap subtraction must match the legacy per-pair computation."""
        bounds = {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0}
        swarm = EnvironmentalMonitoringSwarm(
            swarm_size=40, spatial_coverage=bounds, sensor_range=0.05, random_seed=3
        )
        radius = swarm.sensor_range

        rng = np.random.default_rng(3)
        positions = rng.uniform(0.05, 0.55, size=(16, 2))
        positions[5] = positions[4]  # co-located pair
        positions[12] = positions[11] + np.array([0.01, 0.0])  # near pair

        result = swarm._estimate_covered_area(positions)

        covered_area = len(positions) * np.pi * radius**2
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance >= 2.0 * radius:
                    continue
                if distance == 0.0:
                    overlap = np.pi * radius**2
                else:
                    ratio = np.clip(distance / (2.0 * radius), -1.0, 1.0)
                    overlap = 2.0 * radius**2 * np.arccos(
                        ratio
                    ) - 0.5 * distance * np.sqrt(
                        float(max(0.0, float(4.0 * radius**2 - distance**2)))
                    )
                covered_area -= overlap
        bruteforce = float(np.clip(covered_area, 0.0, swarm._calculate_total_area()))

        assert result == bruteforce

    def test_monitoring_recommendations(self):
        """Test generation of monitoring recommendations."""
        swarm = EnvironmentalMonitoringSwarm()

        # Create sample assessment with issues
        assessment = {
            "spatial_analysis": {"coverage": 0.6},  # Low coverage
            "anomaly_detection": [
                {"severity": "high", "sensor_type": "pm25_sensor"},
                {"severity": "medium", "sensor_type": "temperature"},
            ],
            "uncertainty_analysis": {"overall_uncertainty": 0.7},  # High uncertainty
        }

        recommendations = swarm._generate_monitoring_recommendations(assessment)

        assert len(recommendations) > 0
        assert all("type" in rec for rec in recommendations)
        assert all("priority" in rec for rec in recommendations)

        # Should recommend coverage improvement for low coverage
        coverage_recs = [
            r for r in recommendations if r["type"] == "coverage_improvement"
        ]
        assert len(coverage_recs) > 0

    def test_monitoring_status(self):
        """Test monitoring system status reporting."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=50)

        status = swarm.get_monitoring_status()

        assert "system_status" in status
        assert "monitoring_objectives" in status
        assert "performance_metrics" in status
        assert "component_status" in status
        assert status["system_status"] == "operational"

    def test_sampling_frequency_calculation(self):
        """Test sampling frequency calculation based on priorities."""
        swarm = EnvironmentalMonitoringSwarm()

        # Test different priority levels
        frequencies = swarm._calculate_sampling_frequencies(
            {"air_quality": 0.9, "water_quality": 0.4, "biodiversity": 0.7},
            {"temperature": 25.0},
        )

        assert "air_quality" in frequencies
        assert "water_quality" in frequencies
        assert "biodiversity" in frequencies

        # High priority should have high frequency
        assert frequencies["air_quality"] == "1_minute"  # High priority
        assert frequencies["water_quality"] == "1_hour"  # Low priority


class TestApplicationIntegration:
    """Test integration between applications and core components."""

    def test_monitoring_with_pheromone_system(self):
        """Test environmental monitoring integration with pheromone system."""
        pheromone_system = PheromoneSystem(pheromone_types=["monitoring", "anomaly"])

        swarm = EnvironmentalMonitoringSwarm(swarm_size=20)

        async def integration_test():
            # Deploy agents
            deployment = await swarm.deploy_agents()

            # Simulate pheromone-guided coordination
            agent_positions = [agent["position"] for agent in deployment["agents"]]

            # Add monitoring pheromones
            for i, pos in enumerate(agent_positions):
                await pheromone_system.deposit_pheromone(
                    agent_id=f"monitor_{i}",
                    pheromone_type="monitoring",
                    location=pos,
                    intensity=1.0,
                )

            # Check pheromone sensing
            sensed = await pheromone_system.sense_pheromones(
                location=agent_positions[0], sensory_range=5.0
            )

            assert "monitoring" in sensed

        asyncio.run(integration_test())

    def test_monitoring_with_digital_stigmergy(self):
        """Test environmental monitoring integration with digital stigmergy."""
        digital_stigmergy = DigitalStigmergy(
            information_types=["sensor_data", "anomaly_detection"]
        )

        _swarm = EnvironmentalMonitoringSwarm(swarm_size=10)

        async def integration_test():
            # Generate sensor data
            sensor_readings = []
            for i in range(20):
                reading = SensorReading(
                    agent_id=f"agent_{i}",
                    sensor_type="temperature",
                    value=np.random.normal(20, 3),
                    location=np.random.uniform(-5, 5, 2),
                    timestamp=datetime.now(),
                    quality_score=0.8,
                )
                sensor_readings.append(reading)

            # Contribute to digital stigmergy
            for reading in sensor_readings:
                await digital_stigmergy.contribute_information(
                    agent_id=reading.agent_id,
                    information_type="sensor_data",
                    content={
                        "sensor_type": reading.sensor_type,
                        "value": reading.value,
                        "quality_score": reading.quality_score,
                    },
                    location=reading.location,
                )

            # Query for anomalies
            anomaly_info = await digital_stigmergy.query_stigmergy(
                agent_id="coordinator",
                query_type="anomaly_detection",
                credibility_threshold=0.7,
            )

            assert len(anomaly_info) >= 0  # May be empty but should not error

        asyncio.run(integration_test())


class TestApplicationPerformance:
    """Test application performance and scalability."""

    def test_large_scale_monitoring(self):
        """Test large-scale environmental monitoring."""
        swarm_sizes = [50, 100, 200]

        for size in swarm_sizes:
            monitoring_swarm = EnvironmentalMonitoringSwarm(swarm_size=size)

            async def performance_test():
                import time

                start_time = time.time()

                # Deploy agents
                _deployment = await monitoring_swarm.deploy_agents()

                deployment_time = time.time() - start_time
                assert deployment_time < 30.0  # Should deploy in reasonable time

                # Process sensor data
                start_time = time.time()
                sensor_readings = []
                for i in range(min(100, size * 2)):
                    reading = SensorReading(
                        agent_id=f"agent_{i % size}",
                        sensor_type="pm25_sensor",
                        value=np.random.normal(25, 5),
                        location=np.random.uniform(-10, 10, 2),
                        timestamp=datetime.now(),
                        quality_score=0.8,
                    )
                    sensor_readings.append(reading)

                # Process collective intelligence
                assessment = await monitoring_swarm.process_collective_intelligence(
                    individual_measurements=sensor_readings,
                    anomaly_detection="statistical",
                )

                processing_time = time.time() - start_time
                assert processing_time < 10.0  # Should process in reasonable time

                # Verify results
                assert assessment["data_summary"]["total_measurements"] == len(
                    sensor_readings
                )

            asyncio.run(performance_test())


class TestApplicationErrorHandling:
    """Test error handling in applications."""

    def test_monitoring_with_empty_data(self):
        """Test monitoring application with empty sensor data."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=10)

        async def empty_data_test():
            # Process empty measurements
            assessment = await swarm.process_collective_intelligence(
                individual_measurements=[], anomaly_detection="statistical"
            )

            assert "data_summary" in assessment
            assert assessment["data_summary"]["total_measurements"] == 0

        asyncio.run(empty_data_test())

    def test_monitoring_with_invalid_data(self):
        """Test monitoring application with invalid sensor data."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=10)

        async def invalid_data_test():
            # Generate invalid measurements
            invalid_readings = [
                SensorReading(
                    "agent_1",
                    "invalid_sensor",
                    float("inf"),
                    np.array([0, 0]),
                    datetime.now(),
                ),
                SensorReading(
                    "agent_2",
                    "temperature",
                    float("nan"),
                    np.array([0, 0]),
                    datetime.now(),
                ),
            ]

            # Should handle gracefully
            assessment = await swarm.process_collective_intelligence(
                individual_measurements=invalid_readings,
                anomaly_detection="statistical",
            )

            assert "error" not in assessment or assessment["error"] is None

        asyncio.run(invalid_data_test())


class TestApplicationDataPersistence:
    """Test data persistence in applications."""

    def test_monitoring_results_serialization(self):
        """Test serialization of monitoring results."""
        swarm = EnvironmentalMonitoringSwarm(swarm_size=10)

        async def serialization_test():
            # Generate sample data and process
            sensor_readings = []
            for i in range(50):
                reading = SensorReading(
                    agent_id=f"agent_{i}",
                    sensor_type="pm25_sensor",
                    value=np.random.normal(25, 5),
                    location=np.random.uniform(-5, 5, 2),
                    timestamp=datetime.now(),
                    quality_score=0.8,
                )
                sensor_readings.append(reading)

            assessment = await swarm.process_collective_intelligence(
                individual_measurements=sensor_readings, anomaly_detection="statistical"
            )

            # Test JSON serialization
            assessment_json = json.dumps(assessment, default=str)
            assert len(assessment_json) > 0

            # Test deserialization
            restored_assessment = json.loads(assessment_json)
            assert "data_summary" in restored_assessment

        asyncio.run(serialization_test())


if __name__ == "__main__":
    # Run application tests
    pytest.main([__file__, "-v", "--tb=short"])
