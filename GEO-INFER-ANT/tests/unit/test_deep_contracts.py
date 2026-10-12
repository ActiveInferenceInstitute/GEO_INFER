"""Regression tests for ANT runtime contracts not covered elsewhere."""

import asyncio
import tempfile
from datetime import datetime

import numpy as np
import networkx as nx
import pytest

from geo_infer_ant.algorithms import (
    AntColonyOptimization,
    ArtificialBeeColony,
    ParticleSwarmOptimization,
)
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy
from geo_infer_ant.applications import (
    DisasterResponseSwarm,
    EnvironmentalMonitoringSwarm,
    SensorReading,
    UrbanTrafficSwarm,
)
from geo_infer_ant.utils.config import load_config
from geo_infer_ant.utils.spatial import parse_h3_resolution


def _aco(seed: int) -> AntColonyOptimization:
    optimizer = AntColonyOptimization(
        number_of_ants=8, max_iterations=4, random_seed=seed
    )
    optimizer.initialize_problem(
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        distance_matrix=np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
    )
    return optimizer


def test_aco_uses_matrix_and_private_seeded_rng():
    first = _aco(42).solve()
    second = _aco(42).solve()

    assert first.best_fitness == second.best_fitness
    assert first.best_solution == second.best_solution
    assert first.iterations_completed == 4
    assert first.best_fitness <= 2.0


def test_aco_state_round_trip_accepts_numpy_nodes():
    optimizer = _aco(3)
    optimizer.solve()
    with tempfile.NamedTemporaryFile(suffix=".json") as state_file:
        assert optimizer.save_optimization_state(state_file.name)
        restored = AntColonyOptimization(random_seed=7)
        assert restored.load_optimization_state(state_file.name)
        assert restored.distance_matrix is not None
        assert restored.problem_size == 3
        assert restored.pheromone_matrix == optimizer.pheromone_matrix


def test_pso_constraints_and_coordinate_results_are_real_state():
    optimizer = ParticleSwarmOptimization(
        swarm_size=2,
        dimensions=2,
        max_iterations=2,
        random_seed=42,
        spatial_constraints={"obstacles": [{"center": [0.0, 0.0], "radius": 1.0}]},
    )
    optimizer.initialize_swarm(np.zeros((2, 2)))
    outside = optimizer._avoid_obstacles(np.zeros(2))
    assert np.isclose(np.linalg.norm(outside), 1.0)

    empty = optimizer.coordinate_swarms([ParticleSwarmOptimization(random_seed=1)])
    assert empty["sub_swarm_results"][0]["best_solution"] is None
    assert empty["combined_best_solution"] is None


def test_abc_honors_constructor_iteration_and_seed_controls():
    def sphere(position):
        return float(np.sum(position**2))

    first_optimizer = ArtificialBeeColony(
        colony_size=10, dimensions=2, max_iterations=3, random_seed=9
    )
    second_optimizer = ArtificialBeeColony(
        colony_size=10, dimensions=2, max_iterations=3, random_seed=9
    )
    first = first_optimizer.optimize(sphere)
    second = second_optimizer.optimize(sphere)

    assert len(first_optimizer.convergence_history) == 3
    assert np.allclose(first, second)
    assert np.all(np.isfinite(first))


def test_digital_stigmergy_exact_spatial_filter_and_quality_edge_case():
    digital = DigitalStigmergy()

    async def populate():
        inside = await digital.contribute_information(
            "inside", "sensor_data", {"value": 1}, np.array([0.0, 0.0])
        )
        await digital.contribute_information(
            "outside", "sensor_data", {"value": 2}, np.array([10.0, 10.0])
        )
        results = await digital.query_stigmergy(
            "reader",
            "sensor",
            spatial_bounds={
                "min_lat": -1e-9,
                "max_lat": 1e-9,
                "min_lng": -1e-9,
                "max_lng": 1e-9,
            },
        )
        return inside, results

    inside_id, results = asyncio.run(populate())
    assert [trace.trace_id for trace in results] == [inside_id]


def test_config_example_is_normalized_to_runtime_contract():
    config = load_config("GEO-INFER-ANT/config/example_config.yaml")
    assert config.swarm.population_size == 100
    assert config.spatial.resolution == 8
    assert parse_h3_resolution("h3_r8") == 8


def test_environmental_application_is_seeded_and_spatially_data_driven():
    bounds = {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0}
    first = EnvironmentalMonitoringSwarm(
        swarm_size=4, spatial_coverage=bounds, sensor_range=0.2, random_seed=11
    )
    second = EnvironmentalMonitoringSwarm(
        swarm_size=4, spatial_coverage=bounds, sensor_range=0.2, random_seed=11
    )
    assert np.allclose(
        first._generate_random_position(), second._generate_random_position()
    )
    assert (
        0.0
        < first._calculate_monitoring_coverage(first._generate_grid_positions())
        <= 1.0
    )

    readings = [
        SensorReading(
            agent_id=f"a{index}",
            sensor_type="pm25_sensor",
            value=float(index + 1),
            location=np.array([index / 4, index / 4]),
            timestamp=datetime.now(),
        )
        for index in range(3)
    ]
    spatial = asyncio.run(first._perform_spatial_analysis(readings, "kriging"))
    assert spatial["pm25_sensor"]["interpolated_field"]["method"] == "ordinary_kriging"


def test_urban_application_uses_graph_costs_and_observed_improvements():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", segment_id="ab", travel_time=1.0, congestion=5.0)
    graph.add_edge("B", "D", segment_id="bd", travel_time=1.0, congestion=0.0)
    graph.add_edge("A", "C", segment_id="ac", travel_time=2.0, congestion=0.0)
    graph.add_edge("C", "D", segment_id="cd", travel_time=2.0, congestion=0.0)
    swarm = UrbanTrafficSwarm(vehicle_types=["autonomous_cars"])
    optimization = asyncio.run(
        swarm.optimize_traffic_flow(
            {
                "graph": graph,
                "od_pairs": [{"origin": "A", "destination": "D"}],
                "total_segments": 4,
                "congested_segments": ["ab"],
                "optimized_congestion_rate": 0.1,
                "baseline_flow": 100,
                "optimized_flow": 110,
            }
        )
    )
    route = optimization["route_recommendations"]["optimized_routes"][
        "autonomous_cars"
    ]["primary_routes"][0]
    assert route["segments"] == ["ac", "cd"]
    assert optimization["flow_improvements"]["congestion_reduction"] == 0.15
    assert optimization["flow_improvements"]["throughput_increase"] == pytest.approx(
        0.1
    )


def test_disaster_application_tracks_resources_and_elapsed_time():
    swarm = DisasterResponseSwarm(
        swarm_composition={"drones": 4, "ground_vehicles": 2, "human_teams": 2}
    )
    assessment = asyncio.run(
        swarm.assess_situation(
            "earthquake",
            {"min_lat": 0, "max_lat": 1, "min_lng": 0, "max_lng": 1},
            available_resources={
                "drones": 4,
                "ground_vehicles": 2,
                "human_teams": 2,
                "supplies": 10,
            },
            environmental_conditions={
                "population": 250,
                "vulnerability_factors": ["density"],
            },
        )
    )
    assert assessment["priority_zones"][0]["estimated_population"] == 250.0
    status = swarm.get_response_status()
    assert 0.0 <= status["time_remaining"] <= 7200.0
    coordination = asyncio.run(swarm.coordinate_response(assessment))
    assert coordination["coordination_metrics"]["resource_utilization"] > 0.0
    assert swarm.get_response_status()["resource_status"]["utilization"] > 0.0
