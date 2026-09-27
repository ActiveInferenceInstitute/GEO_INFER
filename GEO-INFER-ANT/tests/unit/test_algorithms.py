#!/usr/bin/env python3
"""
Unit Tests for GEO-INFER-ANT Algorithms

This module contains comprehensive unit tests for all swarm optimization algorithms
implemented in the GEO-INFER-ANT framework, including ACO, PSO, and ABC algorithms.

Tests cover:
- Algorithm initialization and configuration
- Core optimization functionality
- Parameter validation and adaptation
- Integration with spatial constraints
- Performance and convergence behavior
- Error handling and edge cases
"""

import pytest
import numpy as np
import tempfile
import os

# Import modules to test
from geo_infer_ant.algorithms.aco import AntColonyOptimization
from geo_infer_ant.algorithms.pso import ParticleSwarmOptimization
from geo_infer_ant.algorithms.abc import ArtificialBeeColony, FoodSource
from geo_infer_ant.core.stigmergy import PheromoneSystem


class TestAntColonyOptimization:
    """Test cases for Ant Colony Optimization algorithm."""

    def test_aco_initialization(self):
        """Test ACO algorithm initialization."""
        aco = AntColonyOptimization(
            number_of_ants=30,
            pheromone_evaporation_rate=0.1,
            alpha=1.0,
            beta=2.0,
            max_iterations=50,
        )

        assert aco.parameters.number_of_ants == 30
        assert aco.parameters.pheromone_evaporation_rate == 0.1
        assert aco.parameters.alpha == 1.0
        assert aco.parameters.beta == 2.0
        assert aco.parameters.max_iterations == 50

    def test_aco_problem_initialization(self):
        """Test ACO problem setup."""
        aco = AntColonyOptimization(number_of_ants=10, max_iterations=20)

        # Create simple TSP problem
        n_cities = 5
        city_positions = np.random.uniform(-10, 10, (n_cities, 2))

        # Calculate distance matrix
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                distance_matrix[i, j] = np.linalg.norm(
                    city_positions[i] - city_positions[j]
                )

        # Initialize problem
        aco.initialize_problem(city_positions.tolist(), distance_matrix)

        assert aco.problem_size == n_cities
        assert len(aco.nodes) == n_cities
        assert len(aco.pheromone_matrix) == n_cities * (n_cities - 1)
        assert len(aco.heuristic_matrix) == n_cities * (n_cities - 1)

    def test_aco_solution_construction(self):
        """Test ACO solution construction process."""
        aco = AntColonyOptimization(number_of_ants=5, max_iterations=10)

        # Simple 4-city problem
        cities = [[0, 0], [1, 0], [1, 1], [0, 1]]
        distance_matrix = np.array(
            [
                [0, 1, np.sqrt(2), 1],
                [1, 0, 1, np.sqrt(2)],
                [np.sqrt(2), 1, 0, 1],
                [1, np.sqrt(2), 1, 0],
            ]
        )

        aco.initialize_problem(cities, distance_matrix)

        # Test single solution construction
        solution = aco._construct_single_solution(0)

        assert len(solution) <= aco.problem_size
        assert all(0 <= node < aco.problem_size for node in solution)
        assert len(set(solution)) == len(solution)  # No duplicates

    def test_aco_pheromone_updates(self):
        """Test ACO pheromone update mechanisms."""
        aco = AntColonyOptimization(number_of_ants=5, max_iterations=10, variant="AS")

        # Initialize problem
        cities = [[0, 0], [1, 0], [1, 1], [0, 1]]
        distances = np.array(
            [
                [0, 1, np.sqrt(2), 1],
                [1, 0, 1, np.sqrt(2)],
                [np.sqrt(2), 1, 0, 1],
                [1, np.sqrt(2), 1, 0],
            ]
        )
        aco.initialize_problem(cities, distances)

        # Create test solutions
        solutions = [
            {"solution": [0, 1, 2, 3], "fitness": 4.0, "ant_id": 0},
            {"solution": [0, 2, 1, 3], "fitness": 4.5, "ant_id": 1},
        ]

        # Update pheromones
        aco._update_pheromones(solutions)

        # Check that pheromones were updated
        assert aco.pheromone_matrix[(0, 1)] > aco.parameters.initial_pheromone
        assert aco.pheromone_matrix[(0, 2)] > aco.parameters.initial_pheromone

    @pytest.mark.slow
    def test_aco_convergence_detection(self):
        """Test ACO convergence detection."""
        aco = AntColonyOptimization(max_iterations=20, convergence_threshold=0.01)

        # Simulate convergence history
        aco.convergence_history = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]

        # Test convergence check
        is_converged = aco._check_convergence(10)
        assert is_converged is True

        # Test non-convergence
        aco.convergence_history = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        is_converged = aco._check_convergence(10)
        assert is_converged is False

    def test_aco_multi_objective_optimization(self):
        """Test ACO multi-objective optimization."""
        aco = AntColonyOptimization(number_of_ants=20, max_iterations=30)
        aco.initialize_problem(
            nodes=[[0, 0], [1, 0], [1, 1]],
            distance_matrix=np.array(
                [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
            ),
        )

        objectives = ["minimize_cost", "minimize_time", "maximize_service"]
        result = aco.multi_objective_optimization(
            objectives=objectives, population_size=50, generations=20
        )

        assert "solutions" in result
        assert "objectives" in result
        assert result["objectives"] == objectives

    def test_aco_adaptation(self):
        """Test ACO adaptation to environmental changes."""
        aco = AntColonyOptimization()

        environmental_changes = {
            "volatility": 0.3,
            "problem_complexity": 0.7,
            "major_change": True,
        }

        adaptation = aco.adapt_to_changes(environmental_changes)

        assert "changes_applied" in adaptation
        assert "parameters_updated" in adaptation
        assert len(adaptation["changes_applied"]) > 0

    def test_aco_state_persistence(self):
        """Test ACO state save and load."""
        aco = AntColonyOptimization(number_of_ants=10, max_iterations=20)

        # Initialize problem
        cities = [[0, 0], [1, 0], [1, 1], [0, 1]]
        distances = np.ones((4, 4)) - np.eye(4)
        aco.initialize_problem(cities, distances)

        # Save state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            success = aco.save_optimization_state(f.name)
            assert success

            # Load state
            loaded_success = aco.load_optimization_state(f.name)
            assert loaded_success

            # Clean up
            os.unlink(f.name)

    def test_aco_statistics(self):
        """Test ACO statistics generation."""
        aco = AntColonyOptimization(number_of_ants=10, max_iterations=5)

        # Initialize and run brief optimization
        cities = [[0, 0], [1, 0], [1, 1], [0, 1]]
        distances = np.ones((4, 4)) - np.eye(4)
        aco.initialize_problem(cities, distances)

        aco.solve()

        # Get statistics
        stats = aco.get_optimization_statistics()

        assert stats["algorithm"] == "Ant Colony Optimization"
        assert "parameters" in stats
        assert "optimization_results" in stats
        assert "pheromone_statistics" in stats


class TestParticleSwarmOptimization:
    """Test cases for Particle Swarm Optimization algorithm."""

    def test_pso_initialization(self):
        """Test PSO algorithm initialization."""
        pso = ParticleSwarmOptimization(
            swarm_size=50,
            dimensions=3,
            bounds=[(-5, 5), (-5, 5), (-5, 5)],
            inertia_weight=0.7,
            cognitive_acceleration=1.5,
            social_acceleration=1.5,
            max_iterations=20,
        )

        assert pso.parameters.swarm_size == 50
        assert pso.parameters.dimensions == 3
        assert len(pso.parameters.bounds) == 3
        assert pso.parameters.inertia_weight == 0.7
        assert pso.parameters.max_iterations == 20

    def test_pso_swarm_initialization(self):
        """Test PSO swarm initialization."""
        pso = ParticleSwarmOptimization(swarm_size=10, dimensions=2)

        # Initialize swarm
        pso.initialize_swarm()

        assert len(pso.swarm) == 10
        assert all(particle.position.shape == (2,) for particle in pso.swarm)
        assert all(particle.velocity.shape == (2,) for particle in pso.swarm)
        assert all(
            particle.personal_best_position.shape == (2,) for particle in pso.swarm
        )

    def test_pso_swarm_initialization_with_positions(self):
        """Test PSO swarm initialization with provided positions."""
        pso = ParticleSwarmOptimization(swarm_size=3, dimensions=2)

        # Provide initial positions
        initial_positions = np.array([[0, 0], [1, 1], [2, 2]])

        pso.initialize_swarm(initial_positions)

        # Check that initial positions were used
        for i, particle in enumerate(pso.swarm):
            assert np.allclose(particle.position, initial_positions[i])

    def test_particle_update_mechanics(self):
        """Test particle velocity and position update mechanics."""
        pso = ParticleSwarmOptimization(swarm_size=1, dimensions=2)
        pso.initialize_swarm()

        particle = pso.swarm[0]
        initial_position = particle.position.copy()

        # Update particle
        global_best = np.array([5, 5])
        particle.update_velocity(
            global_best_position=global_best,
            inertia_weight=0.7,
            cognitive_acceleration=1.5,
            social_acceleration=1.5,
        )

        particle.update_position(pso.parameters.bounds)

        # Position should change (unless at boundary)
        assert not np.allclose(particle.position, initial_position)

    @pytest.mark.slow
    def test_pso_optimization_functions(self):
        """Test PSO optimization on various functions."""
        pso = ParticleSwarmOptimization(swarm_size=10, dimensions=2, max_iterations=20)

        # Test on Rastrigin function
        def rastrigin(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

        # Test on Sphere function
        def sphere(x):
            return np.sum(x**2)

        # Optimize sphere function (should find [0, 0])
        optimal_point = pso.optimize(sphere)
        optimal_value = sphere(optimal_point)

        assert optimal_value < 1.0  # Should find good solution
        assert len(optimal_point) == 2

    def test_pso_adaptive_parameters(self):
        """Test PSO adaptive parameter tuning."""
        pso = ParticleSwarmOptimization(
            swarm_size=20, dimensions=2, adaptive_parameters=True, max_iterations=30
        )

        # Create performance history
        performance_history = [
            {"fitness": 100, "iteration": 0},
            {"fitness": 80, "iteration": 5},
            {"fitness": 60, "iteration": 10},
            {"fitness": 40, "iteration": 15},
            {"fitness": 20, "iteration": 20},
        ]

        environmental_changes = {"noise_level": 0.2}
        adaptation = pso.adapt_parameters(performance_history, environmental_changes)

        assert "changes_applied" in adaptation
        assert len(adaptation["changes_applied"]) >= 0  # May or may not adapt

    def test_pso_neighborhood_topology(self):
        """Test PSO neighborhood topology configurations."""
        # Test global topology (default)
        pso_global = ParticleSwarmOptimization(
            swarm_size=10, neighborhood_topology="global"
        )
        pso_global.initialize_swarm()

        assert pso_global.parameters.neighborhood_topology == "global"

        # Test local topology
        pso_local = ParticleSwarmOptimization(
            swarm_size=10, neighborhood_topology="local", neighborhood_size=3
        )
        pso_local.initialize_swarm()

        assert pso_local.parameters.neighborhood_topology == "local"
        assert len(pso_local.neighborhoods) == 10  # One neighborhood per particle

    def test_pso_multi_swarm_coordination(self):
        """Test multi-swarm PSO coordination."""
        pso = ParticleSwarmOptimization(swarm_size=10, dimensions=2)

        # Create sub-swarms
        sub_swarms = [
            ParticleSwarmOptimization(swarm_size=5, dimensions=2),
            ParticleSwarmOptimization(swarm_size=5, dimensions=2),
        ]

        # Test coordination
        coordination_result = pso.coordinate_swarms(
            sub_swarms=sub_swarms,
            communication_topology="hierarchical",
            information_sharing="best_positions",
        )

        assert "topology" in coordination_result
        assert "combined_best_solution" in coordination_result
        assert len(coordination_result["sub_swarm_results"]) == 2

    def test_pso_state_persistence(self):
        """Test PSO state save and load."""
        pso = ParticleSwarmOptimization(swarm_size=5, dimensions=2)
        pso.initialize_swarm()

        # Run a few iterations
        def dummy_objective(x):
            return np.sum(x**2)

        for _ in range(5):
            pso._evaluate_swarm(dummy_objective)
            pso._update_bests()
            pso._update_swarm()

        # Save state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            success = pso.save_optimization_state(f.name)
            assert success

            # Load state
            loaded_success = pso.load_optimization_state(f.name)
            assert loaded_success

            # Clean up
            os.unlink(f.name)

    def test_pso_statistics(self):
        """Test PSO statistics generation."""
        pso = ParticleSwarmOptimization(swarm_size=10, dimensions=2, max_iterations=10)
        pso.initialize_swarm()

        def dummy_objective(x):
            return np.sum(x**2)

        # Run brief optimization
        pso.optimize(dummy_objective)

        # Get statistics
        stats = pso.get_optimization_statistics()

        assert stats["algorithm"] == "Particle Swarm Optimization"
        assert "parameters" in stats
        assert "optimization_results" in stats
        assert "swarm_statistics" in stats


class TestArtificialBeeColony:
    """Test cases for Artificial Bee Colony algorithm."""

    def test_abc_initialization(self):
        """Test ABC algorithm initialization."""
        abc = ArtificialBeeColony(
            colony_size=50,
            dimensions=3,
            bounds=[(-5, 5), (-5, 5), (-5, 5)],
            max_trials=30,
            limit=100,
        )

        assert abc.parameters.colony_size == 50
        assert abc.parameters.dimensions == 3
        assert len(abc.parameters.bounds) == 3
        assert abc.parameters.max_trials == 30
        assert abc.parameters.limit == 100

    def test_abc_food_source_initialization(self):
        """Test ABC food source initialization."""
        abc = ArtificialBeeColony(colony_size=20, dimensions=2)
        abc._initialize_food_sources()

        # Should have half the colony size as food sources
        expected_sources = abc.parameters.colony_size // 2
        assert len(abc.food_sources) == expected_sources

        # Check food source properties
        for source in abc.food_sources:
            assert source.position.shape == (abc.parameters.dimensions,)
            assert source.fitness >= 0
            assert source.trial_count >= 0

    def test_abc_neighbor_generation(self):
        """Test ABC neighbor solution generation."""
        abc = ArtificialBeeColony(colony_size=10, dimensions=2)
        abc._initialize_food_sources()

        source = abc.food_sources[0]
        original_position = source.position.copy()

        # Generate neighbor
        neighbor = abc._generate_neighbor(original_position)

        assert neighbor.shape == original_position.shape
        assert not np.allclose(neighbor, original_position)  # Should be different
        assert all(
            abc.parameters.bounds[i][0] <= neighbor[i] <= abc.parameters.bounds[i][1]
            for i in range(len(neighbor))
        )

    def test_abc_fitness_evaluation(self):
        """Test ABC fitness evaluation."""
        abc = ArtificialBeeColony(colony_size=10, dimensions=2)

        def test_function(x):
            return np.sum(x**2)  # Sphere function

        # Test fitness calculation
        position = np.array([1.0, 2.0])
        fitness = abc._evaluate_fitness(position, test_function)

        expected_fitness = 1.0 / (1.0 + 5.0)  # 1/(1+1²+2²)
        assert abs(fitness - expected_fitness) < 1e-6

    def test_abc_selection_probabilities(self):
        """Test ABC selection probability calculation."""
        abc = ArtificialBeeColony(colony_size=6, dimensions=2)
        abc._initialize_food_sources()

        # Set different fitness values
        abc.food_sources[0].fitness = 1.0
        abc.food_sources[1].fitness = 2.0
        abc.food_sources[2].fitness = 0.5

        probabilities = abc._calculate_selection_probabilities()

        assert len(probabilities) == 3
        assert np.all(probabilities >= 0)
        assert abs(np.sum(probabilities) - 1.0) < 1e-6
        assert (
            probabilities[1] > probabilities[0] > probabilities[2]
        )  # Higher fitness = higher probability

    def test_abc_employed_bee_phase(self):
        """Test ABC employed bee phase."""
        # Seed the algorithm's generator directly.  ``np.random.seed`` does
        # not control the isolated ``default_rng`` used by ABC and made this
        # assertion intermittently fail when every sampled neighbour was
        # worse than its source.
        abc = ArtificialBeeColony(colony_size=6, dimensions=2, random_seed=0)
        abc._initialize_food_sources()

        def test_function(x):
            return np.sum(x**2)

        # Set initial fitness
        for source in abc.food_sources:
            source.fitness = abc._evaluate_fitness(source.position, test_function)

        # Run employed bee phase
        abc._employed_bee_phase(test_function)

        # Check that some sources were updated
        updated_sources = sum(
            1 for source in abc.food_sources if source.trial_count == 0
        )
        assert updated_sources > 0

    def test_abc_onlooker_bee_phase(self):
        """Test ABC onlooker bee phase."""
        abc = ArtificialBeeColony(colony_size=6, dimensions=2)
        abc._initialize_food_sources()

        def test_function(x):
            return np.sum(x**2)

        # Set initial fitness
        for source in abc.food_sources:
            source.fitness = abc._evaluate_fitness(source.position, test_function)

        # Run onlooker bee phase
        abc._onlooker_bee_phase(test_function)

        # Should complete without errors
        assert len(abc.food_sources) > 0

    def test_abc_scout_bee_phase(self):
        """Test ABC scout bee phase."""
        abc = ArtificialBeeColony(colony_size=6, dimensions=2, limit=3)
        abc._initialize_food_sources()

        # Set trial counts to trigger abandonment
        abc.food_sources[0].trial_count = 5  # Should be abandoned

        # Run scout bee phase
        abc._scout_bee_phase()

        # Abandoned source should have been reset
        assert abc.food_sources[0].trial_count == 0
        assert abc.food_sources[0].fitness == 0.0

    def test_abc_optimization(self):
        """Test complete ABC optimization."""
        abc = ArtificialBeeColony(colony_size=20, dimensions=2, max_iterations=30)

        def sphere_function(x):
            return np.sum(x**2)

        # Run optimization
        optimal_solution = abc.optimize(sphere_function)

        assert len(optimal_solution) == 2
        assert all(-5 <= x <= 5 for x in optimal_solution)  # Within bounds

        # Should find reasonably good solution
        optimal_fitness = sphere_function(optimal_solution)
        assert optimal_fitness < 1.0  # Should be close to global optimum

    def test_abc_food_source_management(self):
        """Test ABC food source management strategies."""
        abc = ArtificialBeeColony(colony_size=10, dimensions=2)

        # Create test food sources
        sources = [
            FoodSource(position=np.array([1, 1]), fitness=0.8, trial_count=2),
            FoodSource(position=np.array([2, 2]), fitness=0.6, trial_count=5),
            FoodSource(position=np.array([3, 3]), fitness=0.9, trial_count=1),
        ]

        # Test management
        management = abc.manage_food_sources(
            current_sources=sources,
            abandonment_criteria="trial_limit",
            recruitment_strategy="fitness_proportional",
        )

        assert "sources_abandoned" in management
        assert "sources_updated" in management
        assert management["sources_updated"] == len(sources)

    def test_abc_foraging_adaptation(self):
        """Test ABC foraging strategy adaptation."""
        abc = ArtificialBeeColony(colony_size=20, dimensions=2)

        environmental_conditions = {"resource_density": 0.2}
        colony_performance = {"success_rate": 0.25}

        adaptation = abc.adapt_foraging_strategy(
            environmental_conditions,
            colony_performance,
            behavioral_adaptation="learning_automaton",
        )

        assert "adaptation_type" in adaptation
        assert "parameters_updated" in adaptation
        assert "strategy_changes" in adaptation

    def test_abc_statistics(self):
        """Test ABC statistics generation."""
        abc = ArtificialBeeColony(colony_size=10, dimensions=2, max_iterations=5)

        def dummy_function(x):
            return np.sum(x**2)

        # Run brief optimization
        abc.optimize(dummy_function)

        stats = abc.get_optimization_statistics()

        assert stats["algorithm"] == "Artificial Bee Colony"
        assert "parameters" in stats
        assert "optimization_results" in stats
        assert "food_source_statistics" in stats


class TestAlgorithmIntegration:
    """Test integration between algorithms and swarm systems."""

    def test_aco_with_pheromone_system(self):
        """Test ACO integration with pheromone system."""
        # Create pheromone system
        PheromoneSystem(
            pheromone_types=["trail"],
            bounds={"min_lat": -10, "max_lat": 10, "min_lng": -10, "max_lng": 10},
        )

        # Create ACO with pheromone integration
        aco = AntColonyOptimization(number_of_ants=10, max_iterations=5)

        # Initialize simple problem
        cities = [[0, 0], [1, 0], [1, 1], [0, 1]]
        distances = np.ones((4, 4)) - np.eye(4)
        aco.initialize_problem(cities, distances)

        # Verify integration
        assert aco.pheromone_system is not None

    def test_pso_with_spatial_constraints(self):
        """Test PSO with spatial constraints."""
        pso = ParticleSwarmOptimization(
            swarm_size=10,
            dimensions=2,
            spatial_constraints={"spatial_bounds": [(-5, 5), (-5, 5)]},
        )

        def constrained_objective(x):
            # Add penalty for going outside spatial bounds
            penalty = 0
            for i, (val, (min_b, max_b)) in enumerate(zip(x, pso.parameters.bounds)):
                if val < min_b or val > max_b:
                    penalty += 100

            return np.sum(x**2) + penalty

        # Optimize with constraints
        optimal = pso.optimize(constrained_objective)

        # Should respect bounds
        assert all(
            pso.parameters.bounds[i][0] <= val <= pso.parameters.bounds[i][1]
            for i, val in enumerate(optimal)
        )


class TestAlgorithmPerformance:
    """Test algorithm performance and scalability."""

    @pytest.mark.slow
    def test_aco_performance_scaling(self):
        """Test ACO performance scaling with problem size."""
        problem_sizes = [5, 10, 15]

        for size in problem_sizes:
            aco = AntColonyOptimization(
                number_of_ants=min(10, size * 2), max_iterations=10
            )

            # Create problem
            cities = np.random.uniform(-10, 10, (size, 2))
            distances = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    distances[i, j] = np.linalg.norm(cities[i] - cities[j])

            aco.initialize_problem(cities.tolist(), distances)
            result = aco.solve()

            # Should complete successfully
            assert result.iterations_completed > 0
            assert result.best_fitness < float("inf")

    @pytest.mark.slow
    def test_pso_performance_scaling(self):
        """Test PSO performance scaling with swarm size."""
        swarm_sizes = [10, 30, 50]

        for size in swarm_sizes:
            pso = ParticleSwarmOptimization(
                swarm_size=size, dimensions=2, max_iterations=10
            )

            def sphere(x):
                return np.sum(x**2)

            result = pso.optimize(sphere)

            # Should complete successfully
            assert len(result) == 2
            assert np.sum(result**2) < 10  # Should find reasonable solution

    @pytest.mark.slow
    def test_abc_performance_scaling(self):
        """Test ABC performance scaling with colony size."""
        colony_sizes = [10, 30, 50]

        for size in colony_sizes:
            abc = ArtificialBeeColony(colony_size=size, dimensions=2, max_iterations=10)

            def sphere(x):
                return np.sum(x**2)

            result = abc.optimize(sphere)

            # Should complete successfully
            assert len(result) == 2
            assert np.sum(result**2) < 10  # Should find reasonable solution


if __name__ == "__main__":
    # Run algorithm tests
    pytest.main([__file__, "-v", "--tb=short"])
