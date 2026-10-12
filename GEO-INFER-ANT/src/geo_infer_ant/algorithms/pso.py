"""
Particle Swarm Optimization (PSO) Algorithm for GEO-INFER-ANT

This module implements the Particle Swarm Optimization algorithm, which simulates
the social behavior of bird flocking or fish schooling. PSO is particularly
effective for continuous optimization problems and can be adapted for spatial
optimization tasks.

Key Features:
- Continuous and discrete optimization
- Adaptive parameter tuning
- Multi-swarm coordination
- Spatial constraint handling
- Real-time optimization capabilities
- Integration with spatial indexing systems
"""

import numpy as np
import logging
from typing import Any, cast
from collections.abc import Callable
from datetime import datetime
from dataclasses import dataclass, field

# Integration imports
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

logger = logging.getLogger(__name__)


@dataclass
class PSOParameters:
    """Parameters for Particle Swarm Optimization algorithm."""

    swarm_size: int = 100
    dimensions: int = 2
    bounds: list[tuple[float, float]] = field(
        default_factory=lambda: [(-10, 10), (-10, 10)]
    )
    inertia_weight: float = 0.7
    cognitive_acceleration: float = 1.5  # c1 - personal best influence
    social_acceleration: float = 1.5  # c2 - global best influence
    max_velocity: float = 3.0
    min_velocity: float = -3.0
    max_iterations: int = 200
    convergence_threshold: float = 1e-6

    # Advanced parameters
    velocity_clamping: bool = True
    adaptive_parameters: bool = False
    neighborhood_topology: str = "global"  # 'global', 'local', 'adaptive'
    neighborhood_size: int = 5

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.swarm_size <= 0:
            raise ValueError("Swarm size must be positive")
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        if len(self.bounds) != self.dimensions:
            raise ValueError("Bounds must match number of dimensions")
        if not np.isfinite(self.inertia_weight) or self.inertia_weight < 0:
            raise ValueError("inertia_weight must be finite and non-negative")
        for name, value in {
            "cognitive_acceleration": self.cognitive_acceleration,
            "social_acceleration": self.social_acceleration,
            "convergence_threshold": self.convergence_threshold,
        }.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_velocity > self.max_velocity:
            raise ValueError("min_velocity must not exceed max_velocity")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        for lower, upper in self.bounds:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("Each bound must be finite and ordered (min < max)")
        if self.neighborhood_topology not in {"global", "local", "adaptive"}:
            raise ValueError("neighborhood_topology must be global, local, or adaptive")
        if self.neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")


@dataclass
class Particle:
    """Individual particle in the PSO swarm."""

    position: np.ndarray
    velocity: np.ndarray
    personal_best_position: np.ndarray
    personal_best_fitness: float = float("inf")
    fitness: float = float("inf")

    def update_personal_best(self) -> None:
        """Update personal best if current fitness is better."""
        if self.fitness < self.personal_best_fitness:
            self.personal_best_position = self.position.copy()
            self.personal_best_fitness = self.fitness

    def update_velocity(
        self,
        global_best_position: np.ndarray | None,
        inertia_weight: float,
        cognitive_acceleration: float,
        social_acceleration: float,
        neighborhood_best_position: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        velocity_clamping: bool = True,
        max_velocity: float | None = None,
        min_velocity: float | None = None,
    ) -> None:
        """
        Update particle velocity using PSO formula.

        Args:
            global_best_position: Best position found by swarm
            inertia_weight: Inertia weight parameter
            cognitive_acceleration: Personal acceleration coefficient
            social_acceleration: Social acceleration coefficient
            neighborhood_best_position: Best position in neighborhood (for local topology)
        """
        # Random coefficients
        rng = rng or np.random.default_rng()
        r1 = rng.uniform(0, 1, self.position.shape)
        r2 = rng.uniform(0, 1, self.position.shape)

        # Cognitive component (personal best)
        cognitive_component = (
            cognitive_acceleration * r1 * (self.personal_best_position - self.position)
        )

        # Social component (global or neighborhood best)
        if neighborhood_best_position is not None:
            social_target = neighborhood_best_position
        elif global_best_position is not None:
            social_target = global_best_position
        else:
            social_target = self.personal_best_position

        social_component = social_acceleration * r2 * (social_target - self.position)

        # Update velocity
        self.velocity = (
            inertia_weight * self.velocity + cognitive_component + social_component
        )

        # Apply velocity clamping if enabled
        if velocity_clamping:
            upper = (
                getattr(self, "max_velocity", np.inf)
                if max_velocity is None
                else max_velocity
            )
            lower = -upper if min_velocity is None else min_velocity
            np.clip(self.velocity, lower, upper, out=self.velocity)

    def update_position(self, bounds: list[tuple[float, float]]) -> None:
        """Update particle position based on velocity."""
        self.position += self.velocity

        # Apply boundary constraints
        for i in range(len(self.position)):
            min_bound, max_bound = bounds[i]
            self.position[i] = np.clip(self.position[i], min_bound, max_bound)


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization algorithm implementation.

    This class implements the classic PSO algorithm with extensions for:
    - Multi-dimensional continuous optimization
    - Spatial constraint handling
    - Adaptive parameter tuning
    - Multi-swarm coordination
    - Dynamic environment adaptation

    The algorithm simulates social behavior where particles (solutions)
    move through the search space influenced by their personal best
    positions and the global best position found by the swarm.
    """

    def __init__(
        self,
        swarm_size: int = 100,
        dimensions: int = 2,
        bounds: list[tuple[float, float]] | None = None,
        inertia_weight: float = 0.7,
        cognitive_acceleration: float = 1.5,
        social_acceleration: float = 1.5,
        max_velocity: float = 3.0,
        max_iterations: int = 200,
        spatial_constraints: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize PSO algorithm.

        Args:
            swarm_size: Number of particles in the swarm
            dimensions: Number of dimensions in the search space
            bounds: Search space bounds for each dimension
            inertia_weight: Inertia weight for velocity updates
            cognitive_acceleration: Personal best influence coefficient
            social_acceleration: Global best influence coefficient
            max_velocity: Maximum allowed velocity
            max_iterations: Maximum number of optimization iterations
            spatial_constraints: Spatial constraints for optimization
            **kwargs: Additional parameters
        """
        # Set default bounds if not provided
        if bounds is None:
            bounds = [(-10, 10)] * dimensions

        # Extract neighborhood parameters from kwargs before constructing PSOParameters
        neighborhood_topology = kwargs.pop("neighborhood_topology", "global")
        neighborhood_size = kwargs.pop("neighborhood_size", 5)
        min_velocity = kwargs.pop("min_velocity", -max_velocity)
        velocity_clamping = kwargs.pop("velocity_clamping", True)
        adaptive_parameters = kwargs.pop("adaptive_parameters", False)
        seed = kwargs.pop("random_seed", kwargs.pop("seed", None))

        self.parameters = PSOParameters(
            swarm_size=swarm_size,
            dimensions=dimensions,
            bounds=bounds,
            inertia_weight=inertia_weight,
            cognitive_acceleration=cognitive_acceleration,
            social_acceleration=social_acceleration,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            max_iterations=max_iterations,
            velocity_clamping=velocity_clamping,
            adaptive_parameters=adaptive_parameters,
            neighborhood_topology=neighborhood_topology,
            neighborhood_size=neighborhood_size,
        )

        self.spatial_constraints = spatial_constraints or {}
        self.rng = np.random.default_rng(seed)

        # Swarm state
        self.swarm: list[Particle] = []
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness: float = float("inf")

        # Neighborhood structure (for local topology)
        self.neighborhoods: dict[int, list[int]] = {}

        # History tracking
        self.convergence_history: list[float] = []
        self.diversity_history: list[float] = []
        self.parameter_history: list[dict[str, float]] = []

        # Integration components
        self.spatial_indexer = None
        self.spatial_analytics = None

        # Performance tracking
        self.iteration_times: list[float] = []
        self.function_evaluations: int = 0

        # Initialize integrations
        self._initialize_integrations()

        logger.info(
            f"PSO initialized with {swarm_size} particles, {dimensions} dimensions"
        )

    def _initialize_integrations(self) -> None:
        """Initialize integration with other GEO-INFER modules."""
        if SpatialIndexingInterface:
            try:
                self.spatial_indexer = SpatialIndexingInterface(backend="h3")
                logger.info("Spatial indexer initialized for PSO")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial indexer: {e}")

        if SpatialAnalyticsInterface:
            try:
                self.spatial_analytics = SpatialAnalyticsInterface(backend="h3")
                logger.info("Spatial analytics initialized for PSO")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analytics: {e}")

    def initialize_swarm(self, initial_positions: np.ndarray | None = None) -> None:
        """
        Initialize the particle swarm.

        Args:
            initial_positions: Initial positions for particles (random if None)
        """
        self.swarm = []
        if initial_positions is not None:
            positions = np.asarray(initial_positions, dtype=float)
            expected = (self.parameters.swarm_size, self.parameters.dimensions)
            if positions.shape != expected:
                raise ValueError(f"initial_positions must have shape {expected}")
            if not np.all(np.isfinite(positions)):
                raise ValueError("initial_positions must contain finite values")
        else:
            positions = None

        for i in range(self.parameters.swarm_size):
            if positions is not None:
                # Use provided initial position
                position = positions[i].copy()
            else:
                # Generate random initial position within bounds
                position = np.asarray(
                    [
                        self.rng.uniform(min_bound, max_bound)
                        for min_bound, max_bound in self.parameters.bounds
                    ]
                )

            # Generate random initial velocity
            velocity = np.array(
                [
                    self.rng.uniform(
                        self.parameters.min_velocity, self.parameters.max_velocity
                    )
                    for _ in range(self.parameters.dimensions)
                ]
            )

            # Create particle
            particle = Particle(
                position=position,
                velocity=velocity,
                personal_best_position=position.copy(),
                personal_best_fitness=float("inf"),
                fitness=float("inf"),
            )

            self.swarm.append(particle)

        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = float("inf")

        # Initialize neighborhoods if using local topology
        if self.parameters.neighborhood_topology == "local":
            self._initialize_neighborhoods()

        logger.info(f"PSO swarm initialized with {len(self.swarm)} particles")

    def _initialize_neighborhoods(self) -> None:
        """Initialize neighborhood structure for local topology."""
        if self.parameters.neighborhood_topology == "local":
            # Ring topology neighborhoods
            for i in range(self.parameters.swarm_size):
                neighborhood = []
                for j in range(
                    -self.parameters.neighborhood_size // 2,
                    self.parameters.neighborhood_size // 2 + 1,
                ):
                    neighbor_idx = (i + j) % self.parameters.swarm_size
                    if neighbor_idx != i:
                        neighborhood.append(neighbor_idx)
                self.neighborhoods[i] = neighborhood

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_positions: np.ndarray | None = None,
        velocity_bounds: tuple[float, float] | None = None,
        convergence_criteria: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Optimize the objective function using PSO.

        Args:
            objective_function: Function to optimize (minimize)
            initial_positions: Initial positions for particles
            velocity_bounds: Bounds for particle velocities
            convergence_criteria: Custom convergence criteria

        Returns:
            Optimal solution found
        """
        start_time = datetime.now()

        logger.info(
            f"Starting PSO optimization with {self.parameters.max_iterations} max iterations"
        )

        if not callable(objective_function):
            raise TypeError("objective_function must be callable")
        # Update velocity bounds if provided
        if velocity_bounds:
            if len(velocity_bounds) != 2 or velocity_bounds[0] > velocity_bounds[1]:
                raise ValueError("velocity_bounds must be an ordered (min, max) pair")
            self.parameters.min_velocity, self.parameters.max_velocity = velocity_bounds

        # Initialize swarm
        self.initialize_swarm(initial_positions)
        self.convergence_history = []
        self.diversity_history = []
        self.parameter_history = []
        self.iteration_times = []
        self.function_evaluations = 0

        # Set convergence criteria
        if convergence_criteria:
            self.convergence_criteria = convergence_criteria
        else:
            self.convergence_criteria = {
                "tolerance": self.parameters.convergence_threshold,
                "max_iterations": self.parameters.max_iterations,
                "min_improvement": 1e-8,
            }

        # Main optimization loop
        for iteration in range(self.parameters.max_iterations):
            iteration_start = datetime.now()

            # Evaluate all particles
            self._evaluate_swarm(objective_function)

            # Update personal and global bests
            self._update_bests()

            # Update particle velocities and positions
            self._update_swarm()

            # Check convergence
            self._record_iteration_stats(iteration)
            if self._check_convergence(iteration):
                logger.info(f"Convergence achieved at iteration {iteration}")
                break

            # Adaptive parameter tuning
            if self.parameters.adaptive_parameters:
                self._adapt_parameters(iteration)

            iteration_time = (datetime.now() - iteration_start).total_seconds()
            self.iteration_times.append(iteration_time)

        # Final evaluation
        self._evaluate_swarm(objective_function)
        self._update_bests()

        _computation_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"PSO optimization completed: best fitness = {self.global_best_fitness}"
        )
        logger.info(f"Optimal solution: {self.global_best_position}")

        return (
            self.global_best_position.copy()
            if self.global_best_position is not None
            else np.zeros(self.parameters.dimensions)
        )

    def _evaluate_swarm(
        self, objective_function: Callable[[np.ndarray], float]
    ) -> None:
        """Evaluate fitness of all particles."""
        for particle in self.swarm:
            # Apply spatial constraints if any
            constrained_position = self._apply_spatial_constraints(particle.position)

            # Evaluate objective function
            particle.position = constrained_position
            value = objective_function(constrained_position)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError("objective_function must return a finite scalar")
            particle.fitness = float(cast(Any, value))
            particle.update_personal_best()

            self.function_evaluations += 1

    def _apply_spatial_constraints(self, position: np.ndarray) -> np.ndarray:
        """Apply spatial constraints to particle position."""
        constrained_position = position.copy()

        # Example constraint: keep within spatial bounds
        if "spatial_bounds" in self.spatial_constraints:
            bounds = self.spatial_constraints["spatial_bounds"]
            for i in range(len(constrained_position)):
                if i < len(bounds):
                    min_bound, max_bound = bounds[i]
                    constrained_position[i] = np.clip(
                        constrained_position[i], min_bound, max_bound
                    )

        # Example constraint: avoid obstacles
        if "obstacles" in self.spatial_constraints:
            constrained_position = self._avoid_obstacles(constrained_position)

        return constrained_position

    def _avoid_obstacles(self, position: np.ndarray) -> np.ndarray:
        """Apply obstacle avoidance to particle position."""
        adjusted = position.copy()
        obstacles = self.spatial_constraints.get("obstacles", [])
        if isinstance(obstacles, dict):
            obstacles = [obstacles]
        for obstacle in obstacles:
            if isinstance(obstacle, dict) and {"center", "radius"}.issubset(obstacle):
                center = np.asarray(obstacle["center"], dtype=float)
                radius = float(obstacle["radius"])
                if center.shape != adjusted.shape or radius < 0:
                    raise ValueError(
                        "Circular obstacles require a matching center and non-negative radius"
                    )
                delta = adjusted - center
                distance = float(np.linalg.norm(delta))
                if distance < radius:
                    direction = (
                        delta / distance if distance > 1e-12 else np.zeros_like(delta)
                    )
                    if distance <= 1e-12:
                        direction[0] = 1.0
                    adjusted = center + direction * radius
            elif isinstance(obstacle, dict) and {"min", "max"}.issubset(obstacle):
                lower = np.asarray(obstacle["min"], dtype=float)
                upper = np.asarray(obstacle["max"], dtype=float)
                if (
                    lower.shape != adjusted.shape
                    or upper.shape != adjusted.shape
                    or np.any(lower >= upper)
                ):
                    raise ValueError("Box obstacles require ordered min/max vectors")
                if np.all((adjusted >= lower) & (adjusted <= upper)):
                    distances = np.minimum(adjusted - lower, upper - adjusted)
                    axis = int(np.argmin(distances))
                    adjusted[axis] = (
                        lower[axis]
                        if adjusted[axis] - lower[axis] <= upper[axis] - adjusted[axis]
                        else upper[axis]
                    )
            else:
                raise ValueError(
                    "Obstacles must use {'center', 'radius'} or {'min', 'max'}"
                )
        return adjusted

    def _update_bests(self) -> None:
        """Update personal and global best solutions."""
        # Update global best
        for particle in self.swarm:
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()

    def _update_swarm(self) -> None:
        """Update velocities and positions of all particles."""
        for i, particle in enumerate(self.swarm):
            # Get neighborhood best for local topology
            neighborhood_best = self._get_neighborhood_best(i)

            # Update velocity
            particle.update_velocity(
                global_best_position=self.global_best_position,
                inertia_weight=self.parameters.inertia_weight,
                cognitive_acceleration=self.parameters.cognitive_acceleration,
                social_acceleration=self.parameters.social_acceleration,
                neighborhood_best_position=neighborhood_best,
                rng=self.rng,
                velocity_clamping=self.parameters.velocity_clamping,
                max_velocity=self.parameters.max_velocity,
                min_velocity=self.parameters.min_velocity,
            )

            # Update position
            particle.update_position(self.parameters.bounds)

    def _get_neighborhood_best(self, particle_idx: int) -> np.ndarray | None:
        """Get best position in particle's neighborhood."""
        if self.parameters.neighborhood_topology == "global":
            return self.global_best_position

        elif self.parameters.neighborhood_topology == "local":
            if particle_idx in self.neighborhoods:
                neighborhood_indices = self.neighborhoods[particle_idx]
                neighborhood_particles = [self.swarm[i] for i in neighborhood_indices]

                # Find best in neighborhood
                best_in_neighborhood = min(
                    neighborhood_particles, key=lambda p: p.personal_best_fitness
                )
                return best_in_neighborhood.personal_best_position

        return self.global_best_position

    def _check_convergence(self, iteration: int) -> bool:
        """Check if algorithm has converged."""
        if len(self.convergence_history) < 10:
            return False

        # Check fitness improvement
        recent_fitness = self.convergence_history[-10:]
        improvement = recent_fitness[0] - recent_fitness[-1]

        if 0 <= improvement < self.convergence_criteria.get("min_improvement", 1e-8):
            return True

        # Check fitness stability
        fitness_std = np.std(recent_fitness)
        fitness_mean = np.mean(recent_fitness)

        if fitness_mean > 0:
            coefficient_of_variation = fitness_std / fitness_mean
            return bool(
                coefficient_of_variation
                < self.convergence_criteria.get("tolerance", 1e-6)
            )

        return False

    def _record_iteration_stats(self, iteration: int) -> None:
        """Record statistics for current iteration."""
        self.convergence_history.append(self.global_best_fitness)

        # Calculate swarm diversity
        if len(self.swarm) > 1:
            positions = np.array([p.position for p in self.swarm])
            position_mean = np.mean(positions, axis=0)
            diversity = np.mean(
                [np.linalg.norm(pos - position_mean) for pos in positions]
            )
            self.diversity_history.append(diversity)

        # Record parameter values
        self.parameter_history.append(
            {
                "iteration": iteration,
                "inertia_weight": self.parameters.inertia_weight,
                "global_best_fitness": self.global_best_fitness,
            }
        )

    def _adapt_parameters(self, iteration: int) -> None:
        """Adapt PSO parameters based on optimization progress."""
        # Adaptive inertia weight
        if iteration < self.parameters.max_iterations * 0.5:
            # Decrease inertia weight linearly in first half
            progress = iteration / (self.parameters.max_iterations * 0.5)
            self.parameters.inertia_weight = 0.9 - 0.5 * progress
        else:
            # Keep low inertia weight in second half
            self.parameters.inertia_weight = 0.4

        # Adaptive acceleration coefficients
        if len(self.diversity_history) > 5:
            # Increase exploration if diversity is low
            recent_diversity = np.mean(self.diversity_history[-5:])
            initial_diversity = (
                self.diversity_history[0] if self.diversity_history else 1.0
            )

            if recent_diversity < initial_diversity * 0.1:
                # Low diversity - increase cognitive acceleration
                self.parameters.cognitive_acceleration = min(
                    2.5, self.parameters.cognitive_acceleration * 1.1
                )
                self.parameters.social_acceleration = max(
                    0.5, self.parameters.social_acceleration * 0.9
                )

    def coordinate_swarms(
        self,
        sub_swarms: list["ParticleSwarmOptimization"],
        communication_topology: str = "hierarchical",
        information_sharing: str = "best_positions",
        objective_function: Callable[[np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate multiple PSO swarms for complex optimization.

        Args:
            sub_swarms: List of PSO swarm optimizers
            communication_topology: How swarms communicate ('hierarchical', 'ring', 'complete')
            information_sharing: What information to share ('best_positions', 'all_particles')

        Returns:
            Coordination results and combined optimization
        """
        if communication_topology not in {"hierarchical", "ring", "complete"}:
            raise ValueError("Unsupported communication topology")
        if information_sharing not in {"best_positions", "all_particles"}:
            raise ValueError("Unsupported information sharing mode")
        logger.info(f"Coordinating {len(sub_swarms)} PSO swarms")

        coordination_results: dict[str, Any] = {
            "topology": communication_topology,
            "information_sharing": information_sharing,
            "sub_swarm_results": [],
            "combined_best_solution": None,
            "combined_best_fitness": float("inf"),
        }

        # Run each sub-swarm
        for i, sub_swarm in enumerate(sub_swarms):
            logger.info(f"Running sub-swarm {i + 1}/{len(sub_swarms)}")

            if objective_function is not None:
                sub_swarm.optimize(objective_function)
            solution = (
                sub_swarm.global_best_position.copy()
                if sub_swarm.global_best_position is not None
                else None
            )
            result = {
                "sub_swarm_id": i,
                "best_fitness": float(sub_swarm.global_best_fitness),
                "best_solution": solution,
            }

            coordination_results["sub_swarm_results"].append(result)

            # Update combined best
            if result["best_fitness"] < coordination_results["combined_best_fitness"]:
                coordination_results["combined_best_fitness"] = result["best_fitness"]
                coordination_results["combined_best_solution"] = result["best_solution"]

        # Implement inter-swarm communication
        if not sub_swarms:
            return coordination_results
        if communication_topology == "hierarchical":
            self._hierarchical_communication(sub_swarms, coordination_results)
        elif communication_topology == "ring":
            self._ring_communication(sub_swarms, coordination_results)

        logger.info(
            f"Swarm coordination completed: best fitness = {coordination_results['combined_best_fitness']}"
        )
        return coordination_results

    def _hierarchical_communication(
        self, sub_swarms: list["ParticleSwarmOptimization"], results: dict[str, Any]
    ) -> None:
        """Implement hierarchical communication between swarms."""
        # Find best sub-swarm
        valid_results = [
            r for r in results["sub_swarm_results"] if r["best_solution"] is not None
        ]
        if not valid_results:
            return
        best_swarm_idx = int(
            np.argmin([r["best_fitness"] for r in results["sub_swarm_results"]])
        )

        # Share best solution with all other swarms
        best_solution = results["sub_swarm_results"][best_swarm_idx]["best_solution"]

        for i, swarm in enumerate(sub_swarms):
            if i != best_swarm_idx:
                # Update swarm's global best
                if swarm.global_best_fitness > results["combined_best_fitness"]:
                    swarm.global_best_position = best_solution.copy()
                    swarm.global_best_fitness = results["combined_best_fitness"]

    def _ring_communication(
        self, sub_swarms: list["ParticleSwarmOptimization"], results: dict[str, Any]
    ) -> None:
        """Implement ring topology communication between swarms."""
        n_swarms = len(sub_swarms)
        if n_swarms == 0:
            return

        for i in range(n_swarms):
            # Share information with neighbors in ring
            left_neighbor = (i - 1) % n_swarms
            right_neighbor = (i + 1) % n_swarms

            current_swarm = sub_swarms[i]
            left_result = results["sub_swarm_results"][left_neighbor]
            right_result = results["sub_swarm_results"][right_neighbor]

            # Update with better neighbor solution
            for neighbor_result in [left_result, right_result]:
                if neighbor_result["best_fitness"] < current_swarm.global_best_fitness:
                    current_swarm.global_best_position = neighbor_result[
                        "best_solution"
                    ].copy()
                    current_swarm.global_best_fitness = neighbor_result["best_fitness"]

    def adapt_parameters(
        self,
        performance_history: list[dict[str, Any]],
        environmental_changes: dict[str, Any],
        adaptation_strategy: str = "self_tuning",
    ) -> dict[str, Any]:
        """
        Adapt PSO parameters based on performance and environmental changes.

        Args:
            performance_history: History of optimization performance
            environmental_changes: Description of environmental changes
            adaptation_strategy: Strategy for parameter adaptation

        Returns:
            Adaptation results and updated parameters
        """
        logger.info(f"Adapting PSO parameters using {adaptation_strategy} strategy")

        adaptation_results: dict[str, Any] = {
            "strategy": adaptation_strategy,
            "changes_applied": [],
            "parameters_updated": {},
            "performance_improvement": 0.0,
        }

        if adaptation_strategy == "self_tuning":
            # Analyze performance trends
            if len(performance_history) > 10:
                recent_performance = performance_history[-5:]
                older_performance = (
                    performance_history[-15:-10]
                    if len(performance_history) > 15
                    else performance_history[:5]
                )

                recent_avg_fitness = np.mean(
                    [p.get("fitness", 1.0) for p in recent_performance]
                )
                older_avg_fitness = np.mean(
                    [p.get("fitness", 1.0) for p in older_performance]
                )

                improvement_rate = (
                    older_avg_fitness - recent_avg_fitness
                ) / older_avg_fitness

                # Adjust parameters based on improvement rate
                if improvement_rate < 0.01:  # Slow improvement
                    # Increase exploration
                    old_inertia = self.parameters.inertia_weight
                    self.parameters.inertia_weight = min(0.9, old_inertia * 1.1)
                    adaptation_results["parameters_updated"]["inertia_weight"] = (
                        self.parameters.inertia_weight
                    )
                    adaptation_results["changes_applied"].append(
                        "increased_exploration"
                    )

                elif improvement_rate > 0.1:  # Fast improvement
                    # Increase exploitation
                    old_inertia = self.parameters.inertia_weight
                    self.parameters.inertia_weight = max(0.4, old_inertia * 0.9)
                    adaptation_results["parameters_updated"]["inertia_weight"] = (
                        self.parameters.inertia_weight
                    )
                    adaptation_results["changes_applied"].append(
                        "increased_exploitation"
                    )

        elif adaptation_strategy == "environmental":
            # Adapt to environmental changes
            if "noise_level" in environmental_changes:
                noise = environmental_changes["noise_level"]

                # Increase robustness in noisy environments
                if noise > 0.5:
                    self.parameters.cognitive_acceleration = min(
                        2.0, self.parameters.cognitive_acceleration * 1.2
                    )
                    self.parameters.social_acceleration = min(
                        2.0, self.parameters.social_acceleration * 1.2
                    )
                    adaptation_results["parameters_updated"][
                        "cognitive_acceleration"
                    ] = self.parameters.cognitive_acceleration
                    adaptation_results["parameters_updated"]["social_acceleration"] = (
                        self.parameters.social_acceleration
                    )
                    adaptation_results["changes_applied"].append("adapted_to_noise")

        adaptation_results["performance_improvement"] = adaptation_results.get(
            "performance_improvement", 0.0
        )

        logger.info(
            f"PSO adaptation completed: {len(adaptation_results['changes_applied'])} changes applied"
        )
        return adaptation_results

    def get_optimization_statistics(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "algorithm": "Particle Swarm Optimization",
            "parameters": {
                "swarm_size": self.parameters.swarm_size,
                "dimensions": self.parameters.dimensions,
                "inertia_weight": self.parameters.inertia_weight,
                "cognitive_acceleration": self.parameters.cognitive_acceleration,
                "social_acceleration": self.parameters.social_acceleration,
                "max_iterations": self.parameters.max_iterations,
            },
            "optimization_results": {
                "best_fitness": self.global_best_fitness,
                "iterations_completed": len(self.convergence_history),
                "function_evaluations": self.function_evaluations,
                "convergence_achieved": self._check_convergence(
                    self.parameters.max_iterations - 1
                ),
            },
            "swarm_statistics": {},
        }

        # Swarm diversity and distribution
        if self.swarm:
            positions = np.array([p.position for p in self.swarm])
            fitnesses = np.array([p.fitness for p in self.swarm])

            stats["swarm_statistics"] = {
                "position_mean": np.mean(positions, axis=0).tolist(),
                "position_std": np.std(positions, axis=0).tolist(),
                "fitness_mean": np.mean(fitnesses),
                "fitness_std": np.std(fitnesses),
                "best_particle_index": np.argmin(fitnesses),
            }

        # Performance statistics
        if self.iteration_times:
            stats["performance"] = {
                "avg_iteration_time": np.mean(self.iteration_times),
                "total_computation_time": sum(self.iteration_times),
                "iterations_per_second": len(self.iteration_times)
                / sum(self.iteration_times),
            }

        # Convergence analysis
        if len(self.convergence_history) > 10:
            recent_improvement = (
                self.convergence_history[-1] - self.convergence_history[-10]
            )
            stats["convergence"] = {
                "recent_improvement": recent_improvement,
                "improvement_rate": recent_improvement / 10,
                "convergence_stability": np.std(self.convergence_history[-10:]),
            }

        return stats

    def save_optimization_state(self, filepath: str) -> bool:
        """Save optimization state to file."""
        try:
            import json

            state = {
                "parameters": {
                    "swarm_size": self.parameters.swarm_size,
                    "dimensions": self.parameters.dimensions,
                    "bounds": self.parameters.bounds,
                    "inertia_weight": self.parameters.inertia_weight,
                    "cognitive_acceleration": self.parameters.cognitive_acceleration,
                    "social_acceleration": self.parameters.social_acceleration,
                    "max_velocity": self.parameters.max_velocity,
                    "max_iterations": self.parameters.max_iterations,
                },
                "swarm_state": [
                    {
                        "position": particle.position.tolist(),
                        "velocity": particle.velocity.tolist(),
                        "personal_best_position": particle.personal_best_position.tolist(),
                        "personal_best_fitness": particle.personal_best_fitness,
                        "fitness": particle.fitness,
                    }
                    for particle in self.swarm
                ],
                "global_best": {
                    "position": (
                        self.global_best_position.tolist()
                        if self.global_best_position is not None
                        else None
                    ),
                    "fitness": self.global_best_fitness,
                },
                "history": {
                    "convergence_history": self.convergence_history,
                    "diversity_history": self.diversity_history,
                    "parameter_history": self.parameter_history,
                },
            }

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"PSO state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save PSO state: {e}")
            return False

    def load_optimization_state(self, filepath: str) -> bool:
        """Load optimization state from file."""
        try:
            import json

            with open(filepath) as f:
                state = json.load(f)

            # Restore parameters
            params = state["parameters"]
            self.parameters = PSOParameters(**params)

            # Restore swarm
            self.swarm = []
            for particle_data in state["swarm_state"]:
                particle = Particle(
                    position=np.array(particle_data["position"]),
                    velocity=np.array(particle_data["velocity"]),
                    personal_best_position=np.array(
                        particle_data["personal_best_position"]
                    ),
                    personal_best_fitness=particle_data["personal_best_fitness"],
                    fitness=particle_data["fitness"],
                )
                self.swarm.append(particle)

            # Restore global best
            global_best = state["global_best"]
            self.global_best_position = (
                np.array(global_best["position"])
                if global_best["position"] is not None
                else None
            )
            self.global_best_fitness = global_best["fitness"]

            # Restore history
            history = state["history"]
            self.convergence_history = history["convergence_history"]
            self.diversity_history = history["diversity_history"]
            self.parameter_history = history["parameter_history"]

            logger.info(f"PSO state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load PSO state: {e}")
            return False
