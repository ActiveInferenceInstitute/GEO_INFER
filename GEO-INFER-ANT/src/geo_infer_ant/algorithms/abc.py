"""
Artificial Bee Colony (ABC) Algorithm for GEO-INFER-ANT

This module implements the Artificial Bee Colony algorithm, inspired by the
intelligent foraging behavior of honey bee colonies. The algorithm uses
three types of bees (employed, onlooker, and scout) to find optimal solutions.

Key Features:
- Multi-role bee colony simulation
- Adaptive foraging strategies
- Honey source management
- Spatial optimization capabilities
- Integration with environmental constraints
"""

import numpy as np
import logging
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ABCParameters:
    """Parameters for Artificial Bee Colony algorithm."""

    colony_size: int = 100
    employed_bees_ratio: float = 0.5
    scout_bees_ratio: float = 0.1
    dimensions: int = 2
    bounds: list[tuple] = field(default_factory=lambda: [(-10, 10), (-10, 10)])
    max_trials: float = 50.0  # Abandonment threshold
    limit: float = 100.0  # Maximum trials before abandonment
    max_iterations: int = 1000

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.colony_size <= 0:
            raise ValueError("Colony size must be positive")
        if not 0 < self.employed_bees_ratio < 1:
            raise ValueError("Employed bees ratio must be between 0 and 1")
        if not 0 < self.scout_bees_ratio < 1:
            raise ValueError("Scout bees ratio must be between 0 and 1")
        if self.dimensions <= 0 or len(self.bounds) != self.dimensions:
            raise ValueError("dimensions must be positive and match bounds")
        for lower, upper in self.bounds:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("Each bound must be finite and ordered (min < max)")
        if self.max_trials <= 0 or self.limit <= 0:
            raise ValueError("max_trials and limit must be positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")


@dataclass
class FoodSource:
    """Represents a food source (solution) in the ABC algorithm."""

    position: np.ndarray
    fitness: float = 0.0
    trial_count: int = 0
    last_improvement: int = 0

    def update_fitness(self, fitness: float) -> None:
        """Update fitness and trial count."""
        if fitness > self.fitness:
            self.fitness = fitness
            self.trial_count = 0
            self.last_improvement = 0
        else:
            self.trial_count += 1


class ArtificialBeeColony:
    """Artificial Bee Colony optimization algorithm."""

    def __init__(
        self,
        colony_size: int = 100,
        dimensions: int = 2,
        bounds: list[tuple] | None = None,
        max_trials: int = 50,
        limit: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize ABC algorithm.

        Args:
            colony_size: Total number of bees in colony
            dimensions: Number of dimensions in search space
            bounds: Search space bounds
            max_trials: Maximum trials before food source abandonment
            limit: Trial limit for abandonment
            **kwargs: Additional parameters
        """
        if bounds is None:
            bounds = [(-10, 10)] * dimensions

        employed_bees_ratio = kwargs.pop("employed_bees_ratio", 0.5)
        scout_bees_ratio = kwargs.pop("scout_bees_ratio", 0.1)
        max_iterations = kwargs.pop("max_iterations", 1000)
        seed = kwargs.pop("random_seed", kwargs.pop("seed", None))
        self.parameters = ABCParameters(
            colony_size=colony_size,
            dimensions=dimensions,
            bounds=bounds,
            max_trials=max_trials,
            limit=limit,
            employed_bees_ratio=employed_bees_ratio,
            scout_bees_ratio=scout_bees_ratio,
            max_iterations=max_iterations,
        )
        self.rng = np.random.default_rng(seed)

        # Colony state
        self.food_sources: list[FoodSource] = []
        self.employed_bees: int = 0
        self.onlooker_bees: int = 0
        self.scout_bees: int = 0

        # Optimization state
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness: float = 0.0
        self._objective_function: Callable[[np.ndarray], float] | None = None

        # History
        self.convergence_history: list[float] = []
        self.diversity_history: list[float] = []

        logger.info(f"ABC initialized with {colony_size} bees, {dimensions} dimensions")

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        max_iterations: int | None = None,
        spatial_constraints: dict[str, Any] | None = None,
        parallel_computation: bool = True,
    ) -> np.ndarray:
        """
        Optimize using Artificial Bee Colony algorithm.

        Args:
            objective_function: Function to optimize
            max_iterations: Maximum optimization iterations
            spatial_constraints: Spatial constraints for optimization
            parallel_computation: Whether to use parallel computation

        Returns:
            Optimal solution found
        """
        max_iter = (
            self.parameters.max_iterations if max_iterations is None else max_iterations
        )
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")

        logger.info(f"Starting ABC optimization with {max_iter} max iterations")

        self._objective_function = objective_function
        self.employed_bees = max(
            1,
            int(
                round(self.parameters.colony_size * self.parameters.employed_bees_ratio)
            ),
        )
        self.onlooker_bees = max(1, self.parameters.colony_size - self.employed_bees)
        self.scout_bees = max(
            1,
            int(round(self.parameters.colony_size * self.parameters.scout_bees_ratio)),
        )
        self.convergence_history = []
        self.diversity_history = []
        self.global_best_position = None
        self.global_best_fitness = 0.0
        # Initialize food sources and evaluate every source before the first
        # bee phase.  This avoids treating the default fitness of zero as a
        # meaningful objective value.
        self._initialize_food_sources()
        for source in self.food_sources:
            source.fitness = self._evaluate_fitness(source.position, objective_function)

        # Main optimization loop
        for iteration in range(max_iter):
            # Employed bee phase
            self._employed_bee_phase(objective_function)

            # Onlooker bee phase
            self._onlooker_bee_phase(objective_function)

            # Scout bee phase
            self._scout_bee_phase(objective_function)

            # Update global best
            self._update_global_best()

            # Record statistics
            self._record_iteration_stats(iteration)

            # Check for convergence or stagnation
            if self._check_convergence(iteration):
                logger.info(f"Convergence achieved at iteration {iteration}")
                break

        logger.info(
            f"ABC optimization completed: best fitness = {self.global_best_fitness}"
        )
        best = cast(np.ndarray | None, self.global_best_position)
        if best is None:
            raise RuntimeError("ABC completed without a valid food source")
        return best.copy()

    def _initialize_food_sources(self) -> None:
        """Initialize food sources randomly."""
        n_sources = max(1, self.parameters.colony_size // 2)
        self.food_sources = []

        for _ in range(n_sources):
            position = np.asarray(
                [
                    self.rng.uniform(min_bound, max_bound)
                    for min_bound, max_bound in self.parameters.bounds
                ]
            )

            source = FoodSource(position=position)
            self.food_sources.append(source)

    def _employed_bee_phase(
        self, objective_function: Callable[[np.ndarray], float]
    ) -> None:
        """Employed bee phase: local search around food sources."""
        for source in self.food_sources:
            # Generate new solution in neighborhood
            new_position = self._generate_neighbor(source.position)

            # Evaluate new solution
            new_fitness = self._evaluate_fitness(new_position, objective_function)

            # Greedy selection
            if new_fitness > source.fitness:
                source.position = new_position
                source.fitness = new_fitness
                source.trial_count = 0
            else:
                source.trial_count += 1

    def _onlooker_bee_phase(
        self, objective_function: Callable[[np.ndarray], float]
    ) -> None:
        """Onlooker bee phase: probabilistic selection and local search."""
        # Calculate selection probabilities based on fitness
        probabilities = self._calculate_selection_probabilities()

        for _ in range(self.onlooker_bees or len(self.food_sources)):
            # Select food source probabilistically
            selected_idx = self.rng.choice(len(self.food_sources), p=probabilities)
            source = self.food_sources[selected_idx]

            # Generate neighbor and evaluate
            new_position = self._generate_neighbor(source.position)
            new_fitness = self._evaluate_fitness(new_position, objective_function)

            # Update if better
            if new_fitness > source.fitness:
                source.position = new_position
                source.fitness = new_fitness
                source.trial_count = 0

    def _scout_bee_phase(
        self, objective_function: Callable[[np.ndarray], float] | None = None
    ) -> None:
        """Scout bee phase: abandon poor food sources and explore randomly."""
        for source in self.food_sources:
            if source.trial_count >= self.parameters.limit:
                # Abandon this food source
                new_position = np.array(
                    [
                        self.rng.uniform(min_bound, max_bound)
                        for min_bound, max_bound in self.parameters.bounds
                    ]
                )

                source.position = new_position
                source.fitness = (
                    self._evaluate_fitness(new_position, objective_function)
                    if objective_function is not None
                    else 0.0
                )
                source.trial_count = 0

    def _generate_neighbor(self, position: np.ndarray) -> np.ndarray:
        """Generate neighbor solution using ABC formula."""
        # Select random dimension
        if not self.food_sources:
            raise RuntimeError("Initialize food sources before generating a neighbor")
        dimension = int(self.rng.integers(0, len(position)))

        # Find current source index using safe element-wise comparison
        current_idx = next(
            (
                idx
                for idx, s in enumerate(self.food_sources)
                if np.array_equal(s.position, position)
            ),
            None,
        )

        # Select random neighbor food source (different from current)
        neighbor_idx = int(self.rng.integers(0, len(self.food_sources)))
        if current_idx is not None and len(self.food_sources) > 1:
            choices = [
                idx for idx in range(len(self.food_sources)) if idx != current_idx
            ]
            neighbor_idx = int(self.rng.choice(choices))

        neighbor_position = self.food_sources[neighbor_idx].position

        # Generate new position
        phi = self.rng.uniform(-1, 1)  # Random factor
        new_position = position.copy()
        new_position[dimension] = position[dimension] + phi * (
            position[dimension] - neighbor_position[dimension]
        )

        # Ensure bounds
        min_bound, max_bound = self.parameters.bounds[dimension]
        new_position[dimension] = np.clip(new_position[dimension], min_bound, max_bound)

        return new_position

    def _evaluate_fitness(
        self, position: np.ndarray, objective_function: Callable[[np.ndarray], float]
    ) -> float:
        """Evaluate fitness of a solution."""
        # ABC uses maximization, so convert minimization problem
        raw_value = objective_function(position)
        if not np.isscalar(raw_value) or not np.isfinite(raw_value):
            raise ValueError("objective_function must return a finite scalar")
        objective_value = float(cast(Any, raw_value))
        if objective_value >= 0:
            return 1.0 / (1.0 + objective_value)
        else:
            return 1.0 + abs(objective_value)

    def _calculate_selection_probabilities(self) -> np.ndarray:
        """Calculate selection probabilities for onlooker bees."""
        if not self.food_sources:
            return np.array([])

        fitness_values = np.array([source.fitness for source in self.food_sources])
        total_fitness = np.sum(fitness_values)

        if not np.isfinite(total_fitness) or total_fitness <= 0:
            return np.ones(len(self.food_sources)) / len(self.food_sources)

        return np.asarray(fitness_values / total_fitness)

    def _update_global_best(self) -> None:
        """Update global best solution."""
        for source in self.food_sources:
            if source.fitness > self.global_best_fitness:
                self.global_best_fitness = source.fitness
                self.global_best_position = source.position.copy()

    def _record_iteration_stats(self, iteration: int) -> None:
        """Record statistics for current iteration."""
        self.convergence_history.append(self.global_best_fitness)

        # Calculate diversity
        if len(self.food_sources) > 1:
            positions = np.array([s.position for s in self.food_sources])
            position_mean = np.mean(positions, axis=0)
            diversity = np.mean(
                [np.linalg.norm(pos - position_mean) for pos in positions]
            )
            self.diversity_history.append(diversity)

    def _check_convergence(self, iteration: int) -> bool:
        """Check if algorithm has converged."""
        if len(self.convergence_history) < 10:
            return False

        # Simple convergence check based on fitness stability
        recent_fitness = np.array(self.convergence_history[-10:])
        fitness_change = np.abs(np.diff(recent_fitness))
        avg_change = np.mean(fitness_change)

        return bool(avg_change < 1e-6)

    def manage_food_sources(
        self,
        current_sources: list[FoodSource],
        abandonment_criteria: str = "trial_limit",
        recruitment_strategy: str = "fitness_proportional",
        spatial_clustering: bool = True,
    ) -> dict[str, Any]:
        """
        Manage food sources with advanced strategies.

        Args:
            current_sources: Current food sources
            abandonment_criteria: Criteria for abandoning sources
            recruitment_strategy: Strategy for recruiting bees
            spatial_clustering: Whether to consider spatial clustering

        Returns:
            Management results and updated sources
        """
        logger.info(f"Managing {len(current_sources)} food sources")

        management_results = {
            "sources_abandoned": 0,
            "new_sources_created": 0,
            "sources_updated": 0,
            "clustering_applied": spatial_clustering,
        }

        # Apply abandonment criteria
        if abandonment_criteria == "trial_limit":
            sources_to_abandon = [
                s for s in current_sources if s.trial_count >= self.parameters.limit
            ]

            for source in sources_to_abandon:
                current_sources.remove(source)
                management_results["sources_abandoned"] += 1

        # Create new sources if needed
        while len(current_sources) < self.parameters.colony_size // 2:
            new_position = np.array(
                [
                    self.rng.uniform(min_bound, max_bound)
                    for min_bound, max_bound in self.parameters.bounds
                ]
            )

            new_source = FoodSource(position=new_position)
            current_sources.append(new_source)
            management_results["new_sources_created"] += 1

        # Apply recruitment strategy
        if recruitment_strategy == "fitness_proportional":
            management_results["recruits_evaluated"] = (
                self._apply_fitness_proportional_recruitment(current_sources)
            )

        management_results["sources_updated"] = len(current_sources)

        logger.info(f"Food source management completed: {management_results}")
        return management_results

    def _apply_fitness_proportional_recruitment(self, sources: list[FoodSource]) -> int:
        """Return the number of recruits that can be evaluated from current state.

        Recruitment requires an objective function to assign fitness to a new
        candidate.  The management API does not receive one, so it reports no
        evaluated recruits instead of silently inventing fitness values.
        """
        if not sources or self._objective_function is None:
            return 0
        recruitment_probs = self._calculate_selection_probabilities()
        n_recruitment = max(1, len(sources) // 4)
        evaluated = 0
        for _ in range(n_recruitment):
            selected_idx = int(self.rng.choice(len(sources), p=recruitment_probs))
            candidate = self._generate_neighbor(sources[selected_idx].position)
            fitness = self._evaluate_fitness(candidate, self._objective_function)
            if fitness > sources[selected_idx].fitness:
                sources[selected_idx].position = candidate
                sources[selected_idx].fitness = fitness
                sources[selected_idx].trial_count = 0
                evaluated += 1
        return evaluated

    def adapt_foraging_strategy(
        self,
        environmental_conditions: dict[str, Any],
        colony_performance: dict[str, Any],
        behavioral_adaptation: str = "learning_automaton",
    ) -> dict[str, Any]:
        """
        Adapt foraging strategy based on environmental conditions and performance.

        Args:
            environmental_conditions: Current environmental conditions
            colony_performance: Current colony performance metrics
            behavioral_adaptation: Type of behavioral adaptation

        Returns:
            Adaptation results and updated strategy
        """
        logger.info(f"Adapting foraging strategy using {behavioral_adaptation}")

        adaptation_results: dict[str, Any] = {
            "adaptation_type": behavioral_adaptation,
            "parameters_updated": {},
            "strategy_changes": [],
        }

        if behavioral_adaptation == "learning_automaton":
            # Adapt based on success rate
            success_rate = colony_performance.get("success_rate", 0.5)

            if success_rate < 0.3:
                # Low success - increase exploration
                self.parameters.max_trials = min(100, self.parameters.max_trials * 1.2)
                adaptation_results["parameters_updated"]["max_trials"] = (
                    self.parameters.max_trials
                )
                adaptation_results["strategy_changes"].append("increased_exploration")

            elif success_rate > 0.8:
                # High success - increase exploitation
                self.parameters.max_trials = max(20, self.parameters.max_trials * 0.8)
                adaptation_results["parameters_updated"]["max_trials"] = (
                    self.parameters.max_trials
                )
                adaptation_results["strategy_changes"].append("increased_exploitation")

        elif behavioral_adaptation == "environmental_response":
            # Adapt to environmental conditions
            if "resource_density" in environmental_conditions:
                density = environmental_conditions["resource_density"]

                if density < 0.3:
                    # Scarce resources - be more persistent
                    self.parameters.limit = min(200, self.parameters.limit * 1.5)
                    adaptation_results["parameters_updated"]["limit"] = (
                        self.parameters.limit
                    )
                    adaptation_results["strategy_changes"].append(
                        "increased_persistence"
                    )

                elif density > 0.8:
                    # Abundant resources - be less persistent
                    self.parameters.limit = max(50, self.parameters.limit * 0.7)
                    adaptation_results["parameters_updated"]["limit"] = (
                        self.parameters.limit
                    )
                    adaptation_results["strategy_changes"].append(
                        "decreased_persistence"
                    )

        logger.info(
            f"Foraging strategy adaptation completed: {len(adaptation_results['strategy_changes'])} changes"
        )
        return adaptation_results

    def get_optimization_statistics(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "algorithm": "Artificial Bee Colony",
            "parameters": {
                "colony_size": self.parameters.colony_size,
                "dimensions": self.parameters.dimensions,
                "employed_bees_ratio": self.parameters.employed_bees_ratio,
                "scout_bees_ratio": self.parameters.scout_bees_ratio,
                "max_trials": self.parameters.max_trials,
                "limit": self.parameters.limit,
            },
            "optimization_results": {
                "best_fitness": self.global_best_fitness,
                "food_sources": len(self.food_sources),
                "iterations_completed": len(self.convergence_history),
            },
        }

        # Food source statistics
        if self.food_sources:
            fitness_values = [s.fitness for s in self.food_sources]
            trial_counts = [s.trial_count for s in self.food_sources]

            stats["food_source_statistics"] = {
                "max_fitness": np.max(fitness_values),
                "min_fitness": np.min(fitness_values),
                "avg_fitness": np.mean(fitness_values),
                "avg_trials": np.mean(trial_counts),
                "abandoned_sources": len(
                    [
                        s
                        for s in self.food_sources
                        if s.trial_count >= self.parameters.limit
                    ]
                ),
            }

        # Performance statistics
        if len(self.convergence_history) > 1:
            improvement = self.convergence_history[-1] - self.convergence_history[0]
            stats["performance"] = {
                "total_improvement": improvement,
                "improvement_rate": improvement / len(self.convergence_history),
                "convergence_stability": np.std(self.convergence_history[-10:]),
            }

        return stats
