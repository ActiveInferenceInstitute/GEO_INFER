"""
Mathematical Optimization Methods

This module provides optimization algorithms for geospatial applications
in the GEO-INFER framework.
"""

import numpy as np
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from scipy.optimize import minimize, differential_evolution, basinhopping

from ..utils.rng import resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""

    # General parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    random_seed: int | None = None

    # Algorithm-specific parameters
    population_size: int = 50  # For genetic algorithms
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Basin hopping parameters
    n_iter: int = 100
    T: float = 1.0
    stepsize: float = 0.5

    # Gradient descent parameters
    learning_rate: float = 0.01
    momentum: float = 0.9


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(self, config: OptimizationConfig | None = None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.best_solution: np.ndarray | None = None
        self.best_value: float | None = None
        self.convergence_history: list[float] = []
        self.rng = resolve_rng(self.config.random_seed)

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize the objective function."""
        raise RuntimeError("Optimization subclasses must implement optimize()")

    def get_best_solution(self) -> tuple[np.ndarray, float]:
        """Get the best solution found."""
        if self.best_solution is None or self.best_value is None:
            raise ValueError("No optimization has been performed yet")
        return self.best_solution, self.best_value


class GradientDescentOptimizer(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)
        self.gradient_function: Callable | None = None

    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        gradient_function: Callable | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize using gradient descent.

        Args:
            objective_function: Function to minimize
            bounds: Parameter bounds
            initial_guess: Initial parameter values
            gradient_function: Gradient function (optional)
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        obj_fn: Callable
        if isinstance(objective_function, list):
            obj_fn = objective_function[0]
        else:
            obj_fn = objective_function

        if initial_guess is None:
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])

        self.gradient_function = gradient_function
        current_params = initial_guess.copy()
        current_value = float(obj_fn(current_params))

        self.best_solution = current_params.copy()
        self.best_value = current_value
        self.convergence_history = [current_value]

        velocity = np.zeros_like(current_params)
        for iteration in range(self.config.max_iterations):
            # Calculate gradient
            if self.gradient_function is not None:
                gradient = self.gradient_function(current_params)
            else:
                gradient = self._numerical_gradient(obj_fn, current_params)

            # Update velocity with momentum
            velocity = (
                self.config.momentum * velocity - self.config.learning_rate * gradient
            )

            # Update parameters
            new_params = current_params + velocity

            # Apply bounds
            new_params = np.clip(
                new_params, [b[0] for b in bounds], [b[1] for b in bounds]
            )

            # Evaluate new solution
            new_value = float(obj_fn(new_params))

            # Update best solution
            if new_value < self.best_value:
                self.best_solution = new_params.copy()
                self.best_value = new_value

            # Check convergence
            if abs(new_value - current_value) < self.config.tolerance:
                logger.info(f"Converged at iteration {iteration}")
                break

            current_params = new_params
            current_value = new_value
            self.convergence_history.append(current_value)

        # Success requires actual progress on the initial best value and a
        # finite objective: budget-exhausted-without-improvement (including
        # clipped oscillation plateaus) and non-finite objectives are
        # failures, not successes.
        success = bool(
            np.isfinite(self.best_value)
            and self.best_value < self.convergence_history[0]
        )

        return {
            "success": success,
            "x": self.best_solution,
            "fun": self.best_value,
            "nit": len(self.convergence_history),
            "convergence_history": self.convergence_history,
        }

    def _numerical_gradient(
        self, objective_function: Callable, params: np.ndarray, epsilon: float = 1e-7
    ) -> np.ndarray:
        """Calculate numerical gradient."""
        gradient = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon

            gradient[i] = (
                objective_function(params_plus) - objective_function(params_minus)
            ) / (2 * epsilon)

        return gradient


class GeneticAlgorithmOptimizer(Optimizer):
    """Genetic algorithm optimizer."""

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)

    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize using genetic algorithm.

        Args:
            objective_function: Function to minimize
            bounds: Parameter bounds
            initial_guess: Initial parameter values (ignored for GA)

        Returns:
            Optimization results
        """
        if isinstance(objective_function, list):
            obj_fn: Callable = objective_function[0]
        else:
            obj_fn = objective_function

        n_params = len(bounds)

        # Initialize population
        population = self._initialize_population(n_params, bounds)
        fitness = np.array([obj_fn(ind) for ind in population])

        self.best_solution = population[np.argmin(fitness)].copy()
        self.best_value = float(np.min(fitness))
        self.convergence_history = [self.best_value]

        for generation in range(self.config.max_iterations):
            # Selection
            parents = self._selection(population, fitness)

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            offspring = self._mutation(offspring, bounds)

            # Evaluate offspring
            offspring_fitness = np.array([obj_fn(ind) for ind in offspring])

            # Replace worst individuals
            worst_indices = np.argsort(fitness)[-len(offspring) :]
            population[worst_indices] = offspring
            fitness[worst_indices] = offspring_fitness

            # Update best solution
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_value:
                self.best_solution = population[min_fitness_idx].copy()
                self.best_value = float(fitness[min_fitness_idx])

            self.convergence_history.append(self.best_value)

            # Check convergence
            if len(self.convergence_history) > 10:
                recent_improvement = abs(
                    self.convergence_history[-1] - self.convergence_history[-10]
                )
                if recent_improvement < self.config.tolerance:
                    logger.info(f"Converged at generation {generation}")
                    break

        # Success requires actual progress on the initial best value and a
        # finite objective: budget-exhausted-without-improvement and
        # non-finite objectives are failures, not successes.
        success = bool(
            np.isfinite(self.best_value)
            and self.best_value < self.convergence_history[0]
        )

        return {
            "success": success,
            "x": self.best_solution,
            "fun": self.best_value,
            "nit": len(self.convergence_history),
            "convergence_history": self.convergence_history,
        }

    def _initialize_population(
        self, n_params: int, bounds: list[tuple[float, float]]
    ) -> np.ndarray:
        """Initialize random population."""
        population = []
        for _ in range(self.config.population_size):
            individual = np.array([self.rng.uniform(b[0], b[1]) for b in bounds])
            population.append(individual)
        return np.array(population)

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection."""
        n_parents = len(population) // 2
        parents = []

        for _ in range(n_parents):
            # Tournament selection
            tournament_size = 3
            tournament_indices = self.rng.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])

        return np.array(parents)

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Uniform crossover."""
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Uniform crossover
                mask = self.rng.random(len(parent1)) < self.config.crossover_rate
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)

                offspring.extend([child1, child2])

        return np.array(offspring)

    def _mutation(
        self, offspring: np.ndarray, bounds: list[tuple[float, float]]
    ) -> np.ndarray:
        """Gaussian mutation."""
        mutated = offspring.copy()

        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                if self.rng.random() < self.config.mutation_rate:
                    # Gaussian mutation
                    sigma = (bounds[j][1] - bounds[j][0]) * 0.1
                    mutated[i, j] += self.rng.normal(0, sigma)
                    mutated[i, j] = np.clip(mutated[i, j], bounds[j][0], bounds[j][1])

        return mutated


class ScipyOptimizer(Optimizer):
    """Wrapper for scipy optimization methods."""

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)

    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        method: str = "L-BFGS-B",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize using scipy methods.

        Args:
            objective_function: Function to minimize
            bounds: Parameter bounds
            initial_guess: Initial parameter values
            method: Optimization method

        Returns:
            Optimization results
        """
        if isinstance(objective_function, list):
            obj_fn: Callable = objective_function[0]
        else:
            obj_fn = objective_function

        if initial_guess is None:
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])

        try:
            if method == "differential_evolution":
                result = differential_evolution(
                    obj_fn,
                    bounds,
                    maxiter=self.config.max_iterations,
                    tol=self.config.tolerance,
                    seed=self.config.random_seed,
                )
            elif method == "basin_hopping":
                result = basinhopping(
                    obj_fn,
                    initial_guess,
                    niter=self.config.n_iter,
                    T=self.config.T,
                    stepsize=self.config.stepsize,
                    minimizer_kwargs={"bounds": bounds},
                )
            else:
                result = minimize(
                    obj_fn,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": self.config.max_iterations},
                )

            self.best_solution = result.x
            self.best_value = float(result.fun)

            return {
                "success": result.success,
                "x": result.x,
                "fun": result.fun,
                "nit": getattr(result, "nit", 0),
                "message": result.message,
            }

        except Exception as e:
            logger.error(f"Scipy optimization failed: {e}")
            return {
                "success": False,
                "x": initial_guess,
                "fun": obj_fn(initial_guess),
                "error": str(e),
            }


class MultiObjectiveOptimizer(Optimizer):
    """Multi-objective optimization using NSGA-II."""

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)

    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize multiple objectives using NSGA-II.

        Args:
            objective_function: List of functions to minimize (or single function)
            bounds: Parameter bounds
            initial_guess: Initial parameter values (ignored for NSGA-II)

        Returns:
            Optimization results
        """
        if isinstance(objective_function, list):
            objective_functions = objective_function
        else:
            objective_functions = [objective_function]

        n_params = len(bounds)

        # Initialize population
        population = self._initialize_population(n_params, bounds)

        # Evaluate objectives
        objectives = np.array(
            [[f(ind) for f in objective_functions] for ind in population]
        )

        self.best_solution = population[np.argmin(np.sum(objectives, axis=1))].copy()
        self.best_value = float(np.min(np.sum(objectives, axis=1)))
        self.convergence_history = [self.best_value]

        for generation in range(self.config.max_iterations):
            # Generate offspring
            offspring = self._generate_offspring(population, bounds)
            offspring_objectives = np.array(
                [[f(ind) for f in objective_functions] for ind in offspring]
            )

            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_objectives])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_obj)

            # Select next generation
            new_population: list[np.ndarray] = []
            new_objectives: list[np.ndarray] = []

            for front in fronts:
                if len(new_population) + len(front) <= self.config.population_size:
                    new_population.extend(combined_pop[front])
                    new_objectives.extend(combined_obj[front])
                else:
                    # Crowding distance selection
                    remaining = self.config.population_size - len(new_population)
                    front_pop = combined_pop[front]
                    front_obj = combined_obj[front]
                    selected_indices = self._crowding_distance_selection(
                        front_pop, front_obj, remaining
                    )
                    new_population.extend(front_pop[selected_indices])
                    new_objectives.extend(front_obj[selected_indices])
                    break

            population = np.array(new_population)
            objectives = np.array(new_objectives)

            # Update best solution
            total_objectives = np.sum(objectives, axis=1)
            min_idx = np.argmin(total_objectives)
            if total_objectives[min_idx] < self.best_value:
                self.best_solution = population[min_idx].copy()
                self.best_value = float(total_objectives[min_idx])

            self.convergence_history.append(self.best_value)

            # Check convergence (stall on the aggregated objective)
            if len(self.convergence_history) > 10:
                recent_improvement = abs(
                    self.convergence_history[-1] - self.convergence_history[-10]
                )
                if recent_improvement < self.config.tolerance:
                    logger.info(f"Converged at generation {generation}")
                    break

        # Success requires actual progress on the initial best value and a
        # finite objective: budget-exhausted-without-improvement and
        # non-finite objectives are failures, not successes.
        success = bool(
            np.isfinite(self.best_value)
            and self.best_value < self.convergence_history[0]
        )

        return {
            "success": success,
            "x": self.best_solution,
            "fun": self.best_value,
            "pareto_front": objectives,
            "nit": len(self.convergence_history),
        }

    def _initialize_population(
        self, n_params: int, bounds: list[tuple[float, float]]
    ) -> np.ndarray:
        """Initialize random population."""
        population = []
        for _ in range(self.config.population_size):
            individual = np.array([self.rng.uniform(b[0], b[1]) for b in bounds])
            population.append(individual)
        return np.array(population)

    def _generate_offspring(
        self, population: np.ndarray, bounds: list[tuple[float, float]]
    ) -> np.ndarray:
        """Generate offspring using crossover and mutation."""
        offspring = []

        for _ in range(len(population)):
            # Select parents
            parent1, parent2 = self.rng.choice(len(population), 2, replace=False)

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child = (population[parent1] + population[parent2]) / 2
            else:
                child = population[parent1].copy()

            # Mutation
            for j in range(len(child)):
                if self.rng.random() < self.config.mutation_rate:
                    sigma = (bounds[j][1] - bounds[j][0]) * 0.1
                    child[j] += self.rng.normal(0, sigma)
                    child[j] = np.clip(child[j], bounds[j][0], bounds[j][1])

            offspring.append(child)

        return np.array(offspring)

    def _non_dominated_sort(self, objectives: np.ndarray) -> list[list[int]]:
        """Perform non-dominated sorting."""
        n_points = len(objectives)
        domination_count = np.zeros(n_points)
        dominated_solutions: list[list[int]] = [[] for _ in range(n_points)]

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1

        fronts = []
        current_front = np.where(domination_count == 0)[0].tolist()

        while current_front:
            fronts.append(current_front)
            next_front = []

            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front = next_front

        return fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2."""
        return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))

    def _crowding_distance_selection(
        self, population: np.ndarray, objectives: np.ndarray, n_select: int
    ) -> list[int]:
        """Select individuals using crowding distance."""
        if len(population) <= n_select:
            return list(range(len(population)))

        # Calculate crowding distance
        distances = np.zeros(len(population))

        for obj_idx in range(objectives.shape[1]):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_range = (
                objectives[sorted_indices[-1], obj_idx]
                - objectives[sorted_indices[0], obj_idx]
            )
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], obj_idx]
                        - objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range

        # Select individuals with highest crowding distance
        selected_indices = np.argsort(distances)[-n_select:]
        return cast(list[int], selected_indices.tolist())


class OptimizationManager:
    """Manager for multiple optimization methods."""

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        """
        Initialize optimization manager.

        Args:
            config: Configuration for optimization methods
        """
        self.config = config or OptimizationConfig()
        self.optimizers: dict[str, Optimizer] = {}
        self._initialize_optimizers()

    def _initialize_optimizers(self) -> None:
        """Initialize all optimization methods."""
        self.optimizers = {
            "gradient_descent": GradientDescentOptimizer(self.config),
            "genetic_algorithm": GeneticAlgorithmOptimizer(self.config),
            "scipy_lbfgs": ScipyOptimizer(self.config),
            "scipy_de": ScipyOptimizer(self.config),
            "scipy_bh": ScipyOptimizer(self.config),
            "multi_objective": MultiObjectiveOptimizer(self.config),
        }

        logger.info(f"Initialized {len(self.optimizers)} optimization methods")

    def optimize(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        method: str = "scipy_lbfgs",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform optimization.

        Args:
            objective_function: Function to minimize
            bounds: Parameter bounds
            method: Optimization method
            **kwargs: Additional method-specific parameters

        Returns:
            Optimization results
        """
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")

        optimizer = self.optimizers[method]

        if method == "scipy_de":
            return optimizer.optimize(
                objective_function, bounds, method="differential_evolution", **kwargs
            )
        elif method == "scipy_bh":
            return optimizer.optimize(
                objective_function, bounds, method="basin_hopping", **kwargs
            )
        else:
            return optimizer.optimize(objective_function, bounds, **kwargs)

    def compare_methods(
        self,
        objective_function: Callable | list[Callable],
        bounds: list[tuple[float, float]],
        methods: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare different optimization methods.

        Args:
            objective_function: Function to minimize
            bounds: Parameter bounds
            methods: List of methods to compare (uses all if None)

        Returns:
            Dictionary with comparison results
        """
        if methods is None:
            methods = list(self.optimizers.keys())

        results = {}

        for method in methods:
            try:
                result = self.optimize(objective_function, bounds, method)
                results[method] = {
                    "success": result.get("success", False),
                    "best_value": result.get("fun", float("inf")),
                    "iterations": result.get("nit", 0),
                    "best_solution": result.get("x", None),
                }
                logger.info(f"Completed optimization with {method}")

            except Exception as e:
                logger.error(f"Failed to optimize with {method}: {e}")
                results[method] = {
                    "success": False,
                    "best_value": float("inf"),
                    "iterations": 0,
                    "best_solution": None,
                    "error": str(e),
                }

        return results


# Convenience functions
def create_optimization_manager(
    config: OptimizationConfig | None = None,
) -> OptimizationManager:
    """Create a new optimization manager."""
    return OptimizationManager(config)


def optimize_function(
    objective_function: Callable,
    bounds: list[tuple[float, float]],
    method: str = "scipy_lbfgs",
) -> dict[str, Any]:
    """Convenience function for optimization."""
    config = OptimizationConfig()
    manager = OptimizationManager(config)
    return manager.optimize(objective_function, bounds, method)


def compare_optimization_methods(
    objective_function: Callable, bounds: list[tuple[float, float]]
) -> dict[str, dict[str, Any]]:
    """Compare different optimization methods."""
    config = OptimizationConfig()
    manager = OptimizationManager(config)
    return manager.compare_methods(objective_function, bounds)
