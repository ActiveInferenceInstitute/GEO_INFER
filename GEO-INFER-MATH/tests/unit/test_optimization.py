"""
Tests for GEO-INFER-MATH optimization module.

Tests cover: GradientDescentOptimizer, GeneticAlgorithmOptimizer,
ScipyOptimizer, MultiObjectiveOptimizer, and OptimizationManager.
"""

import numpy as np
import pytest


from geo_infer_math.core.optimization import (
    OptimizationConfig,
    GradientDescentOptimizer,
    GeneticAlgorithmOptimizer,
    ScipyOptimizer,
    MultiObjectiveOptimizer,
    OptimizationManager,
    create_optimization_manager,
    optimize_function,
)


def quadratic(x: np.ndarray) -> float:
    """Simple quadratic: f(x) = sum(x^2), minimum at origin."""
    return float(np.sum(x**2))


def rosenbrock_2d(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2."""
    return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


class TestGradientDescentOptimizer:
    """Tests for gradient descent optimizer."""

    def test_optimizes_quadratic(self):
        config = OptimizationConfig(
            max_iterations=500,
            learning_rate=0.1,
            momentum=0.0,
            tolerance=1e-8,
        )
        optimizer = GradientDescentOptimizer(config)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = optimizer.optimize(
            quadratic, bounds, initial_guess=np.array([3.0, 4.0])
        )
        assert result["success"] is True
        assert result["fun"] < 0.01
        assert np.all(np.abs(result["x"]) < 0.1)

    def test_convergence_history_populated(self):
        config = OptimizationConfig(max_iterations=100, learning_rate=0.1, momentum=0.0)
        optimizer = GradientDescentOptimizer(config)
        bounds = [(-5.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds, initial_guess=np.array([3.0]))
        assert len(result["convergence_history"]) > 1
        # Values should generally decrease
        assert result["convergence_history"][-1] <= result["convergence_history"][0]

    def test_custom_gradient_function(self):
        def grad(x):
            return 2 * x

        config = OptimizationConfig(max_iterations=200, learning_rate=0.1, momentum=0.0)
        optimizer = GradientDescentOptimizer(config)
        bounds = [(-5.0, 5.0)]
        result = optimizer.optimize(
            quadratic, bounds, initial_guess=np.array([4.0]), gradient_function=grad
        )
        assert result["fun"] < 0.01

    def test_bounds_respected(self):
        config = OptimizationConfig(max_iterations=100, learning_rate=0.5, momentum=0.0)
        optimizer = GradientDescentOptimizer(config)
        bounds = [(1.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds, initial_guess=np.array([3.0]))
        assert result["x"][0] >= 1.0
        assert result["x"][0] <= 5.0

    def test_divergent_learning_rate_reports_failure(self):
        # lr=10 on a quadratic diverges / plateaus at the clip bounds; the
        # optimizer must report success=False instead of a false positive.
        config = OptimizationConfig(
            max_iterations=100,
            learning_rate=10.0,
            momentum=0.0,
            tolerance=1e-6,
        )
        optimizer = GradientDescentOptimizer(config)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = optimizer.optimize(
            quadratic, bounds, initial_guess=np.array([3.0, 4.0])
        )
        assert result["success"] is False


class TestGeneticAlgorithmOptimizer:
    """Tests for genetic algorithm optimizer."""

    def test_optimizes_quadratic(self):
        config = OptimizationConfig(
            max_iterations=100,
            population_size=30,
            mutation_rate=0.1,
            crossover_rate=0.7,
            tolerance=1e-6,
            random_seed=42,
        )
        optimizer = GeneticAlgorithmOptimizer(config)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds)
        assert result["success"] is True
        assert result["fun"] < 1.0

    def test_get_best_solution_after_optimization(self):
        config = OptimizationConfig(
            max_iterations=50,
            population_size=20,
            random_seed=42,
        )
        optimizer = GeneticAlgorithmOptimizer(config)
        bounds = [(-5.0, 5.0)]
        optimizer.optimize(quadratic, bounds)
        best_x, best_val = optimizer.get_best_solution()
        assert best_val < 5.0
        assert best_x is not None

    def test_raises_before_optimization(self):
        config = OptimizationConfig()
        optimizer = GeneticAlgorithmOptimizer(config)
        with pytest.raises(ValueError):
            optimizer.get_best_solution()


class TestScipyOptimizer:
    """Tests for scipy-backed optimizer."""

    def test_lbfgsb_quadratic(self):
        config = OptimizationConfig(max_iterations=100)
        optimizer = ScipyOptimizer(config)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds, method="L-BFGS-B")
        assert result["success"] is True
        assert result["fun"] < 1e-6

    def test_differential_evolution(self):
        config = OptimizationConfig(max_iterations=100, random_seed=42)
        optimizer = ScipyOptimizer(config)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds, method="differential_evolution")
        assert result["fun"] < 1e-4

    def test_default_initial_guess(self):
        config = OptimizationConfig()
        optimizer = ScipyOptimizer(config)
        bounds = [(-5.0, 5.0)]
        result = optimizer.optimize(quadratic, bounds, method="L-BFGS-B")
        assert "x" in result


class TestMultiObjectiveOptimizer:
    """Tests for multi-objective optimizer."""

    def test_multi_objective_two_functions(self):
        def f1(x):
            return float(x[0] ** 2)

        def f2(x):
            return float((x[0] - 2) ** 2)

        config = OptimizationConfig(
            max_iterations=20,
            population_size=20,
            random_seed=42,
        )
        optimizer = MultiObjectiveOptimizer(config)
        bounds = [(-5.0, 5.0)]
        result = optimizer.optimize([f1, f2], bounds)
        assert result["success"] is True
        assert "pareto_front" in result
        # Optimal trade-off is around x=1 (midpoint)
        assert result["x"][0] > -3 and result["x"][0] < 5


class TestOptimizationManager:
    """Tests for optimization manager."""

    def test_create_manager(self):
        manager = create_optimization_manager()
        assert "gradient_descent" in manager.optimizers
        assert "genetic_algorithm" in manager.optimizers
        assert "scipy_lbfgs" in manager.optimizers

    def test_optimize_with_manager(self):
        manager = OptimizationManager(
            OptimizationConfig(
                max_iterations=100,
                learning_rate=0.1,
                momentum=0.0,
            )
        )
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = manager.optimize(quadratic, bounds, method="gradient_descent")
        assert result["fun"] < 1.0

    def test_unknown_method_raises(self):
        manager = OptimizationManager()
        bounds = [(-5.0, 5.0)]
        with pytest.raises(ValueError, match="Unknown optimization method"):
            manager.optimize(quadratic, bounds, method="nonexistent")

    def test_convenience_function(self):
        result = optimize_function(quadratic, [(-5.0, 5.0), (-5.0, 5.0)])
        assert "x" in result
        assert "fun" in result
