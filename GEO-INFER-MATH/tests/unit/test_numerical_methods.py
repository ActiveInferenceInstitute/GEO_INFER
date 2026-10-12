"""
Tests for the numerical_methods module.
"""

import numpy as np
import pytest
from geo_infer_math.core.numerical_methods import (
    SpatialInterpolator,
    SpatialOptimizer,
    ODESolver,
    PDEsolver,
    numerical_integration,
    find_root,
    minimize_scalar_function,
)


class TestSpatialInterpolator:
    """Test spatial interpolation functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create simple test data
        self.coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        self.values = np.array([10, 20, 30, 40, 100])
        self.query_coords = np.array([[0.25, 0.25], [0.75, 0.75]])

    def test_kriging_interpolation(self):
        """Test Kriging interpolation."""
        interpolator = SpatialInterpolator(method="kriging")
        interpolator.fit(self.coords, self.values)

        predictions = interpolator.predict(self.query_coords)

        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_rbf_interpolation(self):
        """Test RBF interpolation."""
        interpolator = SpatialInterpolator(method="rbf")
        interpolator.fit(self.coords, self.values)

        predictions = interpolator.predict(self.query_coords)

        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_spline_interpolation(self):
        """Test spline interpolation."""
        interpolator = SpatialInterpolator(method="spline")
        interpolator.fit(self.coords, self.values)

        predictions = interpolator.predict(self.query_coords)

        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_invalid_method(self):
        """Test handling of invalid interpolation method."""
        with pytest.raises(ValueError):
            SpatialInterpolator(method="invalid")

    def test_unfitted_prediction(self):
        """Test prediction without fitting."""
        interpolator = SpatialInterpolator()
        with pytest.raises(ValueError):
            interpolator.predict(self.query_coords)


class TestSpatialOptimizer:
    """Test spatial optimization functionality."""

    def test_gradient_descent_optimization(self):
        """Test gradient descent optimization."""

        def objective(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

        def gradient(x):
            return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])

        optimizer = SpatialOptimizer(method="gradient_descent")
        bounds = [(-10, 10), (-10, 10)]

        result = optimizer.minimize(
            objective,
            bounds,
            initial_guess=np.array([0, 0]),
            gradient_function=gradient,
        )

        assert result.success
        assert abs(result.x[0] - 2) < 0.1
        assert abs(result.x[1] - 3) < 0.1

    def test_newton_optimization(self):
        """Test Newton's method optimization."""

        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        optimizer = SpatialOptimizer(method="newton")
        bounds = [(-10, 10), (-10, 10)]

        result = optimizer.minimize(objective, bounds, initial_guess=np.array([5, 5]))

        assert result.success
        assert abs(result.x[0]) < 0.1
        assert abs(result.x[1]) < 0.1

    def test_simulated_annealing(self):
        """Test simulated annealing optimization."""

        def objective(x):
            return (x[0] - 1) ** 2 + (x[1] - 1) ** 2

        optimizer = SpatialOptimizer(method="simulated_annealing")
        bounds = [(-10, 10), (-10, 10)]

        # Seed the annealing trajectory so the quality bounds below are
        # deterministic regardless of test-execution order or xdist worker.
        result = optimizer.minimize(
            objective, bounds, initial_guess=np.array([5, 5]), seed=42
        )
        replay = optimizer.minimize(
            objective, bounds, initial_guess=np.array([5, 5]), seed=42
        )
        np.testing.assert_array_equal(result.x, replay.x)

        assert result.success
        assert abs(result.x[0] - 1) < 1.0  # SA might not be as precise
        assert abs(result.x[1] - 1) < 1.0

    def test_invalid_method(self):
        """Test handling of invalid optimization method."""

        def objective(x):
            return x[0] ** 2

        optimizer = SpatialOptimizer(method="invalid")
        bounds = [(-10, 10)]

        with pytest.raises(ValueError):
            optimizer.minimize(objective, bounds)


class TestODESolver:
    """Test ODE solving functionality."""

    def test_simple_ode(self):
        """Test solving a simple ODE."""

        def ode(t, y):
            return -2 * y  # dy/dt = -2y, solution: y = y0 * exp(-2t)

        solver = ODESolver(method="RK45")
        t_span = (0, 2)
        y0 = [1]

        result = solver.solve(ode, t_span, y0)

        assert result.success
        assert len(result.t) > 1
        assert len(result.y) == 1
        assert len(result.y[0]) == len(result.t)

        # Check that solution decays exponentially
        assert result.y[0][-1] < result.y[0][0]

    def test_system_of_odes(self):
        """Test solving a system of ODEs."""

        def ode_system(t, y):
            return [-y[0], y[0] - y[1]]  # Predator-prey like system

        solver = ODESolver(method="RK45")
        t_span = (0, 1)
        y0 = [1, 0.5]

        result = solver.solve(ode_system, t_span, y0)

        assert result.success
        assert len(result.t) > 1
        assert len(result.y) == 2
        assert len(result.y[0]) == len(result.t)
        assert len(result.y[1]) == len(result.t)


class TestPDEsolver:
    """Test PDE solving functionality."""

    def test_diffusion_equation(self):
        """Test solving 1D diffusion equation."""
        solver = PDEsolver(method="finite_difference")

        # Initial condition: delta function at center
        n_points = 50
        initial_condition = np.zeros(n_points)
        initial_condition[n_points // 2] = 1

        diffusion_coefficient = 0.4
        time_steps = 100
        dt = 0.01
        dx = 0.1

        solution = solver.solve_diffusion(
            initial_condition, diffusion_coefficient, time_steps, dt, dx
        )

        assert solution.shape == (time_steps + 1, n_points)

        # Check conservation of mass (approximately)
        total_mass = np.sum(solution, axis=1)
        assert abs(total_mass[0] - total_mass[-1]) < 0.1

        # Check that solution spreads out over time (peak decreases as distribution flattens)
        peak_values = np.max(solution, axis=1)
        assert peak_values[-1] < peak_values[0]

    def test_wave_equation(self):
        """Test solving 1D wave equation."""
        solver = PDEsolver(method="finite_difference")

        n_points = 50
        initial_displacement = np.zeros(n_points)
        initial_displacement[n_points // 2] = 1  # Initial pulse

        initial_velocity = np.zeros(n_points)

        wave_speed = 1.0
        time_steps = 20
        dt = 0.01
        dx = 0.1

        displacement, velocity = solver.solve_wave_equation(
            initial_displacement, initial_velocity, wave_speed, time_steps, dt, dx
        )

        assert displacement.shape == (time_steps + 1, n_points)
        assert velocity.shape == (time_steps + 1, n_points)

        # Check boundary conditions
        assert np.allclose(displacement[:, 0], 0)
        assert np.allclose(displacement[:, -1], 0)


class TestNumericalIntegration:
    """Test numerical integration functionality."""

    def test_trapezoidal_integration(self):
        """Test trapezoidal rule integration."""

        def func(x):
            return x**2

        result = numerical_integration(func, 0, 1, method="trapezoidal", n_points=1000)
        expected = 1 / 3  # Analytical result

        assert abs(result - expected) < 0.01

    def test_simpson_integration(self):
        """Test Simpson's rule integration."""

        def func(x):
            return np.sin(x)

        result = numerical_integration(func, 0, np.pi, method="simpson", n_points=1000)
        expected = 2  # Analytical result

        assert abs(result - expected) < 0.01

    def test_invalid_method(self):
        """Test handling of invalid integration method."""

        def func(x):
            return x

        with pytest.raises(ValueError):
            numerical_integration(func, 0, 1, method="invalid")


class TestRootFinding:
    """Test root finding functionality."""

    def test_find_root(self):
        """Test root finding."""

        def func(x):
            return x**2 - 4  # Roots at x = -2, 2

        root = find_root(func, (-3, -1), method="brentq")
        assert abs(root + 2) < 0.01

        root = find_root(func, (1, 3), method="brentq")
        assert abs(root - 2) < 0.01


class TestScalarMinimization:
    """Test scalar function minimization."""

    def test_minimize_scalar(self):
        """Test scalar minimization."""

        def func(x):
            return (x - 3) ** 2 + 1

        result = minimize_scalar_function(func, (0, 5), method="bounded")
        assert abs(result - 3) < 0.01

    def test_minimize_with_failure(self):
        """Test minimization with potential failure."""

        def func(x):
            return float("inf")  # Always returns infinity

        result = minimize_scalar_function(func, (0, 1))
        assert np.isnan(result)
