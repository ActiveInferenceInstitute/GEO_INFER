"""
Numerical Methods Module

This module provides specialized numerical algorithms for solving mathematical
problems arising in geospatial contexts, including interpolation, optimization,
and solving differential equations.
"""

import numpy as np
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import solve_ivp, quad, simpson as scipy_simpson, trapezoid
from scipy.interpolate import interp1d, RBFInterpolator
import logging

from geo_infer_math.utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class InterpolationResult:
    """Container for interpolation results."""

    values: np.ndarray
    method: str
    parameters: dict[str, Any]
    error_estimate: np.ndarray | None = None


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    x: np.ndarray
    fun: float
    success: bool
    method: str
    nfev: int
    message: str


@dataclass
class ODEsolution:
    """Container for ODE solution."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    method: str
    message: str


class SpatialInterpolator:
    """Advanced spatial interpolation methods."""

    def __init__(self, method: str = "kriging"):
        """
        Initialize spatial interpolator.

        Args:
            method: Interpolation method ('kriging', 'spline', 'rbf')
        """
        valid_methods = ("kriging", "spline", "rbf")
        if method not in valid_methods:
            raise ValueError(
                f"Unknown interpolation method: {method}. Must be one of {valid_methods}"
            )
        self.method = method
        self.trained = False
        self.training_points: np.ndarray | None = None
        self.training_values: np.ndarray | None = None
        self.parameters: dict[str, Any] = {}

    def fit(
        self, points: np.ndarray, values: np.ndarray, **kwargs: Any
    ) -> "SpatialInterpolator":
        """
        Fit the interpolator to training data.

        Args:
            points: Training point coordinates (n_points, 2)
            values: Training values (n_points,)
            **kwargs: Method-specific parameters

        Returns:
            Self for method chaining
        """
        self.training_points = points.copy()
        self.training_values = values.copy()
        self.parameters.update(kwargs)

        if self.method == "kriging":
            self._fit_kriging()
        elif self.method == "spline":
            self._fit_spline()
        elif self.method == "rbf":
            self._fit_rbf()

        self.trained = True
        return self

    def predict(self, query_points: np.ndarray) -> np.ndarray:
        """
        Predict values at query points.

        Args:
            query_points: Points to interpolate (n_points, 2)

        Returns:
            Interpolated values
        """
        if not self.trained:
            raise ValueError("Interpolator must be fitted before prediction")

        if self.method == "kriging":
            return self._predict_kriging(query_points)
        elif self.method == "spline":
            return self._predict_spline(query_points)
        elif self.method == "rbf":
            return self._predict_rbf(query_points)
        raise ValueError("Interpolator must be fitted and method known")

    def _fit_kriging(self) -> None:
        """Fit Kriging model."""
        assert self.training_points is not None
        assert self.training_values is not None
        # Calculate distances between all training points
        n_points = len(self.training_points)
        distances = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = np.sqrt(
                    np.sum((self.training_points[i] - self.training_points[j]) ** 2)
                )

        # Use spherical variogram model by default
        sill = np.var(self.training_values)
        range_param = np.max(distances) * 0.3
        nugget = sill * 0.1

        self.parameters.update(
            {
                "sill": sill,
                "range": range_param,
                "nugget": nugget,
                "distances": distances,
            }
        )

    def _predict_kriging(self, query_points: np.ndarray) -> np.ndarray:
        """Predict using ordinary kriging (extended-system solve)."""
        assert self.training_points is not None
        assert self.training_values is not None

        n_points = len(self.training_points)
        k_matrix = self._spherical_variogram(self.parameters["distances"])
        np.fill_diagonal(k_matrix, 0.0)
        extended = np.ones((n_points + 1, n_points + 1), dtype=np.float64)
        extended[:n_points, :n_points] = k_matrix
        extended[n_points, n_points] = 0.0

        predictions = []
        for query_point in query_points:
            # Distances from the query point to the training points
            distances = np.sqrt(
                np.sum((self.training_points - query_point) ** 2, axis=1)
            )
            variogram_values = self._spherical_variogram(distances)

            # Ordinary-kriging extended system:
            #   [ K   1 ] [w]   [gamma(h0)]
            #   [ 1^T 0 ] [mu] = [    1    ]
            rhs = np.concatenate([variogram_values, [1.0]])
            try:
                solution = np.linalg.solve(extended, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Kriging system is singular; check for duplicate "
                    "training points or a degenerate variogram"
                ) from exc
            weights = solution[:n_points]
            predictions.append(float(np.sum(weights * self.training_values)))

        return np.array(predictions)

    def _spherical_variogram(self, h: np.ndarray) -> np.ndarray:
        """Spherical variogram model."""
        sill = self.parameters["sill"]
        range_param = self.parameters["range"]
        nugget = self.parameters["nugget"]

        h = np.asarray(h, dtype=np.float64)
        result = np.zeros_like(h)
        mask = (h <= range_param) & (h > 0.0)
        result[mask] = nugget + (sill - nugget) * (
            1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param) ** 3
        )
        result[h > range_param] = sill
        return result

    def _fit_spline(self) -> None:
        """Fit a spline interpolation model.

        For 1-D training data a cubic ``scipy.interpolate.interp1d`` is
        constructed over the sorted abscissae. For scattered 2-D data the
        thin-plate-spline kernel of ``scipy.interpolate.RBFInterpolator``
        is used, which is the natural generalization of the cubic spline to
        scattered points.
        """
        assert self.training_points is not None
        assert self.training_values is not None
        points = np.asarray(self.training_points, dtype=np.float64)
        values = np.asarray(self.training_values, dtype=np.float64)

        if points.ndim == 1 or points.shape[1] == 1:
            x = points.ravel()
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = values[order]
            unique_x, unique_idx = np.unique(x_sorted, return_index=True)
            self.parameters["_spline_1d"] = interp1d(
                unique_x, y_sorted[unique_idx], kind="cubic"
            )
        else:
            self.parameters["_spline_2d"] = RBFInterpolator(
                points, values, kernel="thin_plate_spline"
            )
        self.parameters["fitted"] = True

    def _predict_spline(self, query_points: np.ndarray) -> np.ndarray:
        """Predict using the fitted spline model."""
        assert self.training_points is not None
        assert self.training_values is not None
        queries = np.asarray(query_points, dtype=np.float64)

        spline_1d = self.parameters.get("_spline_1d")
        if spline_1d is not None:
            return np.asarray(spline_1d(queries.ravel()), dtype=np.float64)

        spline_2d = self.parameters.get("_spline_2d")
        if spline_2d is not None:
            return np.asarray(spline_2d(np.atleast_2d(queries)), dtype=np.float64)

        raise ValueError("Spline model has not been fitted")

    def _fit_rbf(self) -> None:
        """Fit Radial Basis Function interpolation (exact kernel solve)."""
        assert self.training_points is not None
        assert self.training_values is not None
        epsilon = self.parameters.get("epsilon", 1.0)
        function = self.parameters.get("function", "multiquadric")

        points = np.asarray(self.training_points, dtype=np.float64)
        values = np.asarray(self.training_values, dtype=np.float64)
        pair_distances = np.sqrt(
            np.sum(
                (points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        kernel_matrix = self._rbf_function(pair_distances, epsilon, function)
        try:
            kernel_weights = np.linalg.solve(kernel_matrix, values)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "RBF kernel matrix is singular for the given training "
                "points; check for duplicates or a degenerate basis"
            ) from exc

        self.parameters.update(
            {
                "epsilon": epsilon,
                "function": function,
                "kernel_weights": kernel_weights,
            }
        )

    def _predict_rbf(self, query_points: np.ndarray) -> np.ndarray:
        """Predict using the solved RBF kernel weights."""
        assert self.training_points is not None
        assert self.training_values is not None
        queries = np.atleast_2d(np.asarray(query_points, dtype=np.float64))
        kernel_weights = self.parameters["kernel_weights"]

        predictions = []
        for query_point in queries:
            distances = np.sqrt(
                np.sum((self.training_points - query_point) ** 2, axis=1)
            )
            basis_values = self._rbf_function(
                distances, self.parameters["epsilon"], self.parameters["function"]
            )
            predictions.append(float(np.dot(basis_values, kernel_weights)))

        return np.array(predictions)

    def _rbf_function(
        self, r: float | np.ndarray, epsilon: float, function: str
    ) -> float | np.ndarray:
        """Radial basis function kernel values."""
        r_arr = np.asarray(r, dtype=np.float64)
        if function == "multiquadric":
            values = np.sqrt(1 + (epsilon * r_arr) ** 2)
        elif function == "inverse_multiquadric":
            values = 1.0 / np.sqrt(1 + (epsilon * r_arr) ** 2)
        elif function == "gaussian":
            values = np.exp(-((epsilon * r_arr) ** 2))
        elif function == "thin_plate":
            values = r_arr**2 * np.log(r_arr + 1e-10)
        else:
            values = np.exp(-r_arr)  # Default exponential
        if np.isscalar(r):
            return float(values)
        return values


class SpatialOptimizer:
    """Optimization methods for spatial problems."""

    def __init__(self, method: str = "gradient_descent"):
        """
        Initialize spatial optimizer.

        Args:
            method: Optimization method
        """
        self.method = method
        self.objective_function: Callable[..., Any] | None = None
        self.constraints: list[Any] = []

    def minimize(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        initial_guess: np.ndarray | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Minimize objective function.

        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds
            initial_guess: Initial parameter values
            **kwargs: Method-specific parameters (``simulated_annealing``
                accepts ``seed`` for reproducible proposals).

        Returns:
            Optimization result
        """
        self.objective_function = objective

        if initial_guess is None:
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])

        if self.method == "gradient_descent":
            return self._gradient_descent(objective, bounds, initial_guess, **kwargs)
        elif self.method == "newton":
            return self._newton_method(objective, bounds, initial_guess, **kwargs)
        elif self.method == "simulated_annealing":
            return self._simulated_annealing(objective, bounds, initial_guess, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def _gradient_descent(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        x0: np.ndarray,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        tolerance: float = 1e-6,
        gradient_function: Callable | None = None,
    ) -> OptimizationResult:
        """Gradient descent optimization."""
        x = x0.copy()
        n_evaluations = 0

        for iteration in range(max_iter):
            # Evaluate objective and gradient
            objective(
                x
            )  # keep the objective call: user functions may count invocations
            if gradient_function is not None:
                gradient = gradient_function(x)
                n_evaluations += 1
            else:
                gradient = self._numerical_gradient(objective, x)
                n_evaluations += len(x) + 1

            # Update parameters
            x_new = x - learning_rate * gradient

            # Apply bounds
            x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])

            # Check convergence
            if np.linalg.norm(x_new - x) < tolerance:
                return OptimizationResult(
                    x=x_new,
                    fun=objective(x_new),
                    success=True,
                    method="gradient_descent",
                    nfev=n_evaluations,
                    message=f"Converged after {iteration} iterations",
                )

            x = x_new

        return OptimizationResult(
            x=x,
            fun=objective(x),
            success=False,
            method="gradient_descent",
            nfev=n_evaluations,
            message="Maximum iterations reached",
        )

    def _newton_method(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        x0: np.ndarray,
        max_iter: int = 100,
    ) -> OptimizationResult:
        """Newton's method optimization."""
        x = x0.copy()
        n_evaluations = 0

        for iteration in range(max_iter):
            # Evaluate objective, gradient, and Hessian
            objective(
                x
            )  # keep the objective call: user functions may count invocations
            gradient = self._numerical_gradient(objective, x)
            hessian = self._numerical_hessian(objective, x)
            n_evaluations += len(x) ** 2 + len(x) + 1

            # Solve for Newton step
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Hessian is singular, use gradient descent step
                step = gradient

            # Update parameters
            x_new = x - step

            # Apply bounds
            x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])

            # Check convergence
            if np.linalg.norm(x_new - x) < 1e-6:
                return OptimizationResult(
                    x=x_new,
                    fun=objective(x_new),
                    success=True,
                    method="newton",
                    nfev=n_evaluations,
                    message=f"Converged after {iteration} iterations",
                )

            x = x_new

        return OptimizationResult(
            x=x,
            fun=objective(x),
            success=False,
            method="newton",
            nfev=n_evaluations,
            message="Maximum iterations reached",
        )

    def _simulated_annealing(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        x0: np.ndarray,
        max_iter: int = 1000,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        seed: SeedLike = None,
    ) -> OptimizationResult:
        """Simulated annealing optimization.

        ``seed`` drives candidate proposals and acceptance draws; see
        :func:`geo_infer_math.utils.rng.resolve_rng`.
        """
        rng = resolve_rng(seed)
        x = x0.copy()
        current_energy = objective(x)
        best_x = x.copy()
        best_energy = current_energy
        temperature = initial_temp
        n_evaluations = 1

        for _ in range(max_iter):
            # Generate candidate solution
            candidate = x + rng.normal(0, temperature / 10, size=len(x))

            # Apply bounds
            candidate = np.clip(
                candidate, [b[0] for b in bounds], [b[1] for b in bounds]
            )

            # Evaluate candidate
            candidate_energy = objective(candidate)
            n_evaluations += 1

            # Accept or reject candidate
            delta_energy = candidate_energy - current_energy

            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / temperature):
                x = candidate
                current_energy = candidate_energy

                # Update best solution
                if current_energy < best_energy:
                    best_x = x.copy()
                    best_energy = current_energy

            # Cool down
            temperature *= cooling_rate

        return OptimizationResult(
            x=best_x,
            fun=best_energy,
            success=True,
            method="simulated_annealing",
            nfev=n_evaluations,
            message=f"Completed {max_iter} iterations",
        )

    def _numerical_gradient(
        self, objective: Callable, x: np.ndarray, epsilon: float = 1e-7
    ) -> np.ndarray:
        """Calculate numerical gradient."""
        gradient = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon

            gradient[i] = (objective(x_plus) - objective(x_minus)) / (2 * epsilon)

        return gradient

    def _numerical_hessian(
        self, objective: Callable, x: np.ndarray, epsilon: float = 1e-7
    ) -> np.ndarray:
        """Calculate numerical Hessian matrix."""
        n = len(x)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += epsilon
                x_pp[j] += epsilon
                x_pm[i] += epsilon
                x_pm[j] -= epsilon
                x_mp[i] -= epsilon
                x_mp[j] += epsilon
                x_mm[i] -= epsilon
                x_mm[j] -= epsilon

                hessian[i, j] = (
                    objective(x_pp)
                    - objective(x_pm)
                    - objective(x_mp)
                    + objective(x_mm)
                ) / (4 * epsilon**2)

        return hessian


class ODESolver:
    """ODE solver for spatial-temporal models."""

    def __init__(self, method: str = "rk45"):
        """
        Initialize ODE solver.

        Args:
            method: Integration method ('rk45', 'rk23', 'dop853', etc.)
        """
        self.method = method

    def solve(
        self,
        ode_function: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ODEsolution:
        """
        Solve ODE system.

        Args:
            ode_function: ODE function dy/dt = f(t, y)
            t_span: Time span (t_start, t_end)
            y0: Initial conditions
            t_eval: Times at which to evaluate solution
            **kwargs: Additional solver options

        Returns:
            ODE solution. Failure contract: any solver exception is logged and
            returned as an ODEsolution with empty ``t``/``y`` arrays and
            ``success=False`` (never raised). Callers MUST check
            ``solution.success`` before using the result; empty arrays are the
            sentinel for solver failure, not a valid numeric result.
        """
        try:
            result = solve_ivp(
                ode_function, t_span, y0, method=self.method, t_eval=t_eval, **kwargs
            )

            return ODEsolution(
                t=result.t,
                y=result.y,
                success=result.success,
                method=self.method,
                message=result.message,
            )

        except Exception as e:
            logger.error(f"ODE solving failed: {e}")
            return ODEsolution(
                t=np.array([]),
                y=np.array([]),
                success=False,
                method=self.method,
                message=str(e),
            )


class PDEsolver:
    """PDE solver for spatial-temporal problems."""

    def __init__(self, method: str = "finite_difference"):
        """
        Initialize PDE solver.

        Args:
            method: Solution method
        """
        self.method = method

    def solve_diffusion(
        self,
        initial_condition: np.ndarray,
        diffusion_coefficient: float,
        time_steps: int,
        dt: float,
        dx: float,
    ) -> np.ndarray:
        """
        Solve 1D diffusion equation using finite differences.

        Args:
            initial_condition: Initial concentration/temperature profile
            diffusion_coefficient: Diffusion coefficient
            time_steps: Number of time steps
            dt: Time step size
            dx: Spatial step size

        Returns:
            Solution at each time step
        """
        n_points = len(initial_condition)
        solution = np.zeros((time_steps + 1, n_points))
        solution[0] = initial_condition

        # Stability check
        stability_param = diffusion_coefficient * dt / dx**2
        if stability_param > 0.5:
            logger.warning(f"Stability condition violated: {stability_param} > 0.5")

        for t in range(time_steps):
            for i in range(1, n_points - 1):
                # Finite difference scheme
                solution[t + 1, i] = solution[t, i] + stability_param * (
                    solution[t, i + 1] - 2 * solution[t, i] + solution[t, i - 1]
                )

            # Boundary conditions (fixed ends)
            solution[t + 1, 0] = solution[t, 0]
            solution[t + 1, -1] = solution[t, -1]

        return solution

    def solve_wave_equation(
        self,
        initial_displacement: np.ndarray,
        initial_velocity: np.ndarray,
        wave_speed: float,
        time_steps: int,
        dt: float,
        dx: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve 1D wave equation using finite differences.

        Args:
            initial_displacement: Initial displacement
            initial_velocity: Initial velocity
            wave_speed: Wave propagation speed
            time_steps: Number of time steps
            dt: Time step size
            dx: Spatial step size

        Returns:
            Tuple of (displacement, velocity) at each time step
        """
        n_points = len(initial_displacement)
        displacement = np.zeros((time_steps + 1, n_points))
        velocity = np.zeros((time_steps + 1, n_points))

        displacement[0] = initial_displacement
        velocity[0] = initial_velocity

        # Stability check
        stability_param = wave_speed * dt / dx
        if stability_param > 1.0:
            logger.warning(f"Stability condition violated: {stability_param} > 1.0")

        for t in range(time_steps):
            for i in range(1, n_points - 1):
                # Update velocity
                velocity[t + 1, i] = velocity[t, i] + stability_param**2 * (
                    displacement[t, i + 1]
                    - 2 * displacement[t, i]
                    + displacement[t, i - 1]
                )

                # Update displacement
                displacement[t + 1, i] = displacement[t, i] + dt * velocity[t + 1, i]

            # Boundary conditions
            displacement[t + 1, 0] = 0
            displacement[t + 1, -1] = 0
            velocity[t + 1, 0] = 0
            velocity[t + 1, -1] = 0

        return displacement, velocity


def numerical_integration(
    func: Callable,
    a: float,
    b: float,
    method: str = "trapezoidal",
    n_points: int = 1000,
) -> float:
    """
    Numerical integration using various methods.

    Args:
        func: Function to integrate
        a: Lower limit
        b: Upper limit
        method: Integration method ('trapezoidal', 'simpson', or 'quad';
            'quad' uses scipy's adaptive Gaussian quadrature via
            scipy.integrate.quad — the former 'romberg' option was renamed
            because scipy.integrate.romberg has been removed from SciPy)
        n_points: Number of integration points (ignored for 'quad')

    Returns:
        Approximate integral value
    """
    if method == "trapezoidal":
        x = np.linspace(a, b, n_points)
        y = np.array([func(xi) for xi in x])
        return float(trapezoid(y, x))

    elif method == "simpson":
        x = np.linspace(a, b, n_points)
        y = np.array([func(xi) for xi in x])
        return float(scipy_simpson(y, x=x))

    elif method == "quad":
        result, _estimated_error = quad(func, a, b)
        return float(result)

    else:
        raise ValueError(f"Unknown integration method: {method}")


def find_root(
    func: Callable, bracket: tuple[float, float], method: str = "brentq", **kwargs: Any
) -> float:
    """
    Find root of a function.

    Args:
        func: Function for which to find root
        bracket: Initial bracket containing the root
        method: Root finding method
        **kwargs: Additional method parameters

    Returns:
        Root value. Failure contract: on failure the error is logged and
        ``np.nan`` is returned silently (no exception). Check ``np.isfinite``
        on the result to detect failure; a genuine nan from ``func`` is
        indistinguishable, so guard the objective function.
    """
    try:
        result = root_scalar(func, bracket=bracket, method=method, **kwargs)
        return float(result.root)
    except Exception as e:
        logger.error(f"Root finding failed: {e}")
        return np.nan


def minimize_scalar_function(
    func: Callable, bounds: tuple[float, float], method: str = "bounded", **kwargs: Any
) -> float:
    """
    Minimize a scalar function.

    Args:
        func: Function to minimize
        bounds: Parameter bounds
        method: Minimization method
        **kwargs: Additional method parameters

    Returns:
        Optimal parameter value. Failure contract: on failure (or a
        non-finite minimum) the error is logged and ``np.nan`` is returned
        silently (no exception). Check ``np.isfinite`` on the result to
        detect failure.
    """
    try:
        result = minimize_scalar(func, bounds=bounds, method=method, **kwargs)
        # If the minimum value is not finite, the optimization effectively failed
        if not np.isfinite(result.fun):
            return float(np.nan)
        return float(result.x)
    except Exception as e:
        logger.error(f"Scalar minimization failed: {e}")
        return np.nan


__all__ = [
    "InterpolationResult",
    "OptimizationResult",
    "ODEsolution",
    "SpatialInterpolator",
    "SpatialOptimizer",
    "ODESolver",
    "PDEsolver",
    "numerical_integration",
    "find_root",
    "minimize_scalar_function",
]
