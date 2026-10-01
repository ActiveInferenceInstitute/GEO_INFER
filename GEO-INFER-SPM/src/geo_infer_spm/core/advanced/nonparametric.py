"""
Nonparametric Methods for Statistical Parametric Mapping

This module implements nonparametric statistical methods for SPM analysis,
providing distribution-free alternatives to parametric GLM approaches.
Nonparametric methods are particularly useful when:

- Data violates parametric assumptions (normality, homoscedasticity)
- Sample sizes are small
- Outliers are present
- Relationships are nonlinear

Implemented Methods:
- Local regression (LOESS/LOWESS)
- Kernel regression
- Splines and smoothing splines
- Generalized additive models (GAM)
- Quantile regression
- Robust regression methods
"""

import numpy as np
from typing import Any, cast

from ...models.data_models import SPMData, SPMResult, DesignMatrix


class NonparametricSPM:
    """
    Nonparametric Statistical Parametric Mapping

    This class implements nonparametric regression and smoothing methods
    for SPM analysis, providing flexible alternatives to parametric GLM.

    Attributes:
        method: Nonparametric method to use
        bandwidth: Smoothing parameter (if applicable)
        kernel: Kernel function for kernel-based methods
        fitted_model: Fitted nonparametric model
    """

    def __init__(
        self,
        method: str = "loess",
        bandwidth: float | None = None,
        kernel: str = "gaussian",
    ):
        """
        Initialize nonparametric SPM.

        Args:
            method: Nonparametric method ('loess', 'kernel', 'spline', 'gam')
            bandwidth: Smoothing bandwidth parameter
            kernel: Kernel function for kernel methods
        """
        self.method = method.lower()
        self.bandwidth = bandwidth
        self.kernel = kernel.lower()
        self.fitted_model: dict[str, Any] | None = None

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate method parameters."""
        valid_methods = ["loess", "lowess", "kernel", "spline", "gam", "robust"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        valid_kernels = ["gaussian", "epanechnikov", "uniform", "triangular"]
        if self.kernel not in valid_kernels:
            raise ValueError(f"Kernel must be one of {valid_kernels}")

    def fit(
        self,
        data: SPMData,
        design_matrix: DesignMatrix,
        response_var: str | None = None,
    ) -> SPMResult:
        """
        Fit nonparametric model to SPM data.

        Args:
            data: SPMData containing response and predictors
            design_matrix: Design matrix (used for structure, not parametric fitting)
            response_var: Name of response variable (if data has multiple)

        Returns:
            SPMResult with nonparametric fit
        """
        # Extract predictors and response
        X = design_matrix.matrix
        y = self._extract_response(data, response_var)

        # Fit nonparametric model
        if self.method in ["loess", "lowess"]:
            y_hat, weights, diagnostics = self._fit_loess(X, y)
        elif self.method == "kernel":
            y_hat, weights, diagnostics = self._fit_kernel_regression(X, y)
        elif self.method == "spline":
            y_hat, weights, diagnostics = self._fit_spline(X, y)
        elif self.method == "gam":
            y_hat, weights, diagnostics = self._fit_gam(X, y)
        elif self.method == "robust":
            y_hat, weights, diagnostics = self._fit_robust_regression(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute residuals
        residuals = y - y_hat

        # Store fitted model
        self.fitted_model = {
            "y_hat": y_hat,
            "weights": weights,
            "diagnostics": diagnostics,
            "method": self.method,
            "bandwidth": self.bandwidth,
        }

        # Create SPMResult
        result = SPMResult(
            spm_data=data,
            design_matrix=design_matrix,
            beta_coefficients=np.array([]),  # Nonparametric, no beta coefficients
            residuals=residuals,
            model_diagnostics={
                "method": f"Nonparametric_{self.method}",
                "r_squared": diagnostics.get("r_squared", 0),
                "bandwidth": self.bandwidth,
                "kernel": self.kernel if self.method == "kernel" else None,
                "converged": diagnostics.get("converged", True),
            },
        )

        return result

    def _extract_response(self, data: SPMData, response_var: str | None) -> np.ndarray:
        """Extract response variable from SPMData."""
        if isinstance(data.data, np.ndarray):
            if data.data.ndim == 1:
                return data.data
            else:
                # Multiple variables - use first column or specified variable
                covariates = data.covariates or {}
                if response_var and response_var in covariates:
                    return covariates[response_var]
                else:
                    return data.data[:, 0]
        else:
            raise TypeError("Nonparametric methods require array data")

    def _fit_loess(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit LOESS (Locally Estimated Scatterplot Smoothing).

        LOESS fits local polynomial regressions at each point,
        weighted by distance from the point.
        """
        n_points = len(y)
        y_hat = np.zeros(n_points)

        # Use first predictor for simplicity (could be extended to multivariate)
        x = X[:, 0] if X.shape[1] > 0 else np.arange(n_points)

        # Determine bandwidth
        if self.bandwidth is None:
            self.bandwidth = min(0.5, 20 / n_points)  # Adaptive bandwidth

        weights = np.zeros((n_points, n_points))

        for i in range(n_points):
            # Compute distances
            distances = np.abs(x - x[i])

            # Compute tricube weights
            max_dist = max(
                np.percentile(distances, self.bandwidth * 100),
                np.finfo(float).eps,
            )
            u = distances / max_dist
            w = (1 - u**3) ** 3
            w[u > 1] = 0  # Only use nearby points

            weights[i, :] = w

            # Local polynomial fit (degree 1)
            W = np.diag(w)
            X_local = np.column_stack([np.ones(n_points), x - x[i]])

            try:
                beta = np.linalg.pinv(X_local.T @ W @ X_local) @ (X_local.T @ W @ y)
                y_hat[i] = beta[0] + beta[1] * 0  # Evaluate at center
            except np.linalg.LinAlgError:
                y_hat[i] = np.mean(y[w > 0]) if np.any(w > 0) else y[i]

        # Compute diagnostics
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics = {
            "r_squared": r_squared,
            "bandwidth": self.bandwidth,
            "converged": True,
        }

        return y_hat, weights, diagnostics

    def _fit_kernel_regression(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit kernel regression model.

        Kernel regression uses kernel-weighted local averaging.
        """
        n_points = len(y)
        y_hat = np.zeros(n_points)

        # Use first predictor
        x = X[:, 0] if X.shape[1] > 0 else np.arange(n_points)

        # Determine bandwidth
        if self.bandwidth is None:
            self.bandwidth = 1.06 * np.std(x) * n_points ** (-1 / 5)  # Scott's rule

        weights = np.zeros((n_points, n_points))

        for i in range(n_points):
            # Compute kernel weights
            u = (x - x[i]) / self.bandwidth

            if self.kernel == "gaussian":
                w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
            elif self.kernel == "epanechnikov":
                w = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
            elif self.kernel == "uniform":
                w = 0.5 * (np.abs(u) <= 1)
            elif self.kernel == "triangular":
                w = (1 - np.abs(u)) * (np.abs(u) <= 1)
            else:
                w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)  # Default to Gaussian

            weights[i, :] = w
            y_hat[i] = np.sum(w * y) / np.sum(w) if np.sum(w) > 0 else y[i]

        # Compute diagnostics
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics = {
            "r_squared": r_squared,
            "bandwidth": self.bandwidth,
            "kernel": self.kernel,
            "converged": True,
        }

        return y_hat, weights, diagnostics

    def _fit_spline(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit smoothing spline.
        Ships a moving-average approximation of a smoothing spline; see
        docs/deferred_statistical_methods.md ("Smoothing spline / GAM").
        """
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Smoothing spline / GAM").
        x = X[:, 0] if X.shape[1] > 0 else np.arange(len(y))

        # Sort data by x
        sort_idx = np.argsort(x)
        _x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        if self.bandwidth is None:
            self.bandwidth = 0.1

        window_size = max(3, int(self.bandwidth * len(y)))
        y_hat_sorted = np.convolve(
            y_sorted, np.ones(window_size) / window_size, mode="same"
        )

        # Unsort results
        y_hat = np.zeros_like(y)
        y_hat[sort_idx] = y_hat_sorted

        # Identity weights (see register: "Smoothing spline / GAM").
        weights = np.eye(len(y))

        # Compute diagnostics
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics = {
            "r_squared": r_squared,
            "bandwidth": self.bandwidth,
            "method": "simplified_spline",
            "converged": True,
        }

        return y_hat, weights, diagnostics

    def _fit_gam(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Fit Generalized Additive Model.

        Ships per-predictor smoothing summed without backfitting; see
        docs/deferred_statistical_methods.md ("Smoothing spline / GAM").
        """
        n_predictors = X.shape[1]
        n_points = len(y)

        # Fit smooth functions for each predictor
        smooth_components = np.zeros((n_points, n_predictors))

        for j in range(n_predictors):
            x_j = X[:, j]

            if self.bandwidth is None:
                bw = 0.1
            else:
                bw = self.bandwidth

            # Local polynomial smoothing
            smooth_components[:, j] = self._local_smooth(x_j, y, bw)

        # Combine smooth components
        y_hat = np.sum(smooth_components, axis=1)
        # Identity weights (see register: "Smoothing spline / GAM").
        weights = np.eye(n_points)

        # Compute diagnostics
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics = {
            "r_squared": r_squared,
            "bandwidth": self.bandwidth,
            "n_predictors": n_predictors,
            "converged": True,
        }

        return y_hat, weights, diagnostics

    def _fit_robust_regression(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit robust regression using iteratively reweighted least squares.

        Robust to outliers in the response variable.
        """
        n_points, n_predictors = X.shape

        # Add intercept if not present
        if n_predictors == 0 or not np.allclose(X[:, 0], 1.0):
            X = np.column_stack([np.ones(n_points), X])
            n_predictors += 1

        # Initial OLS fit
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)

        # Iteratively reweighted least squares with Huber weights
        max_iter = 20
        tol = 1e-6

        n_iter = 0
        for _ in range(max_iter):
            n_iter += 1
            # Compute residuals
            residuals = y - X @ beta

            # Robust scale estimate (MAD)
            scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

            # Huber weights
            k = 1.345  # Tuning constant
            r_norm = residuals / scale
            weights = np.where(np.abs(r_norm) <= k, 1.0, k / np.abs(r_norm))
            weights = np.clip(weights, 0, 1)  # Ensure weights are in [0, 1]

            # Weighted least squares
            W = np.diag(weights)
            beta_new = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

            # Check convergence
            if np.max(np.abs(beta_new - beta)) < tol:
                break

            beta = beta_new

        y_hat = X @ beta

        # Identity weights for compatibility
        weights = np.eye(n_points)

        # Compute robust R-squared
        ss_res = np.sum(weights * (y - y_hat) ** 2)
        ss_tot = np.sum(weights * (y - np.median(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        diagnostics = {
            "r_squared": r_squared,
            "robust_scale": scale,
            "n_iterations": n_iter,
            "converged": n_iter < max_iter,
        }

        return y_hat, weights, diagnostics

    def _local_smooth(
        self, x: np.ndarray, y: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """Local smoothing for GAM components."""
        n_points = len(x)
        y_smooth = np.zeros(n_points)

        for i in range(n_points):
            # Gaussian kernel weights
            weights = np.exp(-0.5 * ((x - x[i]) / bandwidth) ** 2)
            weights /= np.sum(weights)

            y_smooth[i] = np.sum(weights * y)

        return y_smooth

    def temporal_basis_functions(
        self, time_points: np.ndarray, n_basis: int = 5, basis_type: str = "fourier"
    ) -> np.ndarray:
        """
        Generate temporal basis functions for time series modeling.

        Args:
            time_points: Array of time points
            n_basis: Number of basis functions
            basis_type: Type of basis ('fourier', 'polynomial', 'bspline')

        Returns:
            Basis matrix of shape (n_timepoints, n_basis)
        """
        import numpy as np

        n_points = len(time_points)
        t_range = float(time_points.max() - time_points.min())
        t_norm = (time_points - time_points.min()) / (t_range if t_range > 0 else 1.0)

        if basis_type == "fourier":
            basis = np.zeros((n_points, n_basis))
            basis[:, 0] = 1.0
            for k in range(1, n_basis):
                freq = (k + 1) // 2
                if k % 2 == 1:
                    basis[:, k] = np.sin(2 * np.pi * freq * t_norm)
                else:
                    basis[:, k] = np.cos(2 * np.pi * freq * t_norm)
        elif basis_type == "polynomial":
            basis = np.zeros((n_points, n_basis))
            for k in range(n_basis):
                basis[:, k] = t_norm**k
        elif basis_type == "bspline":
            basis = np.zeros((n_points, n_basis))
            knot_positions = np.linspace(0, 1, n_basis)
            width = 1.0 / max(n_basis - 1, 1)
            for k in range(n_basis):
                basis[:, k] = np.exp(-0.5 * ((t_norm - knot_positions[k]) / width) ** 2)
        else:
            raise ValueError(
                f"Unknown basis type: {basis_type}. Must be 'fourier', 'polynomial', or 'bspline'"
            )

        return basis

    def predict(self, new_data: SPMData) -> np.ndarray:
        """
        Make predictions using fitted nonparametric model.

        Args:
            new_data: New data for prediction

        Returns:
            Predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Deferred: see docs/deferred_statistical_methods.md
        # ("Out-of-sample prediction (nonparametric)").
        return cast(np.ndarray, self.fitted_model["y_hat"])

    def get_smooth_components(self) -> np.ndarray | None:
        """
        Get smooth function components (for GAM).

        Returns:
            Array of smooth components or None
        """
        if self.fitted_model is None:
            return None

        return self.fitted_model.get("smooth_components")


def fit_nonparametric(
    data: SPMData, design_matrix: DesignMatrix, method: str = "loess", **kwargs: Any
) -> SPMResult:
    """
    Convenience function to fit nonparametric SPM model.

    Args:
        data: SPMData containing response variable
        design_matrix: Design matrix for predictors
        method: Nonparametric method to use
        **kwargs: Additional arguments passed to NonparametricSPM

    Returns:
        SPMResult with nonparametric fit

    Example:
        >>> result = fit_nonparametric(data, design_matrix, method='kernel', bandwidth=0.1)
    """
    model = NonparametricSPM(method=method, **kwargs)
    return model.fit(data, design_matrix)
