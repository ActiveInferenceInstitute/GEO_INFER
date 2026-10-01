"""
Regression Models Module

This module provides various regression models specialized for geospatial data,
including spatial autoregressive models, geographically weighted regression,
and other spatially-aware regression techniques.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionResults:
    """Container for regression analysis results."""

    coefficients: np.ndarray
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    p_values: np.ndarray
    standard_errors: np.ndarray
    residuals: np.ndarray
    predictions: np.ndarray


class OrdinaryLeastSquares:
    """Ordinary Least Squares regression."""

    def __init__(self) -> None:
        """Initialize OLS regression."""
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquares":
        """
        Fit OLS regression model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            Self for method chaining
        """
        # Add intercept term
        X_design = np.column_stack([np.ones(X.shape[0]), X])

        # Solve normal equations
        try:
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
            self.intercept = beta[0]
            self.coefficients = beta[1:]
            self.is_fitted = True
        except np.linalg.LinAlgError:
            logger.error("Could not solve normal equations")
            self.coefficients = np.zeros(X.shape[1])
            self.intercept = np.mean(y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.coefficients is not None
        assert self.intercept is not None

        return np.asarray(X @ self.coefficients + self.intercept)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            R-squared value
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1 - (ss_res / ss_tot))


class SpatialLagModel:
    """Spatial Lag (SAR) regression model."""

    def __init__(self, weights_matrix: np.ndarray, method: str = "ml"):
        """
        Initialize spatial lag model.

        Args:
            weights_matrix: Spatial weights matrix
            method: Estimation method ('ml' for maximum likelihood, 'iv' for instrumental variables)
        """
        self.weights_matrix = weights_matrix
        self.method = method
        self.rho: float | None = None  # Spatial autoregressive parameter
        self.beta: np.ndarray | None = None  # Regression coefficients
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpatialLagModel":
        """
        Fit spatial lag model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self for method chaining
        """
        if self.method == "ml":
            self._fit_ml(X, y)
        elif self.method == "iv":
            self._fit_iv(X, y)
        else:
            raise ValueError(f"Unknown estimation method: {self.method}")

        self.is_fitted = True
        return self

    def _fit_ml(self, X: np.ndarray, y: np.ndarray) -> None:
        """Maximum likelihood estimation."""

        def log_likelihood(params: np.ndarray) -> float:
            rho = params[0]
            beta = params[1:]

            # Ensure rho is within valid range
            if abs(rho) >= 1:
                return np.inf

            # Create spatially lagged dependent variable
            Wy = self.weights_matrix @ y

            # Model: y = rho * W*y + X*beta + epsilon
            y_hat = rho * Wy + X @ beta
            residuals = y - y_hat

            # Log-likelihood for normal errors
            n = len(y)
            sigma2 = np.sum(residuals**2) / n
            loglik = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (
                2 * sigma2
            )

            return -float(loglik)  # Minimize negative log-likelihood

        # Initial parameter guess
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)
        initial_params[0] = 0.1  # Initial rho

        # Optimize
        result = minimize(
            log_likelihood,
            initial_params,
            bounds=[(-0.99, 0.99)] + [(None, None)] * n_features,
            method="L-BFGS-B",
        )

        self.rho = result.x[0]
        self.beta = result.x[1:]

    def _fit_iv(self, X: np.ndarray, y: np.ndarray) -> None:
        """Instrumental variables estimation."""
        # Create instruments: WX where W is spatial weights matrix
        WX = self.weights_matrix @ X

        # Two-stage least squares
        # First stage: regress WX on X and WX
        Z = np.column_stack([X, WX])  # Instruments
        WZ = self.weights_matrix @ Z

        # Regress y on WZ to get rho estimate
        rho_coef = np.linalg.lstsq(WZ, y, rcond=None)[0]
        self.rho = rho_coef[-1]  # Last coefficient is for WX

        # Second stage: regress (y - rho*Wy) on X
        Wy = self.weights_matrix @ y
        y_tilde = y - self.rho * Wy
        self.beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.beta is not None

        return np.asarray(X @ self.beta)


class GeographicallyWeightedRegression:
    """Geographically Weighted Regression (GWR)."""

    def __init__(self, bandwidth: float | None = None, kernel: str = "gaussian"):
        """
        Initialize GWR model.

        Args:
            bandwidth: Bandwidth parameter (if None, will be estimated)
            kernel: Kernel function ('gaussian', 'bisquare', 'tricube')
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.coordinates: np.ndarray | None = None
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, y: np.ndarray, coordinates: np.ndarray
    ) -> "GeographicallyWeightedRegression":
        """
        Fit GWR model.

        Args:
            X: Feature matrix
            y: Target values
            coordinates: Spatial coordinates (n_samples, 2)

        Returns:
            Self for method chaining
        """
        self.X = X
        self.y = y
        self.coordinates = coordinates

        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X, y, coordinates)

        self.is_fitted = True
        return self

    def _estimate_bandwidth(
        self, X: np.ndarray, y: np.ndarray, coordinates: np.ndarray
    ) -> float:
        """Estimate optimal bandwidth using cross-validation."""

        def cv_score(bandwidth: float) -> float:
            scores: list[float] = []
            for i in range(len(X)):
                # Leave-one-out cross-validation
                mask = np.arange(len(X)) != i

                X_train = X[mask]
                y_train = y[mask]
                coords_train = coordinates[mask]

                # Predict for left-out point
                pred = self._local_regression(
                    X_train, y_train, coords_train, coordinates[i], bandwidth
                )
                scores.append((y[i] - pred) ** 2)

            return float(np.mean(scores))

        # Optimize bandwidth
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(cv_score, bounds=(0.01, 2.0), method="bounded")
        return float(result.x)

    def _local_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        target_coord: np.ndarray,
        bandwidth: float,
    ) -> float:
        """Perform local regression at target coordinate."""
        # Calculate distances
        distances = np.sqrt(np.sum((coords - target_coord) ** 2, axis=1))

        # Calculate weights using kernel function
        weights = self._kernel_function(distances, bandwidth)

        # Remove points with zero weight
        valid_mask = weights > 1e-10
        if not np.any(valid_mask):
            return float(np.mean(y))  # Fallback

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        weights_valid = weights[valid_mask]

        # Weighted least squares
        W = np.diag(weights_valid)
        X_design = np.column_stack([np.ones(len(X_valid)), X_valid])

        # Solve weighted normal equations
        try:
            beta = np.linalg.lstsq(W @ X_design, W @ y_valid, rcond=None)[0]
            return float(beta[0])  # Intercept term
        except np.linalg.LinAlgError:
            return float(np.mean(y_valid))

    def _kernel_function(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Calculate kernel weights."""
        if self.kernel == "gaussian":
            return np.asarray(np.exp(-0.5 * (distances / bandwidth) ** 2))
        elif self.kernel == "bisquare":
            u = distances / bandwidth
            return np.asarray(np.where(u < 1, (1 - u**2) ** 2, 0))
        elif self.kernel == "tricube":
            u = distances / bandwidth
            return np.asarray(np.where(u < 1, (1 - u**3) ** 3, 0))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def predict(self, X: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Make predictions at given coordinates.

        Args:
            X: Feature matrix
            coordinates: Prediction coordinates

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.coordinates is not None
        assert self.bandwidth is not None

        predictions = []

        for coord in coordinates:
            pred = self._local_regression(
                self.X, self.y, self.coordinates, coord, self.bandwidth
            )
            predictions.append(pred)

        return np.array(predictions)


class SpatialErrorModel:
    """Spatial Error Model (SEM)."""

    def __init__(self, weights_matrix: np.ndarray):
        """
        Initialize spatial error model.

        Args:
            weights_matrix: Spatial weights matrix
        """
        self.weights_matrix = weights_matrix
        self.lambda_param: float | None = None  # Spatial error parameter
        self.beta: np.ndarray | None = None  # Regression coefficients
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpatialErrorModel":
        """
        Fit spatial error model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self for method chaining
        """

        def log_likelihood(params: np.ndarray) -> float:
            lambda_param = params[0]
            beta = params[1:]

            # Ensure lambda is within valid range
            if abs(lambda_param) >= 1:
                return np.inf

            # Model: y = X*beta + u, where u = lambda*W*u + epsilon
            y_hat = X @ beta

            # Create spatially autocorrelated errors
            residuals = y - y_hat

            # Log-likelihood
            n = len(y)
            sigma2 = np.sum(residuals**2) / n
            loglik = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (
                2 * sigma2
            )

            return -float(loglik)

        # Initial parameter guess
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)
        initial_params[0] = 0.1  # Initial lambda

        # Optimize
        result = minimize(
            log_likelihood,
            initial_params,
            bounds=[(-0.99, 0.99)] + [(None, None)] * n_features,
            method="L-BFGS-B",
        )

        self.lambda_param = result.x[0]
        self.beta = result.x[1:]
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.beta is not None

        return np.asarray(X @ self.beta)


class SpatialDurbinModel:
    """Spatial Durbin Model (SDM)."""

    def __init__(self, weights_matrix: np.ndarray):
        """
        Initialize spatial Durbin model.

        Args:
            weights_matrix: Spatial weights matrix
        """
        self.weights_matrix = weights_matrix
        self.rho: float | None = None  # Spatial lag parameter
        self.beta: np.ndarray | None = None  # Direct effects
        self.theta: np.ndarray | None = None  # Indirect effects
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpatialDurbinModel":
        """
        Fit spatial Durbin model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self for method chaining
        """
        # Model: y = rho*W*y + X*beta + W*X*theta + epsilon
        WX = self.weights_matrix @ X

        # Create design matrix
        X_design = np.column_stack([X, WX])

        def log_likelihood(params: np.ndarray) -> float:
            rho = params[0]
            other_params = params[1:]

            # Ensure rho is within valid range
            if abs(rho) >= 1:
                return np.inf

            # Spatially lagged dependent variable
            Wy = self.weights_matrix @ y

            # Model prediction
            y_hat = rho * Wy + X_design @ other_params
            residuals = y - y_hat

            # Log-likelihood
            n = len(y)
            sigma2 = np.sum(residuals**2) / n
            loglik = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (
                2 * sigma2
            )

            return -float(loglik)

        # Initial parameter guess
        n_params = X.shape[1] * 2 + 1  # rho + beta + theta
        initial_params = np.zeros(n_params)
        initial_params[0] = 0.1  # Initial rho

        # Optimize
        result = minimize(
            log_likelihood,
            initial_params,
            bounds=[(-0.99, 0.99)] + [(None, None)] * (n_params - 1),
            method="L-BFGS-B",
        )

        self.rho = result.x[0]
        n_features = X.shape[1]
        self.beta = result.x[1 : n_features + 1]
        self.theta = result.x[n_features + 1 :]
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.beta is not None
        assert self.theta is not None

        WX = self.weights_matrix @ X
        X_design = np.column_stack([X, WX])
        other_params = np.concatenate([self.beta, self.theta])

        return np.asarray(X_design @ other_params)


def spatial_regression_analysis(
    X: np.ndarray, y: np.ndarray, coordinates: np.ndarray, model_type: str = "ols"
) -> dict[str, Any]:
    """
    Perform spatial regression analysis.

    Args:
        X: Feature matrix
        y: Target values
        coordinates: Spatial coordinates
        model_type: Type of regression model ('ols', 'sar', 'gwr', 'sem', 'sdm')

    Returns:
        Dictionary containing analysis results
    """
    results: dict[str, Any] = {}

    if model_type == "ols":
        ols_model = OrdinaryLeastSquares()
        ols_model.fit(X, y)
        results["model"] = ols_model
        results["coefficients"] = ols_model.coefficients
        results["intercept"] = ols_model.intercept
        results["r_squared"] = ols_model.score(X, y)

    elif model_type == "sar":
        from ..core.linalg_tensor import MatrixOperations  # noqa: F811

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            coordinates, method="inverse_distance", k=5
        )

        sar_model = SpatialLagModel(weights_matrix)
        sar_model.fit(X, y)
        results["model"] = sar_model
        results["rho"] = sar_model.rho
        results["coefficients"] = sar_model.beta

    elif model_type == "gwr":
        gwr_model = GeographicallyWeightedRegression()
        gwr_model.fit(X, y, coordinates)
        results["model"] = gwr_model
        results["bandwidth"] = gwr_model.bandwidth

    elif model_type == "sem":
        from ..core.linalg_tensor import MatrixOperations  # noqa: F811

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            coordinates, method="inverse_distance", k=5
        )

        sem_model = SpatialErrorModel(weights_matrix)
        sem_model.fit(X, y)
        results["model"] = sem_model
        results["lambda"] = sem_model.lambda_param
        results["coefficients"] = sem_model.beta

    elif model_type == "sdm":
        from ..core.linalg_tensor import MatrixOperations  # noqa: F811

        weights_matrix = MatrixOperations.spatial_weights_matrix(
            coordinates, method="inverse_distance", k=5
        )

        sdm_model = SpatialDurbinModel(weights_matrix)
        sdm_model.fit(X, y)
        results["model"] = sdm_model
        results["rho"] = sdm_model.rho
        results["direct_effects"] = sdm_model.beta
        results["indirect_effects"] = sdm_model.theta

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return results


__all__ = [
    "RegressionResults",
    "OrdinaryLeastSquares",
    "SpatialLagModel",
    "GeographicallyWeightedRegression",
    "SpatialErrorModel",
    "SpatialDurbinModel",
    "spatial_regression_analysis",
]
