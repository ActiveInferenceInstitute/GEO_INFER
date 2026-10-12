"""
Predictive ML models for geospatial forecasting and regression.

This module provides machine learning models specifically designed for
geospatial prediction tasks including land use change, climate impacts,
and resource forecasting.

Interpolation note: ``IDWInterpolator`` and ``OrdinaryKriging`` here are a
deliberate, self-contained ML-oriented implementation kept separate from
``geo_infer_math.core.interpolation`` (``IDWInterpolator`` /
``KrigingInterpolator``). Differences by design: this module takes plain
constructor kwargs (power, min_points, max_distance) instead of a config
object, ``IDWInterpolator.predict`` returns values only, and
``OrdinaryKriging.predict`` returns ``(values, variances)`` so the
estimation variance is directly usable in prediction-uncertainty
workflows. GEO-INFER-MATH remains the owner of general-purpose,
config-driven interpolation; use this module when you need lightweight
interpolators that ship with the AI package.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

logger = logging.getLogger(__name__)


class SpatialPredictor(BaseEstimator, RegressorMixin):
    """
    Spatial predictor for geospatial regression and forecasting tasks.

    Supports multiple regression algorithms optimized for spatial data
    with consideration of spatial autocorrelation and geographic features.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        include_spatial_features: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the spatial predictor.

        Args:
            model_type: Type of model ("linear", "ridge", "random_forest", "gradient_boosting")
            include_spatial_features: Whether to automatically include spatial features
            **kwargs: Additional arguments passed to the underlying model
        """
        self.model_type = model_type
        self.include_spatial_features = include_spatial_features
        self.model: Any | None = None
        self.feature_names_: list[str] | None = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """Initialize the underlying regression model."""
        if self.model_type == "linear":
            linear_params: dict[str, Any] = {"fit_intercept": True}
            linear_params.update(kwargs)
            self.model = LinearRegression(**linear_params)
        elif self.model_type == "ridge":
            ridge_params: dict[str, Any] = {"alpha": 1.0, "fit_intercept": True}
            ridge_params.update(kwargs)
            self.model = Ridge(**ridge_params)
        elif self.model_type == "random_forest":
            rf_params: dict[str, Any] = {
                "n_estimators": 100,
                "max_depth": 20,
                "random_state": 42,
                "n_jobs": -1,
            }
            rf_params.update(kwargs)
            self.model = RandomForestRegressor(**rf_params)
        elif self.model_type == "gradient_boosting":
            gb_params: dict[str, Any] = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42,
            }
            gb_params.update(kwargs)
            self.model = GradientBoostingRegressor(**gb_params)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Must be 'linear', 'ridge', 'random_forest', or 'gradient_boosting'"
            )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
    ) -> "SpatialPredictor":
        """
        Train the spatial predictor.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            sample_weight: Optional sample weights
            coordinates: Optional spatial coordinates (n_samples, 2) for spatial features

        Returns:
            Self for method chaining
        """
        X_arr: np.ndarray
        num_features: int
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = [str(c) for c in X.columns]
            X_arr = np.asarray(X.values)
            num_features = len(X.columns)
        else:
            X_arr = np.asarray(X)
            num_features = X_arr.shape[1] if X_arr.ndim > 1 else 1
            self.feature_names_ = [f"feature_{i}" for i in range(num_features)]

        logger.info(
            f"Training {self.model_type} predictor on {len(X_arr)} samples "
            f"with {num_features} features"
        )

        # Add spatial features if requested and coordinates provided
        if self.include_spatial_features and coordinates is not None:
            X_arr = self._add_spatial_features(X_arr, coordinates)

        # Train the model
        if self.model is None:
            raise ValueError("Model is not initialized")
        if sample_weight is not None:
            self.model.fit(X_arr, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_arr, y)

        logger.info("Training completed")
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        coordinates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features for prediction (n_samples, n_features)
            coordinates: Optional spatial coordinates for spatial features

        Returns:
            Predicted values (n_samples,)
        """
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Model must be trained before prediction")

        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Add spatial features if requested and coordinates provided
        if self.include_spatial_features and coordinates is not None:
            X = self._add_spatial_features(X, coordinates)

        predictions = self.model.predict(X)
        return np.asarray(predictions)

    def _add_spatial_features(
        self, X: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Add spatial features to the feature matrix.

        Args:
            X: Original features (n_samples, n_features)
            coordinates: Spatial coordinates (n_samples, 2) [lon, lat]

        Returns:
            Enhanced feature matrix with spatial features
        """
        # Calculate spatial features
        lon = coordinates[:, 0]
        lat = coordinates[:, 1]

        # Add coordinate features
        spatial_features = np.column_stack([lon, lat])

        # Add distance from origin (centroid)
        centroid_lon = np.mean(lon)
        centroid_lat = np.mean(lat)
        distances = np.sqrt((lon - centroid_lon) ** 2 + (lat - centroid_lat) ** 2)
        spatial_features = np.column_stack([spatial_features, distances])

        # Combine with original features
        X_enhanced = np.column_stack([X, spatial_features])
        return X_enhanced

    def get_feature_importance(self) -> np.ndarray | None:
        """
        Get feature importance scores (for tree-based models).

        Returns:
            Feature importance array or None if not available
        """
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return np.asarray(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficients as importance
            return np.asarray(np.abs(self.model.coef_))
        return None

    def get_feature_names(self) -> list[str] | None:
        """
        Get feature names.

        Returns:
            List of feature names or None
        """
        return self.feature_names_


class IDWInterpolator:
    """
    Inverse Distance Weighting (IDW) spatial interpolation.

    Predicts values at unsampled locations by computing a weighted average
    of observed values, where weights decrease with distance according to
    a power parameter.
    """

    def __init__(
        self,
        power: float = 2.0,
        min_points: int = 3,
        max_distance: float | None = None,
    ) -> None:
        """
        Initialize IDW interpolator.

        Args:
            power: Power parameter controlling distance decay (higher = more local)
            min_points: Minimum number of neighbors to use
            max_distance: Maximum distance for neighbor search (None = unlimited)
        """
        if power <= 0:
            raise ValueError("power must be positive")
        if min_points < 1:
            raise ValueError("min_points must be at least 1")

        self.power = power
        self.min_points = min_points
        self.max_distance = max_distance
        self.coordinates_: np.ndarray | None = None
        self.values_: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "IDWInterpolator":
        """
        Fit the interpolator with known sample locations and values.

        Args:
            coordinates: Sample coordinates (n_samples, 2)
            values: Observed values at sample locations (n_samples,)

        Returns:
            Self for method chaining
        """
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("coordinates and values must have same number of rows")
        self.coordinates_ = coordinates.copy()
        self.values_ = values.copy()
        return self

    def predict(self, target_coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values at target locations using IDW.

        Args:
            target_coordinates: Target coordinates (n_targets, 2)

        Returns:
            Predicted values (n_targets,)
        """
        if self.coordinates_ is None or self.values_ is None:
            raise ValueError("Must call fit before predict")

        n_targets = target_coordinates.shape[0]
        predictions = np.zeros(n_targets)

        for i in range(n_targets):
            target = target_coordinates[i]

            # Compute distances from target to all known points
            diffs = self.coordinates_ - target
            distances = np.sqrt(np.sum(diffs**2, axis=1))

            # Check for exact match (distance ~ 0)
            exact_match = distances < 1e-12
            if np.any(exact_match):
                predictions[i] = np.mean(self.values_[exact_match])
                continue

            # Apply max distance filter
            if self.max_distance is not None:
                mask = distances <= self.max_distance
                if np.sum(mask) < self.min_points:
                    # Fall back to nearest min_points
                    nearest_idx = np.argsort(distances)[: self.min_points]
                    mask = np.zeros(len(distances), dtype=bool)
                    mask[nearest_idx] = True
            else:
                mask = np.ones(len(distances), dtype=bool)

            d = distances[mask]
            v = self.values_[mask]

            weights = 1.0 / (d**self.power)
            predictions[i] = np.sum(weights * v) / np.sum(weights)

        return predictions


class OrdinaryKriging:
    """
    Ordinary Kriging spatial interpolation.

    Uses a variogram model to compute optimal weights for spatial
    prediction, providing both predicted values and estimation variance.
    Implements a simple spherical variogram model.
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        n_lags: int = 15,
        max_range: float | None = None,
    ) -> None:
        """
        Initialize Ordinary Kriging interpolator.

        Args:
            variogram_model: Variogram model type ('spherical', 'exponential', 'gaussian')
            n_lags: Number of lag bins for variogram estimation
            max_range: Maximum range for variogram (None = auto)
        """
        self.variogram_model = variogram_model
        self.n_lags = n_lags
        self.max_range = max_range
        self.coordinates_: np.ndarray | None = None
        self.values_: np.ndarray | None = None
        self.nugget: float = 0.0
        self.sill: float = 1.0
        self.range_param: float = 1.0

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "OrdinaryKriging":
        """
        Fit the kriging model by estimating variogram parameters.

        Args:
            coordinates: Sample coordinates (n_samples, 2)
            values: Observed values (n_samples,)

        Returns:
            Self for method chaining
        """
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("coordinates and values must have same number of rows")
        if coordinates.shape[0] < 3:
            raise ValueError("Need at least 3 sample points for kriging")

        self.coordinates_ = coordinates.copy()
        self.values_ = values.copy()

        # Estimate experimental variogram
        self._estimate_variogram()

        logger.info(
            f"Kriging fitted: nugget={self.nugget:.4f}, sill={self.sill:.4f}, "
            f"range={self.range_param:.4f}"
        )
        return self

    def _estimate_variogram(self) -> None:
        """Estimate experimental variogram and fit model parameters."""
        if self.values_ is None or self.coordinates_ is None:
            return
        n = len(self.values_)

        # Compute pairwise distances and squared differences
        distances_list: list[float] = []
        semivariances_list: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((self.coordinates_[i] - self.coordinates_[j]) ** 2))
                sv = 0.5 * (self.values_[i] - self.values_[j]) ** 2
                distances_list.append(d)
                semivariances_list.append(sv)

        distances_arr = np.array(distances_list)
        semivariances_arr = np.array(semivariances_list)

        if self.max_range is None:
            self.max_range = float(np.max(distances_arr) * 0.5)

        # Bin into lags
        lag_edges = np.linspace(0, self.max_range, self.n_lags + 1)
        lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])
        lag_semivariances = np.zeros(self.n_lags)
        lag_counts = np.zeros(self.n_lags)

        for k in range(self.n_lags):
            mask = (distances_arr >= lag_edges[k]) & (distances_arr < lag_edges[k + 1])
            if np.any(mask):
                lag_semivariances[k] = np.mean(semivariances_arr[mask])
                lag_counts[k] = np.sum(mask)

        # Fit variogram parameters using method of moments
        valid = lag_counts > 0
        if np.sum(valid) < 2:
            # Fallback to data variance
            self.nugget = 0.0
            self.sill = float(np.var(self.values_))
            self.range_param = self.max_range
            return

        valid_centers = lag_centers[valid]
        valid_sv = lag_semivariances[valid]

        # Estimate sill as asymptotic semivariance
        self.sill = float(np.max(valid_sv))
        if self.sill < 1e-10:
            self.sill = float(np.var(self.values_))

        # Estimate nugget from near-origin semivariance
        self.nugget = float(valid_sv[0]) * 0.5

        # Estimate range where semivariance reaches ~95% of sill
        threshold = self.nugget + 0.95 * (self.sill - self.nugget)
        above_threshold = valid_centers[valid_sv >= threshold]
        self.range_param = (
            float(above_threshold[0]) if len(above_threshold) > 0 else self.max_range
        )

    def _variogram_value(self, h: float) -> float:
        """Evaluate the variogram model at distance h."""
        if h < 1e-12:
            return 0.0

        c0 = self.nugget
        c = self.sill - self.nugget
        a = self.range_param

        if self.variogram_model == "spherical":
            if h >= a:
                return float(c0 + c)
            ratio = h / a
            return float(c0 + c * (1.5 * ratio - 0.5 * ratio**3))
        elif self.variogram_model == "exponential":
            return float(c0 + c * (1.0 - np.exp(-3.0 * h / a)))
        elif self.variogram_model == "gaussian":
            return float(c0 + c * (1.0 - np.exp(-3.0 * (h / a) ** 2)))
        else:
            raise ValueError(f"Unknown variogram model: {self.variogram_model}")

    def predict(self, target_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict values and estimation variance at target locations.

        Args:
            target_coordinates: Target coordinates (n_targets, 2)

        Returns:
            Tuple of (predictions, variances) each of shape (n_targets,)
        """
        if self.coordinates_ is None or self.values_ is None:
            raise ValueError("Must call fit before predict")

        n_known = len(self.values_)
        n_targets = target_coordinates.shape[0]

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets)

        # Build kriging matrix (same for all targets)
        # K is (n+1) x (n+1) with variogram values + Lagrange multiplier row/col
        K = np.zeros((n_known + 1, n_known + 1))

        for i in range(n_known):
            for j in range(i + 1, n_known):
                d = np.sqrt(np.sum((self.coordinates_[i] - self.coordinates_[j]) ** 2))
                gamma = self._variogram_value(d)
                K[i, j] = gamma
                K[j, i] = gamma

        # Lagrange multiplier row and column
        K[n_known, :n_known] = 1.0
        K[:n_known, n_known] = 1.0
        K[n_known, n_known] = 0.0

        # Add small regularization for numerical stability
        K[:n_known, :n_known] += np.eye(n_known) * 1e-8

        for t in range(n_targets):
            target = target_coordinates[t]

            # Build right-hand side vector
            k_vec = np.zeros(n_known + 1)
            for i in range(n_known):
                d = np.sqrt(np.sum((self.coordinates_[i] - target) ** 2))
                k_vec[i] = self._variogram_value(d)
            k_vec[n_known] = 1.0  # Lagrange constraint

            # Solve kriging system
            try:
                weights = np.linalg.solve(K, k_vec)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                weights, _, _, _ = np.linalg.lstsq(K, k_vec, rcond=None)

            # Predicted value
            predictions[t] = np.sum(weights[:n_known] * self.values_)

            # Estimation variance
            variances[t] = (
                np.sum(weights[:n_known] * k_vec[:n_known]) + weights[n_known]
            )

        # Clamp negative variances to zero
        variances = np.maximum(variances, 0.0)

        return predictions, variances
