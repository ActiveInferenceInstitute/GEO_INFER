"""
Spatial Interpolation Methods

This module provides various spatial interpolation methods for geospatial data
analysis in the GEO-INFER framework.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from scipy.interpolate import griddata, RBFInterpolator as ScipyRBF

logger = logging.getLogger(__name__)


@dataclass
class InterpolationConfig:
    """Configuration for interpolation methods."""

    # General parameters
    method: str = "idw"  # 'idw', 'kriging', 'rbf', 'linear', 'cubic'
    resolution: float = 0.01  # degrees
    max_distance: float = 1.0  # degrees
    min_points: int = 3

    # IDW parameters
    power: float = 2.0

    # Kriging parameters
    variogram_model: str = "spherical"
    nugget: float = 0.0
    sill: float = 1.0
    range_param: float = 1.0

    # RBF parameters
    rbf_function: str = "multiquadric"
    smoothing: float = 0.0


class SpatialInterpolator(ABC):
    """Abstract base class for spatial interpolators."""

    def __init__(self, config: InterpolationConfig | None = None):
        """
        Initialize spatial interpolator.

        Args:
            config: Interpolation configuration
        """
        self.config = config or InterpolationConfig()
        self.is_fitted = False
        self.training_data = None

    @abstractmethod
    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "SpatialInterpolator":
        """Fit the interpolator to training data."""
        raise RuntimeError("Interpolator subclasses must implement interpolate()")

    @abstractmethod
    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """Predict values at new coordinates."""
        raise RuntimeError("Interpolator subclasses must implement validate_data()")

    def cross_validate(
        self, coordinates: np.ndarray, values: np.ndarray, n_folds: int = 5
    ) -> dict[str, float]:
        """
        Perform cross-validation.

        Args:
            coordinates: Training coordinates
            values: Training values
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation metrics
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        mse_scores = []
        mae_scores = []

        for train_idx, test_idx in kf.split(coordinates):
            # Split data
            train_coords = coordinates[train_idx]
            train_vals = values[train_idx]
            test_coords = coordinates[test_idx]
            test_vals = values[test_idx]

            # Fit model on training data
            model_copy = self.__class__(self.config)
            model_copy.fit(train_coords, train_vals)

            # Predict on test data
            test_pred = model_copy.predict(test_coords)

            # Calculate metrics
            mse = np.mean((test_vals - test_pred) ** 2)
            mae = np.mean(np.abs(test_vals - test_pred))

            mse_scores.append(mse)
            mae_scores.append(mae)

        return {
            "mse_mean": np.mean(mse_scores),
            "mse_std": np.std(mse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
        }


class IDWInterpolator(SpatialInterpolator):
    """Inverse Distance Weighting interpolator."""

    def __init__(self, config: InterpolationConfig | None = None):
        super().__init__(config)
        self.training_coords: np.ndarray | None = None
        self.training_values: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "IDWInterpolator":
        """
        Fit IDW interpolator to training data.

        Args:
            coordinates: Training coordinates (n_samples, 2)
            values: Training values (n_samples,)

        Returns:
            Self for method chaining
        """
        if len(coordinates) < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} points for interpolation"
            )

        self.training_coords = coordinates.copy()
        self.training_values = values.copy()
        self.is_fitted = True

        logger.info(f"Fitted IDW interpolator with {len(coordinates)} points")
        return self

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values using IDW interpolation.

        Args:
            coordinates: Prediction coordinates (n_points, 2)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Interpolator must be fitted before prediction")
        assert self.training_coords is not None
        assert self.training_values is not None

        predictions = []

        for coord in coordinates:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.training_coords - coord) ** 2, axis=1))

            # Apply maximum distance filter
            valid_mask = distances <= self.config.max_distance
            if not np.any(valid_mask):
                # If no points within range, use nearest point
                nearest_idx = np.argmin(distances)
                predictions.append(self.training_values[nearest_idx])
                continue

            valid_distances = distances[valid_mask]
            valid_values = self.training_values[valid_mask]

            # Avoid division by zero
            valid_distances = np.maximum(valid_distances, 1e-10)

            # Calculate weights
            weights = 1.0 / (valid_distances**self.config.power)

            # Calculate weighted average
            prediction = np.sum(weights * valid_values) / np.sum(weights)
            predictions.append(prediction)

        return np.array(predictions)


class KrigingInterpolator(SpatialInterpolator):
    """Ordinary Kriging interpolator."""

    def __init__(self, config: InterpolationConfig | None = None):
        super().__init__(config)
        self.training_coords: np.ndarray | None = None
        self.training_values: np.ndarray | None = None
        self.kriging_weights: np.ndarray | None = None
        self.kriging_variance: float | None = None
        self._extended_matrix: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "KrigingInterpolator":
        """
        Fit Kriging interpolator to training data.

        Builds the ordinary-kriging extended system matrix
        ``[[K, 1], [1^T, 0]]`` where ``K_ij = gamma(h_ij)`` is the variogram
        between training points i and j.

        Args:
            coordinates: Training coordinates (n_samples, 2)
            values: Training values (n_samples,)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If more than 500 training points are supplied or the
                kriging system is singular.
        """
        if len(coordinates) < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} points for interpolation"
            )
        if len(coordinates) > 500:
            raise ValueError(
                "KrigingInterpolator supports at most 500 training points; "
                f"got {len(coordinates)}. Subsample or use IDW interpolation."
            )

        self.training_coords = np.asarray(coordinates, dtype=np.float64).copy()
        self.training_values = np.asarray(values, dtype=np.float64).copy()
        n_points = len(self.training_coords)

        pair_distances = np.sqrt(
            np.sum(
                (
                    self.training_coords[:, np.newaxis, :]
                    - self.training_coords[np.newaxis, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        k_matrix = self._calculate_variogram(pair_distances)
        np.fill_diagonal(k_matrix, 0.0)

        extended = np.ones((n_points + 1, n_points + 1), dtype=np.float64)
        extended[:n_points, :n_points] = k_matrix
        extended[n_points, n_points] = 0.0
        self._extended_matrix = extended
        self.is_fitted = True

        logger.info(f"Fitted Kriging interpolator with {n_points} points")
        return self

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values using ordinary kriging.

        For each query point the semi-variance vector gamma(h0) between the
        query point and all training points is computed and the extended
        kriging system solved for the weights and the Lagrange multiplier.

        Args:
            coordinates: Prediction coordinates (n_points, 2)

        Returns:
            Predicted values

        Raises:
            ValueError: If the interpolator has not been fitted or the
                per-query kriging system is singular.
        """
        if not self.is_fitted:
            raise ValueError("Interpolator must be fitted before prediction")
        assert self.training_coords is not None
        assert self.training_values is not None
        assert self._extended_matrix is not None

        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=np.float64))
        predictions = []
        kriging_variances = []

        for coord in coordinates:
            # Distances from the query point to every training point
            distances = np.sqrt(np.sum((self.training_coords - coord) ** 2, axis=1))

            # Semi-variance between the query point and each training point
            variogram_values = self._calculate_variogram(distances)

            # Solve the ordinary-kriging (OK) extended system:
            #   [ K   1 ] [w]   [gamma(h0)]
            #   [ 1^T 0 ] [mu] = [    1    ]
            rhs = np.concatenate([variogram_values, [1.0]])
            try:
                solution = np.linalg.solve(self._extended_matrix, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Kriging system is singular for the given configuration; "
                    "check for duplicate training points or a degenerate "
                    "variogram model"
                ) from exc
            weights = solution[: len(self.training_values)]
            lagrange_multiplier = float(solution[-1])

            # Kriging prediction and prediction variance
            prediction = float(np.sum(weights * self.training_values))
            kriging_variances.append(
                float(np.dot(weights, variogram_values) + lagrange_multiplier)
            )
            predictions.append(prediction)

        self.kriging_weights = np.asarray(solution[: len(self.training_values)])
        self.kriging_variance = kriging_variances[-1]
        return np.array(predictions)

    def _kriging_variance_at(
        self,
        weights: np.ndarray,
        variogram_values: np.ndarray,
        lagrange_multiplier: float,
    ) -> float:
        """OK prediction variance: w^T gamma(h0) + mu."""
        return float(np.dot(weights, variogram_values) + lagrange_multiplier)

    def _calculate_variogram(self, distances: np.ndarray) -> np.ndarray:
        """Calculate variogram values; gamma(0) is exactly zero."""
        distances = np.asarray(distances, dtype=np.float64)
        variogram = np.zeros_like(distances)
        nonzero = distances > 0.0
        h = distances[nonzero]

        if self.config.variogram_model == "spherical":
            # Spherical variogram model
            hn = h / self.config.range_param
            variogram[nonzero] = np.where(
                hn <= 1,
                self.config.nugget
                + (self.config.sill - self.config.nugget) * (1.5 * hn - 0.5 * hn**3),
                self.config.sill,
            )
        elif self.config.variogram_model == "exponential":
            # Exponential variogram model
            hn = h / self.config.range_param
            variogram[nonzero] = self.config.nugget + (
                self.config.sill - self.config.nugget
            ) * (1 - np.exp(-3 * hn))
        else:
            # Linear variogram model
            variogram[nonzero] = (
                self.config.nugget
                + (self.config.sill - self.config.nugget) * h / self.config.range_param
            )

        return variogram


class RBFInterpolator(SpatialInterpolator):
    """Radial Basis Function interpolator."""

    def __init__(self, config: InterpolationConfig | None = None):
        super().__init__(config)
        self.rbf_model: Any = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "RBFInterpolator":
        """
        Fit RBF interpolator to training data.

        Args:
            coordinates: Training coordinates (n_samples, 2)
            values: Training values (n_samples,)

        Returns:
            Self for method chaining
        """
        if len(coordinates) < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} points for interpolation"
            )

        try:
            self.rbf_model = ScipyRBF(
                coordinates,
                values,
                function=self.config.rbf_function,
                smoothing=self.config.smoothing,
            )
            self.is_fitted = True
            logger.info(f"Fitted RBF interpolator with {len(coordinates)} points")
        except Exception as e:
            logger.error(f"Failed to fit RBF interpolator: {e}")
            # Fallback to IDW
            logger.info("Falling back to IDW interpolation")
            self.rbf_model = IDWInterpolator(self.config)
            self.rbf_model.fit(coordinates, values)
            self.is_fitted = True

        return self

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values using RBF interpolation.

        Args:
            coordinates: Prediction coordinates (n_points, 2)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Interpolator must be fitted before prediction")

        try:
            return np.asarray(self.rbf_model(coordinates))
        except Exception as e:
            logger.error(f"RBF prediction failed: {e}")
            # Fallback to IDW
            if isinstance(self.rbf_model, IDWInterpolator):
                return self.rbf_model.predict(coordinates)
            else:
                idw_model = IDWInterpolator(self.config)
                idw_model.fit(self.rbf_model.y, self.rbf_model.d)
                return idw_model.predict(coordinates)


class LinearInterpolator(SpatialInterpolator):
    """Linear interpolation using scipy's griddata."""

    def __init__(self, config: InterpolationConfig | None = None):
        super().__init__(config)
        self.training_coords: np.ndarray | None = None
        self.training_values: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "LinearInterpolator":
        """
        Fit linear interpolator to training data.

        Args:
            coordinates: Training coordinates (n_samples, 2)
            values: Training values (n_samples,)

        Returns:
            Self for method chaining
        """
        if len(coordinates) < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} points for interpolation"
            )

        self.training_coords = coordinates.copy()
        self.training_values = values.copy()
        self.is_fitted = True

        logger.info(f"Fitted linear interpolator with {len(coordinates)} points")
        return self

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values using linear interpolation.

        Args:
            coordinates: Prediction coordinates (n_points, 2)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Interpolator must be fitted before prediction")
        assert self.training_coords is not None
        assert self.training_values is not None

        try:
            return np.asarray(
                griddata(
                    self.training_coords,
                    self.training_values,
                    coordinates,
                    method="linear",
                    fill_value=np.nan,
                )
            )
        except Exception as e:
            logger.error(f"Linear interpolation failed: {e}")
            # Fallback to nearest neighbor
            return np.asarray(
                griddata(
                    self.training_coords,
                    self.training_values,
                    coordinates,
                    method="nearest",
                )
            )


class CubicInterpolator(SpatialInterpolator):
    """Cubic interpolation using scipy's griddata."""

    def __init__(self, config: InterpolationConfig | None = None):
        super().__init__(config)
        self.training_coords: np.ndarray | None = None
        self.training_values: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "CubicInterpolator":
        """
        Fit cubic interpolator to training data.

        Args:
            coordinates: Training coordinates (n_samples, 2)
            values: Training values (n_samples,)

        Returns:
            Self for method chaining
        """
        if len(coordinates) < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} points for interpolation"
            )

        self.training_coords = coordinates.copy()
        self.training_values = values.copy()
        self.is_fitted = True

        logger.info(f"Fitted cubic interpolator with {len(coordinates)} points")
        return self

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict values using cubic interpolation.

        Args:
            coordinates: Prediction coordinates (n_points, 2)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Interpolator must be fitted before prediction")
        assert self.training_coords is not None
        assert self.training_values is not None

        try:
            return np.asarray(
                griddata(
                    self.training_coords,
                    self.training_values,
                    coordinates,
                    method="cubic",
                    fill_value=np.nan,
                )
            )
        except Exception as e:
            logger.error(f"Cubic interpolation failed: {e}")
            # Fallback to linear interpolation
            return np.asarray(
                griddata(
                    self.training_coords,
                    self.training_values,
                    coordinates,
                    method="linear",
                    fill_value=np.nan,
                )
            )


class InterpolationManager:
    """Manager for multiple interpolation methods."""

    def __init__(self, config: InterpolationConfig | None = None):
        """
        Initialize interpolation manager.

        Args:
            config: Configuration for interpolation methods
        """
        self.config = config or InterpolationConfig()
        self.interpolators: dict[str, SpatialInterpolator] = {}
        self._initialize_interpolators()

    def _initialize_interpolators(self) -> None:
        """Initialize all interpolation methods."""
        self.interpolators = {
            "idw": IDWInterpolator(self.config),
            "kriging": KrigingInterpolator(self.config),
            "rbf": RBFInterpolator(self.config),
            "linear": LinearInterpolator(self.config),
            "cubic": CubicInterpolator(self.config),
        }

        logger.info(f"Initialized {len(self.interpolators)} interpolation methods")

    def interpolate(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        prediction_coords: np.ndarray,
        method: str | None = None,
    ) -> np.ndarray:
        """
        Perform spatial interpolation.

        Args:
            coordinates: Training coordinates
            values: Training values
            prediction_coords: Coordinates for prediction
            method: Interpolation method (uses config default if None)

        Returns:
            Interpolated values
        """
        method = method or self.config.method

        if method not in self.interpolators:
            raise ValueError(f"Unknown interpolation method: {method}")

        interpolator = self.interpolators[method]
        interpolator.fit(coordinates, values)
        return interpolator.predict(prediction_coords)

    def compare_methods(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        test_coordinates: np.ndarray | None = None,
        test_values: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compare different interpolation methods.

        Args:
            coordinates: Training coordinates
            values: Training values
            test_coordinates: Test coordinates (if None, uses cross-validation)
            test_values: Test values (if None, uses cross-validation)

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for method_name, interpolator in self.interpolators.items():
            try:
                if test_coordinates is not None and test_values is not None:
                    # Use provided test data
                    interpolator.fit(coordinates, values)
                    predictions = interpolator.predict(test_coordinates)

                    mse = np.mean((test_values - predictions) ** 2)
                    mae = np.mean(np.abs(test_values - predictions))
                    rmse = np.sqrt(mse)

                    results[method_name] = {"mse": mse, "mae": mae, "rmse": rmse}
                else:
                    # Use cross-validation
                    cv_results = interpolator.cross_validate(coordinates, values)
                    results[method_name] = cv_results

                logger.info(f"Completed comparison for {method_name}")

            except Exception as e:
                logger.error(f"Failed to compare {method_name}: {e}")
                results[method_name] = {
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "rmse": float("inf"),
                }

        return results

    def create_interpolation_grid(
        self, bounds: dict[str, float], resolution: float | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Create a regular grid for interpolation.

        Args:
            bounds: Dictionary with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
            resolution: Grid resolution (uses config default if None)

        Returns:
            Tuple of (grid_coordinates, grid_metadata)
        """
        resolution = resolution or self.config.resolution

        lat_min = bounds["lat_min"]
        lat_max = bounds["lat_max"]
        lon_min = bounds["lon_min"]
        lon_max = bounds["lon_max"]

        # Create grid
        lat_grid = np.arange(lat_min, lat_max + resolution, resolution)
        lon_grid = np.arange(lon_min, lon_max + resolution, resolution)

        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Flatten to coordinate pairs
        grid_coords = np.column_stack([lat_mesh.flatten(), lon_mesh.flatten()])

        grid_metadata = {
            "resolution": resolution,
            "n_points": len(grid_coords),
            "bounds": bounds,
            "shape": (len(lat_grid), len(lon_grid)),
        }

        return grid_coords, grid_metadata

    def interpolate_to_grid(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        bounds: dict[str, float],
        method: str | None = None,
        resolution: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Interpolate data to a regular grid.

        Args:
            coordinates: Training coordinates
            values: Training values
            bounds: Grid bounds
            method: Interpolation method
            resolution: Grid resolution

        Returns:
            Tuple of (grid_coordinates, interpolated_values, metadata)
        """
        # Create grid
        grid_coords, grid_metadata = self.create_interpolation_grid(bounds, resolution)

        # Perform interpolation
        interpolated_values = self.interpolate(coordinates, values, grid_coords, method)

        return grid_coords, interpolated_values, grid_metadata


# Convenience functions
def create_interpolation_manager(
    config: InterpolationConfig | None = None,
) -> InterpolationManager:
    """Create a new interpolation manager."""
    return InterpolationManager(config)


def interpolate_spatial_data(
    coordinates: np.ndarray,
    values: np.ndarray,
    prediction_coords: np.ndarray,
    method: str = "idw",
) -> np.ndarray:
    """Convenience function for spatial interpolation."""
    config = InterpolationConfig(method=method)
    manager = InterpolationManager(config)
    return manager.interpolate(coordinates, values, prediction_coords, method)


def create_interpolation_grid(
    bounds: dict[str, float], resolution: float = 0.01
) -> np.ndarray:
    """Create a regular interpolation grid."""
    config = InterpolationConfig(resolution=resolution)
    manager = InterpolationManager(config)
    grid_coords, _ = manager.create_interpolation_grid(bounds, resolution)
    return grid_coords
