"""Bayesian spatial inference for IoT sensor data.

This module owns the Gaussian-process-backed spatial inference used to
convert point sensor measurements into continuous spatial distributions
over an H3 grid. It integrates with GEO-INFER-BAYES for probabilistic
modeling when that workspace module is available.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
from geo_infer_bayes import GaussianProcess, SpatialCovariance  # type: ignore[import-untyped]
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BayesianSpatialInference:
    """
    Bayesian spatial inference for IoT sensor data.

    Converts point sensor measurements to continuous spatial distributions
    using Gaussian process models and H3 spatial indexing. Integrates with
    GEO-INFER-BAYES for sophisticated probabilistic modeling.
    """

    def __init__(
        self,
        variable: str,
        spatial_resolution: int,
        temporal_window: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.variable = variable
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        self.config = config or {}

        # Integration with GEO-INFER-BAYES
        self.gp_model: Any = None
        self.posterior_cache: dict[str, Any] = {}

        self._setup_bayesian_inference()

    def _setup_bayesian_inference(self) -> None:
        """Configure the GEO-INFER-BAYES Gaussian-process spatial model."""
        # Configure covariance function based on variable characteristics
        if self.variable in ["soil_moisture", "temperature"]:
            # Environmental variables with smooth spatial correlation
            cov_func = SpatialCovariance.matern_52(
                length_scale=self.config.get("length_scale", 1000.0),
                variance=self.config.get("variance", 1.0),
            )
        elif self.variable in ["air_quality", "radiation"]:
            # Variables with more complex spatial patterns
            cov_func = SpatialCovariance.matern_32(
                length_scale=self.config.get("length_scale", 2000.0),
                variance=self.config.get("variance", 0.5),
            )
        else:
            # Default configuration
            cov_func = SpatialCovariance.matern_52(
                length_scale=self.config.get("length_scale", 1500.0),
                variance=self.config.get("variance", 1.0),
            )

        # Initialize Gaussian Process model
        self.gp_model = GaussianProcess(
            covariance_function=cov_func,
            mean_function=self.config.get("mean_function", "constant"),
            noise_variance=self.config.get("noise_variance", 0.01),
        )

    def infer_spatial_distribution(
        self,
        sensor_data: list[dict[str, Any]],
        priors: Any | None = None,
        update_interval: str = "15min",
    ) -> dict[str, Any]:
        """
        Perform Bayesian spatial inference on sensor data.

        Args:
            sensor_data: List of sensor measurements with lat/lon coordinates
            priors: Optional prior beliefs for Bayesian inference
            update_interval: Time interval for updating the model

        Returns:
            Dictionary containing posterior distribution and uncertainty estimates
        """
        try:
            # Extract coordinates and values from sensor data
            coords_list = []
            values_list = []

            for measurement in sensor_data:
                if "latitude" in measurement and "longitude" in measurement:
                    # Convert lat/lon to local coordinate system for GP
                    x = measurement["longitude"] * 111000  # Rough meters per degree
                    y = measurement["latitude"] * 111000
                    coords_list.append([x, y])
                    values_list.append(measurement["value"])

            if len(coords_list) < 3:
                return {"error": "Insufficient data for spatial inference"}

            coords = np.array(coords_list)
            values = np.array(values_list)

            # Perform Bayesian inference
            self.gp_model.fit(coords, values)

            # Generate predictions on H3 grid
            h3_grid = self._generate_h3_prediction_grid(coords)

            if len(h3_grid) > 0:
                predictions, uncertainties = self.gp_model.predict(
                    h3_grid, return_std=True
                )

                # Store results in cache
                self.posterior_cache[self.variable] = {
                    "posterior_mean": predictions,
                    "posterior_std": uncertainties,
                    "h3_grid": h3_grid,
                    "sensor_coords": coords,
                    "sensor_values": values,
                    "timestamp": datetime.now(),
                    "update_interval": update_interval,
                }

                return {
                    "success": True,
                    "posterior_mean": predictions.tolist(),
                    "posterior_std": uncertainties.tolist(),
                    "h3_grid": h3_grid.tolist(),
                    "sensor_count": len(values),
                    "prediction_points": len(predictions),
                }
            else:
                return {"error": "Failed to generate prediction grid"}

        except Exception as e:
            logger.error("Spatial inference failed: %s", e, exc_info=True)
            return {"error": f"Spatial inference failed: {str(e)}"}

    def get_posterior_map(
        self, confidence_intervals: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Get current posterior spatial distribution map.

        Args:
            confidence_intervals: List of confidence levels (e.g., [0.68, 0.95])

        Returns:
            Dictionary with posterior map data and uncertainty bounds
        """
        if self.variable not in self.posterior_cache:
            return {"error": "No posterior distribution available"}

        cache_data = self.posterior_cache[self.variable]

        if confidence_intervals is None:
            confidence_intervals = [0.68, 0.95]

        # Calculate confidence intervals using the exact normal quantile for
        # each requested confidence level (the previous two-value hardcode
        # silently assigned z=1.0 to every CI other than 0.95).
        confidence_bounds = {}
        for ci in confidence_intervals:
            if not 0.0 < ci < 1.0:
                raise ValueError(f"Confidence interval must be in (0, 1), got {ci!r}")
            z_score = float(norm.ppf(0.5 + ci / 2.0))
            confidence_bounds[ci] = {
                "lower": (
                    cache_data["posterior_mean"] - z_score * cache_data["posterior_std"]
                ).tolist(),
                "upper": (
                    cache_data["posterior_mean"] + z_score * cache_data["posterior_std"]
                ).tolist(),
            }

        return {
            "variable": self.variable,
            "posterior_mean": cache_data["posterior_mean"].tolist(),
            "posterior_std": cache_data["posterior_std"].tolist(),
            "confidence_bounds": confidence_bounds,
            "h3_grid": cache_data["h3_grid"].tolist(),
            "timestamp": cache_data["timestamp"].isoformat(),
            "sensor_count": len(cache_data["sensor_values"]),
        }

    def _generate_h3_prediction_grid(self, sensor_coords: np.ndarray) -> np.ndarray:
        """Generate H3 grid for spatial predictions."""
        try:
            # Find bounds of sensor data
            min_x, min_y = np.min(sensor_coords, axis=0)
            max_x, max_y = np.max(sensor_coords, axis=0)

            # Create prediction grid with buffer
            buffer = 0.2  # 20% buffer around sensor locations
            x_range = max_x - min_x
            y_range = max_y - min_y

            grid_min_x = min_x - buffer * x_range
            grid_max_x = max_x + buffer * x_range
            grid_min_y = min_y - buffer * y_range
            grid_max_y = max_y + buffer * y_range

            # Adaptive regular lat/lon grid over the buffered sensor extent
            n_points = min(100, len(sensor_coords) * 10)  # Adaptive grid density
            x_grid = np.linspace(grid_min_x, grid_max_x, int(np.sqrt(n_points)))
            y_grid = np.linspace(grid_min_y, grid_max_y, int(np.sqrt(n_points)))

            # Create mesh grid
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([X.ravel(), Y.ravel()])

            return np.asarray(grid_points)

        except Exception as e:
            logger.error(f"Error generating prediction grid: {e}")
            return np.array([])
