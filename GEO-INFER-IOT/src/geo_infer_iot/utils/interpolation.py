"""
Spatial Interpolation Utilities

This module provides spatial interpolation methods for converting point sensor
measurements to continuous spatial surfaces, with integration to H3 spatial indexing.
"""

import logging
from typing import Any, cast
from datetime import datetime
import numpy as np
import h3

# Optional imports for enhanced functionality
try:
    from scipy import interpolate
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class SpatialInterpolation:
    """
    Spatial interpolation utilities for IoT sensor data.

    Provides methods for:
    - Inverse distance weighted interpolation
    - Kriging interpolation (simplified)
    - H3-based spatial interpolation
    - Cross-validation for interpolation quality assessment
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.interpolation_cache: dict[str, Any] = {}

        # Default interpolation parameters
        self.default_params = {
            "method": "inverse_distance_weighted",
            "power": 2,  # Power parameter for IDW
            "max_distance": 10000,  # Maximum distance in meters
            "min_neighbors": 3,
            "search_radius": 5000,  # Search radius in meters
            "h3_resolution": 8,
        }

        logger.info("SpatialInterpolation initialized")

    def _haversine_meters(
        self, from_latlon: np.ndarray, to_latlon: np.ndarray
    ) -> np.ndarray:
        """Pairwise great-circle distances in meters.

        Args:
            from_latlon: (N, 2) array of (latitude, longitude) degrees
            to_latlon: (M, 2) array of (latitude, longitude) degrees

        Returns:
            (N, M) array of distances in meters
        """
        from_latlon = np.asarray(from_latlon, dtype=float)
        to_latlon = np.asarray(to_latlon, dtype=float)
        from_rad = np.radians(from_latlon)[:, None, :]
        to_rad = np.radians(to_latlon)[None, :, :]
        dlat = to_rad[..., 0] - from_rad[..., 0]
        dlon = to_rad[..., 1] - from_rad[..., 1]
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(from_rad[..., 0])
            * np.cos(to_rad[..., 0])
            * np.sin(dlon / 2.0) ** 2
        )
        earth_radius_m = 6_371_000.0
        return 2.0 * earth_radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def interpolate_to_grid(
        self,
        measurements: list[dict[str, Any]],
        target_grid: list[tuple[float, float]],
        method: str | None = None,
    ) -> dict[str, Any]:
        """
        Interpolate sensor measurements to a target grid.

        Args:
            measurements: List of sensor measurements with lat/lon
            target_grid: List of (lat, lon) tuples for interpolation targets
            method: Interpolation method to use

        Returns:
            Dictionary with interpolated values and metadata
        """
        try:
            if len(measurements) < 3:
                return {"error": "Insufficient measurements for interpolation"}

            method = method or self.config.get("method", "inverse_distance_weighted")

            # Extract sensor coordinates and values. Internal representation
            # is (latitude, longitude) in degrees; all distance thresholds are
            # meters and converted via haversine.
            sensor_latlon_list = []
            sensor_values_list = []

            for measurement in measurements:
                if "latitude" in measurement and "longitude" in measurement:
                    sensor_latlon_list.append(
                        [measurement["latitude"], measurement["longitude"]]
                    )
                    sensor_values_list.append(measurement["value"])

            sensor_latlon = np.array(sensor_latlon_list)
            sensor_values = np.array(sensor_values_list)

            if len(sensor_latlon) < 3:
                return {"error": "Need at least 3 sensor locations for interpolation"}

            # Prepare target grid
            target_latlon = np.array(target_grid)

            # Perform interpolation based on method
            if method == "inverse_distance_weighted":
                interpolated_values = self._inverse_distance_weighted(
                    sensor_latlon, sensor_values, target_latlon
                )
            elif method == "nearest_neighbor":
                interpolated_values = self._nearest_neighbor(
                    sensor_latlon, sensor_values, target_latlon
                )
            elif method == "linear" and HAS_SCIPY:
                interpolated_values = self._linear_interpolation(
                    sensor_latlon, sensor_values, target_latlon
                )
            else:
                # Default to IDW
                interpolated_values = self._inverse_distance_weighted(
                    sensor_latlon, sensor_values, target_latlon
                )

            # Calculate interpolation uncertainty
            uncertainty = self._calculate_interpolation_uncertainty(
                sensor_latlon, target_latlon, method or "inverse_distance_weighted"
            )

            return {
                "interpolated_values": interpolated_values.tolist(),
                "target_coordinates": target_grid,
                "method": method,
                "sensor_count": len(sensor_values),
                "target_points": len(target_grid),
                "uncertainty": uncertainty.tolist(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in grid interpolation: {e}")
            return {"error": f"Interpolation failed: {str(e)}"}

    def _inverse_distance_weighted(
        self,
        sensor_latlon: np.ndarray,
        sensor_values: np.ndarray,
        target_latlon: np.ndarray,
    ) -> np.ndarray:
        """Perform inverse distance weighted interpolation."""
        power = self.config.get("power", 2)
        max_distance = self.config.get("max_distance", 10000)  # meters

        distances = self._haversine_meters(sensor_latlon, target_latlon)  # (N, M)

        interpolated = []
        for j in range(distances.shape[1]):
            target_distances = distances[:, j]

            # Filter points within max distance
            valid_indices = target_distances <= max_distance
            if not np.any(valid_indices):
                interpolated.append(np.nan)
                continue

            valid_distances = target_distances[valid_indices]
            valid_values = sensor_values[valid_indices]

            # Avoid division by zero for coincident points
            valid_distances = np.where(valid_distances == 0, 1e-3, valid_distances)

            # Calculate weights (inverse distance)
            weights = 1.0 / (valid_distances**power)

            # Weighted average
            weighted_sum = np.sum(valid_values * weights)
            weight_sum = np.sum(weights)

            if weight_sum > 0:
                interpolated.append(weighted_sum / weight_sum)
            else:
                interpolated.append(np.nan)

        return np.array(interpolated)

    def _nearest_neighbor(
        self,
        sensor_latlon: np.ndarray,
        sensor_values: np.ndarray,
        target_latlon: np.ndarray,
    ) -> np.ndarray:
        """Perform nearest neighbor interpolation."""
        distances = self._haversine_meters(sensor_latlon, target_latlon)
        nearest_indices = np.argmin(distances, axis=0)
        return sensor_values[nearest_indices]

    def _linear_interpolation(
        self,
        sensor_latlon: np.ndarray,
        sensor_values: np.ndarray,
        target_latlon: np.ndarray,
    ) -> np.ndarray:
        """Perform linear interpolation using scipy on projected planar meters."""
        if not HAS_SCIPY:
            # Fall back to IDW
            return self._inverse_distance_weighted(
                sensor_latlon, sensor_values, target_latlon
            )

        try:
            # Project degrees to a local equirectangular metric plane so
            # scipy's planar griddata operates in consistent units
            mean_lat = float(np.radians(np.mean(sensor_latlon[:, 0])))

            def _project(points: np.ndarray) -> np.ndarray:
                x = np.radians(points[:, 1]) * 6_371_000.0 * np.cos(mean_lat)
                y = np.radians(points[:, 0]) * 6_371_000.0
                return np.column_stack([x, y])

            sensor_xy = _project(sensor_latlon)
            target_xy = _project(target_latlon)

            # Use scipy's griddata for linear interpolation
            interpolated = interpolate.griddata(
                sensor_xy, sensor_values, target_xy, method="linear"
            )

            # Fill NaN values with nearest neighbor
            nan_mask = np.isnan(interpolated)
            if np.any(nan_mask):
                nearest_values = self._nearest_neighbor(
                    sensor_latlon, sensor_values, target_latlon
                )
                interpolated[nan_mask] = nearest_values[nan_mask]

            return cast(np.ndarray, interpolated)

        except Exception as e:
            logger.warning(f"Linear interpolation failed, using IDW: {e}")
            return self._inverse_distance_weighted(
                sensor_latlon, sensor_values, target_latlon
            )

    def _calculate_interpolation_uncertainty(
        self, sensor_latlon: np.ndarray, target_latlon: np.ndarray, method: str
    ) -> np.ndarray:
        """Calculate uncertainty for interpolated values.

        Distances are great-circle meters, so the divisor constants below
        operate on the same unit scale as the max_distance/search_radius
        meter thresholds.
        """
        distances = self._haversine_meters(sensor_latlon, target_latlon)
        min_distances = np.min(distances, axis=0)

        # Uncertainty increases with distance from sensors
        # Also depends on interpolation method
        if method == "nearest_neighbor":
            uncertainty = min_distances / 1000  # Simple distance-based uncertainty
        elif method == "inverse_distance_weighted":
            uncertainty = min_distances / 5000 + 0.1  # IDW has some base uncertainty
        else:
            uncertainty = min_distances / 3000 + 0.2  # Default uncertainty

        return np.clip(uncertainty, 0.01, 1.0)  # Clamp between 0.01 and 1.0

    def interpolate_h3_cells(
        self, measurements: list[dict], target_h3_indices: list[str]
    ) -> dict:
        """
        Interpolate measurements to specific H3 cells.

        Args:
            measurements: List of sensor measurements
            target_h3_indices: List of H3 indices to interpolate to

        Returns:
            Dictionary with H3-interpolated values
        """
        try:
            # Get coordinates for target H3 cells
            target_coords = []
            for h3_index in target_h3_indices:
                lat, lon = h3.cell_to_latlng(h3_index)
                target_coords.append((lat, lon))

            # Perform interpolation
            interpolation_result = self.interpolate_to_grid(
                measurements,
                target_coords,
                method=self.config.get("method", "inverse_distance_weighted"),
            )

            if "error" in interpolation_result:
                return interpolation_result

            # Map results back to H3 indices
            h3_results = {}
            for i, h3_index in enumerate(target_h3_indices):
                h3_results[h3_index] = {
                    "interpolated_value": interpolation_result["interpolated_values"][
                        i
                    ],
                    "uncertainty": interpolation_result["uncertainty"][i],
                    "coordinates": target_coords[i],
                }

            return {
                "h3_interpolations": h3_results,
                "method": interpolation_result["method"],
                "target_cells": len(target_h3_indices),
                "timestamp": interpolation_result["timestamp"],
            }

        except Exception as e:
            logger.error(f"Error in H3 interpolation: {e}")
            return {"error": f"H3 interpolation failed: {str(e)}"}

    def create_interpolation_grid(
        self, bounds: dict[str, float], resolution_km: float = 1.0
    ) -> list[tuple[float, float]]:
        """
        Create a regular grid for interpolation within bounds.

        Args:
            bounds: Geographic bounds (lat_min, lat_max, lon_min, lon_max)
            resolution_km: Grid resolution in kilometers

        Returns:
            List of (lat, lon) tuples for grid points
        """
        try:
            lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
            lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]

            # Calculate grid spacing in degrees
            # Approximate: 1 km ≈ 0.009 degrees at equator
            lat_spacing = resolution_km * 0.009
            lon_spacing = (
                resolution_km * 0.009 / np.cos(np.radians((lat_min + lat_max) / 2))
            )

            # Create grid
            lat_points = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)
            lon_points = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)

            # Limit grid size to prevent memory issues
            max_points = 10000
            if len(lat_points) * len(lon_points) > max_points:
                # Reduce resolution
                factor = np.sqrt(max_points / (len(lat_points) * len(lon_points)))
                lat_points = lat_points[:: int(1 / factor)]
                lon_points = lon_points[:: int(1 / factor)]

            # Create mesh grid
            lat_grid, lon_grid = np.meshgrid(lat_points, lon_points)
            grid_points = list(zip(lat_grid.ravel(), lon_grid.ravel()))

            return grid_points[:max_points]  # Limit to prevent excessive computation

        except Exception as e:
            logger.error(f"Error creating interpolation grid: {e}")
            return []

    def cross_validate_interpolation(
        self, measurements: list[dict], test_fraction: float = 0.2
    ) -> dict:
        """
        Cross-validate interpolation quality using hold-out testing.

        Args:
            measurements: All available measurements
            test_fraction: Fraction of data to use for testing

        Returns:
            Cross-validation results with error metrics
        """
        try:
            if len(measurements) < 10:
                return {"error": "Insufficient measurements for cross-validation"}

            # Split data into training and testing
            np.random.seed(42)  # For reproducibility
            n_test = int(len(measurements) * test_fraction)
            test_indices = np.random.choice(len(measurements), n_test, replace=False)

            train_measurements = [
                measurements[i]
                for i in range(len(measurements))
                if i not in test_indices
            ]
            test_measurements = [measurements[i] for i in test_indices]

            # Create grid from test locations
            test_coords = [(m["latitude"], m["longitude"]) for m in test_measurements]

            # Interpolate using training data
            interpolation_result = self.interpolate_to_grid(
                train_measurements, test_coords
            )

            if "error" in interpolation_result:
                return interpolation_result

            # Calculate errors
            true_values = [m["value"] for m in test_measurements]
            predicted_values = interpolation_result["interpolated_values"]

            # Filter out NaN predictions
            valid_indices = [
                i for i, pred in enumerate(predicted_values) if not np.isnan(pred)
            ]
            if not valid_indices:
                return {"error": "No valid predictions for cross-validation"}

            valid_true = [true_values[i] for i in valid_indices]
            valid_predicted = [predicted_values[i] for i in valid_indices]

            # Calculate error metrics
            mae = np.mean(np.abs(np.array(valid_true) - np.array(valid_predicted)))
            mse = np.mean((np.array(valid_true) - np.array(valid_predicted)) ** 2)
            rmse = np.sqrt(mse)

            # Calculate R-squared
            ss_res = np.sum((np.array(valid_true) - np.array(valid_predicted)) ** 2)
            ss_tot = np.sum((np.array(valid_true) - np.mean(valid_true)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "cross_validation_results": {
                    "mean_absolute_error": mae,
                    "mean_squared_error": mse,
                    "root_mean_squared_error": rmse,
                    "r_squared": r_squared,
                    "test_points": len(valid_true),
                    "training_points": len(train_measurements),
                },
                "method": interpolation_result["method"],
                "test_fraction": test_fraction,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {"error": f"Cross-validation failed: {str(e)}"}

    def get_interpolation_quality(self, sensor_coords: np.ndarray, method: str) -> dict:
        """
        Assess interpolation quality based on sensor distribution.

        Args:
            sensor_coords: Array of sensor coordinates
            method: Interpolation method being used

        Returns:
            Quality assessment metrics
        """
        try:
            if len(sensor_coords) < 3:
                return {
                    "quality_score": 0.0,
                    "issues": ["Insufficient sensors for reliable interpolation"],
                }

            # Calculate sensor density metrics
            distances = cdist(sensor_coords, sensor_coords)
            np.fill_diagonal(distances, np.inf)  # Exclude self-distances

            min_distances = np.min(distances, axis=1)
            mean_distance = np.mean(min_distances)
            max_distance = np.max(min_distances)

            # Calculate spatial coverage
            # Simple bounding box area (in degrees squared)
            lat_range = np.max(sensor_coords[:, 1]) - np.min(sensor_coords[:, 1])
            lon_range = np.max(sensor_coords[:, 0]) - np.min(sensor_coords[:, 0])
            coverage_area = lat_range * lon_range

            # Quality factors
            density_factor = min(
                1.0, len(sensor_coords) / 50
            )  # Normalize to 50 sensors
            distribution_factor = min(
                1.0, mean_distance / max_distance
            )  # Better distribution = higher score
            coverage_factor = min(
                1.0, 1.0 / (1.0 + coverage_area * 100)
            )  # Smaller area = higher score

            # Weighted quality score
            quality_score = (
                0.4 * density_factor + 0.3 * distribution_factor + 0.3 * coverage_factor
            )

            # Identify issues
            issues = []
            if mean_distance > 0.1:  # Sensors too spread out
                issues.append(
                    "Sensors are widely dispersed - may affect interpolation quality"
                )
            if len(sensor_coords) < 10:
                issues.append("Low sensor density - consider adding more sensors")
            if coverage_area > 0.01:  # Large coverage area
                issues.append(
                    "Large coverage area - interpolation may be less accurate"
                )

            return {
                "quality_score": quality_score,
                "sensor_count": len(sensor_coords),
                "mean_neighbor_distance": mean_distance,
                "max_neighbor_distance": max_distance,
                "coverage_area_deg2": coverage_area,
                "density_factor": density_factor,
                "distribution_factor": distribution_factor,
                "coverage_factor": coverage_factor,
                "issues": issues,
                "assessment_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error assessing interpolation quality: {e}")
            return {"quality_score": 0.0, "error": str(e)}
