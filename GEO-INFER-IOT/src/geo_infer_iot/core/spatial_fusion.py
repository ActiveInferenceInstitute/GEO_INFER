"""
Spatial Data Fusion Module

This module handles the fusion of spatial data from multiple IoT sensors,
integrating with GEO-INFER-SPACE for H3 spatial indexing and spatial operations.
"""

import logging
from typing import Any
from datetime import datetime
import numpy as np
import h3

# Optional imports for enhanced functionality
try:
    from geo_infer_space.utils.h3_utils import get_h3_neighbors

    HAS_GEO_SPACE = True
except ImportError:
    HAS_GEO_SPACE = False

logger = logging.getLogger(__name__)

if not HAS_GEO_SPACE:
    logger.warning("GEO-INFER-SPACE not available, using basic spatial operations")


class SpatialDataFusion:
    """
    Spatial data fusion engine for combining IoT sensor measurements.

    This class provides methods for:
    - Spatial interpolation between sensor locations
    - Multi-sensor data fusion with uncertainty quantification
    - H3-based spatial aggregation and analysis
    - Temporal consistency validation across spatial regions
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.fusion_cache: dict[str, Any] = {}
        self.spatial_operations: dict[str, Any] | None = None

        # Initialize spatial operations if available
        if HAS_GEO_SPACE:
            self.spatial_operations = {"h3_neighbors": get_h3_neighbors}

        # Default fusion parameters
        self.default_params = {
            "interpolation_method": "inverse_distance_weighted",
            "max_distance_km": 10.0,
            "min_neighbors": 3,
            "temporal_window_minutes": 15,
            "uncertainty_threshold": 0.1,
        }

        logger.info("SpatialDataFusion initialized")

    def fuse_sensor_data(
        self,
        measurements: list[dict],
        target_variable: str,
        target_location: tuple[float, float] | None = None,
    ) -> dict:
        """
        Fuse sensor data using spatial interpolation and uncertainty quantification.

        Args:
            measurements: List of sensor measurements with lat/lon coordinates
            target_variable: Variable to fuse (e.g., 'temperature', 'pm25')
            target_location: Optional target location for interpolation

        Returns:
            Dictionary with fused value, uncertainty, and metadata
        """
        try:
            if len(measurements) < 2:
                return {"error": "Insufficient measurements for fusion"}

            # Filter measurements for target variable
            variable_measurements = [
                m
                for m in measurements
                if m.get("variable") == target_variable
                and m.get("latitude") is not None
                and m.get("longitude") is not None
            ]

            if len(variable_measurements) < 2:
                return {
                    "error": f"Insufficient measurements for variable {target_variable}"
                }

            # Perform spatial fusion
            if target_location:
                # Interpolate to specific location
                fused_result = self._interpolate_to_location(
                    variable_measurements, target_location
                )
            else:
                # General spatial aggregation
                fused_result = self._aggregate_measurements(variable_measurements)

            # Add metadata
            fused_result.update(
                {
                    "target_variable": target_variable,
                    "measurement_count": len(variable_measurements),
                    "fusion_method": "spatial_interpolation",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return fused_result

        except Exception as e:
            logger.error(f"Error in sensor data fusion: {e}")
            return {"error": f"Fusion failed: {str(e)}"}

    def _interpolate_to_location(
        self, measurements: list[dict], target_location: tuple[float, float]
    ) -> dict:
        """Interpolate measurements to a target location."""
        target_lat, target_lon = target_location

        # Calculate distances and weights
        weighted_sum = 0.0
        total_weight = 0.0
        uncertainty_sum = 0.0

        target_h3 = h3.latlng_to_cell(target_lat, target_lon, 8)

        for measurement in measurements:
            sensor_lat = measurement["latitude"]
            sensor_lon = measurement["longitude"]
            sensor_value = measurement["value"]

            # Calculate distance (simplified - should use proper geodesic distance)
            distance = self._calculate_distance(
                sensor_lat, sensor_lon, target_lat, target_lon
            )

            # Skip if too far
            max_distance = self.config.get("max_distance_km", 10.0)
            if distance > max_distance:
                continue

            # Calculate weight (inverse distance weighted)
            if distance < 0.001:  # Very close to sensor location
                weight = 1.0
            else:
                weight = 1.0 / (distance**2)

            # Apply weight
            weighted_sum += sensor_value * weight
            total_weight += weight

            # Estimate uncertainty contribution
            sensor_uncertainty = measurement.get("uncertainty", 0.1)
            uncertainty_sum += sensor_uncertainty * weight

        if total_weight == 0:
            return {"error": "No valid measurements within range"}

        fused_value = weighted_sum / total_weight
        fused_uncertainty = uncertainty_sum / total_weight

        return {
            "fused_value": fused_value,
            "uncertainty": fused_uncertainty,
            "target_location": target_location,
            "target_h3_index": target_h3,
            "valid_measurements": len(
                [
                    m
                    for m in measurements
                    if self._calculate_distance(
                        m["latitude"], m["longitude"], target_lat, target_lon
                    )
                    <= max_distance
                ]
            ),
        }

    def _aggregate_measurements(self, measurements: list[dict]) -> dict:
        """Aggregate measurements spatially without target location."""
        # Simple spatial averaging with H3-based grouping
        h3_groups: dict[str, list[dict[str, Any]]] = {}

        for measurement in measurements:
            h3_index = h3.latlng_to_cell(
                measurement["latitude"], measurement["longitude"], 8
            )

            if h3_index not in h3_groups:
                h3_groups[h3_index] = []
            h3_groups[h3_index].append(measurement)

        # Aggregate within each H3 cell
        cell_aggregates = []
        for h3_index, cell_measurements in h3_groups.items():
            values = [m["value"] for m in cell_measurements]

            cell_aggregate = {
                "h3_index": h3_index,
                "mean_value": np.mean(values),
                "std_value": np.std(values),
                "count": len(values),
                "min_value": np.min(values),
                "max_value": np.max(values),
                "center_coordinates": h3.cell_to_latlng(h3_index),
            }

            cell_aggregates.append(cell_aggregate)

        # Overall statistics
        all_values = [m["value"] for m in measurements]
        overall_stats = {
            "mean_value": np.mean(all_values),
            "std_value": np.std(all_values),
            "min_value": np.min(all_values),
            "max_value": np.max(all_values),
            "total_measurements": len(measurements),
            "h3_cells": len(cell_aggregates),
        }

        return {
            "overall_statistics": overall_stats,
            "cell_aggregates": cell_aggregates,
            "aggregation_method": "h3_spatial_averaging",
        }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate approximate distance in kilometers between two points."""
        # Simple Euclidean distance (not geodesic, but sufficient for small areas)
        R = 6371  # Earth radius in km

        lat1_rad, lon1_rad = np.radians([lat1, lon1])
        lat2_rad, lon2_rad = np.radians([lat2, lon2])

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)

    def validate_spatial_consistency(
        self, measurements: list[dict], consistency_threshold: float = 2.0
    ) -> dict:
        """
        Validate spatial consistency of measurements using H3-based analysis.

        Args:
            measurements: List of measurements to validate
            consistency_threshold: Maximum allowed standard deviations between neighbors

        Returns:
            Dictionary with consistency validation results
        """
        try:
            # Group measurements by H3 cell
            h3_measurements: dict[str, list[dict[str, Any]]] = {}
            for measurement in measurements:
                h3_index = h3.latlng_to_cell(
                    measurement["latitude"], measurement["longitude"], 8
                )

                if h3_index not in h3_measurements:
                    h3_measurements[h3_index] = []
                h3_measurements[h3_index].append(measurement)

            # Analyze consistency within and between H3 cells
            consistency_results = {}

            for h3_index, cell_measurements in h3_measurements.items():
                if len(cell_measurements) < 2:
                    consistency_results[h3_index] = {
                        "status": "insufficient_data",
                        "consistency_score": 0.0,
                    }
                    continue

                # Calculate within-cell consistency
                values = [m["value"] for m in cell_measurements]
                mean_value = np.mean(values)
                std_value = np.std(values)

                # Get neighboring cells for between-cell consistency
                if HAS_GEO_SPACE:
                    try:
                        neighbors = get_h3_neighbors(h3_index, ring_size=1)
                        neighbor_values = []

                        for neighbor_h3 in neighbors:
                            if neighbor_h3 in h3_measurements:
                                neighbor_vals = [
                                    m["value"] for m in h3_measurements[neighbor_h3]
                                ]
                                neighbor_values.extend(neighbor_vals)

                        if neighbor_values:
                            neighbor_mean = np.mean(neighbor_values)
                            between_cell_diff = abs(mean_value - neighbor_mean)
                            between_cell_consistency = max(
                                0, 1.0 - between_cell_diff / consistency_threshold
                            )
                        else:
                            between_cell_consistency = 1.0
                    except Exception:
                        between_cell_consistency = (
                            1.0  # Default if neighbor analysis fails
                        )
                else:
                    between_cell_consistency = 1.0

                # Overall consistency score
                within_cell_consistency = max(
                    0, 1.0 - std_value / consistency_threshold
                )
                overall_consistency = (
                    0.6 * within_cell_consistency + 0.4 * between_cell_consistency
                )

                consistency_results[h3_index] = {
                    "status": (
                        "consistent" if overall_consistency > 0.7 else "inconsistent"
                    ),
                    "consistency_score": overall_consistency,
                    "within_cell_std": std_value,
                    "within_cell_consistency": within_cell_consistency,
                    "between_cell_consistency": between_cell_consistency,
                    "measurement_count": len(cell_measurements),
                }

            return {
                "overall_consistent": all(
                    r["status"] == "consistent" for r in consistency_results.values()
                ),
                "cell_results": consistency_results,
                "total_cells": len(consistency_results),
                "consistent_cells": len(
                    [
                        r
                        for r in consistency_results.values()
                        if r["status"] == "consistent"
                    ]
                ),
            }

        except Exception as e:
            logger.error(f"Error in spatial consistency validation: {e}")
            return {"error": f"Consistency validation failed: {str(e)}"}
