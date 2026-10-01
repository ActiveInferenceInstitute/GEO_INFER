"""
H3 Machine Learning Integration Module.

Advanced H3 methods for machine learning applications including demand forecasting,
spatial prediction, and feature engineering based on Analytics Vidhya guide:
https://www.analyticsvidhya.com/blog/2025/03/ubers-h3-for-spatial-indexing/
"""

import logging
import math
from typing import Any
from datetime import datetime
from .core import H3Grid, H3Cell

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h3

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3-py package not available. Install with 'uv pip install h3'")


class H3MLFeatureEngine:
    """
    H3 Machine Learning Feature Engineering.

    Creates spatial features from H3 grids for machine learning models.
    Based on methods from Analytics Vidhya H3 guide for demand forecasting
    and spatial prediction applications.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize ML feature engine.

        Args:
            grid: H3Grid instance for feature extraction
        """
        self.grid = grid

    def create_spatial_features(
        self, target_column: str, neighbor_rings: int = 2
    ) -> dict[str, Any]:
        """
        Create spatial features for machine learning models.

        Based on Uber's approach to converting GPS data into hexagonal features
        for demand forecasting as described in Analytics Vidhya guide.

        Args:
            target_column: Column containing target values
            neighbor_rings: Number of neighbor rings to include in features

        Returns:
            Dictionary containing spatial features for each cell

        Example:
            >>> engine = H3MLFeatureEngine(grid)
            >>> features = engine.create_spatial_features('demand', neighbor_rings=2)
            >>> print(f"Created features for {len(features['features'])} cells")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        features = []

        for cell in self.grid.cells:
            if target_column not in cell.properties:
                continue

            cell_features = self._extract_cell_features(
                cell, target_column, neighbor_rings
            )
            features.append(cell_features)

        return {
            "features": features,
            "feature_names": self._get_feature_names(neighbor_rings),
            "target_column": target_column,
            "neighbor_rings": neighbor_rings,
            "method": "H3 Spatial Feature Engineering",
        }

    def _extract_cell_features(
        self, cell: H3Cell, target_column: str, neighbor_rings: int
    ) -> dict[str, Any]:
        """
        Extract features for a single cell.

        Args:
            cell: H3Cell to extract features from
            target_column: Target column name
            neighbor_rings: Number of neighbor rings

        Returns:
            Dictionary of features for the cell
        """
        features = {
            "cell_index": cell.index,
            "target_value": cell.properties.get(target_column, 0),
            "resolution": cell.resolution,
        }

        # Basic cell properties
        try:
            if H3_AVAILABLE:
                # Cell area and geometry features
                features["cell_area_km2"] = h3.cell_area(cell.index, "km^2")
                features["cell_area_m2"] = h3.cell_area(cell.index, "m^2")

                # Cell coordinates
                lat, lng = h3.cell_to_latlng(cell.index)
                features["cell_lat"] = lat
                features["cell_lng"] = lng

                # Distance from equator and prime meridian
                features["distance_from_equator"] = abs(lat)
                features["distance_from_prime_meridian"] = abs(lng)
        except Exception as e:
            logger.warning(f"Failed to extract basic features for {cell.index}: {e}")

        # Neighbor-based features
        neighbor_features = self._extract_neighbor_features(
            cell, target_column, neighbor_rings
        )
        features.update(neighbor_features)

        # Temporal features if timestamp available
        if "timestamp" in cell.properties:
            temporal_features = self._extract_temporal_features(cell)
            features.update(temporal_features)

        return features

    def _extract_neighbor_features(
        self, cell: H3Cell, target_column: str, neighbor_rings: int
    ) -> dict[str, Any]:
        """
        Extract neighbor-based features.

        Args:
            cell: H3Cell to analyze
            target_column: Target column name
            neighbor_rings: Number of neighbor rings

        Returns:
            Dictionary of neighbor features
        """
        neighbor_features = {}

        try:
            if H3_AVAILABLE:
                # Get neighbors at different ring distances
                for ring in range(1, neighbor_rings + 1):
                    ring_cells = h3.grid_ring(cell.index, ring)

                    # Find corresponding cells in grid
                    ring_values = []
                    for ring_cell_idx in ring_cells:
                        for grid_cell in self.grid.cells:
                            if grid_cell.index == ring_cell_idx:
                                if target_column in grid_cell.properties:
                                    ring_values.append(
                                        grid_cell.properties[target_column]
                                    )
                                break

                    if ring_values:
                        neighbor_features[f"ring_{ring}_mean"] = float(
                            np.mean(ring_values)
                        )
                        neighbor_features[f"ring_{ring}_std"] = float(
                            np.std(ring_values)
                        )
                        neighbor_features[f"ring_{ring}_max"] = float(
                            np.max(ring_values)
                        )
                        neighbor_features[f"ring_{ring}_min"] = float(
                            np.min(ring_values)
                        )

                        neighbor_features[f"ring_{ring}_count"] = len(ring_values)
                    else:
                        # No neighbors found at this ring
                        neighbor_features[f"ring_{ring}_mean"] = 0.0
                        neighbor_features[f"ring_{ring}_std"] = 0.0
                        neighbor_features[f"ring_{ring}_max"] = 0.0
                        neighbor_features[f"ring_{ring}_min"] = 0.0
                        neighbor_features[f"ring_{ring}_count"] = 0

                # Overall neighbor statistics
                all_neighbor_values = []
                all_neighbors = list(h3.grid_disk(cell.index, neighbor_rings))
                if cell.index in all_neighbors:
                    all_neighbors.remove(cell.index)  # Remove self

                for neighbor_idx in all_neighbors:
                    for grid_cell in self.grid.cells:
                        if grid_cell.index == neighbor_idx:
                            if target_column in grid_cell.properties:
                                all_neighbor_values.append(
                                    grid_cell.properties[target_column]
                                )
                            break

                if all_neighbor_values:
                    neighbor_features["neighbor_density"] = (
                        len(all_neighbor_values) / len(all_neighbors)
                        if all_neighbors
                        else 0
                    )
                    neighbor_features["neighbor_total"] = float(
                        np.sum(all_neighbor_values)
                    )
                    neighbor_features["neighbor_avg"] = float(
                        np.mean(all_neighbor_values)
                    )
                else:
                    neighbor_features["neighbor_density"] = 0.0
                    neighbor_features["neighbor_total"] = 0.0
                    neighbor_features["neighbor_avg"] = 0.0

        except Exception as e:
            logger.warning(f"Failed to extract neighbor features for {cell.index}: {e}")

        return neighbor_features

    def _extract_temporal_features(self, cell: H3Cell) -> dict[str, Any]:
        """
        Extract temporal features from cell timestamp.

        Args:
            cell: H3Cell with timestamp property

        Returns:
            Dictionary of temporal features
        """
        temporal_features: dict[str, Any] = {}

        try:
            timestamp_str = str(cell.properties["timestamp"])
            timestamp = self._parse_timestamp(timestamp_str)

            if timestamp:
                temporal_features["hour"] = timestamp.hour
                temporal_features["day_of_week"] = timestamp.weekday()
                temporal_features["day_of_month"] = timestamp.day
                temporal_features["month"] = timestamp.month
                temporal_features["quarter"] = (timestamp.month - 1) // 3 + 1
                temporal_features["is_weekend"] = 1 if timestamp.weekday() >= 5 else 0
                temporal_features["is_business_hour"] = (
                    1 if 9 <= timestamp.hour <= 17 else 0
                )
                temporal_features["is_rush_hour"] = (
                    1 if timestamp.hour in [7, 8, 9, 17, 18, 19] else 0
                )

                # Cyclical encoding for temporal features
                temporal_features["hour_sin"] = math.sin(
                    2 * math.pi * timestamp.hour / 24
                )
                temporal_features["hour_cos"] = math.cos(
                    2 * math.pi * timestamp.hour / 24
                )
                temporal_features["day_sin"] = math.sin(
                    2 * math.pi * timestamp.weekday() / 7
                )
                temporal_features["day_cos"] = math.cos(
                    2 * math.pi * timestamp.weekday() / 7
                )
                temporal_features["month_sin"] = math.sin(
                    2 * math.pi * timestamp.month / 12
                )
                temporal_features["month_cos"] = math.cos(
                    2 * math.pi * timestamp.month / 12
                )

        except Exception as e:
            logger.warning(f"Failed to extract temporal features: {e}")

        return temporal_features

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """Parse timestamp string into datetime object."""
        try:
            # Try common timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            # Try parsing as ISO format
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        except Exception:
            return None

    def _get_feature_names(self, neighbor_rings: int) -> list[str]:
        """Get list of feature names."""
        feature_names = [
            "cell_index",
            "target_value",
            "resolution",
            "cell_area_km2",
            "cell_area_m2",
            "cell_lat",
            "cell_lng",
            "distance_from_equator",
            "distance_from_prime_meridian",
        ]

        # Add neighbor features
        for ring in range(1, neighbor_rings + 1):
            feature_names.extend(
                [
                    f"ring_{ring}_mean",
                    f"ring_{ring}_std",
                    f"ring_{ring}_max",
                    f"ring_{ring}_min",
                    f"ring_{ring}_count",
                ]
            )

        feature_names.extend(["neighbor_density", "neighbor_total", "neighbor_avg"])

        # Add temporal features
        temporal_features = [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "is_weekend",
            "is_business_hour",
            "is_rush_hour",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]
        feature_names.extend(temporal_features)

        return feature_names

    def create_demand_forecasting_features(
        self, demand_column: str, time_column: str = "timestamp"
    ) -> dict[str, Any]:
        """
        Create features specifically for demand forecasting models.

        Based on Uber's demand forecasting approach using H3 hexagonal features
        as described in Analytics Vidhya guide.

        Args:
            demand_column: Column containing demand values
            time_column: Column containing timestamps

        Returns:
            Dictionary containing demand forecasting features

        Example:
            >>> features = engine.create_demand_forecasting_features('ride_requests')
            >>> print(f"Created demand features: {len(features['features'])} cells")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Create base spatial features
        spatial_features = self.create_spatial_features(demand_column, neighbor_rings=3)

        if "error" in spatial_features:
            return spatial_features

        # Add demand-specific features
        demand_features = []

        for feature_dict in spatial_features["features"]:
            cell_index = feature_dict["cell_index"]

            # Find corresponding cell
            cell = None
            for grid_cell in self.grid.cells:
                if grid_cell.index == cell_index:
                    cell = grid_cell
                    break

            if not cell:
                continue

            # Add demand-specific features
            demand_dict = feature_dict.copy()

            # Historical demand patterns
            demand_dict.update(self._calculate_demand_patterns(cell, demand_column))

            # Spatial demand gradients
            demand_dict.update(self._calculate_demand_gradients(cell, demand_column))

            # Supply-demand balance indicators
            demand_dict.update(
                self._calculate_supply_demand_balance(cell, demand_column)
            )

            demand_features.append(demand_dict)

        return {
            "features": demand_features,
            "feature_names": self._get_demand_feature_names(),
            "target_column": demand_column,
            "method": "H3 Demand Forecasting Features",
        }

    def _calculate_demand_patterns(
        self, cell: H3Cell, demand_column: str
    ) -> dict[str, Any]:
        """Calculate demand pattern features."""
        patterns = {}

        demand_value = cell.properties.get(demand_column, 0)

        # Basic demand statistics
        patterns["demand_value"] = demand_value
        patterns["demand_log"] = math.log(max(1, demand_value))  # Log transform
        patterns["demand_sqrt"] = math.sqrt(
            max(0, demand_value)
        )  # Square root transform

        # Demand density (demand per unit area)
        try:
            if H3_AVAILABLE:
                area_km2 = h3.cell_area(cell.index, "km^2")
                patterns["demand_density"] = (
                    demand_value / area_km2 if area_km2 > 0 else 0
                )
        except Exception:
            patterns["demand_density"] = 0

        return patterns

    def _calculate_demand_gradients(
        self, cell: H3Cell, demand_column: str
    ) -> dict[str, Any]:
        """Calculate spatial demand gradients."""
        gradients = {}

        try:
            if H3_AVAILABLE:
                cell_demand = cell.properties.get(demand_column, 0)

                # Get immediate neighbors
                neighbors = list(h3.grid_disk(cell.index, 1))
                if cell.index in neighbors:
                    neighbors.remove(cell.index)

                neighbor_demands = []
                for neighbor_idx in neighbors:
                    for grid_cell in self.grid.cells:
                        if grid_cell.index == neighbor_idx:
                            if demand_column in grid_cell.properties:
                                neighbor_demands.append(
                                    grid_cell.properties[demand_column]
                                )
                            break

                if neighbor_demands:
                    avg_neighbor_demand = sum(neighbor_demands) / len(neighbor_demands)
                    gradients["demand_gradient"] = cell_demand - avg_neighbor_demand
                    gradients["demand_gradient_abs"] = abs(gradients["demand_gradient"])
                    gradients["demand_gradient_normalized"] = gradients[
                        "demand_gradient"
                    ] / max(1, cell_demand + avg_neighbor_demand)
                else:
                    gradients["demand_gradient"] = 0
                    gradients["demand_gradient_abs"] = 0
                    gradients["demand_gradient_normalized"] = 0
        except Exception as e:
            logger.warning(f"Failed to calculate demand gradients: {e}")
            gradients["demand_gradient"] = 0
            gradients["demand_gradient_abs"] = 0
            gradients["demand_gradient_normalized"] = 0

        return gradients

    def _calculate_supply_demand_balance(
        self, cell: H3Cell, demand_column: str
    ) -> dict[str, Any]:
        """Calculate supply-demand balance indicators."""
        balance = {}

        demand_value = cell.properties.get(demand_column, 0)
        supply_value = cell.properties.get(
            "supply", demand_value
        )  # Default to demand if no supply

        # Supply-demand ratio
        balance["supply_demand_ratio"] = supply_value / max(1, demand_value)
        balance["demand_supply_gap"] = demand_value - supply_value
        balance["demand_supply_gap_abs"] = abs(balance["demand_supply_gap"])

        # Utilization rate
        balance["utilization_rate"] = min(1.0, demand_value / max(1, supply_value))

        # Scarcity indicator
        balance["scarcity_indicator"] = max(0, demand_value - supply_value) / max(
            1, demand_value
        )

        return balance

    def _get_demand_feature_names(self) -> list[str]:
        """Get demand forecasting feature names."""
        base_names = self._get_feature_names(3)  # 3 neighbor rings

        demand_names = [
            "demand_value",
            "demand_log",
            "demand_sqrt",
            "demand_density",
            "demand_gradient",
            "demand_gradient_abs",
            "demand_gradient_normalized",
            "supply_demand_ratio",
            "demand_supply_gap",
            "demand_supply_gap_abs",
            "utilization_rate",
            "scarcity_indicator",
        ]

        return base_names + demand_names


class H3DisasterResponse:
    """
    H3 methods for disaster response and environmental monitoring.

    Implements spatial analysis methods for emergency response, evacuation planning,
    and environmental monitoring using H3 hexagonal grids.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize disaster response analyzer.

        Args:
            grid: H3Grid instance for analysis
        """
        self.grid = grid

    def analyze_evacuation_zones(
        self,
        hazard_column: str,
        population_column: str = "population",
        evacuation_radius_km: float = 5.0,
    ) -> dict[str, Any]:
        """
        Analyze evacuation zones based on hazard locations.

        Args:
            hazard_column: Column containing hazard intensity values
            population_column: Column containing population data
            evacuation_radius_km: Evacuation radius in kilometers

        Returns:
            Dictionary containing evacuation zone analysis

        Example:
            >>> analyzer = H3DisasterResponse(grid)
            >>> zones = analyzer.analyze_evacuation_zones('flood_risk', 'population')
            >>> print(f"Identified {len(zones['high_risk_zones'])} high-risk zones")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Identify high-risk cells
        high_risk_cells = []
        evacuation_zones = []
        affected_population = 0

        for cell in self.grid.cells:
            if hazard_column not in cell.properties:
                continue

            hazard_level = cell.properties[hazard_column]
            population = cell.properties.get(population_column, 0)

            # Define risk thresholds (can be customized)
            if hazard_level > 0.7:  # High risk threshold
                high_risk_cells.append(
                    {
                        "cell_index": cell.index,
                        "hazard_level": hazard_level,
                        "population": population,
                        "risk_category": "high",
                    }
                )

                # Calculate evacuation zone
                evacuation_zone = self._calculate_evacuation_zone(
                    cell, evacuation_radius_km, population_column
                )
                evacuation_zones.append(evacuation_zone)
                affected_population += evacuation_zone["total_population"]

        # Calculate evacuation routes and capacity
        evacuation_analysis = self._analyze_evacuation_capacity(evacuation_zones)

        return {
            "high_risk_zones": high_risk_cells,
            "evacuation_zones": evacuation_zones,
            "total_affected_population": affected_population,
            "evacuation_analysis": evacuation_analysis,
            "evacuation_radius_km": evacuation_radius_km,
            "method": "H3 Evacuation Zone Analysis",
        }

    def _calculate_evacuation_zone(
        self, hazard_cell: H3Cell, radius_km: float, population_column: str
    ) -> dict[str, Any]:
        """Calculate evacuation zone around a hazard cell."""
        try:
            if not H3_AVAILABLE:
                return {"error": "H3 not available"}

            # Estimate number of rings needed for radius
            # Approximate: each ring adds ~edge_length to radius
            cell_resolution = h3.get_resolution(hazard_cell.index)

            # Real average edge length from the H3 API (valid for resolutions 0-15)
            edge_length_km = float(
                h3.average_hexagon_edge_length(cell_resolution, unit="km")
            )
            estimated_rings = max(1, int(radius_km / edge_length_km))

            # Get cells within evacuation radius
            evacuation_cells = h3.grid_disk(hazard_cell.index, estimated_rings)

            # Calculate evacuation zone statistics
            total_population = 0
            zone_cells = []

            for cell_idx in evacuation_cells:
                # Find corresponding cell in grid
                for grid_cell in self.grid.cells:
                    if grid_cell.index == cell_idx:
                        population = grid_cell.properties.get(population_column, 0)
                        total_population += population

                        zone_cells.append(
                            {
                                "cell_index": cell_idx,
                                "population": population,
                                "distance_rings": h3.grid_distance(
                                    hazard_cell.index, cell_idx
                                ),
                            }
                        )
                        break

            return {
                "hazard_cell": hazard_cell.index,
                "evacuation_cells": zone_cells,
                "total_population": total_population,
                "zone_area_cells": len(evacuation_cells),
                "estimated_radius_km": radius_km,
            }

        except Exception as e:
            logger.error(f"Failed to calculate evacuation zone: {e}")
            return {"error": str(e)}

    def _analyze_evacuation_capacity(
        self, evacuation_zones: list[dict]
    ) -> dict[str, Any]:
        """Analyze evacuation capacity and requirements."""
        total_zones = len(evacuation_zones)
        total_population = sum(zone["total_population"] for zone in evacuation_zones)

        # Estimate evacuation requirements
        # Assumptions: 4 people per vehicle, 30 minutes per evacuation trip
        vehicles_needed = math.ceil(total_population / 4)
        evacuation_time_hours = math.ceil(
            vehicles_needed / 100
        )  # Assume 100 vehicles available

        # Priority zones (highest population)
        priority_zones = sorted(
            evacuation_zones, key=lambda x: x["total_population"], reverse=True
        )[:5]

        return {
            "total_evacuation_zones": total_zones,
            "total_affected_population": total_population,
            "estimated_vehicles_needed": vehicles_needed,
            "estimated_evacuation_time_hours": evacuation_time_hours,
            "priority_zones": priority_zones,
            "average_population_per_zone": (
                total_population / total_zones if total_zones > 0 else 0
            ),
        }

    def monitor_environmental_changes(
        self, baseline_column: str, current_column: str, change_threshold: float = 0.2
    ) -> dict[str, Any]:
        """
        Monitor environmental changes using H3 spatial analysis.

        Args:
            baseline_column: Column containing baseline environmental values
            current_column: Column containing current environmental values
            change_threshold: Threshold for significant change detection

        Returns:
            Dictionary containing environmental change analysis

        Example:
            >>> changes = analyzer.monitor_environmental_changes('baseline_temp', 'current_temp')
            >>> print(f"Detected {len(changes['significant_changes'])} significant changes")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        significant_changes = []
        all_changes = []

        for cell in self.grid.cells:
            if (
                baseline_column not in cell.properties
                or current_column not in cell.properties
            ):
                continue

            baseline_value = cell.properties[baseline_column]
            current_value = cell.properties[current_column]

            # Calculate change metrics
            absolute_change = current_value - baseline_value
            relative_change = (
                absolute_change / baseline_value if baseline_value != 0 else 0
            )

            change_data = {
                "cell_index": cell.index,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "absolute_change": absolute_change,
                "relative_change": relative_change,
                "change_magnitude": abs(relative_change),
            }

            all_changes.append(change_data)

            # Check if change is significant
            if abs(relative_change) > change_threshold:
                change_data["change_type"] = (
                    "increase" if relative_change > 0 else "decrease"
                )
                change_data["significance"] = (
                    "high"
                    if abs(relative_change) > change_threshold * 2
                    else "moderate"
                )
                significant_changes.append(change_data)

        # Spatial clustering of changes
        change_clusters = self._cluster_environmental_changes(significant_changes)

        # Calculate summary statistics
        if all_changes:
            changes_array = np.array([c["relative_change"] for c in all_changes])
            summary_stats = {
                "mean_change": float(np.mean(changes_array)),
                "std_change": float(np.std(changes_array)),
                "max_change": float(np.max(changes_array)),
                "min_change": float(np.min(changes_array)),
                "median_change": float(np.median(changes_array)),
            }
        else:
            summary_stats = {}

        return {
            "significant_changes": significant_changes,
            "all_changes": all_changes,
            "change_clusters": change_clusters,
            "summary_statistics": summary_stats,
            "change_threshold": change_threshold,
            "total_cells_analyzed": len(all_changes),
            "method": "H3 Environmental Change Monitoring",
        }

    def _cluster_environmental_changes(self, changes: list[dict]) -> list[dict]:
        """Cluster spatially adjacent environmental changes."""
        if not changes or not H3_AVAILABLE:
            return []

        clusters: list[dict[str, Any]] = []
        processed_cells = set()

        for change in changes:
            cell_index = change["cell_index"]

            if cell_index in processed_cells:
                continue

            # Find spatially connected changes
            cluster_cells = [change]
            cluster_queue = [cell_index]
            processed_cells.add(cell_index)

            while cluster_queue:
                current_cell = cluster_queue.pop(0)

                try:
                    # Get neighbors
                    neighbors = h3.grid_disk(current_cell, 1)
                    neighbors.discard(current_cell)

                    for neighbor_idx in neighbors:
                        if neighbor_idx in processed_cells:
                            continue

                        # Check if neighbor has significant change
                        for other_change in changes:
                            if (
                                other_change["cell_index"] == neighbor_idx
                                and neighbor_idx not in processed_cells
                            ):
                                cluster_cells.append(other_change)
                                cluster_queue.append(neighbor_idx)
                                processed_cells.add(neighbor_idx)
                                break

                except Exception as e:
                    logger.warning(
                        f"Failed to process neighbors for {current_cell}: {e}"
                    )

            if len(cluster_cells) > 1:  # Only include multi-cell clusters
                # Calculate cluster statistics
                cluster_changes = [c["relative_change"] for c in cluster_cells]

                cluster_stats = {
                    "mean_change": float(np.mean(cluster_changes)),
                    "max_change": float(np.max(cluster_changes)),
                    "min_change": float(np.min(cluster_changes)),
                }

                clusters.append(
                    {
                        "cluster_id": len(clusters),
                        "cells": cluster_cells,
                        "cluster_size": len(cluster_cells),
                        "cluster_statistics": cluster_stats,
                    }
                )

        return clusters


class H3PerformanceOptimizer:
    """
    H3 Performance Optimization and Benchmarking.

    Methods for optimizing H3 operations and benchmarking performance
    for large-scale spatial analysis applications.
    """

    def __init__(self) -> None:
        """Initialize performance optimizer."""
        self.benchmark_results: dict[str, Any] = {}
        self.optimization_history: list[dict[str, Any]] = []

    def benchmark_h3_operations(
        self,
        test_coordinates: list[tuple[float, float]],
        resolutions: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Benchmark H3 operations performance.

        Args:
            test_coordinates: List of (lat, lng) coordinates for testing
            resolutions: List of H3 resolutions to test

        Returns:
            Dictionary containing benchmark results

        Example:
            >>> optimizer = H3PerformanceOptimizer()
            >>> coords = [(37.7749, -122.4194), (40.7128, -74.0060)]
            >>> results = optimizer.benchmark_h3_operations(coords)
            >>> print(f"Coordinate conversion: {results['coordinate_conversion']['avg_time_ms']:.2f}ms")
        """
        if not H3_AVAILABLE:
            return {"error": "H3 not available for benchmarking"}

        if resolutions is None:
            resolutions = [6, 7, 8, 9, 10]

        import time

        benchmark_results = {}

        # Benchmark coordinate to cell conversion
        start_time = time.time()
        for lat, lng in test_coordinates:
            for resolution in resolutions:
                h3.latlng_to_cell(lat, lng, resolution)
        end_time = time.time()

        coord_conversion_time = (
            end_time - start_time
        ) * 1000  # Convert to milliseconds
        benchmark_results["coordinate_conversion"] = {
            "total_time_ms": coord_conversion_time,
            "avg_time_ms": coord_conversion_time
            / (len(test_coordinates) * len(resolutions)),
            "operations_per_second": (len(test_coordinates) * len(resolutions))
            / (coord_conversion_time / 1000),
        }

        # Benchmark neighbor operations
        test_cells = [
            h3.latlng_to_cell(lat, lng, 9) for lat, lng in test_coordinates[:10]
        ]

        start_time = time.time()
        for cell in test_cells:
            h3.grid_disk(cell, 2)  # Get neighbors within 2 rings
        end_time = time.time()

        neighbor_time = (end_time - start_time) * 1000
        benchmark_results["neighbor_operations"] = {
            "total_time_ms": neighbor_time,
            "avg_time_ms": neighbor_time / len(test_cells),
            "operations_per_second": len(test_cells) / (neighbor_time / 1000),
        }

        # Benchmark distance calculations
        if len(test_cells) >= 2:
            start_time = time.time()
            for i in range(len(test_cells) - 1):
                h3.grid_distance(test_cells[i], test_cells[i + 1])
            end_time = time.time()

            distance_time = (end_time - start_time) * 1000
            benchmark_results["distance_calculations"] = {
                "total_time_ms": distance_time,
                "avg_time_ms": distance_time / (len(test_cells) - 1),
                "operations_per_second": (len(test_cells) - 1) / (distance_time / 1000),
            }

        # Memory usage estimation
        benchmark_results["memory_usage"] = self._estimate_memory_usage(test_cells)

        return {
            "benchmark_results": benchmark_results,
            "test_parameters": {
                "num_coordinates": len(test_coordinates),
                "resolutions_tested": resolutions,
                "num_test_cells": len(test_cells),
            },
            "method": "H3 Performance Benchmarking",
        }

    def _estimate_memory_usage(self, test_cells: list[str]) -> dict[str, Any]:
        """Estimate memory usage for H3 operations."""
        import sys

        # Estimate memory usage for different H3 data structures
        single_cell_size = sys.getsizeof(test_cells[0]) if test_cells else 0

        # Estimate neighbor storage
        if test_cells and H3_AVAILABLE:
            neighbors = h3.grid_disk(test_cells[0], 2)
            neighbor_storage = sys.getsizeof(neighbors) + sum(
                sys.getsizeof(cell) for cell in neighbors
            )
        else:
            neighbor_storage = 0

        return {
            "single_cell_bytes": single_cell_size,
            "neighbor_storage_bytes": neighbor_storage,
            "estimated_cells_per_mb": 1024 * 1024 // max(1, single_cell_size),
        }

    def optimize_grid_resolution(
        self,
        area_km2: float,
        target_cells: int | None = None,
        analysis_type: str = "general",
    ) -> dict[str, Any]:
        """
        Recommend optimal H3 resolution for given area and analysis type.

        Args:
            area_km2: Area in square kilometers
            target_cells: Target number of cells (optional)
            analysis_type: Type of analysis ('general', 'ml', 'routing', 'visualization')

        Returns:
            Dictionary containing resolution recommendations

        Example:
            >>> optimizer = H3PerformanceOptimizer()
            >>> rec = optimizer.optimize_grid_resolution(100.0, analysis_type='ml')
            >>> print(f"Recommended resolution: {rec['recommended_resolution']}")
        """
        # H3 resolution statistics computed from the real H3 API
        resolution_stats = {
            res: {
                "avg_area_km2": float(h3.average_hexagon_area(res, unit="km^2")),
                "edge_length_km": float(h3.average_hexagon_edge_length(res, unit="km")),
            }
            for res in range(16)
        }

        recommendations = []

        for resolution, stats in resolution_stats.items():
            estimated_cells = area_km2 / stats["avg_area_km2"]

            # Calculate suitability score based on analysis type
            suitability_score = self._calculate_suitability_score(
                estimated_cells, target_cells, analysis_type, resolution
            )

            recommendations.append(
                {
                    "resolution": resolution,
                    "estimated_cells": int(estimated_cells),
                    "avg_cell_area_km2": stats["avg_area_km2"],
                    "edge_length_km": stats["edge_length_km"],
                    "suitability_score": suitability_score,
                }
            )

        # Sort by suitability score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)

        return {
            "recommended_resolution": recommendations[0]["resolution"],
            "estimated_cells": recommendations[0]["estimated_cells"],
            "all_recommendations": recommendations[:5],  # Top 5
            "analysis_parameters": {
                "area_km2": area_km2,
                "target_cells": target_cells,
                "analysis_type": analysis_type,
            },
        }

    def _calculate_suitability_score(
        self,
        estimated_cells: float,
        target_cells: int | None,
        analysis_type: str,
        resolution: int,
    ) -> float:
        """Calculate suitability score for a resolution."""
        score = 0.0

        # Base score from cell count appropriateness
        if target_cells:
            # Prefer resolutions that give close to target cells
            cell_diff = abs(estimated_cells - target_cells) / target_cells
            score += max(0, 1 - cell_diff) * 40
        else:
            # General preferences for cell counts
            if 100 <= estimated_cells <= 10000:
                score += 40
            elif 10 <= estimated_cells <= 100000:
                score += 30
            else:
                score += 10

        # Analysis type preferences
        if analysis_type == "ml":
            # ML prefers moderate resolutions (7-10) for good feature granularity
            if 7 <= resolution <= 10:
                score += 30
            elif 5 <= resolution <= 12:
                score += 20
            else:
                score += 10
        elif analysis_type == "visualization":
            # Visualization prefers fewer cells for performance
            if estimated_cells <= 1000:
                score += 30
            elif estimated_cells <= 5000:
                score += 20
            else:
                score += 10
        elif analysis_type == "routing":
            # Routing prefers higher resolution for accuracy
            if 8 <= resolution <= 12:
                score += 30
            elif 6 <= resolution <= 14:
                score += 20
            else:
                score += 10
        else:  # general
            # General analysis prefers balanced approach
            if 6 <= resolution <= 10:
                score += 30
            elif 4 <= resolution <= 12:
                score += 20
            else:
                score += 10

        # Performance considerations
        if estimated_cells <= 50000:  # Good performance
            score += 20
        elif estimated_cells <= 200000:  # Moderate performance
            score += 10
        # else: no bonus for large cell counts

        return score
