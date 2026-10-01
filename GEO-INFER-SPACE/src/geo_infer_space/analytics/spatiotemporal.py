"""
Spatio-Temporal Analysis Module for GEO-INFER-SPACE.

This module provides comprehensive temporal-spatial analysis capabilities that
integrate time-series analysis with spatial context using H3 indexing.

Features:
- Space-time clustering (ST-DBSCAN)
- Space-time cubes for 3D aggregation
- Movement pattern analysis
- Emerging hotspot detection
- Spatio-temporal autocorrelation
- Time-aware spatial interpolation
"""

import logging
from typing import Any
from datetime import datetime
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class SpatioTemporalAnalyzer:
    """
    Comprehensive analyzer for spatio-temporal patterns.

    Integrates spatial (H3) and temporal analysis for patterns
    that emerge across both space and time.
    """

    def __init__(self, h3_backend: Any | None = None) -> None:
        """
        Initialize the SpatioTemporalAnalyzer.

        Args:
            h3_backend: Optional H3Backend instance for spatial operations
        """
        self.h3: Any = h3_backend
        if self.h3 is None:
            try:
                from ..backends.h3.h3_backend import H3Backend

                self.h3 = H3Backend()
            except ImportError:
                logger.warning("H3 backend not available")

        logger.info("SpatioTemporalAnalyzer initialized")

    def analyze_spatial_time_series(
        self,
        data: list[dict[str, Any]],
        cell_column: str,
        timestamp_column: str,
        value_column: str,
        temporal_resolution: str = "day",
    ) -> dict[str, Any]:
        """
        Analyze time series for each spatial cell.

        Args:
            data: List of records with cell, timestamp, and value
            cell_column: Column name for H3 cell ID
            timestamp_column: Column name for timestamps
            value_column: Column name for values
            temporal_resolution: 'hour', 'day', 'week', 'month'

        Returns:
            Per-cell time series analysis with trends and patterns
        """
        if not data:
            return {"error": "No data provided"}

        # Group by cell
        cell_series: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for record in data:
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)
            value = record.get(value_column)

            if cell and ts and value is not None:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    cell_series[cell].append(
                        {"timestamp": timestamp, "value": float(value)}
                    )

        # Analyze each cell's time series
        cell_analyses: dict[Any, dict[str, Any]] = {}
        for cell, series in cell_series.items():
            sorted_series = sorted(series, key=lambda x: x["timestamp"])
            values = [s["value"] for s in sorted_series]

            analysis: dict[str, Any] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "first_timestamp": sorted_series[0]["timestamp"].isoformat(),
                "last_timestamp": sorted_series[-1]["timestamp"].isoformat(),
            }

            # Trend detection
            if len(values) >= 3:
                trend = self._detect_trend(values)
                analysis["trend"] = trend

            # Variability
            if len(values) > 1:
                analysis["std"] = float(np.std(values))
                analysis["cv"] = (
                    analysis["std"] / analysis["mean"] if analysis["mean"] != 0 else 0
                )

            cell_analyses[cell] = analysis

        # Spatial summary
        all_means = [a["mean"] for a in cell_analyses.values()]

        return {
            "num_cells": len(cell_analyses),
            "total_records": sum(a["count"] for a in cell_analyses.values()),
            "cell_analyses": cell_analyses,
            "spatial_summary": {
                "mean_across_cells": (
                    sum(all_means) / len(all_means) if all_means else 0
                ),
                "min_mean": min(all_means) if all_means else 0,
                "max_mean": max(all_means) if all_means else 0,
            },
            "temporal_resolution": temporal_resolution,
        }

    def detect_spatiotemporal_clusters(
        self,
        data: list[dict[str, Any]],
        cell_column: str,
        timestamp_column: str,
        spatial_eps: int = 1,
        temporal_eps_hours: float = 24,
        min_points: int = 3,
    ) -> dict[str, Any]:
        """
        Detect spatio-temporal clusters using ST-DBSCAN algorithm.

        Finds clusters of points that are close in both space and time.

        Args:
            data: List of records with cell and timestamp
            cell_column: Column for H3 cell ID
            timestamp_column: Column for timestamps
            spatial_eps: Max grid distance for neighbors
            temporal_eps_hours: Max time difference in hours
            min_points: Minimum points to form cluster

        Returns:
            Cluster assignments and statistics
        """
        if not data or not self.h3:
            return {"error": "No data or H3 backend not available"}

        # Parse data
        points: list[dict[str, Any]] = []
        for i, record in enumerate(data):
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)
            if cell and ts:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    points.append(
                        {
                            "index": i,
                            "cell": cell,
                            "timestamp": timestamp,
                            "cluster": -1,  # -1 = unassigned
                        }
                    )

        if len(points) < min_points:
            return {"error": f"Need at least {min_points} points"}

        temporal_eps_seconds = temporal_eps_hours * 3600

        def are_neighbors(p1: dict[str, Any], p2: dict[str, Any]) -> bool:
            """Check if two points are ST-neighbors."""
            # Temporal check
            time_diff = abs((p1["timestamp"] - p2["timestamp"]).total_seconds())
            if time_diff > temporal_eps_seconds:
                return False

            # Spatial check
            try:
                dist = self.h3.get_cell_distance(p1["cell"], p2["cell"])
                return bool(dist <= spatial_eps)
            except Exception:
                return bool(p1["cell"] == p2["cell"])

        def get_neighbors(point_idx: int) -> list[int]:
            """Get all neighbors of a point."""
            neighbors = []
            for i, p in enumerate(points):
                if i != point_idx and are_neighbors(points[point_idx], p):
                    neighbors.append(i)
            return neighbors

        # ST-DBSCAN
        cluster_id = 0
        for i, point in enumerate(points):
            if point["cluster"] != -1:
                continue

            neighbors = get_neighbors(i)
            if len(neighbors) < min_points - 1:
                point["cluster"] = 0  # Noise
                continue

            # Start new cluster
            cluster_id += 1
            point["cluster"] = cluster_id

            # Expand cluster
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                neighbor_idx = seed_set[j]
                neighbor = points[neighbor_idx]

                if neighbor["cluster"] == 0:  # Was noise
                    neighbor["cluster"] = cluster_id
                elif neighbor["cluster"] == -1:  # Unassigned
                    neighbor["cluster"] = cluster_id
                    neighbor_neighbors = get_neighbors(neighbor_idx)
                    if len(neighbor_neighbors) >= min_points - 1:
                        seed_set.extend(neighbor_neighbors)
                j += 1

        # Summarize clusters
        clusters = defaultdict(list)
        for p in points:
            clusters[p["cluster"]].append(p)

        cluster_summaries = []
        for cid, members in clusters.items():
            if cid == 0:
                continue  # Skip noise

            cells = list(set(m["cell"] for m in members))
            timestamps = [m["timestamp"] for m in members]

            cluster_summaries.append(
                {
                    "cluster_id": cid,
                    "size": len(members),
                    "unique_cells": len(cells),
                    "cells": cells[:10],  # Limit output
                    "start_time": min(timestamps).isoformat(),
                    "end_time": max(timestamps).isoformat(),
                    "duration_hours": (
                        max(timestamps) - min(timestamps)
                    ).total_seconds()
                    / 3600,
                }
            )

        noise_count = len(clusters.get(0, []))

        return {
            "num_clusters": len(cluster_summaries),
            "clusters": cluster_summaries,
            "noise_points": noise_count,
            "total_points": len(points),
            "parameters": {
                "spatial_eps": spatial_eps,
                "temporal_eps_hours": temporal_eps_hours,
                "min_points": min_points,
            },
        }

    def compute_space_time_cube(
        self,
        data: list[dict[str, Any]],
        cell_column: str,
        timestamp_column: str,
        value_column: str,
        temporal_bin_size: str = "day",
        aggregation: str = "mean",
    ) -> dict[str, Any]:
        """
        Create a space-time cube for 3D analysis (x, y, t).

        Aggregates values into spatial-temporal bins for pattern analysis.

        Args:
            data: List of records
            cell_column: H3 cell column
            timestamp_column: Timestamp column
            value_column: Value column
            temporal_bin_size: 'hour', 'day', 'week', 'month'
            aggregation: 'mean', 'sum', 'count', 'max', 'min'

        Returns:
            Space-time cube with binned values
        """
        if not data:
            return {"error": "No data provided"}

        # Build cube: {(cell, time_bin): [values]}
        cube = defaultdict(list)

        for record in data:
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)
            value = record.get(value_column)

            if cell and ts and value is not None:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    time_bin = self._get_time_bin(timestamp, temporal_bin_size)
                    cube[(cell, time_bin)].append(float(value))

        # Aggregate
        aggregated_cube = {}
        for key, values in cube.items():
            if aggregation == "mean":
                agg_value = sum(values) / len(values)
            elif aggregation == "sum":
                agg_value = sum(values)
            elif aggregation == "count":
                agg_value = len(values)
            elif aggregation == "max":
                agg_value = max(values)
            elif aggregation == "min":
                agg_value = min(values)
            else:
                agg_value = sum(values) / len(values)

            aggregated_cube[key] = {"value": agg_value, "count": len(values)}

        # Get unique cells and time bins
        cells = sorted(set(k[0] for k in aggregated_cube.keys()))
        time_bins = sorted(set(k[1] for k in aggregated_cube.keys()))

        # Create time slices
        time_slices = {}
        for time_bin in time_bins:
            slice_data = {}
            for cell in cells:
                if (cell, time_bin) in aggregated_cube:
                    slice_data[cell] = aggregated_cube[(cell, time_bin)]["value"]
            time_slices[time_bin] = {
                "values": slice_data,
                "cell_count": len(slice_data),
                "mean": sum(slice_data.values()) / len(slice_data) if slice_data else 0,
            }

        return {
            "num_cells": len(cells),
            "num_time_bins": len(time_bins),
            "total_bins": len(aggregated_cube),
            "time_bins": time_bins,
            "time_slices": time_slices,
            "aggregation": aggregation,
            "temporal_bin_size": temporal_bin_size,
        }

    def detect_emerging_hotspots(
        self,
        data: list[dict[str, Any]],
        cell_column: str,
        timestamp_column: str,
        value_column: str,
        time_steps: int = 5,
        threshold_percentile: float = 90,
    ) -> dict[str, Any]:
        """
        Detect emerging, intensifying, and diminishing hotspots.

        Analyzes trend of hotspot status over time periods.

        Args:
            data: List of records
            cell_column: H3 cell column
            timestamp_column: Timestamp column
            value_column: Value column
            time_steps: Number of time periods to analyze
            threshold_percentile: Percentile for hotspot threshold

        Returns:
            Hotspot classifications and trends
        """
        # Build space-time cube
        cube_result = self.compute_space_time_cube(
            data,
            cell_column,
            timestamp_column,
            value_column,
            temporal_bin_size="day",
            aggregation="mean",
        )

        if "error" in cube_result:
            return cube_result

        time_slices = cube_result["time_slices"]
        time_bins = sorted(time_slices.keys())[-time_steps:]  # Last N time steps

        if len(time_bins) < 2:
            return {"error": "Need at least 2 time periods"}

        # Calculate hotspot threshold for each time step
        cell_histories = defaultdict(list)

        for time_bin in time_bins:
            slice_data = time_slices[time_bin]["values"]

            if not slice_data:
                continue

            # Calculate threshold
            values = list(slice_data.values())
            threshold = float(np.percentile(values, threshold_percentile))
            # Classify each cell
            for cell, value in slice_data.items():
                is_hotspot = value >= threshold
                cell_histories[cell].append(
                    {"time_bin": time_bin, "value": value, "is_hotspot": is_hotspot}
                )

        # Classify hotspot patterns
        classifications: dict[str, list[dict[str, Any]]] = {
            "new": [],  # Not hot before, hot now
            "consecutive": [],  # Hot in all periods
            "intensifying": [],  # Hot and increasing
            "diminishing": [],  # Hot but decreasing
            "sporadic": [],  # Hot sometimes
            "cold": [],  # Never hot
        }

        for cell, history in cell_histories.items():
            if len(history) < 2:
                continue

            hotspot_count = sum(1 for h in history if h["is_hotspot"])
            recent_hot = history[-1]["is_hotspot"]
            first_hot = history[0]["is_hotspot"]

            # Trend calculation
            values = [h["value"] for h in history]
            trend = (values[-1] - values[0]) / max(abs(values[0]), 0.001)

            cell_info = {
                "cell": cell,
                "hotspot_count": hotspot_count,
                "trend": trend,
                "current_value": values[-1],
            }

            if hotspot_count == 0:
                classifications["cold"].append(cell_info)
            elif hotspot_count == len(history):
                if trend > 0.1:
                    classifications["intensifying"].append(cell_info)
                elif trend < -0.1:
                    classifications["diminishing"].append(cell_info)
                else:
                    classifications["consecutive"].append(cell_info)
            elif recent_hot and not first_hot:
                classifications["new"].append(cell_info)
            else:
                classifications["sporadic"].append(cell_info)

        return {
            "time_steps_analyzed": len(time_bins),
            "threshold_percentile": threshold_percentile,
            "classifications": {
                k: {"count": len(v), "cells": v[:10]}  # Limit output
                for k, v in classifications.items()
            },
            "summary": {
                "emerging_hotspots": len(classifications["new"]),
                "persistent_hotspots": len(classifications["consecutive"]),
                "intensifying": len(classifications["intensifying"]),
                "diminishing": len(classifications["diminishing"]),
            },
        }

    def compute_spatiotemporal_autocorrelation(
        self,
        data: list[dict[str, Any]],
        cell_column: str,
        timestamp_column: str,
        value_column: str,
        spatial_lag: int = 1,
        temporal_lag_hours: float = 24,
    ) -> dict[str, Any]:
        """
        Compute space-time autocorrelation (space-time Moran's I).

        Measures correlation between values at nearby locations in space and time.

        Args:
            data: List of records
            cell_column: H3 cell column
            timestamp_column: Timestamp column
            value_column: Value column
            spatial_lag: Grid distance for spatial neighbors
            temporal_lag_hours: Hours for temporal neighborhood

        Returns:
            Space-time autocorrelation statistics
        """
        if not data or not self.h3:
            return {"error": "No data or H3 backend not available"}

        # Parse data
        points = []
        for record in data:
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)
            value = record.get(value_column)

            if cell and ts and value is not None:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    points.append(
                        {"cell": cell, "timestamp": timestamp, "value": float(value)}
                    )

        if len(points) < 3:
            return {"error": "Need at least 3 points"}

        # Calculate global mean
        values = [p["value"] for p in points]
        mean_value = sum(values) / len(values)

        # Calculate space-time Moran's I
        temporal_lag_seconds = temporal_lag_hours * 3600

        numerator = 0.0
        denominator = 0.0
        weight_sum = 0.0
        neighbor_count = 0

        for i, pi in enumerate(points):
            dev_i = pi["value"] - mean_value
            denominator += dev_i**2

            for j, pj in enumerate(points):
                if i == j:
                    continue

                # Check space-time neighborhood
                time_diff = abs((pi["timestamp"] - pj["timestamp"]).total_seconds())
                if time_diff > temporal_lag_seconds:
                    continue

                try:
                    spatial_dist = self.h3.get_cell_distance(pi["cell"], pj["cell"])
                    if spatial_dist > spatial_lag:
                        continue
                except Exception:
                    if pi["cell"] != pj["cell"]:
                        continue

                # Weight is 1 for neighbors
                weight = 1.0
                dev_j = pj["value"] - mean_value
                numerator += weight * dev_i * dev_j
                weight_sum += weight
                neighbor_count += 1

        n = len(points)

        if weight_sum > 0 and denominator > 0:
            morans_i = (n / weight_sum) * (numerator / denominator)
        else:
            morans_i = 0.0

        # Interpretation
        if morans_i > 0.3:
            interpretation = (
                "Strong positive spatial-temporal autocorrelation (clustered)"
            )
        elif morans_i > 0.1:
            interpretation = "Moderate positive autocorrelation"
        elif morans_i < -0.3:
            interpretation = "Strong negative autocorrelation (dispersed)"
        elif morans_i < -0.1:
            interpretation = "Moderate negative autocorrelation"
        else:
            interpretation = "Random pattern (no significant autocorrelation)"

        return {
            "morans_i": morans_i,
            "interpretation": interpretation,
            "n_points": n,
            "n_neighbors": neighbor_count,
            "avg_neighbors_per_point": neighbor_count / n if n > 0 else 0,
            "parameters": {
                "spatial_lag": spatial_lag,
                "temporal_lag_hours": temporal_lag_hours,
            },
        }

    def analyze_movement_patterns(
        self,
        trajectories: list[dict[str, Any]],
        id_column: str,
        cell_column: str,
        timestamp_column: str,
    ) -> dict[str, Any]:
        """
        Analyze movement patterns from trajectory data.

        Computes flow statistics, common routes, and movement metrics.

        Args:
            trajectories: List of records with entity ID, cell, timestamp
            id_column: Column for entity/trajectory ID
            cell_column: H3 cell column
            timestamp_column: Timestamp column

        Returns:
            Movement pattern analysis
        """
        if not trajectories:
            return {"error": "No trajectory data"}

        # Group by entity
        entity_tracks = defaultdict(list)
        for record in trajectories:
            entity_id = record.get(id_column)
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)

            if entity_id and cell and ts:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    entity_tracks[entity_id].append(
                        {"cell": cell, "timestamp": timestamp}
                    )

        # Sort each track by time
        for entity_id in entity_tracks:
            entity_tracks[entity_id].sort(key=lambda x: x["timestamp"])

        # Analyze flows between cells
        flows: dict[tuple[Any, Any], int] = defaultdict(int)
        entity_stats = []

        for entity_id, track in entity_tracks.items():
            if len(track) < 2:
                continue

            cells_visited = [t["cell"] for t in track]
            unique_cells = len(set(cells_visited))

            # Count transitions
            for i in range(len(track) - 1):
                origin = track[i]["cell"]
                dest = track[i + 1]["cell"]
                if origin != dest:
                    flows[(origin, dest)] += 1

            # Time span
            duration = (
                track[-1]["timestamp"] - track[0]["timestamp"]
            ).total_seconds() / 3600

            entity_stats.append(
                {
                    "entity_id": entity_id,
                    "points": len(track),
                    "unique_cells": unique_cells,
                    "duration_hours": duration,
                    "mobility_ratio": (
                        unique_cells / len(track) if len(track) > 0 else 0
                    ),
                }
            )

        # Top flows
        sorted_flows = sorted(flows.items(), key=lambda x: x[1], reverse=True)
        top_flows = [
            {"origin": f[0][0], "destination": f[0][1], "count": f[1]}
            for f in sorted_flows[:20]
        ]

        # Summary
        avg_duration = (
            sum(e["duration_hours"] for e in entity_stats) / len(entity_stats)
            if entity_stats
            else 0
        )
        avg_cells = (
            sum(e["unique_cells"] for e in entity_stats) / len(entity_stats)
            if entity_stats
            else 0
        )

        return {
            "num_entities": len(entity_tracks),
            "total_flows": len(flows),
            "top_flows": top_flows,
            "summary": {
                "avg_duration_hours": avg_duration,
                "avg_unique_cells": avg_cells,
                "avg_mobility_ratio": (
                    sum(e["mobility_ratio"] for e in entity_stats) / len(entity_stats)
                    if entity_stats
                    else 0
                ),
            },
            "entity_stats": entity_stats[:10],  # Limit output
        }

    def kriging_spatiotemporal(
        self,
        known_data: list[dict[str, Any]],
        target_cells: list[str],
        target_timestamp: datetime,
        cell_column: str,
        timestamp_column: str,
        value_column: str,
        spatial_range: int = 3,
        temporal_range_hours: float = 48,
    ) -> dict[str, Any]:
        """
        Interpolate values using space-time kriging.

        Estimates values at target locations and time using nearby
        observations weighted by space-time distance.

        Args:
            known_data: Records with known values
            target_cells: Cells to interpolate
            target_timestamp: Target time for interpolation
            cell_column: H3 cell column
            timestamp_column: Timestamp column
            value_column: Value column
            spatial_range: Max grid distance to include
            temporal_range_hours: Max time range to include

        Returns:
            Interpolated values for target cells
        """
        if not known_data or not target_cells:
            return {"error": "Missing data or targets"}

        if not self.h3:
            return {"error": "H3 backend not available"}

        # Parse known data
        known_points = []
        for record in known_data:
            cell = record.get(cell_column)
            ts = record.get(timestamp_column)
            value = record.get(value_column)

            if cell and ts and value is not None:
                timestamp = self._parse_timestamp(ts)
                if timestamp:
                    known_points.append(
                        {"cell": cell, "timestamp": timestamp, "value": float(value)}
                    )

        if not known_points:
            return {"error": "No valid known points"}

        temporal_range_seconds = temporal_range_hours * 3600
        interpolated = {}

        for target_cell in target_cells:
            weights = []
            values = []

            for kp in known_points:
                # Temporal distance
                time_diff = abs((kp["timestamp"] - target_timestamp).total_seconds())
                if time_diff > temporal_range_seconds:
                    continue

                # Spatial distance
                try:
                    spatial_dist = self.h3.get_cell_distance(target_cell, kp["cell"])
                    if spatial_dist > spatial_range:
                        continue
                except Exception:
                    if target_cell != kp["cell"]:
                        continue
                    spatial_dist = 0

                # Calculate space-time weight (inverse distance)
                temporal_weight = 1.0 / (1.0 + time_diff / 3600)
                spatial_weight = 1.0 / (1.0 + spatial_dist)
                combined_weight = temporal_weight * spatial_weight

                weights.append(combined_weight)
                values.append(kp["value"])

            if weights:
                total_weight = sum(weights)
                interpolated_value = (
                    sum(w * v for w, v in zip(weights, values)) / total_weight
                )
                interpolated[target_cell] = {
                    "value": interpolated_value,
                    "neighbors_used": len(weights),
                    "total_weight": total_weight,
                }
            else:
                interpolated[target_cell] = {
                    "value": None,
                    "neighbors_used": 0,
                    "error": "No neighbors in range",
                }

        # Summary
        valid_interpolations = [
            v for v in interpolated.values() if v["value"] is not None
        ]

        return {
            "target_timestamp": target_timestamp.isoformat(),
            "num_targets": len(target_cells),
            "num_interpolated": len(valid_interpolations),
            "interpolated": interpolated,
            "parameters": {
                "spatial_range": spatial_range,
                "temporal_range_hours": temporal_range_hours,
            },
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_timestamp(self, ts: Any) -> datetime | None:
        """Parse timestamp from various formats."""
        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(ts)
            except (ValueError, OSError, OverflowError):
                return None

        timestamp_str = str(ts)
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def _get_time_bin(self, timestamp: datetime, bin_size: str) -> str:
        """Get time bin key for a timestamp."""
        if bin_size == "hour":
            return timestamp.strftime("%Y-%m-%d-%H")
        elif bin_size == "day":
            return timestamp.strftime("%Y-%m-%d")
        elif bin_size == "week":
            return timestamp.strftime("%Y-W%W")
        elif bin_size == "month":
            return timestamp.strftime("%Y-%m")
        else:
            return timestamp.strftime("%Y-%m-%d")

    def _detect_trend(self, values: list[float]) -> dict[str, Any]:
        """Detect trend in a series of values."""
        n = len(values)
        if n < 2:
            return {"direction": "insufficient_data"}

        # Simple linear trend
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope: float = 0
        else:
            slope = numerator / denominator

        # Determine direction
        if slope > 0.01 * y_mean:
            direction = "increasing"
        elif slope < -0.01 * y_mean:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": slope,
            "change_percent": (values[-1] - values[0])
            / max(abs(values[0]), 0.001)
            * 100,
        }
