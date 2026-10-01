"""
Modular Spatial Methods for GEO-INFER-SPACE.

This module provides modular, composable spatial operations that work
with H3-indexed data. All methods are designed to be backend-agnostic
and easily chainable.
"""

import logging
from typing import Any, cast
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class SpatialMethods:
    """
    Modular spatial methods for H3-indexed analysis.

    Provides composable operations for buffers, overlays, filtering,
    aggregation, and spatial outlier detection.
    """

    def __init__(self, h3_backend: Any | None = None) -> None:
        """
        Initialize SpatialMethods.

        Args:
            h3_backend: Optional H3Backend instance
        """
        self.h3 = h3_backend
        if self.h3 is None:
            try:
                from ..backends.h3.h3_backend import H3Backend

                self.h3 = H3Backend()
            except ImportError:
                logger.warning("H3 backend not available")

        logger.info("SpatialMethods initialized")

    def buffer_analysis(
        self, cells: list[str], buffer_rings: int = 1, include_center: bool = True
    ) -> dict[str, Any]:
        """
        Create buffer zones around cells.

        Args:
            cells: List of H3 cell IDs
            buffer_rings: Number of rings to buffer
            include_center: Whether to include original cells

        Returns:
            Buffer analysis results with zones
        """
        if not cells or not self.h3:
            raise ValueError("No cells or H3 not available")

        center_cells = set(cells)
        buffer_cells = set()
        ring_cells: dict[int, set[str]] = {}

        for ring in range(1, buffer_rings + 1):
            ring_cells[ring] = set()

        for cell in cells:
            for ring in range(1, buffer_rings + 1):
                try:
                    ring_neighbors = self.h3.get_cell_ring(cell, ring)
                    ring_cells[ring].update(ring_neighbors)
                    buffer_cells.update(ring_neighbors)
                except Exception as e:
                    logger.debug(f"Could not get ring {ring} for H3 cell {cell}: {e}")
                    continue

        # Remove center cells from buffer
        buffer_cells -= center_cells

        # Also remove inner rings from outer rings
        for ring in range(2, buffer_rings + 1):
            for inner_ring in range(1, ring):
                ring_cells[ring] -= ring_cells[inner_ring]
            ring_cells[ring] -= center_cells

        result_cells = buffer_cells.copy()
        if include_center:
            result_cells.update(center_cells)

        return {
            "center_cells": list(center_cells),
            "buffer_cells": list(buffer_cells),
            "all_cells": list(result_cells),
            "rings": {k: list(v) for k, v in ring_cells.items()},
            "buffer_rings": buffer_rings,
            "center_count": len(center_cells),
            "buffer_count": len(buffer_cells),
            "total_count": len(result_cells),
        }

    def overlay_cells(
        self, cells_a: list[str], cells_b: list[str], operation: str = "intersection"
    ) -> dict[str, Any]:
        """
        Perform overlay operations between two cell sets.

        Args:
            cells_a: First set of cells
            cells_b: Second set of cells
            operation: 'intersection', 'union', 'difference', 'symmetric_difference'

        Returns:
            Result of overlay operation
        """
        set_a = set(cells_a)
        set_b = set(cells_b)

        if operation == "intersection":
            result = set_a & set_b
        elif operation == "union":
            result = set_a | set_b
        elif operation == "difference":
            result = set_a - set_b
        elif operation == "symmetric_difference":
            result = set_a ^ set_b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return {
            "operation": operation,
            "input_a_count": len(set_a),
            "input_b_count": len(set_b),
            "result_count": len(result),
            "result_cells": list(result),
            "overlap_ratio": len(set_a & set_b) / len(set_a | set_b)
            if set_a | set_b
            else 0,
        }

    def spatial_filter(
        self,
        cells: list[str],
        values: list[float],
        filter_type: str = "threshold",
        threshold: float | None = None,
        percentile: float | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """
        Filter cells based on spatial criteria.

        Args:
            cells: List of H3 cells
            values: Values at each cell
            filter_type: 'threshold', 'percentile', 'top_n', 'outliers'
            threshold: Value threshold (for 'threshold')
            percentile: Percentile threshold (for 'percentile')
            top_n: Number of top cells (for 'top_n')

        Returns:
            Filtered cells and values
        """
        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")

        cell_values = list(zip(cells, values))

        if filter_type == "threshold" and threshold is not None:
            filtered = [(c, v) for c, v in cell_values if v >= threshold]

        elif filter_type == "percentile" and percentile is not None:
            thresh = float(np.percentile(values, percentile))
            filtered = [(c, v) for c, v in cell_values if v >= thresh]

        elif filter_type == "top_n" and top_n is not None:
            sorted_cv = sorted(cell_values, key=lambda x: x[1], reverse=True)
            filtered = sorted_cv[:top_n]

        elif filter_type == "outliers":
            # IQR-based outlier detection
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [(c, v) for c, v in cell_values if v < lower or v > upper]

        else:
            raise ValueError("Invalid filter configuration")

        return {
            "filter_type": filter_type,
            "input_count": len(cells),
            "filtered_count": len(filtered),
            "filtered_cells": [c for c, v in filtered],
            "filtered_values": [v for c, v in filtered],
            "filter_ratio": len(filtered) / len(cells) if cells else 0,
        }

    def aggregate_to_region(
        self,
        cells: list[str],
        values: list[float],
        target_resolution: int,
        aggregation: str = "mean",
    ) -> dict[str, Any]:
        """
        Aggregate cell values to a coarser resolution.

        Args:
            cells: List of H3 cells
            values: Values at each cell
            target_resolution: Coarser resolution to aggregate to
            aggregation: 'mean', 'sum', 'max', 'min', 'count'

        Returns:
            Aggregated values at target resolution
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")

        # Group by parent cell
        parent_values = defaultdict(list)

        for cell, value in zip(cells, values):
            try:
                cell_res = self.h3.get_cell_resolution(cell)
                if cell_res <= target_resolution:
                    parent = cell
                else:
                    parent = self.h3.get_cell_parent(cell, target_resolution)
                parent_values[parent].append(value)
            except Exception as e:
                logger.debug(f"Failed to aggregate cell {cell}: {e}")
                continue

        # Aggregate
        aggregated = {}
        for parent, vals in parent_values.items():
            if aggregation == "mean":
                agg_val = sum(vals) / len(vals)
            elif aggregation == "sum":
                agg_val = sum(vals)
            elif aggregation == "max":
                agg_val = max(vals)
            elif aggregation == "min":
                agg_val = min(vals)
            elif aggregation == "count":
                agg_val = len(vals)
            else:
                agg_val = sum(vals) / len(vals)

            aggregated[parent] = {"value": agg_val, "child_count": len(vals)}

        return {
            "input_cells": len(cells),
            "output_cells": len(aggregated),
            "target_resolution": target_resolution,
            "aggregation": aggregation,
            "aggregated": aggregated,
            "compression_ratio": len(cells) / len(aggregated) if aggregated else 0,
        }

    def disaggregate_to_cells(
        self,
        parent_cells: list[str],
        values: list[float],
        target_resolution: int,
        method: str = "equal",
    ) -> dict[str, Any]:
        """
        Disaggregate values to finer resolution cells.

        Args:
            parent_cells: Coarser resolution cells
            values: Values at each parent cell
            target_resolution: Finer resolution
            method: 'equal' (split equally) or 'proportional'

        Returns:
            Disaggregated values at target resolution
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        if len(parent_cells) != len(values):
            raise ValueError("Cells and values must have same length")

        disaggregated = {}

        for parent, value in zip(parent_cells, values):
            try:
                children = self.h3.get_cell_children(parent, target_resolution)

                if method == "equal":
                    child_value = value / len(children) if children else 0
                    for child in children:
                        disaggregated[child] = child_value
                elif method == "proportional":
                    # Each child gets same value (density-preserving)
                    for child in children:
                        disaggregated[child] = value
            except Exception as e:
                logger.debug(f"Failed to disaggregate parent {parent}: {e}")
                continue

        return {
            "input_cells": len(parent_cells),
            "output_cells": len(disaggregated),
            "target_resolution": target_resolution,
            "method": method,
            "disaggregated": disaggregated,
            "expansion_ratio": len(disaggregated) / len(parent_cells)
            if parent_cells
            else 0,
        }

    def calculate_coverage(
        self, cells: list[str], region_cells: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Calculate coverage statistics for cell sets.

        Args:
            cells: Cells to measure coverage of
            region_cells: Optional region to calculate coverage within

        Returns:
            Coverage statistics including area
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        cells_set = set(cells)
        total_area = 0.0
        resolution_counts: dict[int, int] = defaultdict(int)

        for cell in cells:
            try:
                area = self.h3.get_cell_area(cell, "km^2")
                total_area += area
                res = self.h3.get_cell_resolution(cell)
                resolution_counts[res] += 1
            except Exception as e:
                logger.debug(f"Failed to calculate area for cell {cell}: {e}")
                continue

        result = {
            "num_cells": len(cells_set),
            "total_area_km2": total_area,
            "resolution_distribution": dict(resolution_counts),
        }

        if region_cells:
            region_set = set(region_cells)
            covered = cells_set & region_set
            result["region_cells"] = len(region_set)
            result["covered_cells"] = len(covered)
            result["coverage_ratio"] = (
                len(covered) / len(region_set) if region_set else 0
            )

        return result

    def find_spatial_outliers(
        self, cells: list[str], values: list[float], k: int = 1
    ) -> dict[str, Any]:
        """
        Find spatial outliers using Local Moran's I.

        Identifies cells that differ significantly from their neighbors.

        Args:
            cells: List of H3 cells
            values: Values at each cell
            k: Neighborhood ring size

        Returns:
            Outlier classifications (HH, LL, HL, LH)
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")

        cell_values = dict(zip(cells, values))
        mean_val = sum(values) / len(values)

        outliers: dict[str, list[dict[str, Any]]] = {
            "HH": [],  # High value, high neighbors (cluster)
            "LL": [],  # Low value, low neighbors (cluster)
            "HL": [],  # High value, low neighbors (outlier)
            "LH": [],  # Low value, high neighbors (outlier)
            "NS": [],  # Not significant
        }

        for cell, value in cell_values.items():
            # Get neighbor values
            try:
                neighbors = self.h3.get_cell_neighbors(cell, k)
                neighbor_vals = [cell_values[n] for n in neighbors if n in cell_values]
            except Exception as e:
                logger.debug(f"Failed to get neighbors for cell {cell}: {e}")
                neighbor_vals = []

            if not neighbor_vals:
                outliers["NS"].append({"cell": cell, "value": value})
                continue

            neighbor_mean = sum(neighbor_vals) / len(neighbor_vals)

            # Classify
            is_high = value > mean_val
            neighbor_high = neighbor_mean > mean_val

            # Calculate local Moran's I contribution
            z_i = value - mean_val
            z_neighbors = neighbor_mean - mean_val
            local_moran = z_i * z_neighbors

            if is_high and neighbor_high:
                category = "HH"
            elif not is_high and not neighbor_high:
                category = "LL"
            elif is_high and not neighbor_high:
                category = "HL"
            else:
                category = "LH"

            outliers[category].append(
                {
                    "cell": cell,
                    "value": value,
                    "neighbor_mean": neighbor_mean,
                    "local_moran": local_moran,
                }
            )

        return {
            "total_cells": len(cells),
            "outliers": {
                "HH_clusters": len(outliers["HH"]),
                "LL_clusters": len(outliers["LL"]),
                "HL_outliers": len(outliers["HL"]),
                "LH_outliers": len(outliers["LH"]),
                "not_significant": len(outliers["NS"]),
            },
            "details": {k: v[:10] for k, v in outliers.items()},  # Limit output
            "spatial_outlier_count": len(outliers["HL"]) + len(outliers["LH"]),
        }

    def compute_accessibility(
        self,
        origin_cells: list[str],
        destination_cells: list[str],
        max_distance: int = 10,
    ) -> dict[str, Any]:
        """
        Compute accessibility from origins to destinations.

        Args:
            origin_cells: Starting cells
            destination_cells: Target cells
            max_distance: Maximum grid distance to consider

        Returns:
            Accessibility scores for each origin
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        accessibility = {}

        for origin in origin_cells:
            reachable_count = 0
            total_distance = 0
            min_distance = float("inf")

            for dest in destination_cells:
                try:
                    dist = self.h3.get_cell_distance(origin, dest)
                    if dist <= max_distance:
                        reachable_count += 1
                        total_distance += dist
                        min_distance = min(min_distance, dist)
                except Exception as e:
                    logger.debug(
                        f"Failed to calc distance from {origin} to {dest}: {e}"
                    )
                    continue

            accessibility[origin] = {
                "reachable_destinations": reachable_count,
                "min_distance": min_distance if min_distance != float("inf") else None,
                "avg_distance": total_distance / reachable_count
                if reachable_count > 0
                else None,
                "accessibility_score": reachable_count / len(destination_cells)
                if destination_cells
                else 0,
            }

        scores: list[float] = [
            cast(float, a["accessibility_score"]) for a in accessibility.values()
        ]

        return {
            "num_origins": len(origin_cells),
            "num_destinations": len(destination_cells),
            "max_distance": max_distance,
            "accessibility": accessibility,
            "summary": {
                "mean_accessibility": sum(scores) / len(scores) if scores else 0,
                "max_accessibility": max(scores) if scores else 0,
                "min_accessibility": min(scores) if scores else 0,
                "fully_accessible_origins": sum(1 for s in scores if s == 1.0),
            },
        }

    def calculate_spatial_weights(
        self, cells: list[str], weight_type: str = "queen", k: int = 1
    ) -> dict[str, Any]:
        """
        Calculate spatial weights matrix for cells.

        Args:
            cells: List of H3 cells
            weight_type: 'queen' (all neighbors), 'rook' (shared edge), 'distance'
            k: Number of rings for neighbors

        Returns:
            Spatial weights for each cell pair
        """
        if not self.h3:
            raise RuntimeError("H3 backend not available")

        cell_set = set(cells)
        weights = {}
        neighbor_counts = {}

        for cell in cells:
            cell_weights = {}

            if weight_type in ["queen", "rook"]:
                try:
                    neighbors = self.h3.get_cell_neighbors(cell, k)
                    valid_neighbors = [n for n in neighbors if n in cell_set]

                    for neighbor in valid_neighbors:
                        cell_weights[neighbor] = 1.0

                    neighbor_counts[cell] = len(valid_neighbors)
                except Exception as e:
                    logger.debug(f"Failed to get weights for {cell}: {e}")
                    neighbor_counts[cell] = 0

            elif weight_type == "distance":
                for other in cells:
                    if other == cell:
                        continue
                    try:
                        dist = self.h3.get_cell_distance(cell, other)
                        if dist <= k:
                            cell_weights[other] = 1.0 / (dist + 1)
                    except Exception as e:
                        logger.debug(
                            f"Failed distance weight for {cell} to {other}: {e}"
                        )
                        continue
                neighbor_counts[cell] = len(cell_weights)

            # Row standardize
            total_weight = sum(cell_weights.values())
            if total_weight > 0:
                weights[cell] = {n: w / total_weight for n, w in cell_weights.items()}
            else:
                weights[cell] = {}

        avg_neighbors = (
            sum(neighbor_counts.values()) / len(neighbor_counts)
            if neighbor_counts
            else 0
        )

        return {
            "num_cells": len(cells),
            "weight_type": weight_type,
            "k": k,
            "weights": weights,
            "summary": {
                "avg_neighbors": avg_neighbors,
                "max_neighbors": max(neighbor_counts.values())
                if neighbor_counts
                else 0,
                "min_neighbors": min(neighbor_counts.values())
                if neighbor_counts
                else 0,
                "isolated_cells": sum(1 for c in neighbor_counts.values() if c == 0),
            },
        }
