"""Validated, deterministic operations on one-resolution H3 cell domains.

Topology failures propagate. Extensive values use sum/equal transfer; intensive
values use mean/proportional transfer. These semantics are explicit so callers
can compose resolution changes without inventing or silently losing mass.
"""

from collections import defaultdict
from typing import Any
import math

import h3
import numpy as np

from .state_space import H3StateSpace


class SpatialMethods:
    """Composable H3 set operations, transfers, weights and accessibility."""

    def __init__(self, h3_backend: Any | None = None) -> None:
        if h3_backend is None:
            from ..backends.h3.h3_backend import H3Backend

            h3_backend = H3Backend()
        self.h3 = h3_backend

    @staticmethod
    def _cells(cells, *, unique=True, allow_empty=True) -> tuple[str, ...]:
        if isinstance(cells, (str, bytes)):
            raise TypeError("cells must be a collection of H3 cells")
        cells = list(cells)
        if not cells:
            if allow_empty:
                return ()
            raise ValueError("cells must not be empty")
        ordered = list(cells) if unique else list(dict.fromkeys(cells))
        return H3StateSpace(ordered).cells

    @staticmethod
    def _integer(value: int, name: str, *, minimum: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    @classmethod
    def _resolution(cls, value: int) -> None:
        cls._integer(value, "target_resolution")
        if value > 15:
            raise ValueError("target_resolution must be <= 15")

    @staticmethod
    def _number(value: float, name: str) -> None:
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{name} must be a finite number")
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not finite:
            raise ValueError(f"{name} must be a finite number")

    @classmethod
    def _values(cls, cells, values) -> None:
        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")
        for value in values:
            cls._number(value, "value")

    def buffer_analysis(
        self,
        cells: list[str],
        buffer_rings: int = 1,
        include_center: bool = True,
        *,
        max_cells: int = 1_000_000,
    ) -> dict[str, Any]:
        """Return minimum-distance H3 rings around the union of input cells.

        Bound the worst-case disk allocation before querying topology. Ring zero
        is the input domain; rings 1..k never contain any source or inner ring.
        """
        cells = self._cells(cells, allow_empty=False)
        self._integer(buffer_rings, "buffer_rings")
        self._integer(max_cells, "max_cells", minimum=1)
        if not isinstance(include_center, bool):
            raise ValueError("include_center must be boolean")
        if len(cells) * (1 + 3 * buffer_rings * (buffer_rings + 1)) > max_cells:
            raise ValueError("Buffer disk allocation exceeds max_cells")
        centers, seen = set(cells), set(cells)
        rings = {}
        for ring in range(1, buffer_rings + 1):
            candidates = set()
            for cell in cells:
                candidates.update(self.h3.get_cell_ring(cell, ring))
            rings[ring] = sorted(candidates - seen)
            seen.update(candidates)
        buffered = seen - centers
        result = seen if include_center else buffered
        return {
            "center_cells": sorted(centers),
            "buffer_cells": sorted(buffered),
            "all_cells": sorted(result),
            "rings": rings,
            "buffer_rings": buffer_rings,
            "center_count": len(centers),
            "buffer_count": len(buffered),
            "total_count": len(result),
        }

    def overlay_cells(
        self, cells_a: list[str], cells_b: list[str], operation: str = "intersection"
    ) -> dict[str, Any]:
        """Apply exact set operations to a common-resolution H3 domain."""
        a = self._cells(cells_a, unique=False)
        b = self._cells(cells_b, unique=False)
        self._cells(list(dict.fromkeys((*a, *b))))
        set_a, set_b = set(a), set(b)
        operations = {
            "intersection": set.intersection,
            "union": set.union,
            "difference": set.difference,
            "symmetric_difference": set.symmetric_difference,
        }
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        result = operations[operation](set_a, set_b)
        union = set_a | set_b
        return {
            "operation": operation,
            "input_a_count": len(set_a),
            "input_b_count": len(set_b),
            "result_count": len(result),
            "result_cells": sorted(result),
            "overlap_ratio": len(set_a & set_b) / len(union) if union else 0,
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
        """Filter finite observations while retaining their original cell order."""
        cells = self._cells(cells)
        self._values(cells, values)
        pairs = list(zip(cells, values))
        if filter_type == "threshold" and threshold is not None:
            self._number(threshold, "threshold")
            filtered = [(c, v) for c, v in pairs if v >= threshold]
        elif filter_type == "percentile" and percentile is not None:
            self._number(percentile, "percentile")
            if not 0 <= percentile <= 100:
                raise ValueError("percentile must be between 0 and 100")
            limit = float(np.percentile(values, percentile)) if values else 0
            filtered = [(c, v) for c, v in pairs if v >= limit]
        elif filter_type == "top_n" and top_n is not None:
            self._integer(top_n, "top_n")
            selected = {
                c for c, _ in sorted(pairs, key=lambda p: p[1], reverse=True)[:top_n]
            }
            filtered = [(c, v) for c, v in pairs if c in selected]
        elif filter_type == "outliers":
            q1, q3 = np.percentile(values, [25, 75]) if values else (0, 0)
            width = 1.5 * (q3 - q1)
            filtered = [(c, v) for c, v in pairs if v < q1 - width or v > q3 + width]
        else:
            raise ValueError("Invalid filter configuration")
        return {
            "filter_type": filter_type,
            "input_count": len(cells),
            "filtered_count": len(filtered),
            "filtered_cells": [c for c, _ in filtered],
            "filtered_values": [v for _, v in filtered],
            "filter_ratio": len(filtered) / len(cells) if cells else 0,
        }

    def aggregate_to_region(
        self,
        cells: list[str],
        values: list[float],
        target_resolution: int,
        aggregation: str = "mean",
    ) -> dict[str, Any]:
        """Aggregate observed cells to parents; refinement requests raise."""
        cells = self._cells(cells)
        self._values(cells, values)
        self._resolution(target_resolution)
        reducers = {
            "mean": lambda a: math.fsum(a) / len(a),
            "sum": math.fsum,
            "min": min,
            "max": max,
            "count": len,
        }
        if aggregation not in reducers:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        if cells and target_resolution > h3.get_resolution(cells[0]):
            raise ValueError(
                "Aggregation target must be coarser than or equal to source resolution"
            )
        groups = defaultdict(list)
        for cell, value in zip(cells, values):
            parent = self.h3.get_cell_parent(cell, target_resolution)
            groups[parent].append(value)
        aggregated = {
            parent: {"value": reducers[aggregation](vals), "child_count": len(vals)}
            for parent, vals in groups.items()
        }
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
        *,
        max_cells: int = 1_000_000,
    ) -> dict[str, Any]:
        """Split extensive totals equally, or replicate intensive values.

        ``proportional`` preserves an intensive per-cell value; it is not an
        area-weighted mass transfer. Both modes require complete H3 children.
        """
        parents = self._cells(parent_cells)
        self._values(parents, values)
        self._resolution(target_resolution)
        self._integer(max_cells, "max_cells", minimum=1)
        if method not in {"equal", "proportional"}:
            raise ValueError(f"Unknown disaggregation method: {method}")
        if parents and target_resolution < h3.get_resolution(parents[0]):
            raise ValueError(
                "Disaggregation target must be finer than or equal to parent resolution"
            )
        counts = [h3.cell_to_children_size(cell, target_resolution) for cell in parents]
        if sum(counts) > max_cells:
            raise ValueError("Disaggregation allocation exceeds max_cells")
        disaggregated = {}
        for parent, value, count in zip(parents, values, counts):
            children = self.h3.get_cell_children(
                parent, target_resolution, max_cells=max_cells
            )
            if (
                len(children) != count
                or len(set(children)) != count
                or any(
                    not h3.is_valid_cell(child)
                    or h3.get_resolution(child) != target_resolution
                    or h3.cell_to_parent(child, h3.get_resolution(parent)) != parent
                    for child in children
                )
            ):
                raise ValueError("Backend returned incomplete or invalid H3 children")
            child_value = value / count if method == "equal" else value
            for child in children:
                disaggregated[child] = child_value
        return {
            "input_cells": len(parents),
            "output_cells": len(disaggregated),
            "target_resolution": target_resolution,
            "method": method,
            "disaggregated": disaggregated,
            "expansion_ratio": len(disaggregated) / len(parents) if parents else 0,
        }

    def calculate_coverage(
        self, cells: list[str], region_cells: list[str] | None = None
    ) -> dict[str, Any]:
        """Measure distinct, nonoverlapping cells at a common resolution."""
        cells = self._cells(cells, unique=False)
        region = (
            self._cells(region_cells, unique=False) if region_cells is not None else ()
        )
        self._cells(list(dict.fromkeys((*cells, *region))))
        result = {
            "num_cells": len(cells),
            "total_area_km2": math.fsum(
                self.h3.get_cell_area(cell, "km^2") for cell in cells
            ),
            "resolution_distribution": {h3.get_resolution(cells[0]): len(cells)}
            if cells
            else {},
        }
        if region_cells is not None:
            covered = set(cells) & set(region)
            result.update(
                region_cells=len(region),
                covered_cells=len(covered),
                coverage_ratio=len(covered) / len(region) if region else 0,
            )
        return result

    def find_spatial_outliers(
        self, cells: list[str], values: list[float], k: int = 1
    ) -> dict[str, Any]:
        """Return descriptive standardized local Moran quadrants.

        These are descriptive quadrants; no significance test is performed.
        ``NS`` also includes zero deviations and cells with no observed neighbor.
        """
        cells = self._cells(cells)
        self._values(cells, values)
        self._integer(k, "k", minimum=1)
        mean = float(np.mean(values)) if values else 0
        variance = float(np.var(values)) if values else 0
        observed = dict(zip(cells, values))
        groups = {label: [] for label in ("HH", "LL", "HL", "LH", "NS")}
        for cell, value in observed.items():
            neighbors = self.h3.get_cells_within_radius(cell, k)
            neighbor_values = [
                observed[n] for n in neighbors if n in observed and n != cell
            ]
            if not neighbor_values or variance == 0 or value == mean:
                groups["NS"].append({"cell": cell, "value": value})
                continue
            neighbor_mean = math.fsum(neighbor_values) / len(neighbor_values)
            if neighbor_mean == mean:
                groups["NS"].append({"cell": cell, "value": value})
                continue
            label = ("H" if value > mean else "L") + (
                "H" if neighbor_mean > mean else "L"
            )
            groups[label].append(
                {
                    "cell": cell,
                    "value": value,
                    "neighbor_mean": neighbor_mean,
                    "local_moran": (value - mean) * (neighbor_mean - mean) / variance,
                }
            )
        return {
            "total_cells": len(cells),
            "significance_tested": False,
            "outliers": {
                "HH_clusters": len(groups["HH"]),
                "LL_clusters": len(groups["LL"]),
                "HL_outliers": len(groups["HL"]),
                "LH_outliers": len(groups["LH"]),
                "not_significant": len(groups["NS"]),
            },
            "details": {
                label: observations[:10] for label, observations in groups.items()
            },
            "spatial_outlier_count": len(groups["HL"]) + len(groups["LH"]),
        }

    def compute_accessibility(
        self,
        origin_cells: list[str],
        destination_cells: list[str],
        max_distance: int = 10,
    ) -> dict[str, Any]:
        """Compute reachable fractions using exact H3 grid distances.

        Unsupported H3 distance pairs raise; they are not labeled unreachable.
        """
        origins = self._cells(origin_cells)
        destinations = self._cells(destination_cells)
        self._cells(list(dict.fromkeys((*origins, *destinations))))
        self._integer(max_distance, "max_distance")
        accessibility = {}
        for origin in origins:
            distances = [
                self.h3.get_cell_distance(origin, dest) for dest in destinations
            ]
            reachable = [d for d in distances if d <= max_distance]
            accessibility[origin] = {
                "reachable_destinations": len(reachable),
                "min_distance": min(reachable) if reachable else None,
                "avg_distance": math.fsum(reachable) / len(reachable)
                if reachable
                else None,
                "accessibility_score": len(reachable) / len(destinations)
                if destinations
                else 0,
            }
        scores = [a["accessibility_score"] for a in accessibility.values()]
        return {
            "num_origins": len(origins),
            "num_destinations": len(destinations),
            "max_distance": max_distance,
            "accessibility": accessibility,
            "summary": {
                "mean_accessibility": math.fsum(scores) / len(scores) if scores else 0,
                "max_accessibility": max(scores) if scores else 0,
                "min_accessibility": min(scores) if scores else 0,
                "fully_accessible_origins": sum(score == 1 for score in scores),
            },
        }

    def calculate_spatial_weights(
        self, cells: list[str], weight_type: str = "queen", k: int = 1
    ) -> dict[str, Any]:
        """Return row-standardized disk or inverse-distance weights.

        H3 hexagonal neighbors share edges, so rook and queen agree. Disk
        neighborhoods include all rings up to k and exclude the focal cell.
        """
        cells = self._cells(cells)
        self._integer(k, "k", minimum=1)
        if weight_type not in {"queen", "rook", "distance"}:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        cell_set, weights, counts = set(cells), {}, {}
        for cell in cells:
            if weight_type in {"queen", "rook"}:
                neighbors = set(self.h3.get_cells_within_radius(cell, k)) & cell_set - {
                    cell
                }
                row = {neighbor: 1.0 for neighbor in sorted(neighbors)}
            else:
                row = {}
                for other in cells:
                    if cell == other:
                        continue
                    distance = self.h3.get_cell_distance(cell, other)
                    if distance <= k:
                        row[other] = 1.0 / (distance + 1)
            total = math.fsum(row.values())
            counts[cell] = len(row)
            weights[cell] = (
                {neighbor: weight / total for neighbor, weight in row.items()}
                if total
                else {}
            )
        return {
            "num_cells": len(cells),
            "weight_type": weight_type,
            "k": k,
            "weights": weights,
            "summary": {
                "avg_neighbors": math.fsum(counts.values()) / len(counts)
                if counts
                else 0,
                "max_neighbors": max(counts.values()) if counts else 0,
                "min_neighbors": min(counts.values()) if counts else 0,
                "isolated_cells": sum(count == 0 for count in counts.values()),
            },
        }
