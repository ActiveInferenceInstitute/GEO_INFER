"""
H3 Utilities module for helper functions and optimization tools.

This module provides utility functions for H3 operations including
converters, optimizers, caching, and general helper functions.
"""

import logging
from typing import Any
from collections.abc import Callable
import time

logger = logging.getLogger(__name__)


class H3Utils:
    """
    General utility functions for H3 operations.

    Provides helper methods for common H3 tasks, data validation,
    and format conversions.
    """

    @staticmethod
    def validate_h3_index_format(h3_index: str) -> bool:
        """
        Validate H3 index format (basic check).

        Args:
            h3_index: H3 index string to validate

        Returns:
            True if format appears valid
        """
        if not isinstance(h3_index, str):
            return False  # type: ignore[unreachable]

        # Basic format check - H3 indices are typically 15 characters
        if len(h3_index) != 15:
            return False

        # Should be hexadecimal
        try:
            int(h3_index, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def estimate_resolution_from_area(area_km2: float) -> int:
        """
        Estimate appropriate H3 resolution based on desired area.

        Args:
            area_km2: Desired area in square kilometers

        Returns:
            Estimated H3 resolution (0-15)
        """
        # Approximate areas for H3 resolutions (very rough estimates)
        resolution_areas = {
            0: 4250000,  # ~4.25M km²
            1: 607000,  # ~607K km²
            2: 86700,  # ~86.7K km²
            3: 12400,  # ~12.4K km²
            4: 1770,  # ~1.77K km²
            5: 253,  # ~253 km²
            6: 36.1,  # ~36.1 km²
            7: 5.16,  # ~5.16 km²
            8: 0.737,  # ~0.737 km²
            9: 0.105,  # ~0.105 km²
            10: 0.015,  # ~0.015 km²
            11: 0.002,  # ~0.002 km²
            12: 0.0003,  # ~0.0003 km²
            13: 0.00004,  # ~0.00004 km²
            14: 0.000006,  # ~0.000006 km²
            15: 0.0000009,  # ~0.0000009 km²
        }

        # Find closest resolution
        best_resolution = 0
        min_diff = float("inf")

        for resolution, res_area in resolution_areas.items():
            diff = abs(res_area - area_km2)
            if diff < min_diff:
                min_diff = diff
                best_resolution = resolution

        return best_resolution

    @staticmethod
    def format_area(area_km2: float) -> str:
        """
        Format area value with appropriate units.

        Args:
            area_km2: Area in square kilometers

        Returns:
            Formatted area string
        """
        if area_km2 >= 1000:
            return f"{area_km2 / 1000:.1f}K km²"
        elif area_km2 >= 1:
            return f"{area_km2:.2f} km²"
        elif area_km2 >= 0.01:
            return f"{area_km2 * 100:.1f} hectares"
        else:
            return f"{area_km2 * 1000000:.0f} m²"

    @staticmethod
    def calculate_grid_bounds(cells: list) -> tuple[float, float, float, float]:
        """
        Calculate bounding box for a list of cells.

        Args:
            cells: List of H3Cell objects

        Returns:
            Tuple of (min_lat, min_lng, max_lat, max_lng)
        """
        if not cells:
            return (0.0, 0.0, 0.0, 0.0)

        lats = [cell.latitude for cell in cells]
        lngs = [cell.longitude for cell in cells]

        return (min(lats), min(lngs), max(lats), max(lngs))

    @staticmethod
    def generate_grid_summary(cells: list) -> dict[str, Any]:
        """
        Generate summary statistics for a grid of cells.

        Args:
            cells: List of H3Cell objects

        Returns:
            Dictionary with summary statistics
        """
        if not cells:
            return {"cell_count": 0}

        # Basic statistics
        resolutions = [cell.resolution for cell in cells]
        areas = [cell.area_km2 for cell in cells]

        # Property analysis
        all_properties = set()
        for cell in cells:
            all_properties.update(cell.properties.keys())

        summary: dict[str, Any] = {
            "cell_count": len(cells),
            "resolutions": {
                "unique": len(set(resolutions)),
                "range": (min(resolutions), max(resolutions)),
                "distribution": {},
            },
            "area": {
                "total_km2": sum(areas),
                "mean_km2": sum(areas) / len(areas),
                "range_km2": (min(areas), max(areas)),
            },
            "bounds": H3Utils.calculate_grid_bounds(cells),
            "properties": list(all_properties),
        }

        # Resolution distribution
        for res in resolutions:
            summary["resolutions"]["distribution"][res] = (
                summary["resolutions"]["distribution"].get(res, 0) + 1
            )

        return summary


class H3Converter:
    """
    Conversion utilities for H3 data formats.

    Provides methods for converting between different H3 data formats
    and coordinate systems.
    """

    @staticmethod
    def cells_to_coordinates(cells: list) -> list[tuple[float, float]]:
        """
        Extract coordinates from H3 cells.

        Args:
            cells: List of H3Cell objects

        Returns:
            List of (latitude, longitude) tuples
        """
        return [(cell.latitude, cell.longitude) for cell in cells]

    @staticmethod
    def cells_to_dict(cells: list) -> list[dict[str, Any]]:
        """
        Convert H3 cells to list of dictionaries.

        Args:
            cells: List of H3Cell objects

        Returns:
            List of cell dictionaries
        """
        result = []
        for cell in cells:
            cell_dict = {
                "h3_index": cell.index,
                "resolution": cell.resolution,
                "latitude": cell.latitude,
                "longitude": cell.longitude,
                "area_km2": cell.area_km2,
                "created_at": cell.created_at.isoformat(),
            }
            cell_dict.update(cell.properties)
            result.append(cell_dict)

        return result

    @staticmethod
    def dict_to_geojson(cell_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert cell dictionary to GeoJSON feature.

        Args:
            cell_dict: Cell data dictionary

        Returns:
            GeoJSON feature dictionary
        """
        # Create point geometry (simplified - would need boundary for polygon)
        geometry = {
            "type": "Point",
            "coordinates": [
                cell_dict.get("longitude", 0),
                cell_dict.get("latitude", 0),
            ],
        }

        # Extract properties (exclude geometry-related fields)
        properties = {}
        for key, value in cell_dict.items():
            if key not in ["latitude", "longitude"]:
                properties[key] = value

        return {"type": "Feature", "geometry": geometry, "properties": properties}


class H3Optimizer:
    """
    Optimization utilities for H3 operations.

    Provides methods for optimizing H3 operations including
    spatial indexing, query optimization, and performance tuning.
    """

    def __init__(self) -> None:
        """Initialize optimizer."""
        self.performance_stats: dict[str, dict[str, Any]] = {}

    def time_operation(
        self,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        """
        Time an operation and record performance statistics.

        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result and execution time
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time

        # Record statistics
        if operation_name not in self.performance_stats:
            self.performance_stats[operation_name] = {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
            }

        stats = self.performance_stats[operation_name]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)

        return result, execution_time

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get performance statistics report.

        Returns:
            Dictionary with performance statistics
        """
        report = {}

        for operation, stats in self.performance_stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0

            report[operation] = {
                "executions": stats["count"],
                "total_time_sec": stats["total_time"],
                "average_time_sec": avg_time,
                "min_time_sec": stats["min_time"],
                "max_time_sec": stats["max_time"],
            }

        return report

    def suggest_optimizations(self, cells: list) -> list[str]:
        """
        Suggest optimizations based on grid characteristics.

        Args:
            cells: List of H3Cell objects

        Returns:
            List of optimization suggestions
        """
        suggestions: list[str] = []

        if not cells:
            return suggestions

        # Analyze grid characteristics
        cell_count = len(cells)
        resolutions = set(cell.resolution for cell in cells)

        # Size-based suggestions
        if cell_count > 10000:
            suggestions.append("Consider using spatial indexing for large grids")
            suggestions.append("Use compaction to reduce memory usage")

        if cell_count > 100000:
            suggestions.append("Consider parallel processing for operations")
            suggestions.append("Use streaming processing for very large datasets")

        # Resolution-based suggestions
        if len(resolutions) > 3:
            suggestions.append(
                "Mixed resolutions detected - consider uncompacting for uniform analysis"
            )

        if max(resolutions) > 12:
            suggestions.append("High resolution detected - may impact performance")

        # Property-based suggestions
        property_counts: dict[str, int] = {}
        for cell in cells:
            for prop in cell.properties:
                property_counts[prop] = property_counts.get(prop, 0) + 1

        if property_counts:
            max_props = max(property_counts.values())
            if max_props < cell_count * 0.8:
                suggestions.append(
                    "Sparse properties detected - consider data validation"
                )

        return suggestions


class H3Cache:
    """
    Caching utilities for H3 operations.

    Provides simple in-memory caching for expensive H3 operations
    to improve performance.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize cache with maximum size.

        Args:
            max_size: Maximum number of cached items
        """
        self.cache: dict[str, Any] = {}
        self.access_times: dict[str, float] = {}
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "keys": list(self.cache.keys()),
        }
