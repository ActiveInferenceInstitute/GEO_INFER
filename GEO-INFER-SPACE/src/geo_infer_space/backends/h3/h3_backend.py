"""
H3 Backend Implementation for GEO-INFER-SPACE.

This module provides the H3-specific implementation of spatial operations
that integrates with the generic spatial methods layer. All operations
require the H3 library to be installed - no simulated implementations.
"""

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast
from collections.abc import Callable

from ...core.interfaces import H3UnavailableError

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

MIN_H3_VERSION = (4, 5, 0)
MAX_H3_MAJOR = 5
F = TypeVar("F", bound=Callable[..., Any])


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    """Parse a semantic H3 version without accepting a legacy major release."""
    try:
        parts = version.lstrip("v").split(".")
        return cast(
            tuple[int, int, int],
            tuple(int(part.split("+")[0].split("-")[0]) for part in parts[:3])
            + (0,) * max(0, 3 - len(parts)),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _require_h3(operation: str) -> Callable[[F], F]:
    """Decorator to require H3 library for an operation."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not self._available:
                logger.error(f"H3 library required for {operation}")
                raise H3UnavailableError(operation)
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


class H3Backend:
    """
    H3 backend implementation for spatial operations.

    This class provides H3-specific implementations of the generic spatial
    interfaces. All operations require the H3 library - operations will
    raise H3UnavailableError if H3 is not installed.

    Implements: IndexingBackendProtocol, AnalyticsBackendProtocol
    """

    def __init__(self) -> None:
        """Initialize the H3 backend and check library availability."""
        self._check_h3_availability()

    def _check_h3_availability(self) -> None:
        """Check if H3 library is available."""
        try:
            import h3

            self.h3 = h3
            version = getattr(h3, "__version__", None)
            parsed_version = _version_tuple(cast(str, version))
            if (
                parsed_version is None
                or parsed_version < MIN_H3_VERSION
                or parsed_version[0] >= MAX_H3_MAJOR
            ):
                self._available = False
                logger.error(
                    "Unsupported H3 version %r; GEO-INFER requires h3-py >=4.5.0,<5",
                    version,
                )
                return

            self._available = True
            logger.info("H3 library v%s loaded successfully", version)
        except ImportError:
            self.h3 = None
            self._available = False
            logger.warning("H3 library is not installed - install with: pip install h3")

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "h3"

    @property
    def version(self) -> str:
        """Return the backend version."""
        if self.h3:
            return getattr(self.h3, "__version__", "unknown")
        return "not-installed"

    def is_available(self) -> bool:
        """Check if the backend is available and functional."""
        return self._available

    def get_capabilities(self) -> dict[str, Any]:
        """Return the backend's capabilities."""
        return {
            "indexing": {
                "latlng_to_cell": self._available,
                "cell_to_latlng": self._available,
                "polygon_to_cells": self._available,
                "get_neighbors": self._available,
                "get_cells_within_radius": self._available,
                "get_distance": self._available,
                "compact_cells": self._available,
                "uncompact_cells": self._available,
                "get_cell_parent": self._available,
                "get_cell_children": self._available,
                "get_cell_path": self._available,
                "get_cell_ring": self._available,
                "get_cell_resolution": self._available,
                "get_cell_boundary": self._available,
                "get_cell_area": self._available,
                "cells_to_multipolygon": self._available,
            },
            "analytics": {
                "analyze_hotspots": self._available,
                "compute_proximity": self._available,
                "cluster_points": self._available,
                "interpolate_values": self._available,
            },
            "geometric": {
                "buffer_geometry": self._available,
                "calculate_area": self._available,
                "calculate_perimeter": self._available,
                "calculate_centroid": self._available,
                "union_geometries": self._available,
                "intersection_geometries": self._available,
                "difference_geometries": self._available,
                "calculate_distance": self._available,
                "contains_geometry": self._available,
                "intersects_geometry": self._available,
                "transform_geometry": self._available,
            },
            "supported_resolutions": list(range(16)),  # H3 supports resolutions 0-15
            "coordinate_system": "WGS84",
            "version": self.version,
            "minimum_version": "4.5.0",
            "available": self._available,
        }

    # SpatialIndexingBackend implementation
    @_require_h3("latlng_to_cell")
    def latlng_to_cell(self, lat: float, lng: float, resolution: int) -> str:
        """
        Convert lat/lng coordinates to H3 cell.

        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)
            resolution: H3 resolution (0-15)

        Returns:
            H3 cell identifier string

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If coordinates or resolution are invalid
        """
        logger.debug(f"Converting ({lat}, {lng}) to H3 cell at resolution {resolution}")
        return cast(str, self.h3.latlng_to_cell(lat, lng, resolution))

    @_require_h3("cell_to_latlng")
    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """
        Convert H3 cell back to lat/lng coordinates.

        Args:
            cell: H3 cell identifier

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifier is invalid
        """
        logger.debug(f"Converting H3 cell {cell} to coordinates")
        return cast(tuple[float, float], self.h3.cell_to_latlng(cell))

    @_require_h3("polygon_to_cells")
    def polygon_to_cells(self, polygon: dict[str, Any], resolution: int) -> list[str]:
        """
        Convert polygon to list of H3 cells.

        Args:
            polygon: GeoJSON-like polygon dictionary with 'coordinates' key
            resolution: H3 resolution (0-15)

        Returns:
            List of H3 cell identifiers covering the polygon

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If polygon format is invalid
        """
        if hasattr(polygon, "__geo_interface__"):
            geometry = polygon.__geo_interface__
        elif isinstance(polygon, dict):
            geometry = polygon.get("geometry", polygon)
        else:
            raise ValueError(
                "polygon must be a GeoJSON geometry, Feature, or object "
                "implementing __geo_interface__"
            )

        if not isinstance(geometry, dict):
            raise ValueError("polygon must be a GeoJSON object")

        if geometry.get("type") == "FeatureCollection":
            features = geometry.get("features", [])
            cells: set[str] = set()
            for feature in features:
                if not isinstance(feature, dict) or feature.get("type") != "Feature":
                    raise ValueError("FeatureCollection contains an invalid feature")
                feature_geometry = feature.get("geometry")
                if not feature_geometry:
                    continue
                cells.update(self.h3.geo_to_cells(feature_geometry, resolution))
            return sorted(cells)

        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("polygon must contain a Polygon or MultiPolygon geometry")

        if not geometry.get("coordinates"):
            logger.warning("Empty polygon coordinates provided")
            return []

        try:
            logger.debug(
                "Converting %s geometry to H3 cells at resolution %s",
                geometry["type"],
                resolution,
            )
            # GeoJSON is [longitude, latitude]. geo_to_cells performs the
            # official H3Shape conversion and preserves holes/multipolygons.
            polygon_cells = sorted(self.h3.geo_to_cells(geometry, resolution))
            logger.debug(f"Generated {len(polygon_cells)} H3 cells from polygon")
            return polygon_cells
        except Exception as e:
            raise ValueError(f"H3 polygon conversion failed: {e}") from e

    @_require_h3("get_cell_neighbors")
    def get_cell_neighbors(self, cell: str, k: int = 1) -> list[str]:
        """
        Get neighboring cells around a given cell.

        Args:
            cell: Central H3 cell identifier
            k: Number of rings of neighbors (default 1)

        Returns:
            List of neighboring H3 cell identifiers

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifier is invalid
        """
        logger.debug(f"Getting k={k} neighbors for cell {cell}")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        # grid_disk is defined for pentagons, while grid_ring can raise for
        # them. Removing the inner disk gives the same exact-ring semantics.
        disk = set(self.h3.grid_disk(cell, k))
        inner_disk = set(self.h3.grid_disk(cell, k - 1))
        return sorted(disk - inner_disk)

    @_require_h3("get_cells_within_radius")
    def get_cells_within_radius(self, cell: str, k: int = 1) -> list[str]:
        """Return every cell within ``k`` H3 grid rings, excluding ``cell``."""
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        return sorted(set(self.h3.grid_disk(cell, k)) - {cell})

    @_require_h3("get_cell_distance")
    def get_cell_distance(self, cell1: str, cell2: str) -> int:
        """
        Calculate the grid distance between two H3 cells.

        Args:
            cell1: First H3 cell identifier
            cell2: Second H3 cell identifier

        Returns:
            Grid distance between cells

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cells are at different resolutions or invalid
        """
        logger.debug(f"Calculating distance between {cell1} and {cell2}")
        return cast(int, self.h3.grid_distance(cell1, cell2))

    @_require_h3("compute_distance_matrix")
    def compute_distance_matrix(
        self,
        cells_a: list[str],
        cells_b: list[str],
    ) -> dict[str, Any]:
        """Return host H3 int64 grid distances; incomparable pairs are -1.

        Exact H3 topology always runs on the CPU, and the returned metadata
        reflects that execution.
        """
        from ..gpu.gpu_acceleration import h3_grid_distance_kernel

        matrix = h3_grid_distance_kernel(cells_a, cells_b, h3_module=self.h3)
        return {
            "distance_matrix": matrix,
            "rows": len(cells_a),
            "cols": len(cells_b),
            "shape": list(matrix.shape),
            "accelerator": [],
            "backend": "cpu",
        }

    @_require_h3("geodesic_distance_matrix")
    def geodesic_distance_matrix(
        self,
        cells_a: list[str],
        cells_b: list[str],
        *,
        backend: str = "auto",
        chunk_size: int = 1024,
    ) -> dict[str, Any]:
        """Return float64 centroid great-circle distances in km and diagnostics.

        Invalid cells raise; they are never replaced by an unrelated location.
        backend="cpu" forces CPU. An explicit unavailable GPU backend raises.
        """
        from ..gpu.gpu_acceleration import pairwise_haversine_kernel

        pts_a = [self.h3.cell_to_latlng(cell) for cell in cells_a]
        pts_b = [self.h3.cell_to_latlng(cell) for cell in cells_b]
        diagnostics: dict[str, Any] = {}
        matrix = pairwise_haversine_kernel(
            pts_a,
            pts_b,
            backend=backend,
            chunk_size=chunk_size,
            diagnostics=diagnostics,
        )
        return {
            "distance_matrix": matrix,
            "rows": len(cells_a),
            "cols": len(cells_b),
            "shape": list(matrix.shape),
            "units": "km",
            "accelerator": [
                name for name in diagnostics["used_backends"] if name != "cpu"
            ],
            "backend": diagnostics["backend"],
            "diagnostics": diagnostics,
        }

    @_require_h3("geodesic_spatial_join")
    def geodesic_spatial_join(
        self,
        points_a: list[tuple[float, float]],
        points_b: list[tuple[float, float]],
        max_distance_km: float,
        *,
        backend: str = "auto",
        chunk_size: int = 1024,
    ) -> dict[str, Any]:
        """Join finite geographic points within a positive distance using tiles.

        Intermediate distance storage is bounded by chunk_size squared. Result
        diagnostics identify actual execution, including any automatic fallback.
        """
        from ..gpu.gpu_acceleration import gpu_spatial_join_by_distance

        diagnostics: dict[str, Any] = {}
        pairs, unmatched_a, unmatched_b = gpu_spatial_join_by_distance(
            points_a,
            points_b,
            max_distance_km=max_distance_km,
            backend=backend,
            chunk_size=chunk_size,
            diagnostics=diagnostics,
        )
        return {
            "pairs": pairs,
            "pair_count": len(pairs),
            "unmatched_a": unmatched_a,
            "unmatched_b": unmatched_b,
            "accelerator": [
                name for name in diagnostics["used_backends"] if name != "cpu"
            ],
            "backend": diagnostics["backend"],
            "diagnostics": diagnostics,
        }

    @_require_h3("compact_cells")
    def compact_cells(self, cells: list[str]) -> list[str]:
        """
        Compact a list of cells into a more efficient representation.

        Args:
            cells: List of H3 cell identifiers

        Returns:
            Compacted list of cell identifiers at mixed resolutions

        Raises:
            H3UnavailableError: If H3 library is not installed
        """
        logger.debug(f"Compacting {len(cells)} cells")
        result = list(self.h3.compact_cells(cells))
        logger.debug(f"Compacted to {len(result)} cells")
        return result

    @_require_h3("uncompact_cells")
    def uncompact_cells(self, compacted_cells: list[str], resolution: int) -> list[str]:
        """
        Uncompact cells back to individual cell identifiers.

        Args:
            compacted_cells: Compacted cell identifiers
            resolution: Target resolution level

        Returns:
            List of individual cell identifiers at target resolution

        Raises:
            H3UnavailableError: If H3 library is not installed
        """
        logger.debug(
            f"Uncompacting {len(compacted_cells)} cells to resolution {resolution}"
        )
        result = list(self.h3.uncompact_cells(compacted_cells, resolution))
        logger.debug(f"Uncompacted to {len(result)} cells")
        return result

    @_require_h3("get_cell_parent")
    def get_cell_parent(self, cell: str, resolution: int) -> str:
        """
        Get the parent of a cell at a coarser resolution.

        Args:
            cell: H3 cell identifier
            resolution: Target resolution

        Returns:
            Parent cell identifier

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If resolutions are incompatible
        """
        logger.debug(f"Getting parent of {cell} at resolution {resolution}")
        try:
            return cast(str, self.h3.cell_to_parent(cell, resolution))
        except Exception as e:
            raise ValueError(f"Failed to get parent: {e}") from e

    @_require_h3("get_cell_children")
    def get_cell_children(self, cell: str, resolution: int) -> list[str]:
        """
        Get children of a cell at a finer resolution.

        Args:
            cell: H3 cell identifier
            resolution: Target resolution

        Returns:
            List of child cell identifiers

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If resolutions are incompatible
        """
        logger.debug(f"Getting children of {cell} at resolution {resolution}")
        try:
            return list(self.h3.cell_to_children(cell, resolution))
        except Exception as e:
            raise ValueError(f"Failed to get children: {e}") from e

    @_require_h3("get_cell_path")
    def get_cell_path(self, start_cell: str, end_cell: str) -> list[str]:
        """
        Get the path of cells between two cells.

        Args:
            start_cell: Start cell identifier
            end_cell: End cell identifier

        Returns:
            List of cell identifiers in the path (inclusive)

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cells are invalid or disconnected
        """
        logger.debug(f"Calculating path from {start_cell} to {end_cell}")
        try:
            # H3 v4 API check: verify precise function name if needed
            return list(self.h3.grid_path_cells(start_cell, end_cell))
        except Exception as e:
            raise ValueError(f"Failed to calculate path: {e}") from e

    @_require_h3("get_cell_ring")
    def get_cell_ring(self, cell: str, k: int) -> list[str]:
        """
        Get the ring of cells at distance k.

        Args:
            cell: Center cell identifier
            k: Distance in grid steps

        Returns:
            List of cell identifiers in the ring

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell or k matches invalid
        """
        logger.debug(f"Getting ring k={k} for {cell}")
        try:
            return list(self.h3.grid_ring(cell, k))
        except Exception as e:
            raise ValueError(f"Failed to get ring: {e}") from e

    # SpatialAnalyticsBackend implementation
    @_require_h3("analyze_hotspots")
    def analyze_hotspots(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze spatial hotspots in H3-indexed data.

        Args:
            data: Dictionary with 'cells' (list of H3 cell IDs) and
                  'values' (corresponding numeric values).
                  Optional 'threshold' key for custom threshold.

        Returns:
            Dictionary with:
                - hotspots: List of hotspot dictionaries
                - threshold: Threshold value used
                - total_cells: Number of input cells
                - hotspot_count: Number of identified hotspots

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cells and values have different lengths
        """
        cells = data.get("cells", [])
        values = data.get("values", [])

        if len(cells) != len(values):
            raise ValueError(
                f"Cells ({len(cells)}) and values ({len(values)}) must have the same length"
            )

        logger.info(f"Analyzing hotspots for {len(cells)} cells")

        # Simple hotspot detection based on value thresholds
        hotspots = []
        threshold = data.get("threshold", "median")

        if threshold == "median":
            threshold_value = sorted(values)[len(values) // 2] if values else 0
        else:
            threshold_value = threshold

        for cell, value in zip(cells, values):
            if value > threshold_value:
                hotspots.append(
                    {
                        "cell": cell,
                        "value": value,
                        "intensity": (
                            "high" if value > threshold_value * 1.5 else "medium"
                        ),
                    }
                )

        logger.info(f"Found {len(hotspots)} hotspots (threshold: {threshold_value})")

        return {
            "hotspots": hotspots,
            "threshold": threshold_value,
            "total_cells": len(cells),
            "hotspot_count": len(hotspots),
        }

    @_require_h3("compute_proximity")
    def compute_proximity(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        """
        Compute proximity analysis between points using H3.

        Args:
            points: List of (latitude, longitude) coordinate tuples

        Returns:
            Dictionary with:
                - proximity_pairs: List of pairwise distance information
                - total_points: Number of input points
                - analyzed_pairs: Number of successfully analyzed pairs

        Raises:
            H3UnavailableError: If H3 library is not installed
        """
        logger.info(f"Computing proximity for {len(points)} points")

        # Convert points to H3 cells and compute proximity
        cells = []
        for lat, lng in points:
            cell = self.h3.latlng_to_cell(lat, lng, 9)  # Use resolution 9
            cells.append(cell)

        # Calculate distances between cells
        distances = []
        for i, cell1 in enumerate(cells):
            for j, cell2 in enumerate(cells[i + 1 :], i + 1):
                try:
                    distance = self.h3.grid_distance(cell1, cell2)
                    distances.append(
                        {
                            "from_cell": cell1,
                            "to_cell": cell2,
                            "distance": distance,
                            "from_point": points[i],
                            "to_point": points[j],
                        }
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not compute distance between {cell1} and {cell2}: {e}"
                    )
                    continue

        logger.info(f"Analyzed {len(distances)} proximity pairs")

        return {
            "proximity_pairs": distances,
            "total_points": len(points),
            "analyzed_pairs": len(distances),
        }

    @_require_h3("get_cell_resolution")
    def get_cell_resolution(self, cell: str) -> int:
        """
        Get the resolution level of an H3 cell.

        Args:
            cell: H3 cell identifier

        Returns:
            Resolution level (0-15)

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifier is invalid
        """
        logger.debug(f"Getting resolution for cell: {cell}")

        try:
            resolution = self.h3.get_resolution(cell)
            logger.debug(f"Cell {cell} has resolution {resolution}")
            return cast(int, resolution)
        except Exception as e:
            raise ValueError(f"Invalid H3 cell identifier: {cell}") from e

    @_require_h3("get_cell_boundary")
    def get_cell_boundary(self, cell: str) -> list[tuple[float, float]]:
        """
        Get the boundary coordinates of an H3 cell.

        Args:
            cell: H3 cell identifier

        Returns:
            List of (latitude, longitude) tuples forming the cell boundary

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifier is invalid
        """
        logger.debug(f"Getting boundary for cell: {cell}")

        try:
            # H3 v4 returns boundary as list of LatLngPair
            boundary = self.h3.cell_to_boundary(cell)
            # Convert to list of tuples
            result = [(lat, lng) for lat, lng in boundary]
            logger.debug(f"Cell {cell} has {len(result)} boundary vertices")
            return result
        except Exception as e:
            raise ValueError(f"Invalid H3 cell identifier: {cell}") from e

    @_require_h3("get_cell_area")
    def get_cell_area(self, cell: str, unit: str = "km^2") -> float:
        """
        Get the area of an H3 cell in square kilometers.

        Args:
            cell: H3 cell identifier
            unit: Unit for area calculation (default 'km^2')

        Returns:
            Area in specified unit

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifier is invalid
        """
        logger.debug(f"Getting area for cell: {cell} in {unit}")

        try:
            # H3 v4 cell_area returns area in km² by default
            area = self.h3.cell_area(cell, unit=unit)
            logger.debug(f"Cell {cell} has area {area:.6f} {unit}")
            return cast(float, area)
        except Exception as e:
            raise ValueError(f"Invalid H3 cell identifier: {cell}") from e

    @_require_h3("cells_to_multipolygon")
    def cells_to_multipolygon(self, cells: list[str]) -> dict[str, Any]:
        """
        Convert a list of H3 cells to a GeoJSON MultiPolygon geometry.

        Args:
            cells: List of H3 cell identifiers

        Returns:
            GeoJSON-like dictionary with 'type' and 'coordinates'

        Raises:
            H3UnavailableError: If H3 library is not installed
            ValueError: If cell identifiers are invalid
        """
        logger.info(f"Converting {len(cells)} cells to MultiPolygon")

        if not cells:
            return {"type": "MultiPolygon", "coordinates": []}

        try:
            # Use H3's native union/topology conversion. Building one polygon
            # per cell leaves internal edges in the result and is not a true
            # coverage geometry for adjacent cells.
            geometry = self.h3.cells_to_geo(set(cells))
            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates")
            if geometry_type == "Polygon":
                polygons = [[list(ring) for ring in coordinates]]
            elif geometry_type == "MultiPolygon":
                polygons = [[list(ring) for ring in polygon] for polygon in coordinates]
            else:
                raise ValueError(
                    f"H3 cells_to_geo returned unsupported geometry type: {geometry_type!r}"
                )

            logger.info("Created MultiPolygon with %s polygon parts", len(polygons))
            return {"type": "MultiPolygon", "coordinates": polygons}
        except Exception as e:
            raise ValueError(f"Invalid H3 cell identifiers: {e}") from e

    def _geometry(self, geometry: Any) -> "BaseGeometry":
        """Convert GeoJSON, Feature, or ``__geo_interface__`` to Shapely."""
        from shapely.geometry import shape

        if hasattr(geometry, "__geo_interface__"):
            geometry = geometry.__geo_interface__
        if not isinstance(geometry, dict):
            raise TypeError(
                "geometry must be GeoJSON-like or implement __geo_interface__"
            )
        if geometry.get("type") == "Feature":
            geometry = geometry.get("geometry")
        if not geometry:
            raise ValueError("geometry is empty")
        return shape(geometry)

    @staticmethod
    def _geojson(geometry: "BaseGeometry") -> dict[str, Any]:
        from shapely.geometry import mapping

        return dict(mapping(geometry))

    @_require_h3("buffer_geometry")
    def buffer_geometry(
        self, geometry: dict[str, Any], distance: float, **kwargs: Any
    ) -> dict[str, Any]:
        """Buffer a GeoJSON geometry in its coordinate units."""
        if distance <= 0:
            raise ValueError("distance must be positive")
        return self._geojson(self._geometry(geometry).buffer(distance, **kwargs))

    @_require_h3("calculate_area")
    def calculate_area(self, geometry: dict[str, Any]) -> float:
        """Return the planar area of a GeoJSON geometry."""
        return float(self._geometry(geometry).area)

    @_require_h3("calculate_perimeter")
    def calculate_perimeter(self, geometry: dict[str, Any]) -> float:
        """Return the planar perimeter/length of a GeoJSON geometry."""
        return float(self._geometry(geometry).length)

    @_require_h3("calculate_centroid")
    def calculate_centroid(self, geometry: dict[str, Any]) -> tuple[float, float]:
        """Return the centroid as a ``(latitude, longitude)`` pair."""
        centroid = self._geometry(geometry).centroid
        return float(centroid.y), float(centroid.x)

    @_require_h3("calculate_distance")
    def calculate_distance(
        self, geometry_a: dict[str, Any], geometry_b: dict[str, Any]
    ) -> float:
        """Return the planar distance between two GeoJSON geometries."""
        return float(self._geometry(geometry_a).distance(self._geometry(geometry_b)))

    @_require_h3("union_geometries")
    def union_geometries(self, geometries: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the unary union of GeoJSON geometries."""
        from shapely.ops import unary_union

        if not geometries:
            raise ValueError("geometries must not be empty")
        return self._geojson(unary_union([self._geometry(g) for g in geometries]))

    @_require_h3("intersection_geometries")
    def intersection_geometries(
        self, geometry_a: dict[str, Any], geometry_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the intersection of two GeoJSON geometries."""
        return self._geojson(
            self._geometry(geometry_a).intersection(self._geometry(geometry_b))
        )

    @_require_h3("difference_geometries")
    def difference_geometries(
        self, geometry_a: dict[str, Any], geometry_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Return ``geometry_a - geometry_b`` as GeoJSON."""
        return self._geojson(
            self._geometry(geometry_a).difference(self._geometry(geometry_b))
        )

    @_require_h3("contains_geometry")
    def contains_geometry(
        self, container: dict[str, Any], contained: dict[str, Any]
    ) -> bool:
        """Return whether one GeoJSON geometry contains another."""
        return bool(self._geometry(container).contains(self._geometry(contained)))

    @_require_h3("intersects_geometry")
    def intersects_geometry(
        self, geometry_a: dict[str, Any], geometry_b: dict[str, Any]
    ) -> bool:
        """Return whether two GeoJSON geometries intersect."""
        return bool(self._geometry(geometry_a).intersects(self._geometry(geometry_b)))

    @_require_h3("transform_geometry")
    def transform_geometry(
        self, geometry: dict[str, Any], from_crs: str, to_crs: str
    ) -> dict[str, Any]:
        """Transform a GeoJSON geometry between CRS definitions."""
        from pyproj import Transformer
        from shapely.ops import transform

        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        return self._geojson(transform(transformer.transform, self._geometry(geometry)))

    @_require_h3("find_clusters")
    def find_clusters(
        self,
        cells: list[str],
        values: list[float],
        min_cluster_size: int = 3,
        distance_threshold: int = 1,
    ) -> dict[str, Any]:
        """
        Find spatial clusters of cells based on values and proximity.

        Uses a grid-based DBSCAN-like algorithm that groups cells that are
        within distance_threshold grid steps of each other.

        Args:
            cells: List of H3 cell identifiers
            values: Corresponding values for each cell
            min_cluster_size: Minimum number of cells to form a cluster
            distance_threshold: Maximum grid distance between cluster members

        Returns:
            Dictionary with clusters, statistics, and noise cells
        """
        if len(cells) != len(values):
            raise ValueError(
                f"Cells ({len(cells)}) and values ({len(values)}) must have the same length"
            )

        logger.info(
            f"Finding clusters in {len(cells)} cells with min_size={min_cluster_size}"
        )

        # Create cell-value mapping
        cell_values = dict(zip(cells, values))
        cell_set = set(cells)
        visited = set()
        clusters: list[dict[str, Any]] = []
        noise = []

        def get_neighbors_in_set(cell: str) -> list[str]:
            """Get neighbors of a cell that are in our cell set."""
            try:
                neighbors = list(self.h3.grid_disk(cell, distance_threshold))
                return [n for n in neighbors if n in cell_set and n != cell]
            except (ValueError, self.h3.H3BaseException) as e:
                # h3-py 4.x raises ValueError for malformed indices and
                # H3BaseException subclasses for library errors (no CellError).
                logger.warning(
                    f"grid_disk failed for cell {cell}; cluster expansion may "
                    f"degrade around it: {e}"
                )
                return []

        def expand_cluster(cell: str, neighbors: list[str], cluster: list[str]) -> None:
            """Expand cluster from seed cell."""
            cluster.append(cell)
            i = 0
            while i < len(neighbors):
                neighbor = neighbors[i]
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_neighbors = get_neighbors_in_set(neighbor)
                    if len(new_neighbors) >= min_cluster_size - 1:
                        neighbors.extend(
                            [n for n in new_neighbors if n not in neighbors]
                        )
                if neighbor not in cluster:
                    cluster.append(neighbor)
                i += 1

        # Main clustering loop
        for cell in cells:
            if cell in visited:
                continue
            visited.add(cell)

            neighbors = get_neighbors_in_set(cell)
            if len(neighbors) < min_cluster_size - 1:
                noise.append(cell)
            else:
                cluster: list[str] = []
                expand_cluster(cell, neighbors, cluster)
                if len(cluster) >= min_cluster_size:
                    cluster_values = [cell_values[c] for c in cluster]
                    clusters.append(
                        {
                            "id": len(clusters),
                            "cells": cluster,
                            "size": len(cluster),
                            "mean_value": sum(cluster_values) / len(cluster_values),
                            "min_value": min(cluster_values),
                            "max_value": max(cluster_values),
                        }
                    )
                else:
                    noise.extend(cluster)

        logger.info(f"Found {len(clusters)} clusters, {len(noise)} noise cells")

        return {
            "clusters": clusters,
            "num_clusters": len(clusters),
            "noise_cells": noise,
            "noise_count": len(noise),
            "total_cells": len(cells),
            "parameters": {
                "min_cluster_size": min_cluster_size,
                "distance_threshold": distance_threshold,
            },
        }

    @_require_h3("cluster_points")
    def cluster_points(
        self,
        points: list[tuple[float, float]],
        values: list[float] | None = None,
        method: str = "dbscan",
        resolution: int = 9,
        min_cluster_size: int = 3,
        distance_threshold: int = 1,
    ) -> dict[str, Any]:
        """Cluster coordinate points after indexing them with native H3."""
        if method != "dbscan":
            raise ValueError("H3 point clustering supports method='dbscan' only")
        if not points:
            return self.find_clusters([], [], min_cluster_size, distance_threshold)
        if values is not None and len(values) != len(points):
            raise ValueError("values must have the same length as points")
        values = values or [1.0] * len(points)

        grouped: dict[str, list[float]] = {}
        for (lat, lng), value in zip(points, values):
            cell = self.h3.latlng_to_cell(float(lat), float(lng), resolution)
            grouped.setdefault(cell, []).append(float(value))
        cells = sorted(grouped)
        cell_values = [sum(grouped[cell]) / len(grouped[cell]) for cell in cells]
        return self.find_clusters(
            cells,
            cell_values,
            min_cluster_size=min_cluster_size,
            distance_threshold=distance_threshold,
        )

    @_require_h3("calculate_density")
    def calculate_density(
        self, cells: list[str], values: list[float], kernel_radius: int = 1
    ) -> dict[str, Any]:
        """
        Calculate density values across cells using kernel smoothing.

        Uses inverse distance weighting within the kernel radius.

        Args:
            cells: List of H3 cell identifiers
            values: Values at each cell location
            kernel_radius: Radius for kernel density estimation in grid steps

        Returns:
            Dictionary with density values and statistics
        """
        if len(cells) != len(values):
            raise ValueError(
                f"Cells ({len(cells)}) and values ({len(values)}) must have the same length"
            )

        logger.info(
            f"Calculating density for {len(cells)} cells with kernel_radius={kernel_radius}"
        )

        cell_values = dict(zip(cells, values))
        cell_set = set(cells)
        densities = {}

        for cell in cells:
            # Get neighbors within kernel radius
            try:
                neighborhood = list(self.h3.grid_disk(cell, kernel_radius))
            except Exception:
                neighborhood = [cell]

            # Calculate weighted average (IDW with distance weights)
            total_weight = 0.0
            weighted_sum = 0.0

            for neighbor in neighborhood:
                if neighbor in cell_set:
                    try:
                        distance = self.h3.grid_distance(cell, neighbor)
                        # Inverse distance weight (avoid div by zero)
                        weight = 1.0 / (distance + 1)
                        weighted_sum += cell_values[neighbor] * weight
                        total_weight += weight
                    except Exception:
                        continue

            densities[cell] = weighted_sum / total_weight if total_weight > 0 else 0.0

        density_values = list(densities.values())

        return {
            "densities": densities,
            "statistics": {
                "mean": (
                    sum(density_values) / len(density_values) if density_values else 0.0
                ),
                "max": max(density_values) if density_values else 0.0,
                "min": min(density_values) if density_values else 0.0,
            },
            "kernel_radius": kernel_radius,
            "cells_processed": len(cells),
        }

    @_require_h3("spatial_join")
    def spatial_join(
        self,
        cells_a: list[str],
        cells_b: list[str],
        join_type: str = "intersects",
    ) -> dict[str, Any]:
        """Join exact H3 host topology with deterministic input ordering.

        Intersects means identical or adjacent cells; contains/within mean
        strict H3 ancestry. These host H3 operations always run on the CPU.
        Invalid cells raise.
        """
        from ..gpu.gpu_acceleration import spatial_join_kernel

        matches, unmatched_a, unmatched_b = spatial_join_kernel(
            cells_a,
            cells_b,
            join_type=join_type,
            h3_module=self.h3,
        )
        return {
            "matches": matches,
            "match_count": len(matches),
            "unmatched_a": unmatched_a,
            "unmatched_b": unmatched_b,
            "join_type": join_type,
            "accelerator": [],
            "backend": "cpu",
        }

    @_require_h3("interpolate_values")
    def interpolate_values(
        self,
        cells: list[str],
        values: list[float],
        target_cells: list[str],
        method: str = "idw",
    ) -> dict[str, Any]:
        """
        Interpolate values at target cell locations using source cells.

        Args:
            cells: List of cells with known values
            values: Known values at each cell location
            target_cells: Cells where values should be interpolated
            method: Interpolation method ('idw', 'nearest', 'linear')

        Returns:
            Dictionary with interpolated values and metadata
        """
        if len(cells) != len(values):
            raise ValueError(
                f"Cells ({len(cells)}) and values ({len(values)}) must have the same length"
            )

        valid_methods = {"idw", "nearest", "linear"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Must be one of {valid_methods}"
            )

        logger.info(
            f"Interpolating {len(target_cells)} target cells from {len(cells)} source cells using {method}"
        )

        cell_values = dict(zip(cells, values))
        interpolated = {}

        for target in target_cells:
            if target in cell_values:
                # Target cell has a known value
                interpolated[target] = cell_values[target]
                continue

            if method == "nearest":
                # Find nearest source cell
                min_distance = float("inf")
                nearest_value = 0.0
                for source, value in cell_values.items():
                    try:
                        dist = self.h3.grid_distance(target, source)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_value = value
                    except Exception:
                        continue
                interpolated[target] = nearest_value

            elif method == "idw" or method == "linear":
                # Inverse distance weighting
                total_weight = 0.0
                weighted_sum = 0.0
                power = 2.0 if method == "idw" else 1.0

                for source, value in cell_values.items():
                    try:
                        dist = self.h3.grid_distance(target, source)
                        if dist == 0:
                            weighted_sum = value
                            total_weight = 1.0
                            break
                        weight = 1.0 / (dist**power)
                        weighted_sum += value * weight
                        total_weight += weight
                    except Exception:
                        continue

                interpolated[target] = (
                    weighted_sum / total_weight if total_weight > 0 else 0.0
                )

        interpolated_values = list(interpolated.values())

        return {
            "interpolated": interpolated,
            "method": method,
            "source_count": len(cells),
            "target_count": len(target_cells),
            "statistics": {
                "mean": (
                    sum(interpolated_values) / len(interpolated_values)
                    if interpolated_values
                    else 0.0
                ),
                "min": min(interpolated_values) if interpolated_values else 0.0,
                "max": max(interpolated_values) if interpolated_values else 0.0,
            },
        }

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    @_require_h3("is_valid_cell")
    def is_valid_cell(self, cell: str) -> bool:
        """
        Check if an H3 cell identifier is valid.

        Args:
            cell: H3 cell identifier to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            return bool(self.h3.is_valid_cell(cell))
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug("Unexpected error validating H3 cell %r", cell, exc_info=True)
            return False

    def validate_resolution(self, resolution: int) -> dict[str, Any]:
        """
        Validate that a resolution is within the valid H3 range.

        Args:
            resolution: Resolution value to validate (should be 0-15)

        Returns:
            Dictionary with validation result and details
        """
        is_valid = isinstance(resolution, int) and 0 <= resolution <= 15
        return {
            "valid": is_valid,
            "resolution": resolution,
            "min_resolution": 0,
            "max_resolution": 15,
            "error": (
                None
                if is_valid
                else f"Resolution must be integer 0-15, got {resolution}"
            ),
        }

    def validate_coordinates(self, lat: float, lng: float) -> dict[str, Any]:
        """
        Validate lat/lng coordinates are within valid ranges.

        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)

        Returns:
            Dictionary with validation result and details
        """
        lat_valid = isinstance(lat, (int, float)) and -90 <= lat <= 90
        lng_valid = isinstance(lng, (int, float)) and -180 <= lng <= 180

        errors = []
        if not lat_valid:
            errors.append(f"Latitude must be -90 to 90, got {lat}")
        if not lng_valid:
            errors.append(f"Longitude must be -180 to 180, got {lng}")

        return {
            "valid": lat_valid and lng_valid,
            "lat": lat,
            "lng": lng,
            "lat_valid": lat_valid,
            "lng_valid": lng_valid,
            "errors": errors if errors else None,
        }

    @_require_h3("are_neighbors")
    def are_neighbors(self, cell1: str, cell2: str) -> bool:
        """
        Check if two H3 cells are neighbors (adjacent).

        Args:
            cell1: First H3 cell identifier
            cell2: Second H3 cell identifier

        Returns:
            True if cells are neighbors, False otherwise
        """
        try:
            return bool(self.h3.are_neighbor_cells(cell1, cell2))
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug(
                "Unexpected error checking H3 neighborship for %r, %r",
                cell1,
                cell2,
                exc_info=True,
            )
            return False

    @_require_h3("is_pentagon")
    def is_pentagon(self, cell: str) -> bool:
        """
        Check if an H3 cell is a pentagon (12 per resolution).

        Args:
            cell: H3 cell identifier

        Returns:
            True if cell is a pentagon, False if hexagon
        """
        try:
            return bool(self.h3.is_pentagon(cell))
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug(
                "Unexpected error checking pentagon status for %r", cell, exc_info=True
            )
            return False

    @_require_h3("is_res_class_iii")
    def is_res_class_iii(self, cell: str) -> bool:
        """
        Check if cell is Class III resolution (aperture 7 rotation).

        Class III cells have a 60° rotation relative to their parent.
        Odd resolutions (1, 3, 5, ...) are Class III.

        Args:
            cell: H3 cell identifier

        Returns:
            True if Class III, False if Class II
        """
        try:
            res = self.h3.get_resolution(cell)
            return bool(res % 2 == 1)
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug(
                "Unexpected error checking resolution class for %r", cell, exc_info=True
            )
            return False

    @_require_h3("get_base_cell")
    def get_base_cell(self, cell: str) -> int:
        """
        Get the base cell number (0-121) for any H3 cell.

        Base cells are the 122 resolution-0 cells that tile the icosahedron.

        Args:
            cell: H3 cell identifier at any resolution

        Returns:
            Base cell number (0-121)
        """
        return cast(int, self.h3.get_base_cell_number(cell))

    @_require_h3("get_icosahedron_faces")
    def get_icosahedron_faces(self, cell: str) -> list[int]:
        """
        Get the icosahedron faces a cell intersects.

        Most cells touch 1 face, but cells near vertices touch multiple.

        Args:
            cell: H3 cell identifier

        Returns:
            List of face numbers (0-19)
        """
        return list(self.h3.get_icosahedron_faces(cell))

    @_require_h3("get_pentagons")
    def get_pentagons(self, resolution: int) -> list[str]:
        """
        Get all 12 pentagon cells at a given resolution.

        Args:
            resolution: H3 resolution (0-15)

        Returns:
            List of 12 pentagon cell identifiers
        """
        res_check = self.validate_resolution(resolution)
        if not res_check["valid"]:
            raise ValueError(res_check["error"])
        return list(self.h3.get_pentagons(resolution))

    @_require_h3("get_cells_at_resolution")
    def get_cells_at_resolution(
        self, cells: list[str], target_resolution: int
    ) -> list[str]:
        """
        Convert a mixed-resolution set of cells to a uniform resolution.

        For cells at higher resolution, gets parent.
        For cells at lower resolution, gets children.

        Args:
            cells: List of H3 cells at any resolutions
            target_resolution: Desired uniform resolution

        Returns:
            List of cells all at target_resolution
        """
        res_check = self.validate_resolution(target_resolution)
        if not res_check["valid"]:
            raise ValueError(res_check["error"])

        result = set()
        for cell in cells:
            cell_res = self.h3.get_resolution(cell)
            if cell_res == target_resolution:
                result.add(cell)
            elif cell_res > target_resolution:
                # Get parent
                result.add(self.h3.cell_to_parent(cell, target_resolution))
            else:
                # Get children
                result.update(self.h3.cell_to_children(cell, target_resolution))
        return list(result)

    # =========================================================================
    # DIRECTED EDGE METHODS
    # =========================================================================

    @_require_h3("get_directed_edge")
    def get_directed_edge(self, origin: str, destination: str) -> str:
        """
        Get the directed edge from origin to destination cell.

        Args:
            origin: Origin H3 cell
            destination: Destination H3 cell (must be neighbor)

        Returns:
            Directed edge identifier

        Raises:
            ValueError: If cells are not neighbors
        """
        if not self.are_neighbors(origin, destination):
            raise ValueError(f"Cells {origin} and {destination} are not neighbors")
        return cast(str, self.h3.cells_to_directed_edge(origin, destination))

    @_require_h3("edge_to_cells")
    def edge_to_cells(self, edge: str) -> tuple[str, str]:
        """
        Get the origin and destination cells of a directed edge.

        Args:
            edge: Directed edge identifier

        Returns:
            Tuple of (origin_cell, destination_cell)
        """
        origin = self.h3.get_directed_edge_origin(edge)
        destination = self.h3.get_directed_edge_destination(edge)
        return (origin, destination)

    @_require_h3("get_cell_edges")
    def get_cell_edges(self, cell: str) -> list[str]:
        """
        Get all directed edges originating from a cell.

        Hexagons have 6 edges, pentagons have 5.

        Args:
            cell: H3 cell identifier

        Returns:
            List of directed edge identifiers
        """
        return list(self.h3.origin_to_directed_edges(cell))

    @_require_h3("get_edge_boundary")
    def get_edge_boundary(self, edge: str) -> list[tuple[float, float]]:
        """
        Get the geographic boundary of a directed edge.

        Args:
            edge: Directed edge identifier

        Returns:
            List of (lat, lng) tuples defining the edge
        """
        return list(self.h3.directed_edge_to_boundary(edge))

    # =========================================================================
    # LOCAL IJ COORDINATE METHODS
    # =========================================================================

    @_require_h3("cell_to_local_ij")
    def cell_to_local_ij(self, origin: str, cell: str) -> tuple[int, int]:
        """
        Get the local IJ coordinates of a cell relative to an origin.

        IJ coordinates are a local 2D coordinate system anchored at origin.

        Args:
            origin: Origin cell for the coordinate system
            cell: Cell to get coordinates for

        Returns:
            Tuple of (i, j) coordinates
        """
        ij = self.h3.cell_to_local_ij(origin, cell)
        return (ij[0], ij[1])

    @_require_h3("local_ij_to_cell")
    def local_ij_to_cell(self, origin: str, i: int, j: int) -> str:
        """
        Convert local IJ coordinates back to a cell identifier.

        Args:
            origin: Origin cell for the coordinate system
            i: I coordinate
            j: J coordinate

        Returns:
            H3 cell identifier
        """
        return cast(str, self.h3.local_ij_to_cell(origin, i, j))

    # =========================================================================
    # GEOMETRIC CALCULATION METHODS
    # =========================================================================

    @_require_h3("great_circle_distance")
    def great_circle_distance(
        self, lat1: float, lng1: float, lat2: float, lng2: float, unit: str = "m"
    ) -> float:
        """
        Calculate great circle distance between two points.

        Args:
            lat1, lng1: First point coordinates
            lat2, lng2: Second point coordinates
            unit: Distance unit ('m', 'km', 'rads')

        Returns:
            Distance in specified unit
        """
        import math

        # Convert to radians
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        if unit == "rads":
            return c

        # Earth radius in meters
        r = 6371000
        distance_m = r * c

        if unit == "km":
            return distance_m / 1000
        return distance_m

    @_require_h3("cell_to_geodesic_area")
    def cell_to_geodesic_area(self, cell: str, unit: str = "km^2") -> float:
        """
        Get the geodesic (accurate) area of an H3 cell.

        More accurate than get_cell_area for cells near poles.

        Args:
            cell: H3 cell identifier
            unit: Area unit ('m^2', 'km^2')

        Returns:
            Area in specified unit
        """
        area_rads = cast(float, self.h3.cell_area(cell, unit="rads^2"))

        # Earth radius squared (meters)
        r2 = 6371000**2
        area_m2 = area_rads * r2

        if unit == "km^2":
            return area_m2 / 1_000_000
        return area_m2

    @_require_h3("average_edge_length")
    def average_edge_length(self, resolution: int, unit: str = "m") -> float:
        """
        Get the average edge length for cells at a resolution.

        Args:
            resolution: H3 resolution (0-15)
            unit: Length unit ('m', 'km')

        Returns:
            Average edge length in specified unit
        """
        res_check = self.validate_resolution(resolution)
        if not res_check["valid"]:
            raise ValueError(res_check["error"])

        length = cast(
            float,
            self.h3.average_hexagon_edge_length(resolution, unit=unit),
        )
        return length

    @_require_h3("line_to_cells")
    def line_to_cells(
        self,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        resolution: int,
    ) -> list[str]:
        """
        Convert a line segment to H3 cells it passes through.

        Args:
            start_lat, start_lng: Start point coordinates
            end_lat, end_lng: End point coordinates
            resolution: H3 resolution

        Returns:
            List of H3 cells along the line
        """
        res_check = self.validate_resolution(resolution)
        if not res_check["valid"]:
            raise ValueError(res_check["error"])

        start_cell = self.h3.latlng_to_cell(start_lat, start_lng, resolution)
        end_cell = self.h3.latlng_to_cell(end_lat, end_lng, resolution)

        try:
            return list(self.h3.grid_path_cells(start_cell, end_cell))
        except (ValueError, self.h3.H3BaseException) as e:
            logger.warning(
                f"grid_path_cells failed for {start_cell} -> {end_cell}: {e}"
            )
            # Keep endpoint cells when H3 cannot construct a continuous path.
            return [start_cell, end_cell]

    @_require_h3("point_distance_to_cell_center")
    def point_distance_to_cell_center(self, lat: float, lng: float, cell: str) -> float:
        """
        Calculate distance from a point to a cell's center.

        Args:
            lat, lng: Point coordinates
            cell: H3 cell identifier

        Returns:
            Distance in meters
        """
        center = self.h3.cell_to_latlng(cell)
        return self.great_circle_distance(lat, lng, center[0], center[1], unit="m")

    @_require_h3("get_resolution_stats")
    def get_resolution_stats(self, resolution: int) -> dict[str, Any]:
        """
        Get statistics about a given H3 resolution level.

        Args:
            resolution: H3 resolution (0-15)

        Returns:
            Dictionary with resolution statistics
        """
        res_check = self.validate_resolution(resolution)
        if not res_check["valid"]:
            raise ValueError(res_check["error"])

        return {
            "resolution": resolution,
            "num_hexagons": self.h3.get_num_cells(resolution) - 12,
            "num_pentagons": 12,
            "total_cells": self.h3.get_num_cells(resolution),
            "average_area_km2": self.h3.average_hexagon_area(resolution, unit="km^2"),
            "average_edge_length_km": self.h3.average_hexagon_edge_length(
                resolution, unit="km"
            ),
            "class": "III" if resolution % 2 == 1 else "II",
        }

    # =========================================================================
    # COMPREHENSIVE VALIDATION
    # =========================================================================

    def validate_cell_set(self, cells: list[str]) -> dict[str, Any]:
        """
        Validate a set of H3 cells comprehensively.

        Args:
            cells: List of H3 cell identifiers

        Returns:
            Comprehensive validation report
        """
        if not self._available:
            return {"error": "H3 library not available"}

        valid_cells = []
        invalid_cells = []
        resolutions = set()
        pentagons = []

        for cell in cells:
            if self.is_valid_cell(cell):
                valid_cells.append(cell)
                resolutions.add(self.h3.get_resolution(cell))
                if self.is_pentagon(cell):
                    pentagons.append(cell)
            else:
                invalid_cells.append(cell)

        return {
            "total_cells": len(cells),
            "valid_count": len(valid_cells),
            "invalid_count": len(invalid_cells),
            "invalid_cells": invalid_cells[:10],  # Limit output
            "resolutions_present": sorted(list(resolutions)),
            "is_uniform_resolution": len(resolutions) <= 1,
            "pentagon_count": len(pentagons),
            "pentagons": pentagons,
            "all_valid": len(invalid_cells) == 0,
        }
