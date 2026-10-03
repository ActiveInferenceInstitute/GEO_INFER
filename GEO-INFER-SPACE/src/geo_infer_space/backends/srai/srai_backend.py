"""
SRAI Backend Implementation for GEO-INFER-SPACE.

This module provides the SRAI-specific implementation of spatial operations
that integrates with the generic spatial methods layer.

SRAI (Spatial Representation for Artificial Intelligence) is a geospatial AI library
that supports multiple spatial indexing systems including H3, S2, administrative boundaries, etc.

All operations require the SRAI library - operations will raise SRAIUnavailableError
if SRAI is not installed.
"""

import logging
from functools import wraps
from typing import Any, TypeVar, cast
from collections.abc import Callable

from ...core.interfaces import (
    SRAIUnavailableError,
)

logger = logging.getLogger(__name__)


# Check SRAI availability once at module load
try:
    import srai  # type: ignore[import-not-found]

    SRAI_AVAILABLE = True
    SRAI_VERSION = getattr(srai, "__version__", "unknown")
    logger.info(f"SRAI library v{SRAI_VERSION} is available")
except ModuleNotFoundError as exc:
    if exc.name != "srai":
        raise
    SRAI_AVAILABLE = False
    SRAI_VERSION = None
    srai = None
    logger.warning("SRAI library is not available")


F = TypeVar("F", bound=Callable[..., Any])


def _require_srai(operation: str) -> Callable[[F], F]:
    """Decorator to require SRAI library for an operation."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not SRAI_AVAILABLE:
                raise SRAIUnavailableError(operation)
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


class SraiBackend:
    """
    SRAI backend implementation for spatial operations.

    This class provides SRAI-specific implementations of the generic spatial
    interfaces defined in the core dispatcher module. All operations require
    the SRAI library - operations will raise SRAIUnavailableError if not installed.

    SRAI supports multiple spatial indexing systems:
    - H3 hexagonal hierarchical geospatial indexing system
    - S2 spherical geometry library
    - Administrative boundary-based regionalization
    - Slippy map tiles (Web Mercator)
    - Voronoi-based regionalization

    Implements: IndexingBackendProtocol, AnalyticsBackendProtocol
    """

    def __init__(self, default_regionalizer: str = "h3") -> None:
        """
        Initialize the SRAI backend.

        Args:
            default_regionalizer: Default regionalizer to use ('h3', 's2', etc.)
        """
        self.default_regionalizer = default_regionalizer
        self._available = SRAI_AVAILABLE
        self.srai = srai if SRAI_AVAILABLE else None

        if SRAI_AVAILABLE:
            logger.info(
                f"SRAI backend initialized with {default_regionalizer} regionalizer"
            )
        else:
            logger.warning("SRAI backend initialized but library is not available")

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "srai"

    @property
    def version(self) -> str:
        """Return the backend version."""
        return SRAI_VERSION if SRAI_VERSION else "0.0.0"

    def is_available(self) -> bool:
        """Check if the backend is available and functional."""
        return self._available

    def get_capabilities(self) -> dict[str, Any]:
        """Return the backend's capabilities."""
        logger.debug("Getting SRAI backend capabilities")

        capabilities: dict[str, Any] = {
            "indexing": {
                "latlng_to_cell": self._available,
                "cell_to_latlng": self._available,
                "polygon_to_cells": self._available,
                "get_neighbors": self._available,
                "get_cells_within_radius": self._available,
                "get_distance": self._available,
                "get_resolution": self._available,
                "get_boundary": self._available,
                "get_area": self._available,
                "cells_to_multipolygon": self._available,
            },
            "analytics": {
                "analyze_hotspots": self._available,
                "compute_proximity": self._available,
                "cluster_points": False,
                "interpolate_values": False,
            },
            "regionalizers": ["h3", "s2", "administrative", "slippy_map", "voronoi"],
            "embedders": ["count", "hex2vec", "gtfs2vec", "highway2vec", "s2vec"],
            "coordinate_system": "WGS84",
        }

        if self._available:
            try:
                from srai.regionalizers import (  # type: ignore[import-not-found]
                    H3Regionalizer as H3Regionalizer,
                )

                capabilities["regionalizers_available"] = True
            except ImportError:
                capabilities["regionalizers_available"] = False

            try:
                from srai.embedders import (  # type: ignore[import-not-found]
                    CountEmbedder as CountEmbedder,
                )

                capabilities["embedders_available"] = True
            except ImportError:
                capabilities["embedders_available"] = False

        return capabilities

    # ==================== Spatial Indexing Methods ====================

    @_require_srai("latlng_to_cell")
    def latlng_to_cell(self, lat: float, lng: float, resolution: int) -> str:
        """
        Convert lat/lng coordinates to SRAI region cell.

        Uses the configured regionalizer (H3 by default) to generate cell IDs.

        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)
            resolution: Resolution level (0-15 for H3)

        Returns:
            Cell identifier string

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Converting ({lat}, {lng}) to cell at resolution {resolution}")

        if self.default_regionalizer == "h3":
            # SRAI uses H3 under the hood for H3 regionalizer
            import h3

            cell = h3.latlng_to_cell(lat, lng, resolution)
            logger.debug(f"Generated H3 cell: {cell}")
            return cast(str, cell)
        else:
            # For other regionalizers, use SRAI's approach
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for latlng_to_cell"
            )

    @_require_srai("cell_to_latlng")
    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """
        Convert SRAI region cell back to lat/lng coordinates.

        Args:
            cell: Cell identifier

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Converting cell {cell} to coordinates")

        if self.default_regionalizer == "h3":
            import h3

            lat, lng = h3.cell_to_latlng(cell)
            logger.debug(f"Cell {cell} center: ({lat}, {lng})")
            return lat, lng
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for cell_to_latlng"
            )

    @_require_srai("polygon_to_cells")
    def polygon_to_cells(self, polygon: dict[str, Any], resolution: int) -> list[str]:
        """
        Convert polygon to list of SRAI region cells.

        Args:
            polygon: GeoJSON-like polygon dictionary
            resolution: Resolution level (0-15 for H3)

        Returns:
            List of cell identifiers covering the polygon

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.info(f"Converting polygon to cells at resolution {resolution}")

        if self.default_regionalizer == "h3":
            from srai.regionalizers import H3Regionalizer
            import geopandas as gpd
            from shapely.geometry import shape

            # Create geometry from GeoJSON
            geom = shape(polygon)
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

            # Use H3Regionalizer
            regionalizer = H3Regionalizer(resolution=resolution)
            regions = regionalizer.transform(gdf)

            cells = list(regions.index)
            logger.info(f"Polygon converted to {len(cells)} cells")
            return cells
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for polygon_to_cells"
            )

    @_require_srai("get_cell_neighbors")
    def get_cell_neighbors(self, cell: str, k: int = 1) -> list[str]:
        """
        Get neighboring cells around a given cell.

        Args:
            cell: Central cell identifier
            k: Number of rings of neighbors (default 1)

        Returns:
            List of neighboring cell identifiers

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting k={k} neighbors for cell {cell}")

        if self.default_regionalizer == "h3":
            import h3

            if not isinstance(k, int) or k < 1:
                raise ValueError("k must be a positive integer")
            neighbors = set(h3.grid_disk(cell, k))
            neighbors.difference_update(h3.grid_disk(cell, k - 1))
            logger.debug(f"Found {len(neighbors)} neighbors")
            return sorted(neighbors)
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_neighbors"
            )

    @_require_srai("get_cells_within_radius")
    def get_cells_within_radius(self, cell: str, k: int = 1) -> list[str]:
        """Return all H3 cells within ``k`` rings, excluding the center."""
        if self.default_regionalizer == "h3":
            import h3

            if not isinstance(k, int) or k < 0:
                raise ValueError("k must be a non-negative integer")
            return sorted(set(h3.grid_disk(cell, k)) - {cell})
        raise ValueError(
            f"Regionalizer '{self.default_regionalizer}' is not supported for "
            "get_cells_within_radius"
        )

    @_require_srai("get_cell_distance")
    def get_cell_distance(self, cell1: str, cell2: str) -> int:
        """
        Calculate the grid distance between two cells.

        Args:
            cell1: First cell identifier
            cell2: Second cell identifier

        Returns:
            Grid distance between cells

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Computing distance between {cell1} and {cell2}")

        if self.default_regionalizer == "h3":
            import h3

            distance = h3.grid_distance(cell1, cell2)
            logger.debug(f"Distance: {distance}")
            return cast(int, distance)
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_distance"
            )

    @_require_srai("compact_cells")
    def compact_cells(self, cells: list[str]) -> list[str]:
        """
        Compact a list of cells into a more efficient representation.

        Args:
            cells: List of cell identifiers

        Returns:
            Compacted list of cell identifiers at mixed resolutions

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.info(f"Compacting {len(cells)} cells")

        if self.default_regionalizer == "h3":
            import h3

            compacted = list(h3.compact_cells(cells))
            logger.info(f"Compacted to {len(compacted)} cells")
            return compacted
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for compact_cells"
            )

    @_require_srai("uncompact_cells")
    def uncompact_cells(self, compacted_cells: list[str], resolution: int) -> list[str]:
        """
        Uncompact cells back to individual cell identifiers.

        Args:
            compacted_cells: Compacted cell identifiers
            resolution: Target resolution level

        Returns:
            List of individual cell identifiers at target resolution

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.info(
            f"Uncompacting {len(compacted_cells)} cells to resolution {resolution}"
        )

        if self.default_regionalizer == "h3":
            import h3

            uncompacted = list(h3.uncompact_cells(compacted_cells, resolution))
            logger.info(f"Uncompacted to {len(uncompacted)} cells")
            return uncompacted
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for uncompact_cells"
            )

    @_require_srai("get_cell_parent")
    def get_cell_parent(self, cell: str, resolution: int) -> str:
        """
        Get the parent of a cell at a coarser resolution.

        Args:
            cell: Cell identifier
            resolution: Target resolution

        Returns:
            Parent cell identifier

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting parent of {cell} at resolution {resolution}")

        if self.default_regionalizer == "h3":
            import h3

            return cast(str, h3.cell_to_parent(cell, resolution))
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_parent"
            )

    @_require_srai("get_cell_children")
    def get_cell_children(self, cell: str, resolution: int) -> list[str]:
        """
        Get children of a cell at a finer resolution.

        Args:
            cell: Cell identifier
            resolution: Target resolution

        Returns:
            List of child cell identifiers

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting children of {cell} at resolution {resolution}")

        if self.default_regionalizer == "h3":
            import h3

            return list(h3.cell_to_children(cell, resolution))
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_children"
            )

    @_require_srai("get_cell_path")
    def get_cell_path(self, start_cell: str, end_cell: str) -> list[str]:
        """
        Get the path of cells between two cells.

        Args:
            start_cell: Start cell identifier
            end_cell: End cell identifier

        Returns:
            List of cell identifiers in the path

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Calculating path from {start_cell} to {end_cell}")

        if self.default_regionalizer == "h3":
            import h3

            # H3 v4 API check
            return list(h3.grid_path_cells(start_cell, end_cell))
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_path"
            )

    @_require_srai("get_cell_ring")
    def get_cell_ring(self, cell: str, k: int) -> list[str]:
        """
        Get the ring of cells at distance k.

        Args:
            cell: Center cell identifier
            k: Distance in grid steps

        Returns:
            List of cell identifiers in the ring

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting ring k={k} for {cell}")

        if self.default_regionalizer == "h3":
            import h3

            return list(h3.grid_ring(cell, k))
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_ring"
            )

    @_require_srai("get_cell_resolution")
    def get_cell_resolution(self, cell: str) -> int:
        """
        Get the resolution level of a cell.

        Args:
            cell: Cell identifier

        Returns:
            Resolution level (0-15 for H3)

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting resolution for cell {cell}")

        if self.default_regionalizer == "h3":
            import h3

            resolution = h3.get_resolution(cell)
            logger.debug(f"Cell {cell} has resolution {resolution}")
            return cast(int, resolution)
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_resolution"
            )

    @_require_srai("get_cell_boundary")
    def get_cell_boundary(self, cell: str) -> list[tuple[float, float]]:
        """
        Get the boundary coordinates of a cell.

        Args:
            cell: Cell identifier

        Returns:
            List of (latitude, longitude) tuples forming the cell boundary

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting boundary for cell {cell}")

        if self.default_regionalizer == "h3":
            import h3

            boundary = h3.cell_to_boundary(cell)
            result = [(lat, lng) for lat, lng in boundary]
            logger.debug(f"Cell {cell} has {len(result)} boundary vertices")
            return result
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_boundary"
            )

    @_require_srai("get_cell_area")
    def get_cell_area(self, cell: str, unit: str = "km^2") -> float:
        """
        Get the area of a cell in square kilometers.

        Args:
            cell: Cell identifier

        Returns:
            Area in km²

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.debug(f"Getting area for cell {cell}")

        if self.default_regionalizer == "h3":
            import h3

            area = h3.cell_area(cell, unit=unit)
            logger.debug(f"Cell {cell} has area {area:.6f} {unit}")
            return cast(float, area)
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for get_cell_area"
            )

    @_require_srai("average_edge_length")
    def average_edge_length(self, resolution: int, unit: str = "m") -> float:
        """Return the native H3 average edge length for a resolution."""
        if self.default_regionalizer == "h3":
            import h3

            return float(h3.average_hexagon_edge_length(resolution, unit=unit))
        raise ValueError(
            f"Regionalizer '{self.default_regionalizer}' is not supported for "
            "average_edge_length"
        )

    @_require_srai("cells_to_multipolygon")
    def cells_to_multipolygon(self, cells: list[str]) -> dict[str, Any]:
        """
        Convert a list of cells to a GeoJSON MultiPolygon geometry.

        Args:
            cells: List of cell identifiers

        Returns:
            GeoJSON-like dictionary with 'type' and 'coordinates'

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.info(f"Converting {len(cells)} cells to MultiPolygon")

        if not cells:
            return {"type": "MultiPolygon", "coordinates": []}

        if self.default_regionalizer == "h3":
            import h3

            polygons = []
            for cell in cells:
                boundary = h3.cell_to_boundary(cell)
                # Convert to GeoJSON format [lng, lat] and close the ring
                ring = [[lng, lat] for lat, lng in boundary]
                ring.append(ring[0])
                polygons.append([ring])

            logger.info(f"Created MultiPolygon with {len(polygons)} polygons")
            return {"type": "MultiPolygon", "coordinates": polygons}
        else:
            raise ValueError(
                f"Regionalizer '{self.default_regionalizer}' is not supported for cells_to_multipolygon"
            )

    # ==================== Analytics Methods ====================

    @_require_srai("analyze_hotspots")
    def analyze_hotspots(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze spatial hotspots using SRAI analytics.

        Args:
            data: Dictionary with 'cells' (list of cell IDs) and
                  'values' (corresponding numeric values)

        Returns:
            Dictionary with hotspot analysis results

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
            ValueError: If data format is invalid
        """
        logger.info("Analyzing hotspots with SRAI")

        cells = data.get("cells", [])
        values = data.get("values", [])

        if len(cells) != len(values):
            raise ValueError("Cells and values must have the same length")

        # Use statistical methods to detect hotspots
        import numpy as np

        values_array = np.array(values)

        # Hotspot detection using standard deviation
        mean_val = float(np.mean(values_array))
        std_val = float(np.std(values_array))
        threshold = mean_val + std_val

        hotspots = []
        for cell, value in zip(cells, values):
            if value > threshold:
                hotspots.append(
                    {
                        "cell": cell,
                        "value": float(value),
                        "intensity": "high"
                        if value > mean_val + 2 * std_val
                        else "medium",
                        "z_score": float((value - mean_val) / std_val)
                        if std_val > 0
                        else 0.0,
                    }
                )

        logger.info(f"Found {len(hotspots)} hotspots (threshold: {threshold:.2f})")

        return {
            "hotspots": hotspots,
            "threshold": threshold,
            "total_cells": len(cells),
            "hotspot_count": len(hotspots),
            "statistics": {
                "mean": mean_val,
                "std": std_val,
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
            },
        }

    @_require_srai("compute_proximity")
    def compute_proximity(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        """
        Compute proximity analysis using SRAI.

        Args:
            points: List of (latitude, longitude) coordinate tuples

        Returns:
            Dictionary with proximity analysis results

        Raises:
            SRAIUnavailableError: If SRAI library is not installed
        """
        logger.info(f"Computing proximity for {len(points)} points")

        # Convert points to regions
        regions = []
        for lat, lng in points:
            region = self.latlng_to_cell(lat, lng, 9)  # Use resolution 9
            regions.append(region)

        # Calculate distances between cells
        proximity_pairs = []
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions[i + 1 :], i + 1):
                try:
                    distance = self.get_cell_distance(region1, region2)
                    proximity_pairs.append(
                        {
                            "from_region": region1,
                            "to_region": region2,
                            "distance": distance,
                            "from_point": points[i],
                            "to_point": points[j],
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not compute distance: {e}")
                    continue

        logger.info(f"Analyzed {len(proximity_pairs)} proximity pairs")

        return {
            "proximity_pairs": proximity_pairs,
            "total_points": len(points),
            "analyzed_pairs": len(proximity_pairs),
            "regionalizer": self.default_regionalizer,
        }
