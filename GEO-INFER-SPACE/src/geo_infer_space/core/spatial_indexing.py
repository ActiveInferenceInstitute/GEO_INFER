"""
Generic spatial indexing interface for GEO-INFER-SPACE.

This module defines the generic interface for spatial indexing operations
that can be implemented by different backends (H3, SRAI, etc.).
"""

from typing import Any, cast
import math
import logging

logger = logging.getLogger(__name__)


class SpatialIndexingInterface:
    """
    Generic interface for spatial indexing operations.

    This class provides a unified API for spatial indexing that can dispatch
    to different backends based on configuration.
    """

    def __init__(self, backend: str | None = None) -> None:
        from .dispatcher import get_backend_dispatcher

        self.dispatcher = get_backend_dispatcher()
        self.backend = backend

    def latlng_to_cell(self, lat: float, lng: float, resolution: int) -> str:
        """
        Convert latitude/longitude coordinates to a spatial index cell.

        Args:
            lat: Latitude coordinate
            lng: Longitude coordinate
            resolution: Resolution level for the spatial index

        Returns:
            Spatial index cell identifier
        """
        return cast(
            str,
            self.dispatcher.dispatch_indexing_operation(
                "latlng_to_cell", lat, lng, resolution, backend=self.backend
            ),
        )

    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """
        Convert a spatial index cell back to latitude/longitude coordinates.

        Args:
            cell: Spatial index cell identifier

        Returns:
            Tuple of (latitude, longitude) coordinates
        """
        return cast(
            tuple[float, float],
            self.dispatcher.dispatch_indexing_operation(
                "cell_to_latlng", cell, backend=self.backend
            ),
        )

    def get_neighbors(
        self,
        position: tuple[float, float],
        radius: float,
        resolution: int = 9,
    ) -> list[str]:
        """
        Get neighboring spatial cells within a radius.

        Args:
            position: Center position (lat, lng)
            radius: Search radius in meters. The result contains all cells in
                the H3 grid disk whose ring distance is covered by this radius.
            resolution: H3 resolution used to discretize ``position``.

        Returns:
            List of neighboring cell identifiers
        """
        if len(position) != 2:
            raise ValueError("position must be a (latitude, longitude) pair")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not 0 <= resolution <= 15:
            raise ValueError("resolution must be between 0 and 15")

        cell = self.latlng_to_cell(position[0], position[1], resolution)
        backend_name = self.backend or self.dispatcher.get_default_backend("indexing")
        backend_instance = self.dispatcher.get_backend(backend_name)
        edge_length = getattr(backend_instance, "average_edge_length", None)
        if edge_length is None:
            raise ValueError(
                f"Backend '{backend_name}' cannot convert a metric radius to cells"
            )
        edge_m = float(edge_length(resolution, unit="m"))
        if not math.isfinite(edge_m) or edge_m <= 0:
            raise ValueError("backend returned an invalid average cell edge length")
        ring_distance = max(1, math.ceil(radius / edge_m))
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "get_cells_within_radius",
                cell,
                ring_distance,
                backend=self.backend,
            ),
        )

    def get_cell_neighbors(self, cell: str, k: int = 1) -> list[str]:
        """
        Get neighboring cells around a given cell.

        Args:
            cell: Central spatial index cell
            k: Number of rings of neighbors to return (default 1)

        Returns:
            List of neighboring cell identifiers
        """
        logger.debug(f"Getting k={k} neighbors for cell {cell}")
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "get_neighbors", cell, k, backend=self.backend
            ),
        )

    def polygon_to_cells(self, polygon: dict[str, Any], resolution: int) -> list[str]:
        """
        Convert a polygon geometry to a list of spatial index cells.

        Args:
            polygon: Polygon geometry as GeoJSON-like dictionary
            resolution: Resolution level for the spatial index

        Returns:
            List of spatial index cell identifiers covering the polygon
        """
        logger.debug(f"Converting polygon to cells at resolution {resolution}")
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "polygon_to_cells", polygon, resolution, backend=self.backend
            ),
        )

    def get_cell_distance(self, cell1: str, cell2: str) -> int:
        """
        Calculate the distance between two spatial index cells.

        Args:
            cell1: First spatial index cell
            cell2: Second spatial index cell

        Returns:
            Distance between cells in grid units
        """
        return cast(
            int,
            self.dispatcher.dispatch_indexing_operation(
                "get_distance", cell1, cell2, backend=self.backend
            ),
        )

    def compact_cells(self, cells: list[str]) -> list[str]:
        """
        Compact a list of cells into a more efficient representation.

        Args:
            cells: List of spatial index cell identifiers

        Returns:
            Compacted list of cell identifiers
        """
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "compact_cells", cells, backend=self.backend
            ),
        )

    def uncompact_cells(self, compacted_cells: list[str], resolution: int) -> list[str]:
        """
        Uncompact cells back to individual cell identifiers.

        Args:
            compacted_cells: Compacted cell identifiers
            resolution: Target resolution level

        Returns:
            List of individual cell identifiers
        """
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "uncompact_cells",
                compacted_cells,
                resolution,
                backend=self.backend,
            ),
        )

    def get_cell_parent(self, cell: str, resolution: int) -> str:
        """
        Get the parent of a cell at a coarser resolution.

        Args:
            cell: Spatial index cell identifier
            resolution: Target resolution

        Returns:
            Parent cell identifier
        """
        return cast(
            str,
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_parent", cell, resolution, backend=self.backend
            ),
        )

    def get_cell_children(self, cell: str, resolution: int) -> list[str]:
        """
        Get children of a cell at a finer resolution.

        Args:
            cell: Spatial index cell identifier
            resolution: Target resolution

        Returns:
            List of child cell identifiers
        """
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_children", cell, resolution, backend=self.backend
            ),
        )

    def get_cell_path(self, start_cell: str, end_cell: str) -> list[str]:
        """
        Get the path of cells between two cells.

        Args:
            start_cell: Start cell identifier
            end_cell: End cell identifier

        Returns:
            List of cell identifiers in the path
        """
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_path", start_cell, end_cell, backend=self.backend
            ),
        )

    def get_cell_ring(self, cell: str, k: int) -> list[str]:
        """
        Get the ring of cells at distance k.

        Args:
            cell: Center cell identifier
            k: Distance in grid steps

        Returns:
            List of cell identifiers in the ring
        """
        return cast(
            list[str],
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_ring", cell, k, backend=self.backend
            ),
        )

    def get_cell_resolution(self, cell: str) -> int:
        """Get resolution of a cell."""
        return cast(
            int,
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_resolution", cell, backend=self.backend
            ),
        )

    def get_cell_boundary(self, cell: str) -> list[tuple[float, float]]:
        """Get boundary coordinates of a cell."""
        return cast(
            list[tuple[float, float]],
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_boundary", cell, backend=self.backend
            ),
        )

    def get_cell_area(self, cell: str, unit: str = "km^2") -> float:
        """Get area of a cell."""
        return cast(
            float,
            self.dispatcher.dispatch_indexing_operation(
                "get_cell_area", cell, unit=unit, backend=self.backend
            ),
        )

    def cells_to_multipolygon(self, cells: list[str]) -> dict[str, Any]:
        """Convert cells to multipolygon boundary."""
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_indexing_operation(
                "cells_to_multipolygon", cells, backend=self.backend
            ),
        )


# Convenience functions that use the default backend
def latlng_to_cell(
    lat: float, lng: float, resolution: int, backend: str | None = None
) -> str:
    """Convert lat/lng to cell using specified or default backend."""
    indexer = SpatialIndexingInterface(backend)
    return indexer.latlng_to_cell(lat, lng, resolution)


def cell_to_latlng(cell: str, backend: str | None = None) -> tuple[float, float]:
    """Convert cell to lat/lng using specified or default backend."""
    indexer = SpatialIndexingInterface(backend)
    return indexer.cell_to_latlng(cell)


def polygon_to_cells(
    polygon: dict[str, Any], resolution: int, backend: str | None = None
) -> list[str]:
    """Convert polygon to cells using specified or default backend."""
    indexer = SpatialIndexingInterface(backend)
    return indexer.polygon_to_cells(polygon, resolution)
