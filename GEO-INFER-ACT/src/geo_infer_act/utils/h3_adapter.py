"""
H3 access helpers for GEO-INFER-ACT.

The adapter prefers the canonical GEO-INFER-SPACE indexing interface when it is
installed (the ``space`` extra) and uses direct h3-py v4 calls (a hard
dependency) for every operation SPACE does not expose.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Iterable

import h3

logger = logging.getLogger(__name__)


class H3Adapter:
    """Adapter over SPACE H3 indexing and direct H3 v4."""

    def __init__(self, prefer_space: bool = True):
        """Initialize SPACE-backed and direct H3 access."""
        self.space_indexer = None
        self.h3 = h3
        self.source = "direct"

        if prefer_space:
            try:
                from geo_infer_space.core.spatial_indexing import (  # noqa: PLC0415
                    SpatialIndexingInterface,
                )

                self.space_indexer = SpatialIndexingInterface(backend="h3")
                self.source = "geo_infer_space"
            except Exception:
                self.space_indexer = None

        version = tuple(
            int(part.split("+")[0].split("-")[0])
            for part in h3.__version__.lstrip("v").split(".")[:3]
        )
        if version < (4, 5, 0) or version >= (5, 0, 0):
            raise RuntimeError(
                f"Unsupported h3-py version {h3.__version__}; "
                "GEO-INFER requires h3-py>=4.5.0,<5"
            )

    def latlng_to_cell(self, lat: float, lng: float, resolution: int) -> str:
        """Convert latitude/longitude to an H3 cell."""
        if self.space_indexer is not None:
            return str(self.space_indexer.latlng_to_cell(lat, lng, resolution))
        return str(self.h3.latlng_to_cell(lat, lng, resolution))

    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """Convert an H3 cell to latitude/longitude."""
        if self.space_indexer is not None:
            res = self.space_indexer.cell_to_latlng(cell)
            return (float(res[0]), float(res[1]))
        res = self.h3.cell_to_latlng(cell)
        return (float(res[0]), float(res[1]))

    def cell_to_boundary(self, cell: str) -> list[tuple[float, float]]:
        """Return the H3 cell boundary as latitude/longitude pairs."""
        return [(lat, lng) for lat, lng in self.h3.cell_to_boundary(cell)]

    def polygon_to_cells(self, polygon: dict[str, Any], resolution: int) -> list[str]:
        """Convert a GeoJSON-like polygon to H3 cells."""
        if self.space_indexer is not None:
            try:
                cells = self.space_indexer.polygon_to_cells(polygon, resolution)
                if cells:
                    return list(cells)
            except Exception:
                logger.debug(
                    "SPACE polygon_to_cells failed; using direct h3", exc_info=True
                )

        if hasattr(polygon, "__geo_interface__"):
            polygon = polygon.__geo_interface__
        if not isinstance(polygon, dict):
            raise ValueError("polygon must be GeoJSON-like")

        geometry = polygon.get("geometry", polygon)
        if geometry.get("type") == "FeatureCollection":
            feature_cells: set[str] = set()
            for feature in geometry.get("features", []):
                feature_geometry = feature.get("geometry")
                if feature_geometry:
                    feature_cells.update(
                        self.h3.geo_to_cells(feature_geometry, resolution)
                    )
            return sorted(feature_cells)
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("polygon must contain a Polygon or MultiPolygon")
        return sorted(self.h3.geo_to_cells(geometry, resolution))

    def grid_disk(self, cell: str, k: int = 1) -> list[str]:
        """Return H3 cells within k grid steps of a cell."""
        return list(self.h3.grid_disk(cell, k))

    def grid_ring(self, cell: str, k: int = 1) -> list[str]:
        """Return H3 cells exactly k grid steps from a cell."""
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        return sorted(
            set(self.h3.grid_disk(cell, k)) - set(self.h3.grid_disk(cell, k - 1))
        )

    def get_resolution(self, cell: str) -> int:
        """Return the H3 resolution of a cell."""
        if self.space_indexer is not None:
            return int(self.space_indexer.get_cell_resolution(cell))
        return int(self.h3.get_resolution(cell))

    def cell_to_parent(self, cell: str, resolution: int) -> str:
        """Return the parent cell at a coarser resolution."""
        if self.space_indexer is not None:
            return str(self.space_indexer.get_cell_parent(cell, resolution))
        return str(self.h3.cell_to_parent(cell, resolution))

    def cell_to_children(self, cell: str, resolution: int) -> list[str]:
        """Return child cells at a finer resolution."""
        if self.space_indexer is not None:
            return [
                str(c) for c in self.space_indexer.get_cell_children(cell, resolution)
            ]
        return [str(c) for c in self.h3.cell_to_children(cell, resolution)]

    def is_valid_cell(self, cell: str) -> bool:
        """Return true when a value is a valid H3 cell identifier."""
        try:
            return bool(self.h3.is_valid_cell(cell))
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug("Unexpected error validating H3 cell %r", cell, exc_info=True)
            return False

    def validate_cells(self, cells: Iterable[str]) -> list[str]:
        """Validate and normalize H3 cell identifiers."""
        normalized = [str(cell) for cell in cells]
        invalid = [cell for cell in normalized if not self.is_valid_cell(cell)]
        if invalid:
            raise ValueError(f"Invalid H3 cell identifiers: {invalid[:5]}")
        return normalized


def get_h3_adapter(prefer_space: bool = True) -> H3Adapter:
    """Create an H3 adapter for ACT spatial methods."""
    return H3Adapter(prefer_space=prefer_space)


def get_nested_h3_grid_class() -> Any:
    """Return SPACE's ``NestedH3Grid`` class.

    Raises:
        ImportError: If ``geo_infer_space`` (the ACT ``space`` extra) is not
            installed.
    """
    try:
        from geo_infer_space.nested import NestedH3Grid  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Nested H3 grids require geo_infer_space; install geo-infer-act[space]"
        ) from exc
    return NestedH3Grid


def normalize_belief_vector(values: Any) -> Any:
    """Normalize a belief vector with a finite, nonnegative distribution."""
    import numpy as np

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("Belief vectors must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("Belief vectors must contain finite values")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= 1e-12:
        return np.ones_like(array) / array.size
    return array / total


def edge_count_from_graph(graph: dict[str, Iterable[str]] | None) -> int:
    """Count undirected edges in a cell-neighbor graph."""
    if not graph:
        return 0
    edges = {
        tuple(sorted((str(cell), str(neighbor))))
        for cell, neighbors in graph.items()
        for neighbor in neighbors
    }
    return len(edges)
