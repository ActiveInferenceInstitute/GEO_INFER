"""
Core H3 classes and data structures for advanced hexagonal grid operations.

This module provides the fundamental building blocks for H3 operations including
grid management, cell representation, analytics, and validation using H3 v4 API.
"""

import logging
from typing import (
    Any,
    Optional,
    cast,
)
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, UTC
import json
import math

import h3
import numpy as np
import pandas as pd
from geo_infer_time import normalize_timestamp

logger = logging.getLogger(__name__)


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be an integer in the supported range")


def _coordinates(lat: float, lng: float) -> None:
    if any(
        isinstance(value, (bool, str, bytes))
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (lat, lng)
    ) or not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("Expected finite WGS84 latitude/longitude")


@dataclass(frozen=True)
class H3Cell:
    """
    Represents a single H3 hexagonal cell with comprehensive metadata and operations.

    This class encapsulates all information about an H3 cell including its index,
    coordinates, properties, and provides methods for analysis and manipulation.
    """

    index: str
    resolution: int
    latitude: float = field(default=0.0)
    longitude: float = field(default=0.0)
    area_km2: float = field(default=0.0)
    boundary: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Initialize cell properties after creation."""
        if (
            not isinstance(self.index, str)
            or not h3.is_valid_cell(self.index)
            or h3.int_to_str(h3.str_to_int(self.index)) != self.index
        ):
            raise ValueError("index must be a canonical H3 cell string")
        if (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, int)
            or self.resolution != h3.get_resolution(self.index)
        ):
            raise ValueError("resolution must match the H3 cell index")
        if not isinstance(self.properties, dict):
            raise ValueError("properties must be a mapping")
        latitude, longitude = h3.cell_to_latlng(self.index)
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "area_km2", h3.cell_area(self.index, "km^2"))
        object.__setattr__(self, "boundary", tuple(h3.cell_to_boundary(self.index)))
        object.__setattr__(self, "properties", deepcopy(self.properties))

    @classmethod
    def from_coordinates(
        cls, lat: float, lng: float, resolution: int, **properties: Any
    ) -> "H3Cell":
        """
        Create H3Cell from latitude/longitude coordinates.

        Args:
            lat: Latitude in degrees
            lng: Longitude in degrees
            resolution: H3 resolution (0-15)
            **properties: Additional cell properties

        Returns:
            H3Cell instance
        """
        _coordinates(lat, lng)
        _integer(resolution, "resolution", 0, 15)
        index = h3.latlng_to_cell(lat, lng, resolution)
        return cls(
            index=index,
            resolution=resolution,
            latitude=lat,
            longitude=lng,
            properties=properties,
        )

    def neighbors(self, k: int = 1, *, max_cells: int = 1_000_000) -> list["H3Cell"]:
        """Return real disk neighbors excluding self; topology errors propagate."""
        _integer(k, "k", 0)
        _integer(max_cells, "max_cells", 1)
        if 1 + 3 * k * (k + 1) > max_cells:
            raise ValueError("Neighborhood allocation exceeds max_cells")
        return [
            H3Cell(index=index, resolution=self.resolution)
            for index in h3.grid_disk(self.index, k)
            if index != self.index
        ]

    def parent(self, parent_resolution: int | None = None) -> Optional["H3Cell"]:
        """Return a coarser parent, or None at resolution zero by default."""
        if parent_resolution is None:
            if self.resolution == 0:
                return None
            parent_resolution = self.resolution - 1
        _integer(parent_resolution, "parent_resolution", 0, 15)
        if parent_resolution >= self.resolution:
            raise ValueError("Parent resolution must be coarser than the cell")
        return H3Cell(
            index=h3.cell_to_parent(self.index, parent_resolution),
            resolution=parent_resolution,
        )

    def children(
        self, child_resolution: int | None = None, *, max_cells: int = 1_000_000
    ) -> list["H3Cell"]:
        """Return complete finer children, bounded before topology allocation."""
        if child_resolution is None:
            if self.resolution == 15:
                return []
            child_resolution = self.resolution + 1
        _integer(child_resolution, "child_resolution", 0, 15)
        _integer(max_cells, "max_cells", 1)
        if child_resolution <= self.resolution:
            raise ValueError("Child resolution must be finer than the cell")
        if h3.cell_to_children_size(self.index, child_resolution) > max_cells:
            raise ValueError("Children allocation exceeds max_cells")
        return [
            H3Cell(index=index, resolution=child_resolution)
            for index in h3.cell_to_children(self.index, child_resolution)
        ]

    def distance_to(self, other: "H3Cell") -> int:
        """Return exact grid distance; mixed resolutions and unsupported pairs raise."""
        if not isinstance(other, H3Cell):
            raise TypeError("other must be an H3Cell")
        if self.resolution != other.resolution:
            raise ValueError("Grid distance requires equal resolutions")
        return cast(int, h3.grid_distance(self.index, other.index))

    def is_neighbor(self, other: "H3Cell") -> bool:
        """Return real adjacency for equal-resolution cells."""
        if not isinstance(other, H3Cell):
            raise TypeError("other must be an H3Cell")
        if self.resolution != other.resolution:
            raise ValueError("Adjacency requires equal resolutions")
        return cast(bool, h3.are_neighbor_cells(self.index, other.index))

    def to_geojson(self) -> dict[str, Any]:
        """
        Convert cell to GeoJSON feature.

        Returns:
            GeoJSON feature dictionary
        """
        # Ensure boundary is closed (first point = last point)
        boundary_coords = list(self.boundary)
        if boundary_coords and boundary_coords[0] != boundary_coords[-1]:
            boundary_coords.append(boundary_coords[0])

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in boundary_coords]],
            },
            "properties": {
                **deepcopy(self.properties),
                "h3_index": self.index,
                "resolution": self.resolution,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "area_km2": self.area_km2,
                "created_at": self.created_at.isoformat(),
            },
        }

    def __str__(self) -> str:
        return f"H3Cell(index={self.index}, resolution={self.resolution}, lat={self.latitude:.6f}, lng={self.longitude:.6f})"

    def __repr__(self) -> str:
        return self.__str__()


class H3Grid:
    """
    Manages collections of H3 cells with advanced operations and analytics.

    This class provides high-level operations for working with multiple H3 cells
    including spatial analysis, aggregation, and visualization support.
    """

    def __init__(
        self,
        cells: list[H3Cell] | None = None,
        name: str = "H3Grid",
    ) -> None:
        """
        Initialize H3Grid with optional cells.

        Args:
            cells: List of H3Cell instances
            name: Grid name for identification
        """
        self._cells = list(cells) if cells is not None else []
        self._cells_snapshot: tuple[H3Cell, ...] | None = None
        if any(not isinstance(cell, H3Cell) for cell in self._cells):
            raise TypeError("cells must contain H3Cell instances")
        if len({cell.index for cell in self.cells}) != len(self.cells):
            raise ValueError("Grid cell indexes must be unique")
        self.name = name
        self.created_at = datetime.now(UTC)
        self._cell_index: dict[str, H3Cell] = {}
        self._build_index()

    @property
    def cells(self) -> tuple[H3Cell, ...]:
        """Read-only membership; use add_cell/remove_cell to preserve the index."""
        if self._cells_snapshot is None:
            self._cells_snapshot = tuple(self._cells)
        return self._cells_snapshot

    def _build_index(self) -> None:
        """Build internal index for fast cell lookup."""
        self._cell_index = {cell.index: cell for cell in self.cells}

    def add_cell(self, cell: H3Cell) -> None:
        """Add a cell to the grid."""
        if not isinstance(cell, H3Cell):
            raise TypeError("cell must be an H3Cell")
        if cell.index not in self._cell_index:
            self._cells.append(cell)
            self._cells_snapshot = None
            self._cell_index[cell.index] = cell

    def remove_cell(self, cell_index: str) -> bool:
        """
        Remove a cell from the grid.

        Args:
            cell_index: H3 cell index to remove

        Returns:
            True if cell was removed
        """
        if cell_index in self._cell_index:
            cell = self._cell_index[cell_index]
            self._cells.remove(cell)
            self._cells_snapshot = None
            del self._cell_index[cell_index]
            return True
        return False

    def get_cell(self, cell_index: str) -> H3Cell | None:
        """Get cell by index."""
        return self._cell_index.get(cell_index)

    def has_cell(self, cell_index: str) -> bool:
        """Check if grid contains cell."""
        return cell_index in self._cell_index

    @classmethod
    def from_polygon(
        cls,
        polygon_coords: list[tuple[float, float]],
        resolution: int,
        name: str = "PolygonGrid",
    ) -> "H3Grid":
        """Create cells covering a WGS84 (lat, lng) ring; invalid inputs raise."""
        _integer(resolution, "resolution", 0, 15)
        if len(polygon_coords) < 3:
            raise ValueError("A polygon needs at least three positions")
        for lat, lng in polygon_coords:
            _coordinates(lat, lng)
        coords = [[lng, lat] for lat, lng in polygon_coords]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        from shapely.geometry import Polygon

        polygon = Polygon(coords)
        if not polygon.is_valid or polygon.area == 0:
            raise ValueError("Polygon must have valid topology and positive area")
        indexes = h3.geo_to_cells(
            {"type": "Polygon", "coordinates": [coords]}, resolution
        )
        return cls(
            [H3Cell(index=index, resolution=resolution) for index in indexes], name=name
        )

    @classmethod
    def from_center(
        cls,
        lat: float,
        lng: float,
        resolution: int,
        k: int = 1,
        name: str = "CenterGrid",
        *,
        max_cells: int = 1_000_000,
    ) -> "H3Grid":
        """Create a real, bounded H3 disk centered on WGS84 coordinates."""
        _coordinates(lat, lng)
        _integer(resolution, "resolution", 0, 15)
        _integer(k, "k", 0)
        _integer(max_cells, "max_cells", 1)
        if 1 + 3 * k * (k + 1) > max_cells:
            raise ValueError("Disk allocation exceeds max_cells")
        indexes = h3.grid_disk(h3.latlng_to_cell(lat, lng, resolution), k)
        return cls(
            [H3Cell(index=index, resolution=resolution) for index in indexes], name=name
        )

    def compact(self) -> "H3Grid":
        """Compact real cells; unsupported or overlapping domains raise."""
        indexes = h3.compact_cells([cell.index for cell in self.cells])
        return H3Grid(
            [
                H3Cell(index=index, resolution=h3.get_resolution(index))
                for index in indexes
            ],
            name=f"{self.name}_compacted",
        )

    def uncompact(
        self, target_resolution: int, *, max_cells: int = 1_000_000
    ) -> "H3Grid":
        """Expand a nonoverlapping mixed-resolution domain within an allocation budget."""
        _integer(target_resolution, "target_resolution", 0, 15)
        _integer(max_cells, "max_cells", 1)
        indexes = [cell.index for cell in self.cells]
        if any(cell.resolution > target_resolution for cell in self.cells):
            raise ValueError("Target resolution cannot be coarser than a source cell")
        ancestors = set(indexes)
        if any(
            h3.cell_to_parent(cell.index, level) in ancestors
            for cell in self.cells
            for level in range(cell.resolution)
        ):
            raise ValueError("Source cells cannot overlap their ancestors")
        if (
            sum(h3.cell_to_children_size(index, target_resolution) for index in indexes)
            > max_cells
        ):
            raise ValueError("Uncompact allocation exceeds max_cells")
        expanded = h3.uncompact_cells(indexes, target_resolution)
        return H3Grid(
            [H3Cell(index=index, resolution=target_resolution) for index in expanded],
            name=f"{self.name}_uncompacted",
        )

    def total_area(self) -> float:
        """
        Calculate total area of all cells in km².

        Returns:
            Total area in square kilometers
        """
        return sum(cell.area_km2 for cell in self.cells)

    def bounds(self) -> tuple[float, float, float, float]:
        """
        Get coordinate extrema of all cell boundary vertices.

        Longitude extrema use the conventional [-180, 180] representation.
        A grid crossing the antimeridian therefore has a wide longitude span;
        this tuple does not encode a wrapped longitude interval.

        Returns:
            (min_lat, min_lng, max_lat, max_lng)
        """
        if not self.cells:
            return (0.0, 0.0, 0.0, 0.0)

        lats = [latitude for cell in self.cells for latitude, _ in cell.boundary]
        lngs = [longitude for cell in self.cells for _, longitude in cell.boundary]

        return (min(lats), min(lngs), max(lats), max(lngs))

    def center(self) -> tuple[float, float]:
        """
        Get center coordinates of the grid.

        Returns:
            (center_lat, center_lng)
        """
        if not self.cells:
            return (0.0, 0.0)

        lats = [cell.latitude for cell in self.cells]
        lngs = [cell.longitude for cell in self.cells]

        return (sum(lats) / len(lats), sum(lngs) / len(lngs))

    def resolutions(self) -> set[int]:
        """Get set of all resolutions in the grid."""
        return {cell.resolution for cell in self.cells}

    def filter_by_resolution(self, resolution: int) -> "H3Grid":
        """
        Filter cells by resolution.

        Args:
            resolution: Target resolution

        Returns:
            New H3Grid with only cells of specified resolution
        """
        filtered_cells = [cell for cell in self.cells if cell.resolution == resolution]
        return H3Grid(cells=filtered_cells, name=f"{self.name}_res{resolution}")

    def to_geojson(self) -> dict[str, Any]:
        """
        Convert grid to GeoJSON FeatureCollection.

        Returns:
            GeoJSON FeatureCollection
        """
        features = [cell.to_geojson() for cell in self.cells]

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "name": self.name,
                "cell_count": len(self.cells),
                "total_area_km2": self.total_area(),
                "resolutions": list(self.resolutions()),
                "created_at": self.created_at.isoformat(),
            },
        }

    def to_dataframe(
        self,
    ) -> "pd.DataFrame":
        """
        Convert grid to pandas DataFrame.

        Returns:
            DataFrame with cell information
        """
        data = []
        for cell in self.cells:
            row = {
                **deepcopy(cell.properties),
                "h3_index": cell.index,
                "resolution": cell.resolution,
                "latitude": cell.latitude,
                "longitude": cell.longitude,
                "area_km2": cell.area_km2,
                "created_at": cell.created_at,
            }
            data.append(row)

        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[H3Cell]:
        return iter(self.cells)

    def __str__(self) -> str:
        return f"H3Grid(name={self.name}, cells={len(self.cells)}, area={self.total_area():.2f}km²)"

    def __repr__(self) -> str:
        return self.__str__()


class H3Analytics:
    """
    Advanced analytics for H3 grids and cells.

    Provides statistical analysis, spatial relationships, and pattern detection
    for H3 hexagonal grids with comprehensive metrics and insights.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize analytics for an H3Grid.

        Args:
            grid: H3Grid instance to analyze
        """
        self.grid = grid
        self.stats_cache: dict[str, Any] = {}

    def basic_statistics(self) -> dict[str, Any]:
        """
        Calculate basic grid statistics.

        Returns:
            Dictionary with basic statistics
        """
        if not self.grid.cells:
            return {}

        areas = [cell.area_km2 for cell in self.grid.cells]
        resolutions = [cell.resolution for cell in self.grid.cells]

        stats = {
            "cell_count": len(self.grid.cells),
            "total_area_km2": sum(areas),
            "mean_area_km2": np.mean(areas),
            "std_area_km2": np.std(areas),
            "min_area_km2": min(areas),
            "max_area_km2": max(areas),
            "unique_resolutions": len(set(resolutions)),
            "resolution_distribution": {
                res: resolutions.count(res) for res in set(resolutions)
            },
            "bounds": self.grid.bounds(),
            "center": self.grid.center(),
        }

        self.stats_cache["basic_stats"] = stats
        return stats

    def connectivity_analysis(self) -> dict[str, Any]:
        """
        Analyze connectivity between cells.

        Returns:
            Dictionary with connectivity metrics
        """
        if not self.grid.cells:
            return {}

        # Build adjacency information
        adjacency_count = 0
        isolated_cells = 0
        cell_neighbors = {}

        for cell in self.grid.cells:
            neighbors_in_grid = []

            # Get all neighbors of this cell
            neighbor_indices = h3.grid_disk(cell.index, 1)
            for neighbor_index in neighbor_indices:
                if neighbor_index != cell.index and self.grid.has_cell(neighbor_index):
                    neighbors_in_grid.append(neighbor_index)
                    adjacency_count += 1

            cell_neighbors[cell.index] = neighbors_in_grid

            if not neighbors_in_grid:
                isolated_cells += 1

        # Calculate connectivity metrics
        avg_neighbors = adjacency_count / len(self.grid.cells) if self.grid.cells else 0
        connectivity_ratio = (
            (len(self.grid.cells) - isolated_cells) / len(self.grid.cells)
            if self.grid.cells
            else 0
        )

        return {
            "total_adjacencies": adjacency_count // 2,  # Each adjacency counted twice
            "average_neighbors": avg_neighbors,
            "isolated_cells": isolated_cells,
            "connectivity_ratio": connectivity_ratio,
            "cell_neighbors": cell_neighbors,
        }

    def density_analysis(
        self, reference_area_km2: float | None = None
    ) -> dict[str, Any]:
        """
        Analyze cell density patterns.

        Args:
            reference_area_km2: Reference area for density calculation

        Returns:
            Dictionary with density metrics
        """
        if not self.grid.cells:
            return {}

        bounds = self.grid.bounds()
        if reference_area_km2 is None:
            # Calculate bounding box area (approximate)
            lat_diff = bounds[2] - bounds[0]  # max_lat - min_lat
            lng_diff = bounds[3] - bounds[1]  # max_lng - min_lng

            # Rough conversion to km² (not accurate for large areas)
            reference_area_km2 = (
                lat_diff
                * lng_diff
                * 111.32
                * 111.32
                * np.cos(np.radians((bounds[0] + bounds[2]) / 2))
            )

        total_cell_area = self.grid.total_area()

        return {
            "cells_per_km2": (
                len(self.grid.cells) / reference_area_km2
                if reference_area_km2 > 0
                else 0
            ),
            "coverage_ratio": (
                total_cell_area / reference_area_km2 if reference_area_km2 > 0 else 0
            ),
            "reference_area_km2": reference_area_km2,
            "total_cell_area_km2": total_cell_area,
            "average_cell_area_km2": (
                total_cell_area / len(self.grid.cells) if self.grid.cells else 0
            ),
        }

    def resolution_analysis(self) -> dict[str, Any]:
        """
        Analyze resolution distribution and patterns.

        Returns:
            Dictionary with resolution analysis
        """
        if not self.grid.cells:
            return {}

        resolutions = [cell.resolution for cell in self.grid.cells]
        resolution_counts: dict[int, int] = {}
        resolution_areas: dict[int, float] = {}

        for cell in self.grid.cells:
            res = cell.resolution
            resolution_counts[res] = resolution_counts.get(res, 0) + 1
            resolution_areas[res] = resolution_areas.get(res, 0) + cell.area_km2

        return {
            "resolution_counts": resolution_counts,
            "resolution_areas_km2": resolution_areas,
            "min_resolution": min(resolutions),
            "max_resolution": max(resolutions),
            "resolution_range": max(resolutions) - min(resolutions),
            "dominant_resolution": max(resolution_counts.items(), key=lambda x: x[1])[
                0
            ],
            "resolution_diversity": len(set(resolutions)),
        }

    def spatial_distribution(self) -> dict[str, Any]:
        """
        Analyze spatial distribution patterns.

        Returns:
            Dictionary with spatial distribution metrics
        """
        if not self.grid.cells:
            return {}

        # Calculate centroid distances
        center_lat, center_lng = self.grid.center()
        distances = []

        for cell in self.grid.cells:
            # Simple Euclidean distance (not geodesic)
            dist = np.sqrt(
                (cell.latitude - center_lat) ** 2 + (cell.longitude - center_lng) ** 2
            )
            distances.append(dist)

        # Calculate spatial spread metrics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        return {
            "center_coordinates": (center_lat, center_lng),
            "mean_distance_from_center": mean_dist,
            "std_distance_from_center": std_dist,
            "max_distance_from_center": max(distances) if distances else 0,
            "spatial_compactness": std_dist / mean_dist if mean_dist > 0 else 0,
            "bounding_box_area_deg2": (self.grid.bounds()[2] - self.grid.bounds()[0])
            * (self.grid.bounds()[3] - self.grid.bounds()[1]),
        }

    def generate_report(self) -> dict[str, Any]:
        """
        Generate comprehensive analytics report.

        Returns:
            Complete analytics report
        """
        return {
            "grid_info": {
                "name": self.grid.name,
                "created_at": self.grid.created_at.isoformat(),
                "analysis_timestamp": datetime.now(UTC).isoformat(),
            },
            "basic_statistics": self.basic_statistics(),
            "connectivity_analysis": self.connectivity_analysis(),
            "density_analysis": self.density_analysis(),
            "resolution_analysis": self.resolution_analysis(),
            "spatial_distribution": self.spatial_distribution(),
        }


class H3Visualizer:
    """
    Visualization utilities for H3 grids and analytics.

    Provides methods for creating static and interactive visualizations
    of H3 hexagonal grids with various styling and analysis overlays.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize visualizer for an H3Grid.

        Args:
            grid: H3Grid instance to visualize
        """
        self.grid = grid

    def create_folium_map(self, **kwargs: Any) -> Any:
        """
        Create interactive Folium map of the H3 grid.

        Args:
            **kwargs: Additional arguments for map styling

        Returns:
            Folium map object
        """
        try:
            import folium
        except ImportError as exc:
            raise ImportError(
                "folium package required for interactive maps. Install with 'uv pip install folium'"
            ) from exc

        if not self.grid.cells:
            # Create empty map
            return folium.Map(location=[0, 0], zoom_start=2)

        # Get grid center and bounds
        center_lat, center_lng = self.grid.center()
        bounds = self.grid.bounds()

        # Create map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=kwargs.get("zoom_start", 10),
            tiles=kwargs.get("tiles", "OpenStreetMap"),
        )

        # Add cells to map
        for cell in self.grid.cells:
            # Create polygon from boundary
            boundary_coords = [[lat, lng] for lat, lng in cell.boundary]

            # Cell styling
            cell_color = kwargs.get("cell_color", "blue")
            cell_opacity = kwargs.get("cell_opacity", 0.6)
            cell_weight = kwargs.get("cell_weight", 2)

            # Create popup with cell information
            popup_html = f"""
            <b>H3 Cell Information</b><br>
            Index: {cell.index}<br>
            Resolution: {cell.resolution}<br>
            Coordinates: ({cell.latitude:.6f}, {cell.longitude:.6f})<br>
            Area: {cell.area_km2:.6f} km²<br>
            """

            # Add custom properties to popup
            if cell.properties:
                popup_html += "<br><b>Properties:</b><br>"
                for key, value in cell.properties.items():
                    popup_html += f"{key}: {value}<br>"

            folium.Polygon(
                locations=boundary_coords,
                color=cell_color,
                weight=cell_weight,
                opacity=cell_opacity,
                fillOpacity=kwargs.get("fill_opacity", 0.3),
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"H3: {cell.index}",
            ).add_to(m)

        # Fit map to bounds
        if len(self.grid.cells) > 1:
            m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])

        return m

    def save_geojson(self, filepath: str) -> None:
        """
        Save grid as GeoJSON file.

        Args:
            filepath: Output file path
        """
        geojson_data = self.grid.to_geojson()

        with open(filepath, "w") as f:
            json.dump(geojson_data, f, indent=2)

        logger.info(f"H3Grid saved as GeoJSON: {filepath}")


class H3Validator:
    """
    Validation utilities for H3 operations and data integrity.

    Provides comprehensive validation for H3 indices, coordinates,
    and grid operations to ensure data quality and correctness.
    """

    @staticmethod
    def validate_h3_index(h3_index: str) -> dict[str, Any]:
        """
        Validate H3 index format and properties.

        Args:
            h3_index: H3 cell index to validate

        Returns:
            Validation result dictionary
        """
        result: dict[str, Any] = {
            "valid": False,
            "index": h3_index,
            "errors": [],
            "warnings": [],
            "properties": {},
        }

        try:
            # Check if index is valid
            if not h3.is_valid_cell(h3_index):
                result["errors"].append("Invalid H3 index format")
                return result

            # Get properties
            resolution = h3.get_resolution(h3_index)
            lat, lng = h3.cell_to_latlng(h3_index)
            area = h3.cell_area(h3_index, "km^2")

            result["valid"] = True
            result["properties"] = {
                "resolution": resolution,
                "latitude": lat,
                "longitude": lng,
                "area_km2": area,
            }

            # Add warnings for edge cases
            if resolution < 0 or resolution > 15:
                result["warnings"].append(f"Unusual resolution: {resolution}")

            if abs(lat) > 90:
                result["warnings"].append(f"Invalid latitude: {lat}")

            if abs(lng) > 180:
                result["warnings"].append(f"Invalid longitude: {lng}")

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")

        return result

    @staticmethod
    def validate_coordinates(lat: float, lng: float) -> dict[str, Any]:
        """
        Validate latitude/longitude coordinates.

        Args:
            lat: Latitude in degrees
            lng: Longitude in degrees

        Returns:
            Validation result dictionary
        """
        result: dict[str, Any] = {
            "valid": True,
            "latitude": lat,
            "longitude": lng,
            "errors": [],
            "warnings": [],
        }

        # Check latitude bounds
        if not -90 <= lat <= 90:
            result["valid"] = False
            result["errors"].append(f"Latitude {lat} out of bounds [-90, 90]")

        # Check longitude bounds
        if not -180 <= lng <= 180:
            result["valid"] = False
            result["errors"].append(f"Longitude {lng} out of bounds [-180, 180]")

        # Add warnings for extreme coordinates
        if abs(lat) > 85:
            result["warnings"].append(f"Extreme latitude: {lat}")

        return result

    @staticmethod
    def validate_resolution(resolution: int) -> dict[str, Any]:
        """
        Validate H3 resolution parameter.

        Args:
            resolution: H3 resolution (0-15)

        Returns:
            Validation result dictionary
        """
        result: dict[str, Any] = {
            "valid": True,
            "resolution": resolution,
            "errors": [],
            "warnings": [],
        }

        if not isinstance(resolution, int):
            result["valid"] = False  # type: ignore[unreachable]
            result["errors"].append(
                f"Resolution must be integer, got {type(resolution)}"
            )
            return result

        if not 0 <= resolution <= 15:
            result["valid"] = False
            result["errors"].append(f"Resolution {resolution} out of bounds [0, 15]")

        # Add performance warnings
        if resolution > 12:
            result["warnings"].append(
                f"High resolution {resolution} may impact performance"
            )

        return result

    @classmethod
    def validate_grid(cls, grid: H3Grid) -> dict[str, Any]:
        """
        Validate entire H3Grid for consistency and integrity.

        Args:
            grid: H3Grid instance to validate

        Returns:
            Comprehensive validation report
        """
        result: dict[str, Any] = {
            "valid": True,
            "grid_name": grid.name,
            "cell_count": len(grid.cells),
            "errors": [],
            "warnings": [],
            "cell_validations": {},
            "statistics": {},
        }

        if not grid.cells:
            result["warnings"].append("Grid contains no cells")
            return result

        # Validate individual cells
        invalid_cells = 0
        resolutions = set()

        for i, cell in enumerate(grid.cells):
            cell_validation = cls.validate_h3_index(cell.index)
            result["cell_validations"][cell.index] = cell_validation

            if not cell_validation["valid"]:
                invalid_cells += 1
                result["errors"].extend(
                    [f"Cell {i}: {error}" for error in cell_validation["errors"]]
                )
            else:
                resolutions.add(cell_validation["properties"]["resolution"])

        # Overall validation
        if invalid_cells > 0:
            result["valid"] = False
            result["errors"].append(f"{invalid_cells} invalid cells found")

        # Statistics
        result["statistics"] = {
            "invalid_cells": invalid_cells,
            "valid_cells": len(grid.cells) - invalid_cells,
            "unique_resolutions": len(resolutions),
            "resolutions": list(resolutions),
            "total_area_km2": grid.total_area(),
        }

        # Performance warnings
        if len(grid.cells) > 10000:
            result["warnings"].append(
                f"Large grid ({len(grid.cells)} cells) may impact performance"
            )

        if len(resolutions) > 5:
            result["warnings"].append(
                f"Mixed resolutions ({len(resolutions)}) may complicate analysis"
            )

        return result
