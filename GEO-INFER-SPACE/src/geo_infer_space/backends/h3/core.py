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
from dataclasses import dataclass, field
from datetime import datetime
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import h3

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3-py package not available. Install with 'uv pip install h3'")


@dataclass
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
    boundary: list[tuple[float, float]] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Initialize cell properties after creation."""
        if H3_AVAILABLE and self.index:
            try:
                # Get coordinates
                self.latitude, self.longitude = h3.cell_to_latlng(self.index)

                # Get area
                self.area_km2 = h3.cell_area(self.index, "km^2")

                # Get boundary
                self.boundary = list(h3.cell_to_boundary(self.index))

                # Validate resolution
                actual_resolution = h3.get_resolution(self.index)
                if self.resolution != actual_resolution:
                    logger.warning(
                        f"Resolution mismatch: expected {self.resolution}, got {actual_resolution}"
                    )
                    self.resolution = actual_resolution

            except Exception as e:
                logger.error(f"Failed to initialize H3Cell {self.index}: {e}")

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
        if not H3_AVAILABLE:
            raise ImportError("h3-py package required for H3Cell operations")

        index = h3.latlng_to_cell(lat, lng, resolution)
        return cls(
            index=index,
            resolution=resolution,
            latitude=lat,
            longitude=lng,
            properties=properties,
        )

    def neighbors(self, k: int = 1) -> list["H3Cell"]:
        """
        Get neighboring cells within k distance.

        Args:
            k: Distance (number of rings) for neighbors

        Returns:
            List of neighboring H3Cell instances
        """
        if not H3_AVAILABLE:
            return []

        try:
            neighbor_indices = h3.grid_disk(self.index, k)
            neighbors = []

            for neighbor_index in neighbor_indices:
                if neighbor_index != self.index:  # Exclude self
                    neighbor = H3Cell(index=neighbor_index, resolution=self.resolution)
                    neighbors.append(neighbor)

            return neighbors

        except Exception as e:
            logger.error(f"Failed to get neighbors for {self.index}: {e}")
            return []

    def parent(self, parent_resolution: int | None = None) -> Optional["H3Cell"]:
        """
        Get parent cell at coarser resolution.

        Args:
            parent_resolution: Target parent resolution (must be < current resolution)

        Returns:
            Parent H3Cell or None if invalid
        """
        if not H3_AVAILABLE:
            return None

        if parent_resolution is None:
            parent_resolution = max(0, self.resolution - 1)

        if parent_resolution >= self.resolution:
            logger.warning(
                f"Parent resolution {parent_resolution} must be < current resolution {self.resolution}"
            )
            return None

        try:
            parent_index = h3.cell_to_parent(self.index, parent_resolution)
            return H3Cell(index=parent_index, resolution=parent_resolution)
        except Exception as e:
            logger.error(f"Failed to get parent for {self.index}: {e}")
            return None

    def children(self, child_resolution: int | None = None) -> list["H3Cell"]:
        """
        Get child cells at finer resolution.

        Args:
            child_resolution: Target child resolution (must be > current resolution)

        Returns:
            List of child H3Cell instances
        """
        if not H3_AVAILABLE:
            return []

        if child_resolution is None:
            child_resolution = min(15, self.resolution + 1)

        if child_resolution <= self.resolution:
            logger.warning(
                f"Child resolution {child_resolution} must be > current resolution {self.resolution}"
            )
            return []

        try:
            child_indices = h3.cell_to_children(self.index, child_resolution)
            children = []

            for child_index in child_indices:
                child = H3Cell(index=child_index, resolution=child_resolution)
                children.append(child)

            return children

        except Exception as e:
            logger.error(f"Failed to get children for {self.index}: {e}")
            return []

    def distance_to(self, other: "H3Cell") -> int:
        """
        Calculate grid distance to another cell.

        Args:
            other: Another H3Cell instance

        Returns:
            Grid distance (number of cells)
        """
        if not H3_AVAILABLE:
            return -1

        if self.resolution != other.resolution:
            logger.warning(
                "Distance calculation between different resolutions may be inaccurate"
            )

        try:
            return cast(int, h3.grid_distance(self.index, other.index))
        except Exception as e:
            logger.error(
                f"Failed to calculate distance between {self.index} and {other.index}: {e}"
            )
            return -1

    def is_neighbor(self, other: "H3Cell") -> bool:
        """
        Check if another cell is a direct neighbor.

        Args:
            other: Another H3Cell instance

        Returns:
            True if cells are neighbors
        """
        if not H3_AVAILABLE:
            return False

        try:
            return cast(bool, h3.are_neighbor_cells(self.index, other.index))
        except Exception as e:
            logger.error(f"Failed to check neighbor relationship: {e}")
            return False

    def to_geojson(self) -> dict[str, Any]:
        """
        Convert cell to GeoJSON feature.

        Returns:
            GeoJSON feature dictionary
        """
        # Ensure boundary is closed (first point = last point)
        boundary_coords = self.boundary.copy()
        if boundary_coords and boundary_coords[0] != boundary_coords[-1]:
            boundary_coords.append(boundary_coords[0])

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in boundary_coords]],
            },
            "properties": {
                "h3_index": self.index,
                "resolution": self.resolution,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "area_km2": self.area_km2,
                "created_at": self.created_at.isoformat(),
                **self.properties,
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
        self.cells: list[H3Cell] = cells or []
        self.name = name
        self.created_at = datetime.now()
        self._cell_index: dict[str, H3Cell] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build internal index for fast cell lookup."""
        self._cell_index = {cell.index: cell for cell in self.cells}

    def add_cell(self, cell: H3Cell) -> None:
        """Add a cell to the grid."""
        if cell.index not in self._cell_index:
            self.cells.append(cell)
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
            self.cells.remove(cell)
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
        """
        Create H3Grid from polygon coordinates.

        Args:
            polygon_coords: List of (lat, lng) coordinate pairs
            resolution: H3 resolution
            name: Grid name

        Returns:
            H3Grid instance covering the polygon
        """
        if not H3_AVAILABLE:
            raise ImportError("h3-py package required for H3Grid operations")

        # Convert to GeoJSON format (lng, lat order)
        geojson_coords = [[lng, lat] for lat, lng in polygon_coords]

        # Ensure polygon is closed
        if geojson_coords[0] != geojson_coords[-1]:
            geojson_coords.append(geojson_coords[0])

        polygon_geojson = {"type": "Polygon", "coordinates": [geojson_coords]}

        try:
            cell_indices = h3.geo_to_cells(polygon_geojson, resolution)
            cells = []

            for cell_index in cell_indices:
                cell = H3Cell(index=cell_index, resolution=resolution)
                cells.append(cell)

            return cls(cells=cells, name=name)

        except Exception as e:
            logger.error(f"Failed to create H3Grid from polygon: {e}")
            return cls(name=name)

    @classmethod
    def from_center(
        cls,
        lat: float,
        lng: float,
        resolution: int,
        k: int = 1,
        name: str = "CenterGrid",
    ) -> "H3Grid":
        """
        Create H3Grid centered on coordinates with a grid-disk neighborhood.

        Args:
            lat: Center latitude
            lng: Center longitude
            resolution: H3 resolution
            k: Ring distance
            name: Grid name

        Returns:
            H3Grid instance
        """
        if not H3_AVAILABLE:
            raise ImportError("h3-py package required for H3Grid operations")

        try:
            center_index = h3.latlng_to_cell(lat, lng, resolution)
            cell_indices = h3.grid_disk(center_index, k)
            cells = []

            for cell_index in cell_indices:
                cell = H3Cell(index=cell_index, resolution=resolution)
                cells.append(cell)

            return cls(cells=cells, name=name)

        except Exception as e:
            logger.error(f"Failed to create H3Grid from center: {e}")
            return cls(name=name)

    def compact(self) -> "H3Grid":
        """
        Compact cells to mixed resolutions for efficiency.

        Returns:
            New H3Grid with compacted cells
        """
        if not H3_AVAILABLE or not self.cells:
            return H3Grid(name=f"{self.name}_compacted")

        try:
            cell_indices = [cell.index for cell in self.cells]
            compacted_indices = h3.compact_cells(cell_indices)

            compacted_cells = []
            for cell_index in compacted_indices:
                resolution = h3.get_resolution(cell_index)
                cell = H3Cell(index=cell_index, resolution=resolution)
                compacted_cells.append(cell)

            return H3Grid(cells=compacted_cells, name=f"{self.name}_compacted")

        except Exception as e:
            logger.error(f"Failed to compact H3Grid: {e}")
            return H3Grid(name=f"{self.name}_compacted")

    def uncompact(self, target_resolution: int) -> "H3Grid":
        """
        Uncompact cells to uniform resolution.

        Args:
            target_resolution: Target resolution for all cells

        Returns:
            New H3Grid with uniform resolution
        """
        if not H3_AVAILABLE or not self.cells:
            return H3Grid(name=f"{self.name}_uncompacted")

        try:
            cell_indices = [cell.index for cell in self.cells]
            uncompacted_indices = h3.uncompact_cells(cell_indices, target_resolution)

            uncompacted_cells = []
            for cell_index in uncompacted_indices:
                cell = H3Cell(index=cell_index, resolution=target_resolution)
                uncompacted_cells.append(cell)

            return H3Grid(cells=uncompacted_cells, name=f"{self.name}_uncompacted")

        except Exception as e:
            logger.error(f"Failed to uncompact H3Grid: {e}")
            return H3Grid(name=f"{self.name}_uncompacted")

    def total_area(self) -> float:
        """
        Calculate total area of all cells in km².

        Returns:
            Total area in square kilometers
        """
        return sum(cell.area_km2 for cell in self.cells)

    def bounds(self) -> tuple[float, float, float, float]:
        """
        Get bounding box of all cells.

        Returns:
            (min_lat, min_lng, max_lat, max_lng)
        """
        if not self.cells:
            return (0.0, 0.0, 0.0, 0.0)

        lats = [cell.latitude for cell in self.cells]
        lngs = [cell.longitude for cell in self.cells]

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
                "h3_index": cell.index,
                "resolution": cell.resolution,
                "latitude": cell.latitude,
                "longitude": cell.longitude,
                "area_km2": cell.area_km2,
                "created_at": cell.created_at,
            }
            # Add custom properties
            row.update(cell.properties)
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
        if "basic_stats" in self.stats_cache:
            return cast(dict[str, Any], self.stats_cache["basic_stats"])

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
        if not H3_AVAILABLE or not self.grid.cells:
            return {}

        # Build adjacency information
        adjacency_count = 0
        isolated_cells = 0
        cell_neighbors = {}

        for cell in self.grid.cells:
            neighbors_in_grid = []

            # Get all neighbors of this cell
            try:
                neighbor_indices = h3.grid_disk(cell.index, 1)
                for neighbor_index in neighbor_indices:
                    if neighbor_index != cell.index and self.grid.has_cell(
                        neighbor_index
                    ):
                        neighbors_in_grid.append(neighbor_index)
                        adjacency_count += 1
            except Exception as e:
                logger.error(f"Failed to analyze connectivity for {cell.index}: {e}")
                continue

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
                "analysis_timestamp": datetime.now().isoformat(),
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
        except ImportError:
            raise ImportError(
                "folium package required for interactive maps. Install with 'uv pip install folium'"
            )

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

        if not H3_AVAILABLE:
            result["errors"].append("h3-py package not available")
            return result

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
