"""
Boundary Detection for Nested Systems.

This module provides sophisticated boundary detection algorithms for identifying
and classifying boundaries in nested geospatial systems.
"""

import logging
import math
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from enum import Enum
from collections import defaultdict

import numpy as np
import h3

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.nested_grid import NestedH3Grid, NestedSystem


class BoundaryType(Enum):
    """Types of boundaries in nested systems."""

    EXTERNAL = "external"  # System boundary with external environment
    INTERNAL = "internal"  # Boundary between subsystems
    INTERFACE = "interface"  # Interface between different system types
    GRADIENT = "gradient"  # Gradual transition boundary
    SHARP = "sharp"  # Sharp discontinuous boundary
    PERMEABLE = "permeable"  # Allows flow/exchange
    IMPERMEABLE = "impermeable"  # Blocks flow/exchange
    DYNAMIC = "dynamic"  # Changes over time
    STATIC = "static"  # Fixed boundary


class BoundaryDetectionMethod(Enum):
    """Methods for boundary detection."""

    NEIGHBOR_ANALYSIS = "neighbor_analysis"
    GRADIENT_DETECTION = "gradient_detection"
    CLUSTERING = "clustering"
    EDGE_DETECTION = "edge_detection"
    TOPOLOGICAL = "topological"
    STATISTICAL = "statistical"


@dataclass
class BoundarySegment:
    """
    Represents a segment of a boundary.
    """

    segment_id: str
    cell_indices: list[str]
    boundary_type: BoundaryType
    strength: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    # Geometric properties
    length: float = 0.0
    curvature: float = 0.0
    orientation: float = 0.0

    # Connectivity
    connected_segments: set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Calculate geometric properties after creation."""
        if self.cell_indices:
            self._calculate_geometric_properties()

    def _calculate_geometric_properties(self) -> None:
        """Calculate length, curvature, and orientation."""
        if len(self.cell_indices) < 2:
            return

        try:
            # Get coordinates of boundary cells
            coordinates = []
            for cell_index in self.cell_indices:
                lat, lng = h3.cell_to_latlng(cell_index)
                coordinates.append((lat, lng))

            # Calculate length (sum of distances between consecutive points)
            total_length = 0.0
            for i in range(len(coordinates) - 1):
                lat1, lng1 = coordinates[i]
                lat2, lng2 = coordinates[i + 1]

                # Haversine distance
                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(lat1))
                    * math.cos(math.radians(lat2))
                    * math.sin(dlng / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))
                distance = 6371 * c  # Earth radius in km
                total_length += distance

            self.length = total_length

            # Calculate average orientation
            if len(coordinates) >= 2:
                orientations = []
                for i in range(len(coordinates) - 1):
                    lat1, lng1 = coordinates[i]
                    lat2, lng2 = coordinates[i + 1]

                    # Calculate bearing
                    dlng = math.radians(lng2 - lng1)
                    lat1_rad = math.radians(lat1)
                    lat2_rad = math.radians(lat2)

                    y = math.sin(dlng) * math.cos(lat2_rad)
                    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(
                        lat1_rad
                    ) * math.cos(lat2_rad) * math.cos(dlng)

                    bearing = math.atan2(y, x)
                    orientations.append(bearing)

                if orientations:
                    # Circular mean for angles
                    sin_sum = sum(math.sin(angle) for angle in orientations)
                    cos_sum = sum(math.cos(angle) for angle in orientations)
                    self.orientation = math.atan2(sin_sum, cos_sum)

            # Simple curvature estimation
            if len(coordinates) >= 3:
                curvatures = []
                for i in range(1, len(coordinates) - 1):
                    # Calculate angle change
                    prev_coord = coordinates[i - 1]
                    curr_coord = coordinates[i]
                    next_coord = coordinates[i + 1]

                    # Vectors
                    v1 = (curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
                    v2 = (next_coord[0] - curr_coord[0], next_coord[1] - curr_coord[1])

                    # Angle between vectors
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                        angle = math.acos(cos_angle)
                        curvatures.append(angle)

                if curvatures:
                    self.curvature = sum(curvatures) / len(curvatures)

        except Exception as e:
            logger.warning(f"Failed to calculate geometric properties: {e}")


class BoundaryDetector:
    """
    Advanced boundary detection for nested geospatial systems.

    Provides multiple algorithms for detecting and classifying boundaries
    in H3 hexagonal grids with support for different boundary types
    and detection methods.
    """

    def __init__(self, name: str = "BoundaryDetector") -> None:
        """
        Initialize boundary detector.

        Args:
            name: Detector name for identification
        """
        self.name = name

        # Detection parameters
        self.detection_methods: dict[str, dict[str, Any]] = {
            "neighbor_analysis": {"threshold": 0.5, "min_boundary_length": 3},
            "gradient_detection": {"gradient_threshold": 0.3, "smoothing_radius": 1},
            "clustering": {"min_cluster_size": 5, "boundary_width": 2},
        }

        # Results storage
        self.detected_boundaries: dict[str, list[BoundarySegment]] = {}
        self.boundary_statistics: dict[str, dict[str, Any]] = {}

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def detect_boundaries(
        self,
        nested_grid: "NestedH3Grid",
        method: BoundaryDetectionMethod = (BoundaryDetectionMethod.NEIGHBOR_ANALYSIS),
        system_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[BoundarySegment]]:
        """
        Detect boundaries in nested systems.

        Args:
            nested_grid: NestedH3Grid instance
            method: Detection method to use
            system_ids: Specific systems to analyze (None for all)
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping system IDs to boundary segments
        """
        if system_ids is None:
            system_ids = list(nested_grid.systems.keys())

        detected_boundaries = {}

        for system_id in system_ids:
            system = nested_grid.get_system_by_id(system_id)
            if not system:
                continue

            if method == BoundaryDetectionMethod.NEIGHBOR_ANALYSIS:
                boundaries = self._detect_neighbor_boundaries(system, **kwargs)
            elif method == BoundaryDetectionMethod.GRADIENT_DETECTION:
                boundaries = self._detect_gradient_boundaries(system, **kwargs)
            elif method == BoundaryDetectionMethod.CLUSTERING:
                boundaries = self._detect_clustering_boundaries(system, **kwargs)
            elif method == BoundaryDetectionMethod.EDGE_DETECTION:
                boundaries = self._detect_edge_boundaries(system, **kwargs)
            elif method == BoundaryDetectionMethod.TOPOLOGICAL:
                boundaries = self._detect_topological_boundaries(system, **kwargs)
            elif method == BoundaryDetectionMethod.STATISTICAL:
                boundaries = self._detect_statistical_boundaries(system, **kwargs)
            else:
                logger.warning(  # type: ignore[unreachable]
                    f"Unknown detection method: {method}"
                )
                boundaries = []

            detected_boundaries[system_id] = boundaries

        self.detected_boundaries.update(detected_boundaries)
        self._calculate_boundary_statistics()
        self.updated_at = datetime.now(UTC)

        return detected_boundaries

    def _detect_neighbor_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries based on neighbor analysis."""
        threshold = kwargs.get(
            "threshold", self.detection_methods["neighbor_analysis"]["threshold"]
        )
        min_length = kwargs.get(
            "min_boundary_length",
            self.detection_methods["neighbor_analysis"]["min_boundary_length"],
        )

        boundary_cells = []

        # Find cells with external neighbors
        for cell_index, cell in system.cells.items():
            external_neighbors = 0
            total_neighbors = len(cell.neighbor_cells)

            for neighbor_idx in cell.neighbor_cells:
                if neighbor_idx not in system.cells:
                    external_neighbors += 1

            # Calculate boundary strength
            if total_neighbors > 0:
                boundary_strength = external_neighbors / total_neighbors
                if boundary_strength >= threshold:
                    boundary_cells.append((cell_index, boundary_strength))

        # Group connected boundary cells into segments
        segments = self._group_boundary_cells(boundary_cells, system)

        # Filter by minimum length
        filtered_segments = []
        for segment in segments:
            if len(segment.cell_indices) >= min_length:
                segment.boundary_type = BoundaryType.EXTERNAL
                filtered_segments.append(segment)

        return filtered_segments

    def _detect_gradient_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries based on gradient analysis."""
        gradient_threshold = kwargs.get(
            "gradient_threshold",
            self.detection_methods["gradient_detection"]["gradient_threshold"],
        )
        value_field = kwargs.get("value_field", "value")

        boundary_cells = []

        # Calculate gradients for cells with the specified value field
        for cell_index, cell in system.cells.items():
            if value_field not in cell.state_variables:
                continue

            cell_value = cell.state_variables[value_field]
            neighbor_values = []

            for neighbor_idx in cell.neighbor_cells:
                if neighbor_idx in system.cells:
                    neighbor_cell = system.cells[neighbor_idx]
                    if value_field in neighbor_cell.state_variables:
                        neighbor_values.append(
                            neighbor_cell.state_variables[value_field]
                        )

            if neighbor_values:
                # Calculate gradient magnitude
                gradient_magnitude = np.std(neighbor_values + [cell_value])

                if gradient_magnitude >= gradient_threshold:
                    boundary_cells.append((cell_index, gradient_magnitude))

        # Group into segments
        segments = self._group_boundary_cells(boundary_cells, system)

        for segment in segments:
            segment.boundary_type = BoundaryType.GRADIENT

        return segments

    def _detect_clustering_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries based on clustering analysis."""
        value_field = kwargs.get("value_field", "cluster_id")

        boundary_cells = []

        # Find cells at cluster boundaries
        for cell_index, cell in system.cells.items():
            if value_field not in cell.state_variables:
                continue

            cell_cluster = cell.state_variables[value_field]
            different_neighbors = 0

            for neighbor_idx in cell.neighbor_cells:
                if neighbor_idx in system.cells:
                    neighbor_cell = system.cells[neighbor_idx]
                    if (
                        value_field in neighbor_cell.state_variables
                        and neighbor_cell.state_variables[value_field] != cell_cluster
                    ):
                        different_neighbors += 1

            if different_neighbors > 0:
                boundary_strength = different_neighbors / len(cell.neighbor_cells)
                boundary_cells.append((cell_index, boundary_strength))

        # Group into segments
        segments = self._group_boundary_cells(boundary_cells, system)

        for segment in segments:
            segment.boundary_type = BoundaryType.INTERNAL

        return segments

    def _detect_edge_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries for the EDGE method.

        On the hexagonal H3 lattice, edge detection is the neighbour-difference
        test, so this delegates to ``_detect_neighbor_boundaries``.
        """
        return self._detect_neighbor_boundaries(system, **kwargs)

    def _detect_topological_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries based on topological analysis."""
        # Analyze topological features like holes, islands, etc.
        boundary_cells = []

        # Find topological irregularities
        for cell_index, cell in system.cells.items():
            # Count connected components in neighborhood
            neighbor_in_system = [n for n in cell.neighbor_cells if n in system.cells]

            if len(neighbor_in_system) != len(cell.neighbor_cells):
                # Cell has external connections - potential boundary
                boundary_strength = 1.0 - (
                    len(neighbor_in_system) / len(cell.neighbor_cells)
                )
                boundary_cells.append((cell_index, boundary_strength))

        segments = self._group_boundary_cells(boundary_cells, system)

        for segment in segments:
            segment.boundary_type = (
                BoundaryType.TOPOLOGICAL  # type: ignore[attr-defined]
            )

        return segments

    def _detect_statistical_boundaries(
        self, system: "NestedSystem", **kwargs: Any
    ) -> list[BoundarySegment]:
        """Detect boundaries using statistical methods."""
        value_field = kwargs.get("value_field", "value")
        z_threshold = kwargs.get("z_threshold", 2.0)

        # Collect values
        values = []
        cell_indices = []

        for cell_index, cell in system.cells.items():
            if value_field in cell.state_variables:
                values.append(cell.state_variables[value_field])
                cell_indices.append(cell_index)

        if len(values) < 3:
            return []

        values = np.array(values)  # type: ignore[assignment]
        mean_val = np.mean(values)
        std_val = np.std(values)

        boundary_cells = []

        # Find statistical outliers
        for i, cell_index in enumerate(cell_indices):
            z_score = abs((values[i] - mean_val) / std_val) if std_val > 0 else 0

            if z_score >= z_threshold:
                boundary_cells.append((cell_index, z_score / z_threshold))

        segments = self._group_boundary_cells(boundary_cells, system)

        for segment in segments:
            segment.boundary_type = (
                BoundaryType.STATISTICAL  # type: ignore[attr-defined]
            )

        return segments

    def _group_boundary_cells(
        self,
        boundary_cells: list[tuple[str, float]],
        system: "NestedSystem",
    ) -> list[BoundarySegment]:
        """Group connected boundary cells into segments."""
        if not boundary_cells:
            return []

        # Create adjacency map
        cell_strengths = {cell_idx: strength for cell_idx, strength in boundary_cells}
        boundary_indices = set(cell_strengths.keys())

        # Find connected components
        visited = set()
        segments = []
        segment_counter = 0

        for cell_idx in boundary_indices:
            if cell_idx in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [cell_idx]
            visited.add(cell_idx)

            while queue:
                current = queue.pop(0)
                component.append(current)

                # Check neighbors
                if current in system.cells:
                    cell = system.cells[current]
                    for neighbor_idx in cell.neighbor_cells:
                        if (
                            neighbor_idx in boundary_indices
                            and neighbor_idx not in visited
                        ):
                            queue.append(neighbor_idx)
                            visited.add(neighbor_idx)

            # Create segment
            if component:
                avg_strength = sum(cell_strengths[idx] for idx in component) / len(
                    component
                )

                segment = BoundarySegment(
                    segment_id=f"{system.system_id}_boundary_{segment_counter}",
                    cell_indices=component,
                    boundary_type=BoundaryType.EXTERNAL,
                    strength=avg_strength,
                )

                segments.append(segment)
                segment_counter += 1

        return segments

    def _calculate_boundary_statistics(self) -> None:
        """Calculate statistics for detected boundaries."""
        self.boundary_statistics.clear()

        for system_id, segments in self.detected_boundaries.items():
            if not segments:
                continue

            # Calculate segment statistics
            segment_lengths = [seg.length for seg in segments]
            segment_strengths = [seg.strength for seg in segments]

            stats: dict[str, Any] = {
                "num_segments": len(segments),
                "total_boundary_length": sum(segment_lengths),
                "average_segment_length": sum(segment_lengths) / len(segment_lengths)
                if segment_lengths
                else 0,
                "average_boundary_strength": sum(segment_strengths)
                / len(segment_strengths)
                if segment_strengths
                else 0,
                "boundary_types": {},
            }

            # Count boundary types
            type_counts: dict[str, int] = defaultdict(int)
            for segment in segments:
                type_counts[segment.boundary_type.value] += 1

            stats["boundary_types"] = dict(type_counts)

            self.boundary_statistics[system_id] = stats

    def get_boundary_summary(self) -> dict[str, Any]:
        """Get summary of boundary detection results."""
        total_segments = sum(
            len(segments) for segments in self.detected_boundaries.values()
        )
        total_length = sum(
            sum(seg.length for seg in segments)
            for segments in self.detected_boundaries.values()
        )

        return {
            "detector_name": self.name,
            "total_systems_analyzed": len(self.detected_boundaries),
            "total_boundary_segments": total_segments,
            "total_boundary_length": total_length,
            "detection_methods_used": list(self.detection_methods.keys()),
            "boundary_statistics": self.boundary_statistics,
            "updated_at": self.updated_at.isoformat(),
        }
