"""
Nested H3 Grid System.

This module provides the core classes for managing nested H3 hexagonal grids
with hierarchical relationships, boundary management, and system-level operations.
"""

import logging
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum
from collections import defaultdict

import h3
import numpy as np

from ...backends.h3.core import H3Grid as H3Grid, H3Cell
from ...core import SpatialIndexingInterface

logger = logging.getLogger(__name__)

# Create module-level interface for convenience
_spatial = SpatialIndexingInterface()


def grid_disk(cell_index: str, k: int = 1) -> list[str]:
    """Get cells within grid distance k using the unified interface."""
    return _spatial.get_cell_neighbors(cell_index, k)


def grid_distance(cell1: str, cell2: str) -> int:
    """Get grid distance using unified interface."""
    return _spatial.get_cell_distance(cell1, cell2)


def neighbor_cells(cell_index: str) -> list[str]:
    """Get immediate neighbors using unified interface."""
    return _spatial.get_cell_neighbors(cell_index, 1)


class NestedCellType(Enum):
    """Types of nested cells based on their role in the system."""

    CORE = "core"  # Interior cells
    BOUNDARY = "boundary"  # Cells on system boundaries
    INTERFACE = "interface"  # Cells at interfaces between subsystems
    BRIDGE = "bridge"  # Cells connecting different hierarchical levels
    ISOLATED = "isolated"  # Cells with no connections


class NestedSystemState(Enum):
    """States of nested systems."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRANSITIONING = "transitioning"
    MERGING = "merging"
    SPLITTING = "splitting"
    STABLE = "stable"


@dataclass
class NestedCell:
    """
    Enhanced H3 cell with nested system capabilities.

    Extends the basic H3Cell with hierarchical relationships,
    boundary information, and message passing capabilities.
    """

    # Core H3 properties
    h3_cell: H3Cell

    # Nested system properties
    system_id: str | None = None
    parent_cells: set[str] = field(default_factory=set)
    child_cells: set[str] = field(default_factory=set)
    neighbor_cells: set[str] = field(default_factory=set)

    # Cell classification
    cell_type: NestedCellType = NestedCellType.CORE
    hierarchy_level: int = 0

    # Boundary information
    is_boundary: bool = False
    boundary_ids: set[str] = field(default_factory=set)
    boundary_strength: float = 0.0

    # Message passing
    message_queue: list[Any] = field(default_factory=list)
    message_history: list[Any] = field(default_factory=list)

    # System dynamics
    state_variables: dict[str, Any] = field(default_factory=dict)
    flow_variables: dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Initialize nested cell after creation."""
        if self.h3_cell:
            # Initialize neighbor relationships
            self._update_neighbors()

            # Set initial state
            self.updated_at = datetime.now(UTC)

    @property
    def index(self) -> str:
        """Get H3 index of the cell."""
        return self.h3_cell.index if self.h3_cell else ""

    @property
    def resolution(self) -> int:
        """Get H3 resolution of the cell."""
        return self.h3_cell.resolution if self.h3_cell else 0

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get latitude, longitude coordinates."""
        if self.h3_cell:
            return (self.h3_cell.latitude, self.h3_cell.longitude)
        return (0.0, 0.0)

    @property
    def area_km2(self) -> float:
        """Get area of the cell in square kilometers."""
        if self.h3_cell and hasattr(self.h3_cell, "area_km2"):
            return self.h3_cell.area_km2
        return 1.0  # Default area when H3Cell area not available

    @property
    def latitude(self) -> float:
        """Get latitude of the cell."""
        if self.h3_cell and hasattr(self.h3_cell, "latitude"):
            return self.h3_cell.latitude
        return 0.0

    @property
    def longitude(self) -> float:
        """Get longitude of the cell."""
        if self.h3_cell and hasattr(self.h3_cell, "longitude"):
            return self.h3_cell.longitude
        return 0.0

    def _update_neighbors(self) -> None:
        """Update neighbor cell relationships."""
        if not self.h3_cell:
            return

        neighbors = neighbor_cells(self.h3_cell.index)
        self.neighbor_cells = set(neighbors)

    def add_parent(self, parent_index: str) -> None:
        """Add a parent cell relationship."""
        self.parent_cells.add(parent_index)
        self.updated_at = datetime.now(UTC)

    def add_child(self, child_index: str) -> None:
        """Add a child cell relationship."""
        self.child_cells.add(child_index)
        self.updated_at = datetime.now(UTC)

    def set_boundary(self, boundary_id: str, strength: float = 1.0) -> None:
        """Mark cell as boundary with specified strength."""
        self.is_boundary = True
        self.boundary_ids.add(boundary_id)
        self.boundary_strength = max(self.boundary_strength, strength)
        self.cell_type = NestedCellType.BOUNDARY
        self.updated_at = datetime.now(UTC)

    def add_message(self, message: Any) -> None:
        """Add message to the cell's queue."""
        self.message_queue.append(message)
        self.updated_at = datetime.now(UTC)

    def process_messages(self) -> list[Any]:
        """Process and return all queued messages."""
        messages = self.message_queue.copy()
        self.message_history.extend(messages)
        self.message_queue.clear()
        self.updated_at = datetime.now(UTC)
        return messages

    def update_state(self, variable: str, value: Any) -> None:
        """Update a state variable."""
        self.state_variables[variable] = value
        self.updated_at = datetime.now(UTC)

    def update_flow(self, variable: str, value: float) -> None:
        """Update a flow variable."""
        self.flow_variables[variable] = value
        self.updated_at = datetime.now(UTC)

    def get_connectivity_degree(self) -> int:
        """Get the connectivity degree (number of connections)."""
        return len(self.neighbor_cells) + len(self.parent_cells) + len(self.child_cells)

    def is_connected_to(self, other_index: str) -> bool:
        """Check if connected to another cell."""
        return (
            other_index in self.neighbor_cells
            or other_index in self.parent_cells
            or other_index in self.child_cells
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "index": self.index,
            "resolution": self.resolution,
            "coordinates": self.coordinates,
            "system_id": self.system_id,
            "parent_cells": list(self.parent_cells),
            "child_cells": list(self.child_cells),
            "neighbor_cells": list(self.neighbor_cells),
            "cell_type": self.cell_type.value,
            "hierarchy_level": self.hierarchy_level,
            "is_boundary": self.is_boundary,
            "boundary_ids": list(self.boundary_ids),
            "boundary_strength": self.boundary_strength,
            "state_variables": self.state_variables,
            "flow_variables": self.flow_variables,
            "connectivity_degree": self.get_connectivity_degree(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class NestedSystem:
    """
    A nested system representing a collection of connected H3 cells.

    Systems can contain subsystems and be part of larger systems,
    forming hierarchical nested structures.
    """

    def __init__(self, system_id: str, name: str = "", description: str = "") -> None:
        """
        Initialize nested system.

        Args:
            system_id: Unique system identifier
            name: Human-readable system name
            description: System description
        """
        self.system_id = system_id
        self.name = name or system_id
        self.description = description

        # System composition
        self.cells: dict[str, NestedCell] = {}
        self.subsystems: dict[str, NestedSystem] = {}
        self.parent_system: NestedSystem | None = None

        # System properties
        self.state = NestedSystemState.ACTIVE
        self.hierarchy_level = 0
        self.boundary_cells: set[str] = set()

        # System metrics
        self.total_area: float = 0.0
        self.center_coordinates: tuple[float, float] = (0.0, 0.0)
        self.bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

        # System dynamics
        self.system_variables: dict[str, Any] = {}
        self.flow_balance: dict[str, float] = {}

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def add_cell(self, cell: NestedCell) -> None:
        """Add a cell to the system."""
        cell.system_id = self.system_id
        cell.hierarchy_level = self.hierarchy_level
        self.cells[cell.index] = cell
        self._update_system_metrics()
        self.updated_at = datetime.now(UTC)

    def remove_cell(self, cell_index: str) -> bool:
        """Remove a cell from the system."""
        if cell_index in self.cells:
            del self.cells[cell_index]
            self.boundary_cells.discard(cell_index)
            self._update_system_metrics()
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def add_subsystem(self, subsystem: "NestedSystem") -> None:
        """Add a subsystem."""
        subsystem.parent_system = self
        subsystem.hierarchy_level = self.hierarchy_level + 1
        self.subsystems[subsystem.system_id] = subsystem
        self.updated_at = datetime.now(UTC)

    def remove_subsystem(self, system_id: str) -> bool:
        """Remove a subsystem."""
        if system_id in self.subsystems:
            self.subsystems[system_id].parent_system = None
            del self.subsystems[system_id]
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def get_all_cells(self, include_subsystems: bool = True) -> dict[str, NestedCell]:
        """Get all cells in the system and optionally subsystems."""
        all_cells = self.cells.copy()

        if include_subsystems:
            for subsystem in self.subsystems.values():
                all_cells.update(subsystem.get_all_cells(include_subsystems=True))

        return all_cells

    def get_boundary_cells(self) -> dict[str, NestedCell]:
        """Get all boundary cells in the system."""
        return {idx: cell for idx, cell in self.cells.items() if cell.is_boundary}

    def detect_boundaries(self, external_cells: set[str] | None = None) -> None:
        """Detect and mark boundary cells."""
        if not self.cells:
            return

        self.boundary_cells.clear()

        for cell_index, cell in self.cells.items():
            is_boundary = False

            # Check if cell has neighbors outside the system
            for neighbor_idx in cell.neighbor_cells:
                if neighbor_idx not in self.cells:
                    if external_cells is None or neighbor_idx in external_cells:
                        is_boundary = True
                        break

            if is_boundary:
                cell.set_boundary(f"{self.system_id}_boundary")
                self.boundary_cells.add(cell_index)

        self.updated_at = datetime.now(UTC)

    def calculate_connectivity(self) -> dict[str, Any]:
        """Calculate system connectivity metrics."""
        if not self.cells:
            return {"error": "No cells in system"}

        # Internal connections
        internal_connections = 0
        external_connections = 0

        for cell in self.cells.values():
            for neighbor_idx in cell.neighbor_cells:
                if neighbor_idx in self.cells:
                    internal_connections += 1
                else:
                    external_connections += 1

        # Avoid double counting internal connections
        internal_connections //= 2

        total_possible = len(self.cells) * 6  # Max 6 neighbors per hexagon
        connectivity_ratio = internal_connections / max(1, total_possible)

        return {
            "internal_connections": internal_connections,
            "external_connections": external_connections,
            "total_connections": internal_connections + external_connections,
            "connectivity_ratio": connectivity_ratio,
            "boundary_ratio": len(self.boundary_cells) / len(self.cells),
            "system_compactness": internal_connections / max(1, external_connections),
        }

    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        if not self.cells:
            return

        # Calculate total area
        self.total_area = sum(cell.area_km2 for cell in self.cells.values())

        # Calculate center coordinates
        if self.cells:
            lats = [cell.coordinates[0] for cell in self.cells.values()]
            lngs = [cell.coordinates[1] for cell in self.cells.values()]
            self.center_coordinates = (sum(lats) / len(lats), sum(lngs) / len(lngs))

            # Calculate bounding box
            self.bounding_box = (
                min(lngs),
                min(lats),  # min_lng, min_lat
                max(lngs),
                max(lats),  # max_lng, max_lat
            )

    def merge_with(self, other_system: "NestedSystem") -> "NestedSystem":
        """Merge with another system."""
        merged_system = NestedSystem(
            system_id=f"{self.system_id}_merged_{other_system.system_id}",
            name=f"{self.name} + {other_system.name}",
            description=f"Merged system: {self.description} | {other_system.description}",
        )

        # Add all cells from both systems
        for cell in self.cells.values():
            merged_system.add_cell(cell)

        for cell in other_system.cells.values():
            merged_system.add_cell(cell)

        # Add subsystems
        for subsystem in self.subsystems.values():
            merged_system.add_subsystem(subsystem)

        for subsystem in other_system.subsystems.values():
            merged_system.add_subsystem(subsystem)

        # Update state
        merged_system.state = NestedSystemState.MERGING

        return merged_system

    def split_by_criteria(
        self, criteria_func: Callable[[NestedCell], Any]
    ) -> list["NestedSystem"]:
        """Split system based on criteria function."""
        if not self.cells:
            return [self]

        # Group cells by criteria
        groups = defaultdict(list)
        for cell in self.cells.values():
            group_key = criteria_func(cell)
            groups[group_key].append(cell)

        # Create new systems for each group
        split_systems = []
        for i, (group_key, group_cells) in enumerate(groups.items()):
            new_system = NestedSystem(
                system_id=f"{self.system_id}_split_{i}",
                name=f"{self.name}_split_{i}",
                description=f"Split from {self.system_id} by criteria: {group_key}",
            )

            for cell in group_cells:
                new_system.add_cell(cell)

            new_system.state = NestedSystemState.SPLITTING
            split_systems.append(new_system)

        return split_systems

    def get_system_summary(self) -> dict[str, Any]:
        """Get comprehensive system summary."""
        connectivity = self.calculate_connectivity()

        return {
            "system_id": self.system_id,
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "hierarchy_level": self.hierarchy_level,
            "num_cells": len(self.cells),
            "num_subsystems": len(self.subsystems),
            "num_boundary_cells": len(self.boundary_cells),
            "total_area_km2": self.total_area,
            "center_coordinates": self.center_coordinates,
            "bounding_box": self.bounding_box,
            "connectivity_metrics": connectivity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class NestedH3Grid:
    """
    Advanced H3 grid with nested system capabilities.

    Manages multiple nested systems, hierarchical relationships,
    and provides comprehensive analysis and manipulation tools.
    """

    def __init__(self, name: str = "NestedH3Grid") -> None:
        """
        Initialize nested H3 grid.

        Args:
            name: Grid name for identification
        """
        self.name = name

        # Grid components
        self.cells: dict[str, NestedCell] = {}
        self.systems: dict[str, NestedSystem] = {}

        # Hierarchical structure
        self.hierarchy_levels: dict[int, set[str]] = defaultdict(set)
        self.root_systems: set[str] = set()

        # Grid properties
        self.resolutions: set[int] = set()
        self.total_area: float = 0.0
        self.bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def add_cell(
        self, h3_cell: H3Cell | NestedCell, system_id: str | None = None
    ) -> NestedCell:
        """Add an H3 cell or existing nested cell to the grid."""
        if isinstance(h3_cell, NestedCell):
            nested_cell = h3_cell
            if system_id is not None:
                nested_cell.system_id = system_id
        else:
            nested_cell = NestedCell(h3_cell=h3_cell, system_id=system_id)
        self.cells[nested_cell.index] = nested_cell
        self.resolutions.add(nested_cell.resolution)
        self._update_grid_metrics()
        self.updated_at = datetime.now(UTC)
        return nested_cell

    def build_h3_hierarchy_from_boundary(
        self,
        boundary: dict[str, Any],
        resolutions: list[int],
        system_prefix: str = "nested_h3",
    ) -> dict[str, Any]:
        """
        Build a deterministic nested H3 hierarchy from a GeoJSON-like boundary.

        The finest configured resolution is filled from the boundary and all
        coarser levels are derived from ``cell_to_parent`` so the hierarchy is
        closed by construction.
        """
        ordered = self._validate_h3_resolution_sequence(resolutions)
        finest = ordered[-1]
        cells: list[str] = []
        cells = list(_spatial.polygon_to_cells(boundary, finest))
        if not cells:
            cells = self._cells_from_boundary_vertices(boundary, finest)
        if not cells:
            raise ValueError("Boundary did not produce any H3 cells")
        return self.build_h3_hierarchy_from_cells(cells, ordered, system_prefix)

    def build_h3_hierarchy_from_cells(
        self,
        cells: list[str],
        resolutions: list[int],
        system_prefix: str = "nested_h3",
    ) -> dict[str, Any]:
        """
        Build a nested H3 hierarchy from root or leaf H3 cells.

        Input cells may be at any requested resolution. Coarser input cells are
        expanded to the finest requested resolution, finer input cells are
        reduced to the finest requested resolution, and cells at resolutions not
        in the configured sequence fail clearly.
        """
        ordered = self._validate_h3_resolution_sequence(resolutions)
        if not cells:
            raise ValueError("At least one H3 cell is required")

        finest = ordered[-1]
        finest_cells = self._normalize_cells_to_resolution(cells, finest, ordered)
        cells_by_resolution: dict[int, list[str]] = {}
        for resolution in ordered:
            if resolution == finest:
                level_cells = finest_cells
            else:
                level_cells = sorted(
                    {h3.cell_to_parent(cell, resolution) for cell in finest_cells}
                )
            cells_by_resolution[resolution] = level_cells
            self._register_h3_level_cells(level_cells, resolution, system_prefix)

        parent_child_map: dict[str, list[str]] = {}
        child_parent_map: dict[str, str] = {}
        for parent_res, child_res in zip(ordered, ordered[1:]):
            parents = set(cells_by_resolution[parent_res])
            for child in cells_by_resolution[child_res]:
                parent = h3.cell_to_parent(child, parent_res)
                if parent not in parents:
                    raise ValueError(
                        f"Parent {parent} for child {child} is missing at resolution {parent_res}"
                    )
                parent_child_map.setdefault(parent, []).append(child)
                child_parent_map[child] = parent

        parent_child_map = {
            parent: sorted(children)
            for parent, children in sorted(parent_child_map.items())
        }
        same_level_neighbors = self._build_same_level_neighbor_map(cells_by_resolution)
        level_edge_counts = {
            str(resolution): self._count_neighbor_edges(neighbors)
            for resolution, neighbors in same_level_neighbors.items()
        }

        for parent, children in parent_child_map.items():
            parent_cell = self.cells.get(parent)
            if parent_cell:
                for child in children:
                    parent_cell.add_child(child)
            for child in children:
                child_cell = self.cells.get(child)
                if child_cell:
                    child_cell.add_parent(parent)

        hierarchy = {
            "schema_version": "geo-infer-space-nested-h3-hierarchy/v1",
            "resolutions": ordered,
            "cells_by_resolution": {
                str(resolution): cells_by_resolution[resolution]
                for resolution in ordered
            },
            "parent_child_map": parent_child_map,
            "child_parent_map": dict(sorted(child_parent_map.items())),
            "same_level_neighbors": same_level_neighbors,
            "level_edge_counts": level_edge_counts,
            "root_cells": cells_by_resolution[ordered[0]],
            "leaf_cells": cells_by_resolution[finest],
            "system_prefix": system_prefix,
            "validation": {},
        }
        hierarchy["validation"] = self.validate_h3_hierarchy(hierarchy)
        self.h3_hierarchy = hierarchy
        self.updated_at = datetime.now(UTC)
        return hierarchy

    def validate_h3_hierarchy(
        self, hierarchy: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate H3 hierarchy closure, resolution consistency, and adjacency.

        Returns a JSON-compatible validation report instead of raising so the
        method can be used by contracts, tests, and generated diagnostics.
        """
        hierarchy = hierarchy or getattr(self, "h3_hierarchy", None)
        if not hierarchy:
            return {
                "is_valid": False,
                "issues": ["No H3 hierarchy is available"],
                "warnings": [],
                "orphan_count": 0,
                "parent_count": 0,
                "child_count": 0,
                "multi_child_parent": False,
                "validated_at": datetime.now(UTC).isoformat(),
            }

        issues: list[str] = []
        warnings: list[str] = []
        resolutions = [int(value) for value in hierarchy.get("resolutions", [])]
        try:
            ordered = self._validate_h3_resolution_sequence(resolutions)
        except ValueError as exc:
            issues.append(str(exc))
            ordered = resolutions

        cells_by_resolution = {
            int(resolution): list(cells)
            for resolution, cells in hierarchy.get("cells_by_resolution", {}).items()
        }
        parent_child_map = {
            str(parent): [str(child) for child in children]
            for parent, children in hierarchy.get("parent_child_map", {}).items()
        }
        child_parent_map = {
            str(child): str(parent)
            for child, parent in hierarchy.get("child_parent_map", {}).items()
        }

        all_cells: set[str] = set()
        for resolution in ordered:
            for cell in cells_by_resolution.get(resolution, []):
                all_cells.add(cell)
                if not h3.is_valid_cell(cell):
                    issues.append(f"Invalid H3 cell: {cell}")
                    continue
                actual = h3.get_resolution(cell)
                if actual != resolution:
                    issues.append(
                        f"Cell {cell} has resolution {actual}, expected {resolution}"
                    )

        orphan_count = 0
        for resolution in ordered[1:]:
            for child in cells_by_resolution.get(resolution, []):
                parent = child_parent_map.get(child)
                if not parent:
                    orphan_count += 1
                    issues.append(f"Child {child} has no parent mapping")
                    continue
                parent_resolution = max(
                    candidate for candidate in ordered if candidate < resolution
                )
                expected = h3.cell_to_parent(child, parent_resolution)
                if parent != expected:
                    issues.append(
                        f"Child {child} maps to {parent}, expected parent {expected}"
                    )
                if child not in parent_child_map.get(parent, []):
                    issues.append(
                        f"Parent-child maps disagree for parent {parent} and child {child}"
                    )

        same_level_neighbors = hierarchy.get("same_level_neighbors", {})
        for resolution in ordered:
            level_cells = set(cells_by_resolution.get(resolution, []))
            for cell, neighbors in same_level_neighbors.get(
                str(resolution), {}
            ).items():
                if cell not in level_cells:
                    issues.append(f"Neighbor map includes cell outside level: {cell}")
                for neighbor in neighbors:
                    if neighbor not in level_cells:
                        issues.append(
                            f"Neighbor {neighbor} for {cell} is outside resolution {resolution}"
                        )
                    elif h3.get_resolution(neighbor) != resolution:
                        issues.append(
                            f"Neighbor {neighbor} for {cell} has wrong resolution"
                        )

        parent_count = len(parent_child_map)
        child_count = len(child_parent_map)
        multi_child_parent = any(
            len(children) > 1 for children in parent_child_map.values()
        )
        if child_count > 1 and not multi_child_parent:
            warnings.append("No parent has multiple children in this hierarchy")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "orphan_count": orphan_count,
            "parent_count": parent_count,
            "child_count": child_count,
            "multi_child_parent": multi_child_parent,
            "validated_at": datetime.now(UTC).isoformat(),
        }

    def get_h3_parent_child_map(self) -> dict[str, list[str]]:
        """Return the current H3 parent-child map."""
        hierarchy = getattr(self, "h3_hierarchy", {})
        return {
            parent: list(children)
            for parent, children in hierarchy.get("parent_child_map", {}).items()
        }

    def get_h3_same_level_neighbors(self) -> dict[str, dict[str, list[str]]]:
        """Return same-resolution H3 neighbor maps for the current hierarchy."""
        hierarchy = getattr(self, "h3_hierarchy", {})
        return {
            str(resolution): {
                cell: list(neighbors) for cell, neighbors in level.items()
            }
            for resolution, level in hierarchy.get("same_level_neighbors", {}).items()
        }

    def aggregate_child_values_to_parents(
        self,
        values_by_child: dict[str, list[float]],
        parent_resolution: int,
        weights_by_child: dict[str, float] | None = None,
    ) -> dict[str, list[float]]:
        """
        Aggregate numeric child vectors to H3 parents with finite normalized means.
        """
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_weights: dict[str, list[float]] = defaultdict(list)
        for child, values in values_by_child.items():
            if not h3.is_valid_cell(child):
                raise ValueError(f"Invalid H3 child cell: {child}")
            if h3.get_resolution(child) <= parent_resolution:
                raise ValueError(
                    f"Child {child} is not finer than parent resolution {parent_resolution}"
                )
            array = np.asarray(values, dtype=float).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"Child {child} has non-finite or empty values")
            parent = h3.cell_to_parent(child, parent_resolution)
            grouped[parent].append(array)
            grouped_weights[parent].append(
                float((weights_by_child or {}).get(child, 1.0))
            )

        aggregated: dict[str, list[float]] = {}
        for parent, arrays in grouped.items():
            weights = np.asarray(grouped_weights[parent], dtype=float)
            weights = np.maximum(weights, 0.0)
            if float(np.sum(weights)) <= 1e-12:
                weights = np.ones_like(weights)
            stacked = np.vstack(arrays)
            mean = np.average(stacked, axis=0, weights=weights)
            total = float(np.sum(mean))
            if total > 1e-12:
                mean = mean / total
            aggregated[parent] = mean.tolist()
        return dict(sorted(aggregated.items()))

    def _validate_h3_resolution_sequence(self, resolutions: list[int]) -> list[int]:
        """Validate an ordered unique H3 resolution sequence."""
        ordered = [int(resolution) for resolution in resolutions]
        if len(ordered) < 2:
            raise ValueError("Nested H3 hierarchy requires at least two resolutions")
        if ordered != sorted(ordered) or len(ordered) != len(set(ordered)):
            raise ValueError("Nested H3 resolutions must be unique and ascending")
        for resolution in ordered:
            if resolution < 0 or resolution > 15:
                raise ValueError(f"Invalid H3 resolution: {resolution}")
        return ordered

    def _normalize_cells_to_resolution(
        self, cells: list[str], target_resolution: int, allowed_resolutions: list[int]
    ) -> list[str]:
        """Normalize valid H3 seed cells to the finest target resolution."""
        normalized: set[str] = set()
        for raw_cell in cells:
            cell = str(raw_cell)
            if not h3.is_valid_cell(cell):
                raise ValueError(f"Invalid H3 cell: {cell}")
            source_resolution = h3.get_resolution(cell)
            if (
                source_resolution not in allowed_resolutions
                and source_resolution != target_resolution
            ):
                raise ValueError(
                    f"Input cell {cell} has resolution {source_resolution}, outside {allowed_resolutions}"
                )
            if source_resolution == target_resolution:
                normalized.add(cell)
            elif source_resolution < target_resolution:
                normalized.update(h3.cell_to_children(cell, target_resolution))
            else:
                normalized.add(h3.cell_to_parent(cell, target_resolution))
        return sorted(normalized)

    def _register_h3_level_cells(
        self, cells: list[str], resolution: int, system_prefix: str
    ) -> None:
        """Register cells and a per-resolution system for hierarchy exports."""
        cell_indices = []
        for cell in cells:
            if cell not in self.cells:
                self.add_cell(H3Cell(index=cell, resolution=resolution))
            cell_indices.append(cell)
        system_id = f"{system_prefix}_res_{resolution}"
        system = self.create_system(
            system_id,
            cell_indices,
            name=f"Nested H3 resolution {resolution}",
            description=f"H3 hierarchy level at resolution {resolution}",
        )
        system.hierarchy_level = resolution
        self.hierarchy_levels[resolution].add(system_id)

    def _build_same_level_neighbor_map(
        self, cells_by_resolution: dict[int, list[str]]
    ) -> dict[str, dict[str, list[str]]]:
        """Build deterministic same-resolution neighbor maps."""
        result: dict[str, dict[str, list[str]]] = {}
        for resolution, cells in cells_by_resolution.items():
            cell_set = set(cells)
            level_neighbors: dict[str, list[str]] = {}
            for cell in cells:
                try:
                    neighbors = sorted(
                        neighbor
                        for neighbor in h3.grid_ring(cell, 1)
                        if neighbor in cell_set
                    )
                except Exception:
                    neighbors = []
                level_neighbors[cell] = neighbors
            result[str(resolution)] = level_neighbors
        return result

    def _count_neighbor_edges(self, neighbor_map: dict[str, list[str]]) -> int:
        """Count undirected edges in a same-level neighbor map."""
        edges = {
            tuple(sorted((str(cell), str(neighbor))))
            for cell, neighbors in neighbor_map.items()
            for neighbor in neighbors
        }
        return len(edges)

    def _cells_from_boundary_vertices(
        self, boundary: dict[str, Any], resolution: int
    ) -> list[str]:
        """Fallback boundary cell extraction from GeoJSON coordinate vertices."""
        coordinates = boundary.get("coordinates", [])

        def is_pair(value: Any) -> bool:
            return (
                isinstance(value, (list, tuple))
                and len(value) >= 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            )

        def walk(value: Any) -> list[tuple[float, float]]:
            if not isinstance(value, (list, tuple)):
                return []
            if value and is_pair(value[0]):
                return [
                    (float(item[1]), float(item[0])) for item in value if is_pair(item)
                ]
            points: list[tuple[float, float]] = []
            for child in value:
                points.extend(walk(child))
            return points

        return sorted(
            {h3.latlng_to_cell(lat, lng, resolution) for lat, lng in walk(coordinates)}
        )

    def create_system(
        self,
        system_id: str,
        cell_indices: list[str],
        name: str = "",
        description: str = "",
    ) -> NestedSystem:
        """Create a new nested system from cells."""
        system = NestedSystem(system_id, name, description)

        for cell_index in cell_indices:
            if cell_index in self.cells:
                system.add_cell(self.cells[cell_index])

        self.systems[system_id] = system
        self.hierarchy_levels[system.hierarchy_level].add(system_id)

        if system.parent_system is None:
            self.root_systems.add(system_id)

        system.detect_boundaries()
        self.updated_at = datetime.now(UTC)

        return system

    def create_hierarchical_system(
        self,
        base_resolution: int,
        target_resolutions: list[int],
        bounds: tuple[float, float, float, float],
    ) -> dict[int, NestedSystem]:
        """Create hierarchical nested systems across multiple resolutions."""
        hierarchical_systems = {}

        try:
            # Create base system at target resolution
            min_lat, min_lng, max_lat, max_lng = bounds

            for resolution in sorted(target_resolutions):
                system_id = f"hierarchical_res_{resolution}"

                # Generate H3 cells for this resolution within bounds
                # This is a simplified approach - in practice, you'd use more sophisticated methods
                center_lat = (min_lat + max_lat) / 2
                center_lng = (min_lng + max_lng) / 2

                center_cell = H3Cell.from_coordinates(
                    center_lat, center_lng, resolution
                )

                # Get surrounding cells
                surrounding_indices = grid_disk(
                    center_cell.index, 5
                )  # 5-ring neighborhood

                # Create cells and system
                cell_indices = []
                for h3_index in surrounding_indices:
                    h3_cell = H3Cell(index=h3_index, resolution=resolution)
                    self.add_cell(h3_cell, system_id)
                    cell_indices.append(h3_index)

                # Create the system
                system = self.create_system(
                    system_id,
                    cell_indices,
                    name=f"Hierarchical System Resolution {resolution}",
                    description=f"System at H3 resolution {resolution}",
                )

                hierarchical_systems[resolution] = system

        except Exception as e:
            logger.error(f"Failed to create hierarchical system: {e}")

        return hierarchical_systems

    def detect_all_boundaries(self) -> None:
        """Detect boundaries for all systems in the grid."""
        for system in self.systems.values():
            system.detect_boundaries()
        self.updated_at = datetime.now(UTC)

    def get_system_by_id(self, system_id: str) -> NestedSystem | None:
        """Get system by ID."""
        return self.systems.get(system_id)

    def get_systems_at_level(self, level: int) -> list[NestedSystem]:
        """Get all systems at a specific hierarchy level."""
        system_ids = self.hierarchy_levels.get(level, set())
        return [self.systems[sid] for sid in system_ids if sid in self.systems]

    def get_root_systems(self) -> list[NestedSystem]:
        """Get all root-level systems."""
        return [self.systems[sid] for sid in self.root_systems if sid in self.systems]

    def merge_systems(self, system_id1: str, system_id2: str) -> NestedSystem | None:
        """Merge two systems."""
        system1 = self.systems.get(system_id1)
        system2 = self.systems.get(system_id2)

        if not system1 or not system2:
            return None

        merged_system = system1.merge_with(system2)
        self.systems[merged_system.system_id] = merged_system

        # Remove original systems
        del self.systems[system_id1]
        del self.systems[system_id2]

        self.updated_at = datetime.now(UTC)
        return merged_system

    def split_system(
        self,
        system_id: str,
        criteria_func: Callable[[NestedCell], Any],
    ) -> list[NestedSystem]:
        """Split a system based on criteria."""
        system = self.systems.get(system_id)
        if not system:
            return []

        split_systems = system.split_by_criteria(criteria_func)

        # Remove original system and add split systems
        del self.systems[system_id]
        for split_system in split_systems:
            self.systems[split_system.system_id] = split_system

        self.updated_at = datetime.now(UTC)
        return split_systems

    def _update_grid_metrics(self) -> None:
        """Update grid-level metrics."""
        if not self.cells:
            return

        # Calculate total area
        self.total_area = sum(cell.area_km2 for cell in self.cells.values())

        # Calculate bounding box
        if self.cells:
            lats = [cell.coordinates[0] for cell in self.cells.values()]
            lngs = [cell.coordinates[1] for cell in self.cells.values()]
            self.bounding_box = (
                min(lngs),
                min(lats),  # min_lng, min_lat
                max(lngs),
                max(lats),  # max_lng, max_lat
            )

    def get_grid_summary(self) -> dict[str, Any]:
        """Get comprehensive grid summary."""
        return {
            "name": self.name,
            "num_cells": len(self.cells),
            "num_systems": len(self.systems),
            "num_hierarchy_levels": len(self.hierarchy_levels),
            "resolutions": sorted(list(self.resolutions)),
            "total_area_km2": self.total_area,
            "bounding_box": self.bounding_box,
            "root_systems": list(self.root_systems),
            "hierarchy_distribution": {
                level: len(systems) for level, systems in self.hierarchy_levels.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def export_to_geojson(self) -> dict[str, Any]:
        """Export grid to GeoJSON format."""
        features = []

        for cell in self.cells.values():
            if cell.h3_cell:
                feature = cell.h3_cell.to_geojson()

                # Add nested system properties
                feature["properties"].update(
                    {
                        "system_id": cell.system_id,
                        "cell_type": cell.cell_type.value,
                        "hierarchy_level": cell.hierarchy_level,
                        "is_boundary": cell.is_boundary,
                        "boundary_strength": cell.boundary_strength,
                        "connectivity_degree": cell.get_connectivity_degree(),
                    }
                )

                features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "grid_name": self.name,
                "grid_summary": self.get_grid_summary(),
            },
        }
