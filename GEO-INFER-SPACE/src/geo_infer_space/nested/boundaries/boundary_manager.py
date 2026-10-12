"""
Boundary Management for Nested H3 Hexagonal Grid Systems.

This module provides comprehensive boundary management capabilities including
detection, analysis, manipulation, and monitoring of boundaries in nested
geospatial systems.
"""

import logging
import uuid
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from collections.abc import Callable
from enum import Enum
from collections import defaultdict

from .detector import (
    BoundaryDetector,
    BoundarySegment,
    BoundaryType,
    BoundaryDetectionMethod,
)

if TYPE_CHECKING:
    from ..core.nested_grid import NestedH3Grid

logger = logging.getLogger(__name__)


class BoundaryOperation(Enum):
    """Types of boundary operations."""

    SPLIT = "split"
    MERGE = "merge"
    DISSOLVE = "dissolve"
    BUFFER = "buffer"
    SIMPLIFY = "simplify"
    SMOOTH = "smooth"


class FlowDirection(Enum):
    """Direction of flow across boundaries."""

    INWARD = "inward"
    OUTWARD = "outward"
    BIDIRECTIONAL = "bidirectional"
    BLOCKED = "blocked"


@dataclass
class BoundaryFlow:
    """
    Represents flow across a boundary.
    """

    flow_id: str
    source_system: str
    target_system: str
    boundary_segment_id: str

    # Flow properties
    flow_rate: float = 0.0
    flow_direction: FlowDirection = FlowDirection.BIDIRECTIONAL
    flow_type: str = "generic"  # e.g., "mass", "energy", "information"

    # Flow data
    flow_data: dict[str, Any] = field(default_factory=dict)

    # Constraints
    max_flow_rate: float | None = None
    min_flow_rate: float | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_flow(
        self,
        new_rate: float,
        flow_data: dict[str, Any] | None = None,
    ) -> None:
        """Update flow rate and data."""
        # Apply constraints
        if self.max_flow_rate is not None:
            new_rate = min(new_rate, self.max_flow_rate)
        if self.min_flow_rate is not None:
            new_rate = max(new_rate, self.min_flow_rate)

        self.flow_rate = new_rate

        if flow_data:
            self.flow_data.update(flow_data)

        self.last_updated = datetime.now(UTC)


@dataclass
class BoundaryConstraint:
    """
    Represents constraints on boundary behavior.
    """

    constraint_id: str
    boundary_segment_id: str
    constraint_type: str  # e.g., "permeability", "capacity", "resistance"

    # Constraint parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Validation function
    validation_function: Callable | None = None

    # Status
    is_active: bool = True

    def validate(self, value: Any) -> bool:
        """Validate a value against this constraint."""
        if not self.is_active:
            return True

        if self.validation_function:
            try:
                return cast(bool, self.validation_function(value, self.parameters))
            except Exception as e:
                logger.warning(f"Constraint validation failed: {e}")
                return False

        return True


class H3BoundaryManager:
    """
    Comprehensive boundary management for nested H3 systems.

    Provides capabilities for:
    - Boundary detection and analysis
    - Boundary manipulation (splitting, merging, etc.)
    - Flow management across boundaries
    - Boundary constraint enforcement
    - Dynamic boundary monitoring
    """

    def __init__(self, name: str = "H3BoundaryManager") -> None:
        """
        Initialize boundary manager.

        Args:
            name: Manager name for identification
        """
        self.name = name

        # Core components
        self.detector = BoundaryDetector(f"{name}_detector")

        # Storage
        self.boundaries: dict[str, BoundarySegment] = {}
        self.flows: dict[str, BoundaryFlow] = {}
        self.constraints: dict[str, BoundaryConstraint] = {}

        # System relationships
        self.system_boundaries: dict[str, set[str]] = defaultdict(
            set
        )  # system_id -> boundary_ids
        self.boundary_systems: dict[str, set[str]] = defaultdict(
            set
        )  # boundary_id -> system_ids

        # Flow networks
        self.flow_networks: dict[
            str, dict[str, list[str]]
        ] = {}  # network_id -> {source -> [targets]}

        # Monitoring
        self.boundary_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.flow_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def detect_boundaries(
        self,
        nested_grid: "NestedH3Grid",
        method: BoundaryDetectionMethod = (BoundaryDetectionMethod.NEIGHBOR_ANALYSIS),
        **kwargs: Any,
    ) -> dict[str, list[BoundarySegment]]:
        """
        Detect boundaries in nested grid systems.

        Args:
            nested_grid: NestedH3Grid instance
            method: Detection method to use
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping system IDs to detected boundaries
        """
        detected = self.detector.detect_boundaries(nested_grid, method, **kwargs)

        # Store boundaries and update relationships
        for system_id, segments in detected.items():
            for segment in segments:
                self.boundaries[segment.segment_id] = segment
                self.system_boundaries[system_id].add(segment.segment_id)
                self.boundary_systems[segment.segment_id].add(system_id)

        self.updated_at = datetime.now(UTC)
        return detected

    def get_shared_boundaries(
        self, system_id1: str, system_id2: str
    ) -> list[BoundarySegment]:
        """
        Find boundaries shared between two systems.

        Args:
            system_id1: First system ID
            system_id2: Second system ID

        Returns:
            List of shared boundary segments
        """
        boundaries1 = self.system_boundaries.get(system_id1, set())
        boundaries2 = self.system_boundaries.get(system_id2, set())

        shared_boundary_ids = boundaries1.intersection(boundaries2)

        return [
            self.boundaries[bid]
            for bid in shared_boundary_ids
            if bid in self.boundaries
        ]

    def split_boundary(
        self, boundary_id: str, split_points: list[str]
    ) -> list[BoundarySegment]:
        """
        Split a boundary segment at specified points.

        Args:
            boundary_id: ID of boundary to split
            split_points: List of cell indices where to split

        Returns:
            List of new boundary segments
        """
        if boundary_id not in self.boundaries:
            raise ValueError(f"Boundary {boundary_id} not found")

        original_boundary = self.boundaries[boundary_id]
        cell_indices = original_boundary.cell_indices.copy()

        # Find split positions
        split_positions = []
        for split_point in split_points:
            if split_point in cell_indices:
                split_positions.append(cell_indices.index(split_point))

        split_positions.sort()

        # Create new segments
        new_segments: list[BoundarySegment] = []
        start_idx = 0

        for split_pos in split_positions + [len(cell_indices)]:
            if start_idx < split_pos:
                segment_cells = cell_indices[start_idx:split_pos]

                new_segment = BoundarySegment(
                    segment_id=f"{boundary_id}_split_{len(new_segments)}",
                    cell_indices=segment_cells,
                    boundary_type=original_boundary.boundary_type,
                    strength=original_boundary.strength,
                    properties=original_boundary.properties.copy(),
                )

                new_segments.append(new_segment)
                self.boundaries[new_segment.segment_id] = new_segment

                # Update system relationships
                for system_id in self.boundary_systems[boundary_id]:
                    self.system_boundaries[system_id].add(new_segment.segment_id)
                    self.boundary_systems[new_segment.segment_id].add(system_id)

            start_idx = split_pos

        # Remove original boundary
        self._remove_boundary(boundary_id)

        # Record operation
        self._record_boundary_operation(
            BoundaryOperation.SPLIT,
            {
                "original_boundary": boundary_id,
                "split_points": split_points,
                "new_segments": [seg.segment_id for seg in new_segments],
            },
        )

        return new_segments

    def merge_boundaries(self, boundary_ids: list[str]) -> BoundarySegment:
        """
        Merge multiple boundary segments into one.

        Args:
            boundary_ids: List of boundary IDs to merge

        Returns:
            New merged boundary segment
        """
        if len(boundary_ids) < 2:
            raise ValueError("At least 2 boundaries required for merging")

        # Collect all cells and properties
        all_cells = []
        all_systems = set()
        total_strength = 0.0
        merged_properties = {}

        for boundary_id in boundary_ids:
            if boundary_id not in self.boundaries:
                continue

            boundary = self.boundaries[boundary_id]
            all_cells.extend(boundary.cell_indices)
            all_systems.update(self.boundary_systems[boundary_id])
            total_strength += boundary.strength
            merged_properties.update(boundary.properties)

        # Remove duplicates while preserving order
        unique_cells = []
        seen = set()
        for cell in all_cells:
            if cell not in seen:
                unique_cells.append(cell)
                seen.add(cell)

        # Create merged boundary
        merged_boundary = BoundarySegment(
            segment_id=f"merged_{uuid.uuid4().hex[:8]}",
            cell_indices=unique_cells,
            boundary_type=BoundaryType.INTERNAL,  # Default for merged boundaries
            strength=total_strength / len(boundary_ids),
            properties=merged_properties,
        )

        # Store new boundary
        self.boundaries[merged_boundary.segment_id] = merged_boundary

        # Update system relationships
        for system_id in all_systems:
            self.system_boundaries[system_id].add(merged_boundary.segment_id)
            self.boundary_systems[merged_boundary.segment_id].add(system_id)

        # Remove original boundaries
        for boundary_id in boundary_ids:
            self._remove_boundary(boundary_id)

        # Record operation
        self._record_boundary_operation(
            BoundaryOperation.MERGE,
            {
                "merged_boundaries": boundary_ids,
                "new_boundary": merged_boundary.segment_id,
            },
        )

        return merged_boundary

    def create_flow(
        self,
        source_system: str,
        target_system: str,
        flow_type: str = "generic",
        **kwargs: Any,
    ) -> BoundaryFlow:
        """
        Create a flow between two systems.

        Args:
            source_system: Source system ID
            target_system: Target system ID
            flow_type: Type of flow (e.g., "mass", "energy", "information")
            **kwargs: Additional flow parameters

        Returns:
            Created BoundaryFlow instance
        """
        # Find shared boundary
        shared_boundaries = self.get_shared_boundaries(source_system, target_system)

        if not shared_boundaries:
            raise ValueError(
                f"No shared boundary found between {source_system} and {target_system}"
            )

        # Use first shared boundary
        boundary_segment_id = shared_boundaries[0].segment_id

        flow = BoundaryFlow(
            flow_id=f"flow_{uuid.uuid4().hex[:8]}",
            source_system=source_system,
            target_system=target_system,
            boundary_segment_id=boundary_segment_id,
            flow_type=flow_type,
            **kwargs,
        )

        self.flows[flow.flow_id] = flow

        return flow

    def update_flow(
        self, flow_id: str, new_rate: float, flow_data: dict[str, Any] | None = None
    ) -> None:
        """
        Update flow rate and data.

        Args:
            flow_id: Flow ID to update
            new_rate: New flow rate
            flow_data: Additional flow data
        """
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")

        flow = self.flows[flow_id]

        # Apply constraints
        boundary_constraints = [
            c
            for c in self.constraints.values()
            if c.boundary_segment_id == flow.boundary_segment_id
        ]

        for constraint in boundary_constraints:
            if not constraint.validate(new_rate):
                logger.warning(
                    f"Flow update violates constraint {constraint.constraint_id}"
                )
                return

        # Record flow history
        self.flow_history[flow_id].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "old_rate": flow.flow_rate,
                "new_rate": new_rate,
                "flow_data": flow_data,
            }
        )

        flow.update_flow(new_rate, flow_data)

    def add_constraint(
        self,
        boundary_segment_id: str,
        constraint_type: str,
        parameters: dict[str, Any],
        validation_function: Callable | None = None,
    ) -> BoundaryConstraint:
        """
        Add a constraint to a boundary.

        Args:
            boundary_segment_id: Boundary to constrain
            constraint_type: Type of constraint
            parameters: Constraint parameters
            validation_function: Custom validation function

        Returns:
            Created BoundaryConstraint instance
        """
        constraint = BoundaryConstraint(
            constraint_id=f"constraint_{uuid.uuid4().hex[:8]}",
            boundary_segment_id=boundary_segment_id,
            constraint_type=constraint_type,
            parameters=parameters,
            validation_function=validation_function,
        )

        self.constraints[constraint.constraint_id] = constraint

        return constraint

    def get_boundary_permeability(
        self, boundary_id: str, flow_type: str = "generic"
    ) -> float:
        """
        Calculate boundary permeability for a specific flow type.

        Args:
            boundary_id: Boundary ID
            flow_type: Type of flow

        Returns:
            Permeability value (0.0 = impermeable, 1.0 = fully permeable)
        """
        if boundary_id not in self.boundaries:
            return 0.0

        boundary = self.boundaries[boundary_id]

        # Check for permeability constraints
        permeability_constraints = [
            c
            for c in self.constraints.values()
            if (
                c.boundary_segment_id == boundary_id
                and c.constraint_type == "permeability"
                and c.parameters.get("flow_type") == flow_type
            )
        ]

        if permeability_constraints:
            # Use most restrictive constraint
            return cast(
                float,
                min(
                    c.parameters.get("permeability", 1.0)
                    for c in permeability_constraints
                ),
            )

        # Default permeability based on boundary type
        type_permeability = {
            BoundaryType.PERMEABLE: 1.0,
            BoundaryType.IMPERMEABLE: 0.0,
            BoundaryType.INTERFACE: 0.5,
            BoundaryType.GRADIENT: 0.7,
            BoundaryType.SHARP: 0.3,
        }

        return type_permeability.get(boundary.boundary_type, 0.5)

    def analyze_flow_network(self, network_id: str) -> dict[str, Any]:
        """
        Analyze flow patterns in a network.

        Args:
            network_id: Network ID to analyze

        Returns:
            Network analysis results
        """
        if network_id not in self.flow_networks:
            return {}

        network = self.flow_networks[network_id]

        # Calculate network metrics
        total_flows = sum(len(targets) for targets in network.values())
        num_nodes = len(
            set(
                list(network.keys())
                + [target for targets in network.values() for target in targets]
            )
        )

        # Find flow bottlenecks
        bottlenecks = []
        for source, targets in network.items():
            for target in targets:
                shared_boundaries = self.get_shared_boundaries(source, target)
                for boundary in shared_boundaries:
                    permeability = self.get_boundary_permeability(boundary.segment_id)
                    if permeability < 0.3:  # Low permeability threshold
                        bottlenecks.append(
                            {
                                "source": source,
                                "target": target,
                                "boundary_id": boundary.segment_id,
                                "permeability": permeability,
                            }
                        )

        return {
            "network_id": network_id,
            "total_flows": total_flows,
            "num_nodes": num_nodes,
            "connectivity": total_flows / num_nodes if num_nodes > 0 else 0,
            "bottlenecks": bottlenecks,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

    def get_boundary_statistics(self) -> dict[str, Any]:
        """Get comprehensive boundary statistics."""
        total_boundaries = len(self.boundaries)
        total_flows = len(self.flows)
        total_constraints = len(self.constraints)

        # Boundary type distribution
        type_counts: dict[str, int] = defaultdict(int)
        for boundary in self.boundaries.values():
            type_counts[boundary.boundary_type.value] += 1

        # Flow type distribution
        flow_type_counts: dict[str, int] = defaultdict(int)
        for flow in self.flows.values():
            flow_type_counts[flow.flow_type] += 1

        # Average flow rates by type
        avg_flow_rates = {}
        for flow_type in flow_type_counts:
            rates = [
                f.flow_rate for f in self.flows.values() if f.flow_type == flow_type
            ]
            avg_flow_rates[flow_type] = sum(rates) / len(rates) if rates else 0

        return {
            "manager_name": self.name,
            "total_boundaries": total_boundaries,
            "total_flows": total_flows,
            "total_constraints": total_constraints,
            "boundary_types": dict(type_counts),
            "flow_types": dict(flow_type_counts),
            "average_flow_rates": avg_flow_rates,
            "systems_managed": len(self.system_boundaries),
            "updated_at": self.updated_at.isoformat(),
        }

    def _remove_boundary(self, boundary_id: str) -> None:
        """Remove a boundary and clean up relationships."""
        if boundary_id not in self.boundaries:
            return

        # Remove from system relationships
        for system_id in self.boundary_systems[boundary_id]:
            self.system_boundaries[system_id].discard(boundary_id)

        del self.boundary_systems[boundary_id]
        del self.boundaries[boundary_id]

        # Remove associated flows
        flows_to_remove = [
            fid
            for fid, flow in self.flows.items()
            if flow.boundary_segment_id == boundary_id
        ]

        for flow_id in flows_to_remove:
            del self.flows[flow_id]

        # Remove associated constraints
        constraints_to_remove = [
            cid
            for cid, constraint in self.constraints.items()
            if constraint.boundary_segment_id == boundary_id
        ]

        for constraint_id in constraints_to_remove:
            del self.constraints[constraint_id]

    def _record_boundary_operation(
        self,
        operation: BoundaryOperation,
        details: dict[str, Any],
    ) -> None:
        """Record boundary operation in history."""
        record = {
            "operation": operation.value,
            "details": details,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add to relevant boundary histories
        if "original_boundary" in details:
            self.boundary_history[details["original_boundary"]].append(record)

        if "new_boundary" in details:
            self.boundary_history[details["new_boundary"]].append(record)

        if "new_segments" in details:
            for segment_id in details["new_segments"]:
                self.boundary_history[segment_id].append(record)
