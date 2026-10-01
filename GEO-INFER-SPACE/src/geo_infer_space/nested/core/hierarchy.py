"""
Hierarchical System Management.

This module provides classes for managing hierarchical relationships
between nested systems, including parent-child relationships,
level management, and hierarchical operations.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning(
        "NetworkX not available. Graph-based hierarchy analysis will be limited."
    )


class RelationshipType(Enum):
    """Types of hierarchical relationships."""

    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    ANCESTOR_DESCENDANT = "ancestor_descendant"
    PEER = "peer"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    ADJACENT = "adjacent"


class HierarchyDirection(Enum):
    """Direction of hierarchy traversal."""

    UP = "up"  # Toward root/parent
    DOWN = "down"  # Toward leaves/children
    LATERAL = "lateral"  # Same level
    ALL = "all"  # All directions


@dataclass
class HierarchicalRelationship:
    """
    Represents a relationship between two systems in a hierarchy.
    """

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate relationship after creation."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Relationship strength must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }


class HierarchyManager:
    """
    Manages hierarchical relationships between nested systems.

    Provides functionality for creating, maintaining, and analyzing
    hierarchical structures with support for multiple hierarchy types
    and complex relationship patterns.
    """

    def __init__(self, name: str = "HierarchyManager") -> None:
        """
        Initialize hierarchy manager.

        Args:
            name: Manager name for identification
        """
        self.name = name

        # Hierarchy structure
        self.relationships: dict[str, HierarchicalRelationship] = {}
        self.parent_child_map: dict[str, set[str]] = defaultdict(
            set
        )  # parent -> children
        self.child_parent_map: dict[str, str] = {}  # child -> parent

        # Level management
        self.level_assignments: dict[str, int] = {}
        self.levels: dict[int, set[str]] = defaultdict(set)
        self.max_level: int = 0

        # System registry
        self.systems: set[str] = set()
        self.root_systems: set[str] = set()
        self.leaf_systems: set[str] = set()

        # Graph representation (if NetworkX available)
        self.hierarchy_graph: nx.DiGraph | None = None
        if NETWORKX_AVAILABLE:
            self.hierarchy_graph = nx.DiGraph()

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_system(self, system_id: str, level: int | None = None) -> None:
        """Add a system to the hierarchy."""
        self.systems.add(system_id)

        if level is not None:
            self.set_system_level(system_id, level)

        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            self.hierarchy_graph.add_node(system_id)

        self._update_system_classification()
        self.updated_at = datetime.now()

    def remove_system(self, system_id: str) -> None:
        """Remove a system from the hierarchy."""
        if system_id not in self.systems:
            return

        # Remove all relationships involving this system
        relationships_to_remove = []
        for rel_id, relationship in self.relationships.items():
            if (
                relationship.source_id == system_id
                or relationship.target_id == system_id
            ):
                relationships_to_remove.append(rel_id)

        for rel_id in relationships_to_remove:
            self.remove_relationship(rel_id)

        # Remove from system sets
        self.systems.discard(system_id)
        self.root_systems.discard(system_id)
        self.leaf_systems.discard(system_id)

        # Remove from level assignments
        if system_id in self.level_assignments:
            level = self.level_assignments[system_id]
            self.levels[level].discard(system_id)
            del self.level_assignments[system_id]

        # Remove from parent-child maps
        if system_id in self.child_parent_map:
            parent_id = self.child_parent_map[system_id]
            self.parent_child_map[parent_id].discard(system_id)
            del self.child_parent_map[system_id]

        if system_id in self.parent_child_map:
            for child_id in self.parent_child_map[system_id]:
                if child_id in self.child_parent_map:
                    del self.child_parent_map[child_id]
            del self.parent_child_map[system_id]

        # Remove from graph
        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            if self.hierarchy_graph.has_node(system_id):
                self.hierarchy_graph.remove_node(system_id)

        self._update_system_classification()
        self.updated_at = datetime.now()

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add a hierarchical relationship."""
        if source_id not in self.systems:
            self.add_system(source_id)
        if target_id not in self.systems:
            self.add_system(target_id)

        relationship = HierarchicalRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            properties=properties or {},
        )

        rel_id = f"{source_id}_{target_id}_{relationship_type.value}"
        self.relationships[rel_id] = relationship

        # Update parent-child maps for hierarchical relationships
        if relationship_type == RelationshipType.PARENT_CHILD:
            self.parent_child_map[source_id].add(target_id)
            self.child_parent_map[target_id] = source_id

            # Update levels
            if source_id in self.level_assignments:
                child_level = self.level_assignments[source_id] + 1
                self.set_system_level(target_id, child_level)

        # Update graph
        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            self.hierarchy_graph.add_edge(
                source_id,
                target_id,
                relationship_type=relationship_type.value,
                strength=strength,
                **properties or {},
            )

        self._update_system_classification()
        self.updated_at = datetime.now()

        return rel_id

    def remove_relationship(self, rel_id: str) -> None:
        """Remove a hierarchical relationship."""
        if rel_id not in self.relationships:
            return

        relationship = self.relationships[rel_id]

        # Update parent-child maps
        if relationship.relationship_type == RelationshipType.PARENT_CHILD:
            self.parent_child_map[relationship.source_id].discard(
                relationship.target_id
            )
            if relationship.target_id in self.child_parent_map:
                del self.child_parent_map[relationship.target_id]

        # Remove from graph
        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            if self.hierarchy_graph.has_edge(
                relationship.source_id, relationship.target_id
            ):
                self.hierarchy_graph.remove_edge(
                    relationship.source_id, relationship.target_id
                )

        del self.relationships[rel_id]
        self._update_system_classification()
        self.updated_at = datetime.now()

    def set_system_level(self, system_id: str, level: int) -> None:
        """Set the hierarchical level of a system."""
        if system_id not in self.systems:
            self.add_system(system_id)

        # Remove from old level
        if system_id in self.level_assignments:
            old_level = self.level_assignments[system_id]
            self.levels[old_level].discard(system_id)

        # Add to new level
        self.level_assignments[system_id] = level
        self.levels[level].add(system_id)
        self.max_level = max(self.max_level, level)

        self.updated_at = datetime.now()

    def get_children(self, system_id: str) -> set[str]:
        """Get direct children of a system."""
        return self.parent_child_map.get(system_id, set()).copy()

    def get_parent(self, system_id: str) -> str | None:
        """Get parent of a system."""
        return self.child_parent_map.get(system_id)

    def get_ancestors(self, system_id: str) -> list[str]:
        """Get all ancestors of a system (path to root)."""
        ancestors = []
        current = system_id

        while current in self.child_parent_map:
            parent = self.child_parent_map[current]
            ancestors.append(parent)
            current = parent

        return ancestors

    def get_descendants(self, system_id: str) -> set[str]:
        """Get all descendants of a system."""
        descendants = set()
        queue = deque([system_id])

        while queue:
            current = queue.popleft()
            children = self.parent_child_map.get(current, set())

            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def get_siblings(self, system_id: str) -> set[str]:
        """Get siblings of a system (same parent)."""
        parent = self.get_parent(system_id)
        if not parent:
            return set()

        siblings = self.get_children(parent).copy()
        siblings.discard(system_id)
        return siblings

    def get_systems_at_level(self, level: int) -> set[str]:
        """Get all systems at a specific level."""
        return self.levels.get(level, set()).copy()

    def get_level(self, system_id: str) -> int | None:
        """Get the level of a system."""
        return self.level_assignments.get(system_id)

    def find_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Find path between two systems in the hierarchy."""
        if not NETWORKX_AVAILABLE or not self.hierarchy_graph:
            # Fallback to simple traversal
            return self._find_path_simple(source_id, target_id)

        try:
            return cast(
                list[str],
                nx.shortest_path(self.hierarchy_graph, source_id, target_id),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_path_simple(self, source_id: str, target_id: str) -> list[str] | None:
        """Simple path finding without NetworkX."""
        if source_id == target_id:
            return [source_id]

        # Try going up to common ancestor then down
        source_ancestors = [source_id] + self.get_ancestors(source_id)
        target_ancestors = [target_id] + self.get_ancestors(target_id)

        # Find common ancestor
        common_ancestor = None
        for ancestor in source_ancestors:
            if ancestor in target_ancestors:
                common_ancestor = ancestor
                break

        if not common_ancestor:
            return None

        # Build path: source -> common_ancestor -> target
        source_to_ancestor = []
        current = source_id
        while current != common_ancestor:
            source_to_ancestor.append(current)
            source_parent = self.get_parent(current)
            if not source_parent:
                break
            current = source_parent
        source_to_ancestor.append(common_ancestor)

        ancestor_to_target = []
        current = target_id
        while current != common_ancestor:
            ancestor_to_target.append(current)
            target_parent = self.get_parent(current)
            if not target_parent:
                break
            current = target_parent

        # Combine paths
        path = source_to_ancestor + ancestor_to_target[::-1][1:]
        return path if len(path) > 1 else None

    def calculate_hierarchy_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive hierarchy metrics."""
        metrics = {
            "total_systems": len(self.systems),
            "total_relationships": len(self.relationships),
            "num_levels": len(self.levels),
            "max_level": self.max_level,
            "num_root_systems": len(self.root_systems),
            "num_leaf_systems": len(self.leaf_systems),
            "level_distribution": {
                level: len(systems) for level, systems in self.levels.items()
            },
            "relationship_type_distribution": {},
        }

        # Count relationship types
        rel_type_counts: dict[str, int] = defaultdict(int)
        for relationship in self.relationships.values():
            rel_type_counts[relationship.relationship_type.value] += 1
        metrics["relationship_type_distribution"] = dict(rel_type_counts)

        # Calculate branching factors
        branching_factors = []
        for system_id in self.systems:
            children_count = len(self.get_children(system_id))
            if children_count > 0:
                branching_factors.append(children_count)

        if branching_factors:
            metrics["average_branching_factor"] = sum(branching_factors) / len(
                branching_factors
            )
            metrics["max_branching_factor"] = max(branching_factors)
        else:
            metrics["average_branching_factor"] = 0
            metrics["max_branching_factor"] = 0

        # Calculate depth metrics
        depths = []
        for system_id in self.leaf_systems:
            depth = len(self.get_ancestors(system_id))
            depths.append(depth)

        if depths:
            metrics["average_depth"] = sum(depths) / len(depths)
            metrics["max_depth"] = max(depths)
        else:
            metrics["average_depth"] = 0
            metrics["max_depth"] = 0

        # NetworkX-based metrics
        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            try:
                metrics["is_tree"] = nx.is_tree(self.hierarchy_graph)
                metrics["is_forest"] = nx.is_forest(self.hierarchy_graph)
                metrics["num_connected_components"] = (
                    nx.number_weakly_connected_components(self.hierarchy_graph)
                )
            except Exception as e:
                logger.warning(f"Failed to calculate NetworkX metrics: {e}")

        return metrics

    def _update_system_classification(self) -> None:
        """Update classification of systems (root, leaf, etc.)."""
        self.root_systems.clear()
        self.leaf_systems.clear()

        for system_id in self.systems:
            # Root systems have no parents
            if system_id not in self.child_parent_map:
                self.root_systems.add(system_id)

            # Leaf systems have no children
            if (
                system_id not in self.parent_child_map
                or not self.parent_child_map[system_id]
            ):
                self.leaf_systems.add(system_id)

    def validate_hierarchy(self) -> dict[str, Any]:
        """Validate the hierarchy structure and return issues."""
        issues = []
        warnings = []

        # Check for cycles
        if NETWORKX_AVAILABLE and self.hierarchy_graph:
            try:
                cycles = list(nx.simple_cycles(self.hierarchy_graph))
                if cycles:
                    issues.append(f"Cycles detected: {cycles}")
            except Exception as e:
                warnings.append(f"Could not check for cycles: {e}")

        # Check for orphaned systems
        orphaned = []
        for system_id in self.systems:
            if (
                system_id not in self.child_parent_map
                and system_id not in self.root_systems
            ):
                orphaned.append(system_id)

        if orphaned:
            warnings.append(f"Orphaned systems: {orphaned}")

        # Check level consistency
        level_inconsistencies = []
        for system_id in self.systems:
            parent = self.get_parent(system_id)
            if parent:
                parent_level = self.get_level(parent)
                system_level = self.get_level(system_id)

                if parent_level is not None and system_level is not None:
                    if system_level != parent_level + 1:
                        level_inconsistencies.append(
                            f"{system_id} (level {system_level}) parent {parent} (level {parent_level})"
                        )

        if level_inconsistencies:
            warnings.append(f"Level inconsistencies: {level_inconsistencies}")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "validation_timestamp": datetime.now().isoformat(),
        }

    def export_hierarchy(self) -> dict[str, Any]:
        """Export hierarchy structure."""
        return {
            "name": self.name,
            "systems": list(self.systems),
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "level_assignments": self.level_assignments,
            "root_systems": list(self.root_systems),
            "leaf_systems": list(self.leaf_systems),
            "metrics": self.calculate_hierarchy_metrics(),
            "validation": self.validate_hierarchy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def get_hierarchy_summary(self) -> dict[str, Any]:
        """Get a summary of the hierarchy."""
        metrics = self.calculate_hierarchy_metrics()
        validation = self.validate_hierarchy()

        return {
            "name": self.name,
            "total_systems": metrics["total_systems"],
            "total_relationships": metrics["total_relationships"],
            "hierarchy_depth": metrics["max_depth"],
            "branching_factor": metrics["average_branching_factor"],
            "is_valid": validation["is_valid"],
            "num_issues": len(validation["issues"]),
            "num_warnings": len(validation["warnings"]),
            "updated_at": self.updated_at.isoformat(),
        }
