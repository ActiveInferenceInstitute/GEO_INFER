"""
Hierarchy Metrics for H3 Nested Systems.

This module provides comprehensive metrics and analysis for hierarchical
structures in nested geospatial systems, including depth analysis,
balance metrics, and structural properties.
"""

import logging
import uuid
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from collections import defaultdict, deque

import numpy as np
import h3

logger = logging.getLogger(__name__)


class HierarchyMetric(Enum):
    """Types of hierarchy metrics."""

    DEPTH = "depth"
    BREADTH = "breadth"
    BALANCE = "balance"
    DENSITY = "density"
    CONNECTIVITY = "connectivity"
    EFFICIENCY = "efficiency"
    COMPLEXITY = "complexity"
    STABILITY = "stability"


class HierarchyStructure(Enum):
    """Types of hierarchy structures."""

    TREE = "tree"
    DAG = "dag"  # Directed Acyclic Graph
    NETWORK = "network"
    LATTICE = "lattice"
    HYBRID = "hybrid"


@dataclass
class HierarchyNode:
    """
    Represents a node in the hierarchy.
    """

    node_id: str
    level: int

    # Relationships
    parent_id: str | None = None
    children_ids: set[str] = field(default_factory=set)

    # Properties
    properties: dict[str, Any] = field(default_factory=dict)

    # Metrics
    subtree_size: int = 1
    depth_from_root: int = 0
    height_to_leaves: int = 0

    # H3-specific properties
    h3_index: str | None = None
    h3_resolution: int | None = None

    def add_child(self, child_id: str) -> None:
        """Add a child node."""
        self.children_ids.add(child_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child node."""
        self.children_ids.discard(child_id)

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        """Check if node is a root."""
        return self.parent_id is None


@dataclass
class HierarchyMetrics:
    """
    Comprehensive metrics for a hierarchy.
    """

    hierarchy_id: str

    # Basic structure metrics
    total_nodes: int = 0
    total_levels: int = 0
    max_depth: int = 0
    max_breadth: int = 0

    # Balance metrics
    balance_factor: float = 0.0
    height_variance: float = 0.0
    breadth_variance: float = 0.0

    # Density metrics
    node_density: float = 0.0
    connection_density: float = 0.0

    # Efficiency metrics
    path_efficiency: float = 0.0
    information_efficiency: float = 0.0

    # Complexity metrics
    structural_complexity: float = 0.0
    branching_complexity: float = 0.0

    # Stability metrics
    structural_stability: float = 0.0

    # H3-specific metrics
    resolution_distribution: dict[int, int] = field(default_factory=dict)
    spatial_coherence: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    calculation_time: float = 0.0


@dataclass
class HierarchyAnalysisResult:
    """
    Result of hierarchy analysis.
    """

    analysis_id: str
    hierarchy_metrics: HierarchyMetrics

    # Detailed analysis
    level_statistics: dict[int, dict[str, Any]] = field(default_factory=dict)
    node_rankings: dict[str, dict[str, float]] = field(default_factory=dict)

    # Structural analysis
    detected_structure: HierarchyStructure = HierarchyStructure.TREE
    structure_confidence: float = 0.0

    # Anomaly detection
    structural_anomalies: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    optimization_suggestions: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processing_time: float = 0.0


class H3HierarchyAnalyzer:
    """
    Advanced hierarchy analyzer for H3 nested systems.

    Provides comprehensive analysis of hierarchical structures including:
    - Structural metrics and properties
    - Balance and efficiency analysis
    - Anomaly detection
    - Optimization recommendations
    """

    def __init__(self, name: str = "H3HierarchyAnalyzer") -> None:
        """
        Initialize hierarchy analyzer.

        Args:
            name: Analyzer name for identification
        """
        self.name = name

        # Hierarchy storage
        self.hierarchies: dict[str, dict[str, HierarchyNode]] = {}

        # Analysis results
        self.analysis_results: dict[str, HierarchyAnalysisResult] = {}

        # Configuration
        self.analysis_config: dict[str, Any] = {
            "balance_threshold": 0.8,
            "efficiency_threshold": 0.7,
            "anomaly_threshold": 2.0,  # Standard deviations
            "max_analysis_depth": 20,
        }

        # Statistics
        self.analysis_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def create_hierarchy(self, hierarchy_id: str) -> dict[str, HierarchyNode]:
        """
        Create a new hierarchy.

        Args:
            hierarchy_id: Unique hierarchy identifier

        Returns:
            Empty hierarchy dictionary
        """
        self.hierarchies[hierarchy_id] = {}
        self.updated_at = datetime.now(UTC)
        return self.hierarchies[hierarchy_id]

    def add_node(
        self,
        hierarchy_id: str,
        node_id: str,
        level: int,
        parent_id: str | None = None,
        h3_index: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> HierarchyNode:
        """
        Add a node to a hierarchy.

        Args:
            hierarchy_id: Hierarchy identifier
            node_id: Node identifier
            level: Hierarchy level
            parent_id: Parent node ID
            h3_index: H3 cell index
            properties: Node properties

        Returns:
            Created HierarchyNode instance
        """
        if hierarchy_id not in self.hierarchies:
            self.create_hierarchy(hierarchy_id)

        hierarchy = self.hierarchies[hierarchy_id]

        # Get H3 resolution if available
        h3_resolution = None
        if h3_index:
            try:
                h3_resolution = h3.get_resolution(h3_index)
            except Exception:
                h3_resolution = None

        node = HierarchyNode(
            node_id=node_id,
            level=level,
            parent_id=parent_id,
            h3_index=h3_index,
            h3_resolution=h3_resolution,
            properties=properties or {},
        )

        hierarchy[node_id] = node

        # Update parent-child relationships
        if parent_id and parent_id in hierarchy:
            hierarchy[parent_id].add_child(node_id)

        self.updated_at = datetime.now(UTC)
        return node

    def analyze_hierarchy(
        self, hierarchy_id: str, **kwargs: Any
    ) -> HierarchyAnalysisResult:
        """
        Perform comprehensive hierarchy analysis.

        Args:
            hierarchy_id: Hierarchy to analyze
            **kwargs: Analysis parameters

        Returns:
            HierarchyAnalysisResult instance
        """
        start_time = datetime.now(UTC)
        analysis_id = f"hierarchy_analysis_{uuid.uuid4().hex[:8]}"

        if hierarchy_id not in self.hierarchies:
            raise ValueError(f"Hierarchy {hierarchy_id} not found")

        hierarchy = self.hierarchies[hierarchy_id]

        if not hierarchy:
            return HierarchyAnalysisResult(
                analysis_id=analysis_id,
                hierarchy_metrics=HierarchyMetrics(hierarchy_id=hierarchy_id),
                processing_time=0.0,
            )

        # Calculate comprehensive metrics
        metrics = self._calculate_hierarchy_metrics(hierarchy_id, hierarchy)

        # Analyze structure
        structure, confidence = self._detect_hierarchy_structure(hierarchy)

        # Calculate level statistics
        level_stats = self._calculate_level_statistics(hierarchy)

        # Rank nodes
        node_rankings = self._rank_nodes(hierarchy)

        # Detect anomalies
        anomalies = self._detect_structural_anomalies(hierarchy, metrics)

        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(hierarchy, metrics)

        # Create result
        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        result = HierarchyAnalysisResult(
            analysis_id=analysis_id,
            hierarchy_metrics=metrics,
            level_statistics=level_stats,
            node_rankings=node_rankings,
            detected_structure=structure,
            structure_confidence=confidence,
            structural_anomalies=anomalies,
            optimization_suggestions=suggestions,
            processing_time=processing_time,
        )

        # Store result
        self.analysis_results[analysis_id] = result
        self.analysis_stats["hierarchies_analyzed"] += 1
        self.updated_at = datetime.now(UTC)

        return result

    def _calculate_hierarchy_metrics(
        self, hierarchy_id: str, hierarchy: dict[str, HierarchyNode]
    ) -> HierarchyMetrics:
        """Calculate comprehensive hierarchy metrics."""
        start_time = datetime.now(UTC)

        metrics = HierarchyMetrics(hierarchy_id=hierarchy_id)

        if not hierarchy:
            return metrics

        # Basic structure metrics
        metrics.total_nodes = len(hierarchy)

        # Calculate depths and levels
        self._calculate_node_depths(hierarchy)

        levels = [node.level for node in hierarchy.values()]
        depths = [node.depth_from_root for node in hierarchy.values()]

        metrics.total_levels = len(set(levels))
        metrics.max_depth = max(depths) if depths else 0

        # Calculate breadth at each level
        level_counts: dict[int, int] = defaultdict(int)
        for node in hierarchy.values():
            level_counts[node.level] += 1

        metrics.max_breadth = max(level_counts.values()) if level_counts else 0

        # Balance metrics
        metrics.balance_factor = self._calculate_balance_factor(hierarchy)

        heights = [node.height_to_leaves for node in hierarchy.values()]
        breadths = list(level_counts.values())

        metrics.height_variance = (
            np.var(heights) if heights else 0.0  # type: ignore[assignment]
        )
        metrics.breadth_variance = (
            np.var(breadths) if breadths else 0.0  # type: ignore[assignment]
        )

        # Density metrics
        metrics.node_density = self._calculate_node_density(hierarchy)
        metrics.connection_density = self._calculate_connection_density(hierarchy)

        # Efficiency metrics
        metrics.path_efficiency = self._calculate_path_efficiency(hierarchy)
        metrics.information_efficiency = self._calculate_information_efficiency(
            hierarchy
        )

        # Complexity metrics
        metrics.structural_complexity = self._calculate_structural_complexity(hierarchy)
        metrics.branching_complexity = self._calculate_branching_complexity(hierarchy)

        # Stability metrics
        metrics.structural_stability = self._calculate_structural_stability(hierarchy)

        # H3-specific metrics
        metrics.resolution_distribution = self._calculate_resolution_distribution(
            hierarchy
        )
        metrics.spatial_coherence = self._calculate_spatial_coherence(hierarchy)

        metrics.calculation_time = (datetime.now(UTC) - start_time).total_seconds()

        return metrics

    def _calculate_node_depths(self, hierarchy: dict[str, HierarchyNode]) -> None:
        """Calculate depth and height for all nodes."""
        # Find root nodes
        roots = [node for node in hierarchy.values() if node.is_root()]

        # Calculate depths from roots
        for root in roots:
            self._calculate_depths_recursive(hierarchy, root.node_id, 0)

        # Calculate heights to leaves
        leaves = [node for node in hierarchy.values() if node.is_leaf()]
        for leaf in leaves:
            self._calculate_heights_recursive(hierarchy, leaf.node_id, 0)

    def _calculate_depths_recursive(
        self, hierarchy: dict[str, HierarchyNode], node_id: str, depth: int
    ) -> None:
        """Recursively calculate depths from root."""
        if node_id not in hierarchy:
            return

        node = hierarchy[node_id]
        node.depth_from_root = depth

        for child_id in node.children_ids:
            self._calculate_depths_recursive(hierarchy, child_id, depth + 1)

    def _calculate_heights_recursive(
        self, hierarchy: dict[str, HierarchyNode], node_id: str, height: int
    ) -> None:
        """Recursively calculate heights to leaves."""
        if node_id not in hierarchy:
            return

        node = hierarchy[node_id]
        node.height_to_leaves = max(node.height_to_leaves, height)

        if node.parent_id:
            self._calculate_heights_recursive(hierarchy, node.parent_id, height + 1)

    def _calculate_balance_factor(self, hierarchy: dict[str, HierarchyNode]) -> float:
        """Calculate hierarchy balance factor."""
        if not hierarchy:
            return 0.0

        # Calculate balance based on subtree sizes
        balance_scores: list[float] = []

        for node in hierarchy.values():
            if not node.is_leaf():
                child_sizes = []
                for child_id in node.children_ids:
                    if child_id in hierarchy:
                        child_sizes.append(
                            self._calculate_subtree_size(hierarchy, child_id)
                        )

                if len(child_sizes) > 1:
                    balance: float = 1.0 / (  # type: ignore[assignment]
                        1.0 + np.var(child_sizes)
                    )

                    balance_scores.append(balance)

        return sum(balance_scores) / len(balance_scores) if balance_scores else 1.0

    def _calculate_subtree_size(
        self, hierarchy: dict[str, HierarchyNode], node_id: str
    ) -> int:
        """Calculate size of subtree rooted at node."""
        if node_id not in hierarchy:
            return 0

        node = hierarchy[node_id]
        size = 1  # Count the node itself

        for child_id in node.children_ids:
            size += self._calculate_subtree_size(hierarchy, child_id)

        node.subtree_size = size
        return size

    def _calculate_node_density(self, hierarchy: dict[str, HierarchyNode]) -> float:
        """Calculate node density metric."""
        if not hierarchy:
            return 0.0

        total_nodes = len(hierarchy)
        max_depth = max(node.depth_from_root for node in hierarchy.values())

        if max_depth == 0:
            return 1.0

        # Density is nodes per level
        return total_nodes / (max_depth + 1)

    def _calculate_connection_density(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate connection density metric."""
        if not hierarchy:
            return 0.0

        total_connections = sum(len(node.children_ids) for node in hierarchy.values())
        total_nodes = len(hierarchy)

        if total_nodes <= 1:
            return 0.0

        # Maximum possible connections in a tree
        max_connections = total_nodes - 1

        return total_connections / max_connections if max_connections > 0 else 0.0

    def _calculate_path_efficiency(self, hierarchy: dict[str, HierarchyNode]) -> float:
        """Calculate path efficiency metric."""
        if not hierarchy:
            return 0.0

        # Calculate average path length from roots to leaves
        roots = [node for node in hierarchy.values() if node.is_root()]
        leaves = [node for node in hierarchy.values() if node.is_leaf()]

        if not roots or not leaves:
            return 1.0

        total_path_length = 0
        path_count = 0

        for root in roots:
            for leaf in leaves:
                path_length = self._find_path_length(
                    hierarchy, root.node_id, leaf.node_id
                )
                if path_length > 0:
                    total_path_length += path_length
                    path_count += 1

        if path_count == 0:
            return 1.0

        avg_path_length = total_path_length / path_count
        max_depth = max(node.depth_from_root for node in hierarchy.values())

        # Efficiency is inverse of normalized path length
        return 1.0 / (1.0 + avg_path_length / max(1, max_depth))

    def _find_path_length(
        self, hierarchy: dict[str, HierarchyNode], start_id: str, end_id: str
    ) -> int:
        """Find path length between two nodes."""
        if start_id == end_id:
            return 0

        # Simple BFS to find path
        queue = deque([(start_id, 0)])
        visited = {start_id}

        while queue:
            current_id, distance = queue.popleft()

            if current_id not in hierarchy:
                continue

            current_node = hierarchy[current_id]

            # Check children
            for child_id in current_node.children_ids:
                if child_id == end_id:
                    return distance + 1

                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, distance + 1))

        return -1  # No path found

    def _calculate_information_efficiency(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate information efficiency metric."""
        # This would calculate how efficiently information can flow through the hierarchy
        # For now, return a simple metric based on branching factor
        if not hierarchy:
            return 0.0

        branching_factors = []
        for node in hierarchy.values():
            if not node.is_leaf():
                branching_factors.append(len(node.children_ids))

        if not branching_factors:
            return 1.0

        avg_branching = sum(branching_factors) / len(branching_factors)

        # Optimal branching factor is around 2-4 for most applications
        optimal_branching = 3.0
        efficiency = 1.0 / (1.0 + abs(avg_branching - optimal_branching))

        return efficiency

    def _calculate_structural_complexity(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate structural complexity metric."""
        if not hierarchy:
            return 0.0

        # Complexity based on number of levels and branching variation
        levels = set(node.level for node in hierarchy.values())
        branching_factors = [
            len(node.children_ids) for node in hierarchy.values() if not node.is_leaf()
        ]

        level_complexity = len(levels) / len(hierarchy)

        branching_complexity: float = (
            np.var(branching_factors) if branching_factors else 0.0  # type: ignore[assignment]
        )

        return level_complexity + branching_complexity

    def _calculate_branching_complexity(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate branching complexity metric."""
        if not hierarchy:
            return 0.0

        branching_factors = [len(node.children_ids) for node in hierarchy.values()]

        if not branching_factors:
            return 0.0

        return np.var(branching_factors)  # type: ignore[return-value]

    def _calculate_structural_stability(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate structural stability metric."""
        # This would analyze how stable the hierarchy structure is
        # For now, return a simple metric based on balance
        return self._calculate_balance_factor(hierarchy)

    def _calculate_resolution_distribution(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> dict[int, int]:
        """Calculate H3 resolution distribution."""
        distribution: dict[int, int] = defaultdict(int)

        for node in hierarchy.values():
            if node.h3_resolution is not None:
                distribution[node.h3_resolution] += 1

        return dict(distribution)

    def _calculate_spatial_coherence(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> float:
        """Calculate spatial coherence as fraction of parent-child pairs with valid H3 nesting.

        A hierarchy is spatially coherent when every parent node's H3 cell is a
        direct ancestor (at the next coarser resolution) of its children's H3 cells.
        The score is the fraction of parent-child edges that satisfy this criterion.

        Nodes without H3 indices are skipped; if no H3 nodes are present, returns 0.0.
        """
        coherent_edges = 0
        total_edges = 0

        for node in hierarchy.values():
            if not node.h3_index or node.h3_resolution is None:
                continue
            for child_id in node.children_ids:
                child = hierarchy.get(child_id)
                if child is None or not child.h3_index or child.h3_resolution is None:
                    continue
                total_edges += 1
                # A coherent edge: child is at finer resolution and parent contains it.
                # h3.cell_to_parent(child_h3, parent_resolution) == parent_h3
                try:
                    expected_parent = h3.cell_to_parent(
                        child.h3_index, node.h3_resolution
                    )
                    if expected_parent == node.h3_index:
                        coherent_edges += 1
                except Exception:
                    pass  # Incompatible resolutions count as incoherent

        if total_edges == 0:
            # No H3-indexed edges — fall back to structural proxy:
            # fraction of nodes that are leaves (perfectly nested implies many leaves)
            leaf_count = sum(1 for n in hierarchy.values() if n.is_leaf())
            return leaf_count / len(hierarchy) if hierarchy else 0.0

        return coherent_edges / total_edges

    def _detect_hierarchy_structure(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> tuple[HierarchyStructure, float]:
        """Detect the type of hierarchy structure."""
        if not hierarchy:
            return HierarchyStructure.TREE, 0.0

        # Check for tree structure (each node has at most one parent)
        is_tree = all(
            sum(1 for other in hierarchy.values() if node.node_id in other.children_ids)
            <= 1
            for node in hierarchy.values()
        )

        if is_tree:
            return HierarchyStructure.TREE, 1.0
        else:
            return HierarchyStructure.DAG, 0.8  # Assume DAG if not tree

    def _calculate_level_statistics(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> dict[int, dict[str, Any]]:
        """Calculate statistics for each level."""
        level_stats: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "node_count": 0,
                "avg_children": 0.0,
                "max_children": 0,
                "min_children": float("inf"),
            }
        )

        # Group nodes by level
        level_nodes = defaultdict(list)
        for node in hierarchy.values():
            level_nodes[node.level].append(node)

        # Calculate statistics for each level
        for level, nodes in level_nodes.items():
            stats = level_stats[level]
            stats["node_count"] = len(nodes)

            child_counts = [len(node.children_ids) for node in nodes]

            if child_counts:
                stats["avg_children"] = sum(child_counts) / len(child_counts)
                stats["max_children"] = max(child_counts)
                stats["min_children"] = min(child_counts)
            else:
                stats["min_children"] = 0

        return dict(level_stats)

    def _rank_nodes(
        self, hierarchy: dict[str, HierarchyNode]
    ) -> dict[str, dict[str, float]]:
        """Rank nodes by various metrics."""
        rankings = {}

        for node in hierarchy.values():
            rankings[node.node_id] = {
                "centrality": self._calculate_node_centrality(hierarchy, node),
                "importance": self._calculate_node_importance(hierarchy, node),
                "connectivity": len(node.children_ids),
            }

        return rankings

    def _calculate_node_centrality(
        self, hierarchy: dict[str, HierarchyNode], node: HierarchyNode
    ) -> float:
        """Calculate node centrality metric."""
        # Simple centrality based on position in hierarchy
        max_depth = max(n.depth_from_root for n in hierarchy.values())

        if max_depth == 0:
            return 1.0

        # Nodes in the middle have higher centrality
        normalized_depth = node.depth_from_root / max_depth
        centrality = 1.0 - abs(normalized_depth - 0.5) * 2

        return centrality

    def _calculate_node_importance(
        self, hierarchy: dict[str, HierarchyNode], node: HierarchyNode
    ) -> float:
        """Calculate node importance metric."""
        # Importance based on subtree size and level
        total_nodes = len(hierarchy)

        if total_nodes <= 1:
            return 1.0

        subtree_importance = node.subtree_size / total_nodes
        level_importance = 1.0 / (1.0 + node.level)

        return (subtree_importance + level_importance) / 2.0

    def _detect_structural_anomalies(
        self, hierarchy: dict[str, HierarchyNode], metrics: HierarchyMetrics
    ) -> list[dict[str, Any]]:
        """Detect structural anomalies in the hierarchy."""
        anomalies = []
        threshold = self.analysis_config["anomaly_threshold"]

        # Detect nodes with unusual branching factors
        branching_factors = [
            len(node.children_ids) for node in hierarchy.values() if not node.is_leaf()
        ]

        if branching_factors:
            mean_branching = np.mean(branching_factors)
            std_branching = np.std(branching_factors)

            for node in hierarchy.values():
                if not node.is_leaf():
                    z_score = abs(  # type: ignore[call-overload, operator]
                        len(node.children_ids) - mean_branching
                    ) / max(std_branching, 0.1)

                    if z_score > threshold:
                        anomalies.append(
                            {
                                "type": "unusual_branching",
                                "node_id": node.node_id,
                                "value": len(node.children_ids),
                                "expected": mean_branching,
                                "z_score": z_score,
                            }
                        )

        return anomalies

    def _generate_optimization_suggestions(
        self, hierarchy: dict[str, HierarchyNode], metrics: HierarchyMetrics
    ) -> list[str]:
        """Generate optimization suggestions."""
        suggestions = []

        # Check balance
        if metrics.balance_factor < self.analysis_config["balance_threshold"]:
            suggestions.append(
                "Consider rebalancing the hierarchy to improve balance factor"
            )

        # Check efficiency
        if metrics.path_efficiency < self.analysis_config["efficiency_threshold"]:
            suggestions.append(
                "Consider reducing hierarchy depth to improve path efficiency"
            )

        # Check branching complexity
        if metrics.branching_complexity > 2.0:
            suggestions.append(
                "Consider standardizing branching factors to reduce complexity"
            )

        return suggestions

    def get_hierarchy_statistics(self) -> dict[str, Any]:
        """Get hierarchy analyzer statistics."""
        total_hierarchies = len(self.hierarchies)
        total_nodes = sum(len(h) for h in self.hierarchies.values())

        return {
            "analyzer_name": self.name,
            "total_hierarchies": total_hierarchies,
            "total_nodes": total_nodes,
            "analysis_stats": dict(self.analysis_stats),
            "stored_results": len(self.analysis_results),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
