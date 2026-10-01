"""
Aggregation Operations for H3 Nested Systems.

This module provides algorithms for aggregating data across H3 cells
in nested systems, supporting various aggregation functions and
hierarchical data processing.
"""

import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h3

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3-py package not available")


class AggregationFunction(Enum):
    """Types of aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    VAR = "var"
    WEIGHTED_MEAN = "weighted_mean"
    PERCENTILE = "percentile"
    MODE = "mode"
    CUSTOM = "custom"


class AggregationScope(Enum):
    """Scope of aggregation operations."""

    CELL_NEIGHBORS = "cell_neighbors"
    RESOLUTION_LEVEL = "resolution_level"
    SYSTEM_WIDE = "system_wide"
    BOUNDARY_REGION = "boundary_region"
    CUSTOM_REGION = "custom_region"
    HIERARCHICAL = "hierarchical"


@dataclass
class AggregationRule:
    """
    Defines rules for data aggregation.
    """

    rule_id: str
    source_field: str
    target_field: str
    function: AggregationFunction
    scope: AggregationScope

    # Function parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Scope parameters
    scope_parameters: dict[str, Any] = field(default_factory=dict)

    # Weights (for weighted aggregations)
    weight_field: str | None = None

    # Custom function
    custom_function: Callable | None = None

    # Metadata
    is_active: bool = True
    priority: int = 1

    def apply(
        self, cells: list, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Apply aggregation rule to cells.

        Args:
            cells: List of cells to aggregate
            context: Additional context for aggregation

        Returns:
            Aggregated results
        """
        if not self.is_active or not cells:
            return {}

        # Extract values for aggregation
        values = []
        weights = []

        for cell in cells:
            if self.source_field in cell.state_variables:
                value = cell.state_variables[self.source_field]
                values.append(value)

                # Extract weight if specified
                if self.weight_field and self.weight_field in cell.state_variables:
                    weight = cell.state_variables[self.weight_field]
                    weights.append(weight)
                else:
                    weights.append(1.0)

        if not values:
            return {}

        # Apply aggregation function
        try:
            if self.function == AggregationFunction.CUSTOM and self.custom_function:
                result = self.custom_function(values, weights, self.parameters, context)
            else:
                result = self._apply_standard_function(values, weights)

            return {self.target_field: result}

        except Exception as e:
            logger.warning(f"Aggregation rule {self.rule_id} failed: {e}")
            return {}

    def _apply_standard_function(self, values: list, weights: list) -> Any:
        """Apply standard aggregation function."""
        values_array = np.array(values)
        weights_array = np.array(weights)

        if self.function == AggregationFunction.SUM:
            return np.sum(values_array)
        elif self.function == AggregationFunction.MEAN:
            return np.mean(values_array)
        elif self.function == AggregationFunction.MEDIAN:
            return np.median(values_array)
        elif self.function == AggregationFunction.MIN:
            return np.min(values_array)
        elif self.function == AggregationFunction.MAX:
            return np.max(values_array)
        elif self.function == AggregationFunction.COUNT:
            return len(values_array)
        elif self.function == AggregationFunction.STD:
            return np.std(values_array)
        elif self.function == AggregationFunction.VAR:
            return np.var(values_array)
        elif self.function == AggregationFunction.WEIGHTED_MEAN:
            if np.sum(weights_array) > 0:
                return np.average(values_array, weights=weights_array)
            else:
                return np.mean(values_array)
        elif self.function == AggregationFunction.PERCENTILE:
            percentile = self.parameters.get("percentile", 50)
            return np.percentile(values_array, percentile)
        elif self.function == AggregationFunction.MODE:
            unique_values, counts = np.unique(values_array, return_counts=True)
            return unique_values[np.argmax(counts)]
        else:
            return np.mean(values_array)  # Default

    def _apply_simple_function(self, values: list, weights: list) -> Any:
        """Apply simple aggregation function without NumPy."""
        if self.function == AggregationFunction.SUM:
            return sum(values)
        elif self.function == AggregationFunction.MEAN:
            return sum(values) / len(values)
        elif self.function == AggregationFunction.MIN:
            return min(values)
        elif self.function == AggregationFunction.MAX:
            return max(values)
        elif self.function == AggregationFunction.COUNT:
            return len(values)
        elif self.function == AggregationFunction.WEIGHTED_MEAN:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                return weighted_sum / total_weight
            else:
                return sum(values) / len(values)
        else:
            return sum(values) / len(values)  # Default to mean


@dataclass
class AggregationResult:
    """
    Result of an aggregation operation.
    """

    operation_id: str
    rules_applied: list[str] = field(default_factory=list)

    # Aggregated data
    aggregated_data: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # cell_id -> aggregated_values

    # Summary statistics
    summary_stats: dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    coverage_ratio: float = 0.0
    completeness_score: float = 0.0

    # Statistics
    cells_processed: int = 0
    rules_succeeded: int = 0
    rules_failed: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class H3AggregationEngine:
    """
    Advanced aggregation engine for H3 nested systems.

    Provides multiple algorithms for aggregating data across H3 cells
    with support for hierarchical processing, custom functions, and
    various aggregation scopes.
    """

    def __init__(self, name: str = "H3AggregationEngine") -> None:
        """
        Initialize aggregation engine.

        Args:
            name: Engine name for identification
        """
        self.name = name

        # Aggregation rules
        self.rules: dict[str, AggregationRule] = {}

        # Results storage
        self.aggregation_results: dict[str, AggregationResult] = {}

        # Statistics
        self.operation_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_rule(self, rule: AggregationRule) -> str:
        """
        Add an aggregation rule.

        Args:
            rule: Aggregation rule to add

        Returns:
            Rule ID
        """
        self.rules[rule.rule_id] = rule
        self.updated_at = datetime.now()
        return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an aggregation rule.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.updated_at = datetime.now()
            return True
        return False

    def aggregate_data(
        self,
        nested_grid: Any,
        system_id: str | None = None,
        rule_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AggregationResult:
        """
        Aggregate data in a nested grid system.

        Args:
            nested_grid: NestedH3Grid instance
            system_id: Specific system to aggregate (None for all)
            rule_ids: Specific rules to apply (None for all active rules)
            **kwargs: Additional parameters

        Returns:
            AggregationResult instance
        """
        start_time = datetime.now()
        operation_id = f"agg_{uuid.uuid4().hex[:8]}"

        # Get cells to aggregate
        if system_id:
            system = nested_grid.get_system_by_id(system_id)
            if not system:
                raise ValueError(f"System {system_id} not found")
            cells = list(system.cells.values())
        else:
            cells = []
            for system in nested_grid.systems.values():
                cells.extend(system.cells.values())

        if not cells:
            return AggregationResult(operation_id=operation_id, processing_time=0.0)

        # Get rules to apply
        if rule_ids:
            rules_to_apply = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            rules_to_apply = [rule for rule in self.rules.values() if rule.is_active]

        # Sort rules by priority
        rules_to_apply.sort(key=lambda r: r.priority, reverse=True)

        # Apply aggregation rules
        aggregated_data: dict[str, dict[str, Any]] = {}
        rules_applied = []
        rules_succeeded = 0
        rules_failed = 0

        for rule in rules_to_apply:
            try:
                if rule.scope == AggregationScope.CELL_NEIGHBORS:
                    cell_results = self._aggregate_by_neighbors(cells, rule)
                elif rule.scope == AggregationScope.RESOLUTION_LEVEL:
                    cell_results = self._aggregate_by_resolution(cells, rule)
                elif rule.scope == AggregationScope.SYSTEM_WIDE:
                    cell_results = self._aggregate_system_wide(cells, rule)
                elif rule.scope == AggregationScope.BOUNDARY_REGION:
                    cell_results = self._aggregate_by_boundary(cells, rule, nested_grid)
                elif rule.scope == AggregationScope.HIERARCHICAL:
                    cell_results = self._aggregate_hierarchical(
                        cells, rule, nested_grid
                    )
                else:
                    cell_results = self._aggregate_custom_region(cells, rule)

                # Merge results
                for cell_id, values in cell_results.items():
                    if cell_id not in aggregated_data:
                        aggregated_data[cell_id] = {}
                    aggregated_data[cell_id].update(values)

                rules_applied.append(rule.rule_id)
                rules_succeeded += 1

            except Exception as e:
                logger.warning(f"Aggregation rule {rule.rule_id} failed: {e}")
                rules_failed += 1
                continue

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Create result
        result = AggregationResult(
            operation_id=operation_id,
            rules_applied=rules_applied,
            aggregated_data=aggregated_data,
            cells_processed=len(cells),
            rules_succeeded=rules_succeeded,
            rules_failed=rules_failed,
            processing_time=processing_time,
        )

        # Calculate quality metrics
        result.coverage_ratio = len(aggregated_data) / len(cells) if cells else 0.0
        result.completeness_score = (
            rules_succeeded / len(rules_to_apply) if rules_to_apply else 0.0
        )

        # Calculate summary statistics
        result.summary_stats = self._calculate_summary_stats(aggregated_data)

        # Store result
        self.aggregation_results[operation_id] = result
        self.operation_stats["total_operations"] += 1
        self.updated_at = datetime.now()

        return result

    def _aggregate_by_neighbors(
        self, cells: list, rule: AggregationRule
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data using cell neighbors."""
        neighbor_radius = rule.scope_parameters.get("radius", 1)

        if not H3_AVAILABLE:
            logger.warning("h3-py required for neighbor-based aggregation")
            return {}

        results = {}

        # Create cell index lookup
        cell_lookup = {cell.index: cell for cell in cells}

        for cell in cells:
            try:
                # Get neighbors
                neighbors = h3.grid_disk(cell.index, neighbor_radius)

                # Find neighbor cells in our dataset
                neighbor_cells = []
                for neighbor_idx in neighbors:
                    if neighbor_idx in cell_lookup:
                        neighbor_cells.append(cell_lookup[neighbor_idx])

                if neighbor_cells:
                    aggregated = rule.apply(neighbor_cells)
                    if aggregated:
                        results[cell.index] = aggregated

            except Exception as e:
                logger.warning(
                    f"Failed to aggregate neighbors for cell {cell.index}: {e}"
                )
                continue

        return results

    def _aggregate_by_resolution(
        self, cells: list, rule: AggregationRule
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data by resolution level."""
        target_resolution = rule.scope_parameters.get("resolution")

        if not H3_AVAILABLE:
            logger.warning("h3-py required for resolution-based aggregation")
            return {}

        # Group cells by resolution
        resolution_groups = defaultdict(list)

        for cell in cells:
            try:
                resolution = h3.get_resolution(cell.index)

                if target_resolution is None or resolution == target_resolution:
                    resolution_groups[resolution].append(cell)

            except Exception as e:
                logger.warning(f"Failed to get resolution for cell {cell.index}: {e}")
                continue

        # Aggregate within each resolution group
        results = {}

        for resolution_cells in resolution_groups.values():
            aggregated = rule.apply(resolution_cells)

            # Assign aggregated values to all cells in the group
            for cell in resolution_cells:
                if aggregated:
                    results[cell.index] = aggregated

        return results

    def _aggregate_system_wide(
        self, cells: list, rule: AggregationRule
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data across entire system."""
        aggregated = rule.apply(cells)

        # Assign aggregated values to all cells
        results = {}
        for cell in cells:
            if aggregated:
                results[cell.index] = aggregated

        return results

    def _aggregate_by_boundary(
        self,
        cells: list[Any],
        rule: AggregationRule,
        nested_grid: Any,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data by boundary regions."""

        # This would require boundary information from the nested grid
        # For now, fall back to neighbor-based aggregation
        return self._aggregate_by_neighbors(cells, rule)

    def _aggregate_hierarchical(
        self,
        cells: list[Any],
        rule: AggregationRule,
        nested_grid: Any,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data hierarchically."""
        if not H3_AVAILABLE:
            logger.warning("h3-py required for hierarchical aggregation")
            return {}

        # Group cells by parent-child relationships
        hierarchy_groups = defaultdict(list)

        for cell in cells:
            try:
                resolution = h3.get_resolution(cell.index)

                if resolution > 0:
                    parent = h3.cell_to_parent(cell.index, resolution - 1)
                    hierarchy_groups[parent].append(cell)
                else:
                    hierarchy_groups[cell.index].append(cell)

            except Exception as e:
                logger.warning(f"Failed to get hierarchy for cell {cell.index}: {e}")
                continue

        # Aggregate within each hierarchy group
        results = {}

        for child_cells in hierarchy_groups.values():
            aggregated = rule.apply(child_cells)

            # Assign aggregated values to all cells in the group
            for cell in child_cells:
                if aggregated:
                    results[cell.index] = aggregated

        return results

    def _aggregate_custom_region(
        self, cells: list, rule: AggregationRule
    ) -> dict[str, dict[str, Any]]:
        """Aggregate data in custom regions."""
        # This would implement custom region-based aggregation
        # For now, fall back to system-wide aggregation
        return self._aggregate_system_wide(cells, rule)

    def _calculate_summary_stats(
        self, aggregated_data: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate summary statistics for aggregated data."""
        if not aggregated_data:
            return {}

        # Collect all field values
        field_values = defaultdict(list)

        for cell_data in aggregated_data.values():
            for field_name, value in cell_data.items():
                if isinstance(value, (int, float)):
                    field_values[field_name].append(value)

        # Calculate statistics for each field
        summary = {}

        for field_name, values in field_values.items():
            if values:
                values_array = np.array(values)
                summary[field_name] = {
                    "count": len(values),
                    "mean": np.mean(values_array),
                    "std": np.std(values_array),
                    "min": np.min(values_array),
                    "max": np.max(values_array),
                    "median": np.median(values_array),
                }

        return summary

    def get_aggregation_statistics(self) -> dict[str, Any]:
        """Get aggregation engine statistics."""
        return {
            "engine_name": self.name,
            "operation_stats": dict(self.operation_stats),
            "active_rules": len([r for r in self.rules.values() if r.is_active]),
            "total_rules": len(self.rules),
            "stored_results": len(self.aggregation_results),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
