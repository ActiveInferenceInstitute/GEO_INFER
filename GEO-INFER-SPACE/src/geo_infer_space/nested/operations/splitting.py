"""
Splitting Operations for H3 Nested Systems.

This module provides algorithms for splitting H3 cells into smaller units
based on various criteria such as resolution refinement, load balancing,
and adaptive subdivision.
"""

import logging
import uuid
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import Any, cast
from collections.abc import Callable
from enum import Enum
from collections import defaultdict

import numpy as np
import h3

logger = logging.getLogger(__name__)


class SplittingStrategy(Enum):
    """Strategies for splitting H3 cells."""

    RESOLUTION_REFINEMENT = "resolution_refinement"
    LOAD_BALANCING = "load_balancing"
    ADAPTIVE_SUBDIVISION = "adaptive_subdivision"
    GRADIENT_BASED = "gradient_based"
    THRESHOLD_BASED = "threshold_based"
    UNIFORM_SUBDIVISION = "uniform_subdivision"
    CUSTOM = "custom"


class SplittingCriterion(Enum):
    """Criteria for determining when to split cells."""

    LOAD_THRESHOLD = "load_threshold"
    GRADIENT_THRESHOLD = "gradient_threshold"
    SIZE_THRESHOLD = "size_threshold"
    DENSITY_THRESHOLD = "density_threshold"
    ERROR_THRESHOLD = "error_threshold"
    CUSTOM = "custom"


@dataclass
class SplittingRule:
    """
    Defines rules for splitting cells.
    """

    rule_id: str
    criterion: SplittingCriterion

    # Rule parameters
    threshold: float = 1.0
    min_threshold: float | None = None
    max_threshold: float | None = None

    # Target properties
    target_resolution: int | None = None
    max_children: int = 7  # H3 cells have 7 children

    # Evaluation function
    evaluation_function: Callable | None = None

    # Metadata
    weight: float = 1.0
    is_active: bool = True

    def should_split(
        self,
        cell: Any,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Determine if a cell should be split based on this rule.

        Args:
            cell: Cell to evaluate
            context: Additional context for evaluation

        Returns:
            True if cell should be split
        """
        if not self.is_active:
            return False

        if self.evaluation_function:
            try:
                return cast(bool, self.evaluation_function(cell, self, context))
            except Exception as e:
                logger.warning(f"Rule evaluation failed: {e}")
                return False

        # Default evaluation based on criterion
        if self.criterion == SplittingCriterion.LOAD_THRESHOLD:
            load_value = cell.state_variables.get("load", 0.0)
            return cast(bool, load_value > self.threshold)

        elif self.criterion == SplittingCriterion.SIZE_THRESHOLD:
            size_value = cell.state_variables.get("size", 0.0)
            return cast(bool, size_value > self.threshold)

        elif self.criterion == SplittingCriterion.DENSITY_THRESHOLD:
            density_value = cell.state_variables.get("density", 0.0)
            return cast(bool, density_value > self.threshold)

        return False


@dataclass
class SplittingResult:
    """
    Result of a splitting operation.
    """

    operation_id: str
    strategy: SplittingStrategy

    # Input cells
    input_cells: list[str] = field(default_factory=list)

    # Output cells (parent -> children mapping)
    split_cells: dict[str, list[str]] = field(default_factory=dict)

    # Cell properties
    cell_properties: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Quality metrics
    quality_score: float = 0.0
    balance_score: float = 0.0
    refinement_score: float = 0.0

    # Statistics
    num_input_cells: int = 0
    num_output_cells: int = 0
    expansion_ratio: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processing_time: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived statistics."""
        self.num_input_cells = len(self.input_cells)
        self.num_output_cells = sum(
            len(children) for children in self.split_cells.values()
        )

        if self.num_input_cells > 0:
            self.expansion_ratio = self.num_output_cells / self.num_input_cells


class H3SplittingEngine:
    """
    Advanced splitting engine for H3 nested systems.

    Provides multiple algorithms for subdividing H3 cells into smaller units
    based on load, gradients, thresholds, and other criteria.
    """

    def __init__(self, name: str = "H3SplittingEngine") -> None:
        """
        Initialize splitting engine.

        Args:
            name: Engine name for identification
        """
        self.name = name

        # Splitting rules
        self.rules: dict[str, SplittingRule] = {}

        # Results storage
        self.splitting_results: dict[str, SplittingResult] = {}

        # Statistics
        self.operation_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def add_rule(self, rule: SplittingRule) -> str:
        """
        Add a splitting rule.

        Args:
            rule: Splitting rule to add

        Returns:
            Rule ID
        """
        self.rules[rule.rule_id] = rule
        self.updated_at = datetime.now(UTC)
        return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a splitting rule.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def split_cells(
        self,
        nested_grid: Any,
        strategy: SplittingStrategy = SplittingStrategy.RESOLUTION_REFINEMENT,
        system_id: str | None = None,
        **kwargs: Any,
    ) -> SplittingResult:
        """
        Split cells in a nested grid system.

        Args:
            nested_grid: NestedH3Grid instance
            strategy: Splitting strategy to use
            system_id: Specific system to split (None for all)
            **kwargs: Strategy-specific parameters

        Returns:
            SplittingResult instance
        """
        start_time = datetime.now(UTC)
        operation_id = f"split_{uuid.uuid4().hex[:8]}"

        # Get cells to split
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
            return SplittingResult(
                operation_id=operation_id, strategy=strategy, processing_time=0.0
            )

        # Apply splitting strategy
        if strategy == SplittingStrategy.RESOLUTION_REFINEMENT:
            split_cells = self._split_by_resolution(cells, **kwargs)
        elif strategy == SplittingStrategy.LOAD_BALANCING:
            split_cells = self._split_by_load_balancing(cells, **kwargs)
        elif strategy == SplittingStrategy.ADAPTIVE_SUBDIVISION:
            split_cells = self._split_adaptive(cells, **kwargs)
        elif strategy == SplittingStrategy.GRADIENT_BASED:
            split_cells = self._split_by_gradient(cells, **kwargs)
        elif strategy == SplittingStrategy.THRESHOLD_BASED:
            split_cells = self._split_by_threshold(cells, **kwargs)
        elif strategy == SplittingStrategy.UNIFORM_SUBDIVISION:
            split_cells = self._split_uniform(cells, **kwargs)
        else:
            logger.warning(f"Unknown splitting strategy: {strategy}")
            split_cells = self._split_by_resolution(cells, **kwargs)

        # Calculate processing time
        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        # Create result
        result = SplittingResult(
            operation_id=operation_id,
            strategy=strategy,
            input_cells=[cell.index for cell in cells],
            split_cells=split_cells,
            processing_time=processing_time,
        )

        # Calculate quality metrics
        result.quality_score = self._calculate_quality_score(cells, split_cells)
        result.balance_score = self._calculate_balance_score(split_cells)
        result.refinement_score = self._calculate_refinement_score(cells, split_cells)

        # Store result
        self.splitting_results[operation_id] = result
        self.operation_stats[strategy.value] += 1
        self.updated_at = datetime.now(UTC)

        return result

    def _split_by_resolution(
        self, cells: list[Any], **kwargs: Any
    ) -> dict[str, list[str]]:
        """Split cells by increasing resolution."""
        target_resolution = kwargs.get("target_resolution")

        split_cells = {}

        for cell in cells:
            try:
                current_resolution = h3.get_resolution(cell.index)

                # Determine target resolution
                if target_resolution is None:
                    target_res = min(
                        15, current_resolution + 1
                    )  # H3 max resolution is 15
                else:
                    target_res = min(15, target_resolution)

                if target_res > current_resolution:
                    # Get children at target resolution
                    children = h3.cell_to_children(cell.index, target_res)
                    split_cells[cell.index] = list(children)

            except Exception as e:
                logger.warning(f"Failed to split cell {cell.index} by resolution: {e}")
                continue

        return split_cells

    def _split_by_load_balancing(
        self, cells: list[Any], **kwargs: Any
    ) -> dict[str, list[str]]:
        """Split cells based on load balancing."""
        load_threshold = kwargs.get("load_threshold", 1.0)
        target_load = kwargs.get("target_load", 0.5)

        split_cells = {}

        for cell in cells:
            current_load = cell.state_variables.get("load", 0.0)

            if current_load > load_threshold:
                try:
                    # Split to next resolution
                    current_resolution = h3.get_resolution(cell.index)
                    target_resolution = min(15, current_resolution + 1)

                    children = h3.cell_to_children(cell.index, target_resolution)

                    # Distribute load among children
                    child_load = current_load / len(children)

                    if child_load <= target_load:
                        split_cells[cell.index] = list(children)

                except Exception as e:
                    logger.warning(
                        f"Failed to split cell {cell.index} for load balancing: {e}"
                    )
                    continue

        return split_cells

    def _split_adaptive(self, cells: list[Any], **kwargs: Any) -> dict[str, list[str]]:
        """Split cells using adaptive subdivision."""
        adaptation_field = kwargs.get("adaptation_field", "gradient")
        adaptation_threshold = kwargs.get("adaptation_threshold", 0.5)

        split_cells = {}

        for cell in cells:
            adaptation_value = cell.state_variables.get(adaptation_field, 0.0)

            if adaptation_value > adaptation_threshold:
                try:
                    # Apply splitting rules
                    should_split = False
                    for rule in self.rules.values():
                        if rule.should_split(cell):
                            should_split = True
                            break

                    if should_split:
                        current_resolution = h3.get_resolution(cell.index)
                        target_resolution = min(15, current_resolution + 1)

                        children = h3.cell_to_children(cell.index, target_resolution)
                        split_cells[cell.index] = list(children)

                except Exception as e:
                    logger.warning(f"Failed to adaptively split cell {cell.index}: {e}")
                    continue

        return split_cells

    def _split_by_gradient(
        self, cells: list[Any], **kwargs: Any
    ) -> dict[str, list[str]]:
        """Split cells based on gradient analysis."""
        gradient_field = kwargs.get("gradient_field", "value")
        gradient_threshold = kwargs.get("gradient_threshold", 0.3)

        split_cells = {}

        # Calculate gradients for each cell
        for cell in cells:
            if gradient_field not in cell.state_variables:
                continue

            try:
                # Get neighbors and calculate gradient
                neighbors = h3.grid_ring(cell.index, 1)
                cell_value = cell.state_variables[gradient_field]

                neighbor_values = []
                for neighbor_idx in neighbors:
                    # Find neighbor cell in our system
                    neighbor_cell = None
                    for c in cells:
                        if c.index == neighbor_idx:
                            neighbor_cell = c
                            break

                    if (
                        neighbor_cell
                        and gradient_field in neighbor_cell.state_variables
                    ):
                        neighbor_values.append(
                            neighbor_cell.state_variables[gradient_field]
                        )

                if neighbor_values:
                    # Calculate gradient magnitude
                    gradient_magnitude = np.std(neighbor_values + [cell_value])

                    if gradient_magnitude > gradient_threshold:
                        current_resolution = h3.get_resolution(cell.index)
                        target_resolution = min(15, current_resolution + 1)

                        children = h3.cell_to_children(cell.index, target_resolution)
                        split_cells[cell.index] = list(children)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate gradient for cell {cell.index}: {e}"
                )
                continue

        return split_cells

    def _split_by_threshold(
        self, cells: list[Any], **kwargs: Any
    ) -> dict[str, list[str]]:
        """Split cells based on threshold criteria."""
        threshold_field = kwargs.get("threshold_field", "value")
        threshold_value = kwargs.get("threshold_value", 1.0)

        split_cells = {}

        for cell in cells:
            if threshold_field not in cell.state_variables:
                continue

            field_value = cell.state_variables[threshold_field]

            if field_value > threshold_value:
                try:
                    current_resolution = h3.get_resolution(cell.index)
                    target_resolution = min(15, current_resolution + 1)

                    children = h3.cell_to_children(cell.index, target_resolution)
                    split_cells[cell.index] = list(children)

                except Exception as e:
                    logger.warning(
                        f"Failed to split cell {cell.index} by threshold: {e}"
                    )
                    continue

        return split_cells

    def _split_uniform(self, cells: list[Any], **kwargs: Any) -> dict[str, list[str]]:
        """Split all cells uniformly."""
        target_resolution = kwargs.get("target_resolution")

        split_cells = {}

        for cell in cells:
            try:
                current_resolution = h3.get_resolution(cell.index)

                if target_resolution is None:
                    target_res = min(15, current_resolution + 1)
                else:
                    target_res = min(15, target_resolution)

                if target_res > current_resolution:
                    children = h3.cell_to_children(cell.index, target_res)
                    split_cells[cell.index] = list(children)

            except Exception as e:
                logger.warning(f"Failed to uniformly split cell {cell.index}: {e}")
                continue

        return split_cells

    def _calculate_quality_score(
        self, cells: list, split_cells: dict[str, list[str]]
    ) -> float:
        """Calculate overall quality score for splitting result."""
        if not split_cells:
            return 0.0

        # Quality based on successful splits vs total cells
        successful_splits = len(split_cells)
        total_cells = len(cells)

        if total_cells == 0:
            return 0.0

        return successful_splits / total_cells

    def _calculate_balance_score(self, split_cells: dict[str, list[str]]) -> float:
        """Calculate balance score for split results."""
        if not split_cells:
            return 0.0

        # Balance based on uniformity of children counts
        children_counts = [len(children) for children in split_cells.values()]

        if not children_counts:
            return 0.0

        # Lower variance means better balance
        mean_count: float = cast(float, np.mean(children_counts))
        variance: float = cast(float, np.var(children_counts))

        if mean_count > 0:
            balance = 1.0 / (1.0 + variance / mean_count)
        else:
            balance = 0.0

        return min(1.0, max(0.0, balance))

    def _calculate_refinement_score(
        self, cells: list, split_cells: dict[str, list[str]]
    ) -> float:
        """Calculate refinement score for split results."""
        if not split_cells:
            return 0.0

        refinement_scores = []

        for cell in cells:
            if cell.index in split_cells:
                try:
                    original_resolution = h3.get_resolution(cell.index)
                    children = split_cells[cell.index]

                    if children:
                        child_resolution = h3.get_resolution(children[0])
                        refinement_level = child_resolution - original_resolution

                        # Higher refinement is better (up to a point)
                        refinement_score = min(1.0, refinement_level / 3.0)
                        refinement_scores.append(refinement_score)

                except Exception as e:
                    logger.warning(
                        f"Failed to calculate refinement for cell {cell.index}: {e}"
                    )
                    continue

        return (
            sum(refinement_scores) / len(refinement_scores)
            if refinement_scores
            else 0.0
        )

    def get_splitting_statistics(self) -> dict[str, Any]:
        """Get splitting engine statistics."""
        total_operations = sum(self.operation_stats.values())

        return {
            "engine_name": self.name,
            "total_operations": total_operations,
            "operation_stats": dict(self.operation_stats),
            "active_rules": len(self.rules),
            "stored_results": len(self.splitting_results),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
