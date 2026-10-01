"""
Cellular Automata for GEO-INFER-SIM.

This module provides cellular automata modeling for spatial processes
based on local rules applied to grid cells.
"""

import logging
from typing import Any
from collections.abc import Callable
import numpy as np

logger = logging.getLogger(__name__)


class CellularAutomata:
    """
    Cellular Automata model for geospatial simulations.

    Models spatial processes using a grid of cells where each cell's
    state changes based on local rules and neighbor states.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        initial_states: np.ndarray | None = None,
        num_states: int = 2,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the cellular automata.

        Args:
            grid_shape: Shape of the grid (height, width)
            initial_states: Optional initial cell states
            num_states: Number of possible cell states
            random_seed: Optional seed for the isolated random Generator
        """
        # Deterministic-by-default: isolated Generator instead of global state
        self.rng: np.random.Generator = np.random.default_rng(random_seed)

        self.grid_shape = grid_shape
        self.num_states = num_states

        if initial_states is not None:
            if initial_states.shape != grid_shape:
                raise ValueError(
                    f"Initial states shape {initial_states.shape} "
                    f"does not match grid shape {grid_shape}"
                )
            self.grid = initial_states.copy()
        else:
            # Initialize with random states
            self.grid = self.rng.integers(0, num_states, size=grid_shape)

        self.time = 0.0
        self.history: list[np.ndarray] = []

    def get_neighbors(
        self, row: int, col: int, neighborhood: str = "moore"
    ) -> list[tuple[int, int]]:
        """
        Get neighbor cell coordinates.

        Args:
            row: Cell row
            col: Cell column
            neighborhood: Neighborhood type ("moore" for 8 neighbors, "von_neumann" for 4)

        Returns:
            List of (row, col) tuples for neighbors
        """
        neighbors = []

        if neighborhood == "moore":
            # 8 neighbors (including diagonals)
            offsets = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        elif neighborhood == "von_neumann":
            # 4 neighbors (no diagonals)
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        else:
            raise ValueError(f"Unknown neighborhood type: {neighborhood}")

        for dr, dc in offsets:
            new_row = row + dr
            new_col = col + dc

            if 0 <= new_row < self.grid_shape[0] and 0 <= new_col < self.grid_shape[1]:
                neighbors.append((new_row, new_col))

        return neighbors

    def apply_rule(
        self,
        rule_func: Callable[[int, list[int]], int],
        neighborhood: str = "moore",
    ) -> None:
        """
        Apply a transition rule to all cells.

        Args:
            rule_func: Function that takes (current_state, neighbor_states) and returns new_state
            neighborhood: Neighborhood type
        """
        new_grid = self.grid.copy()

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                current_state = self.grid[row, col]
                neighbors = self.get_neighbors(row, col, neighborhood)
                neighbor_states = [self.grid[r, c] for r, c in neighbors]

                new_state = rule_func(current_state, neighbor_states)
                new_grid[row, col] = new_state

        self.grid = new_grid

    def step(
        self,
        rule_func: Callable[[int, list[int]], int] | None = None,
        neighborhood: str = "moore",
    ) -> None:
        """
        Execute one simulation step.

        Args:
            rule_func: Optional transition rule function
            neighborhood: Neighborhood type
        """
        if rule_func:
            self.apply_rule(rule_func, neighborhood)
        else:
            # Default: Game of Life rule
            def game_of_life_rule(current: int, neighbors: list[int]) -> int:
                alive_neighbors = sum(1 for n in neighbors if n == 1)
                if current == 1:
                    return 1 if 2 <= alive_neighbors <= 3 else 0
                else:
                    return 1 if alive_neighbors == 3 else 0

            self.apply_rule(game_of_life_rule, neighborhood)

        self.time += 1.0

        # Save history (every 10 steps to save memory)
        if int(self.time) % 10 == 0:
            self.history.append(self.grid.copy())

    def get_state(self) -> dict[str, Any]:
        """
        Get current model state.

        Returns:
            Model state dictionary
        """
        return {
            "time": self.time,
            "grid_shape": self.grid_shape,
            "num_states": self.num_states,
            "state_counts": {
                state: int(np.sum(self.grid == state))
                for state in range(self.num_states)
            },
        }

    def reset(self, initial_states: np.ndarray | None = None) -> None:
        """
        Reset the model to initial state.

        Args:
            initial_states: Optional new initial states
        """
        if initial_states is not None:
            self.grid = initial_states.copy()
        else:
            self.grid = self.rng.integers(0, self.num_states, size=self.grid_shape)

        self.time = 0.0
        self.history = []
        logger.info("Cellular automata reset")
