"""Ordered H3 state spaces with bounded, stochastic movement operators."""

from dataclasses import dataclass
import math
from collections.abc import Iterable

import h3
import numpy as np
from scipy.sparse import csc_matrix, eye


@dataclass(frozen=True, init=False)
class H3StateSpace:
    """An immutable state order; indices retain caller order across artifacts."""

    cells: tuple[str, ...]

    def __init__(self, cells: Iterable[str], *, max_cells: int = 100_000):
        if (
            isinstance(max_cells, bool)
            or not isinstance(max_cells, int)
            or max_cells < 1
        ):
            raise ValueError("max_cells must be a positive integer")
        ordered: list[str] = []
        for cell in cells:
            if len(ordered) >= max_cells:
                raise ValueError("H3 state count exceeds max_cells")
            if (
                not isinstance(cell, str)
                or not h3.is_valid_cell(cell)
                or h3.int_to_str(h3.str_to_int(cell)) != cell
            ):
                raise ValueError("State IDs must be canonical H3 cell strings")
            ordered.append(cell)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("H3 cells must be nonempty and unique")
        if len({h3.get_resolution(cell) for cell in ordered}) != 1:
            raise ValueError("H3 state cells must share a resolution")
        object.__setattr__(self, "cells", tuple(ordered))

    def locate(self, latitude: float, longitude: float) -> int:
        """Return the state for a WGS84 point; outside-domain points raise."""
        if (
            isinstance(latitude, bool)
            or isinstance(longitude, bool)
            or not isinstance(latitude, (int, float))
            or not isinstance(longitude, (int, float))
            or not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise ValueError("Expected finite WGS84 latitude/longitude")
        cell = h3.latlng_to_cell(latitude, longitude, h3.get_resolution(self.cells[0]))
        try:
            return self.cells.index(cell)
        except ValueError as exc:
            raise ValueError("Observation lies outside the H3 state space") from exc

    def transitions(self) -> tuple[csc_matrix, csc_matrix]:
        """Return stay and diffuse operators indexed [next, current].

        Diffuse chooses uniformly among the real H3 neighbors. Probability for
        excluded neighbors stays at the source (a reflecting boundary).
        Pentagon degree comes from H3 topology, never an assumed six neighbors.
        """
        index = {cell: i for i, cell in enumerate(self.cells)}
        rows, columns, weights = [], [], []
        for column, cell in enumerate(self.cells):
            neighbors = sorted(set(h3.grid_disk(cell, 1)) - {cell})
            if not neighbors or any(
                not h3.is_valid_cell(neighbor)
                or h3.get_resolution(neighbor) != h3.get_resolution(cell)
                or not h3.are_neighbor_cells(cell, neighbor)
                for neighbor in neighbors
            ):
                raise ValueError("H3 topology returned an invalid neighbor domain")
            for neighbor in neighbors:
                rows.append(index.get(neighbor, column))
                columns.append(column)
                weights.append(1.0 / len(neighbors))
        size = len(self.cells)
        diffuse = csc_matrix((weights, (rows, columns)), shape=(size, size))
        return eye(size, format="csc"), diffuse

    def dense_transition_tensor(self, *, max_entries: int = 1_000_000) -> np.ndarray:
        """Materialize [next, current, action] only within an explicit budget."""
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")
        if 2 * len(self.cells) ** 2 > max_entries:
            raise ValueError("Dense transition tensor exceeds max_entries")
        return np.stack([matrix.toarray() for matrix in self.transitions()], axis=2)
