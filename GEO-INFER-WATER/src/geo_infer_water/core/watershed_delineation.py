"""Watershed delineation with flow direction and accumulation algorithms.

Implements the D8 flow direction algorithm and flow accumulation
for extracting drainage basins from digital elevation models.
"""

import logging
from typing import cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# D8 direction encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
D8_DIRECTIONS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]
D8_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
D8_DISTANCES = [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]


class WatershedDelineator:
    """Delineate watersheds from digital elevation models using D8 algorithm.

    Implements flow direction calculation, flow accumulation, and
    basin extraction from raster DEMs.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize watershed delineator.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    def calculate_flow_direction_d8(
        self,
        dem: np.ndarray,
    ) -> np.ndarray:
        """Calculate D8 flow direction from a DEM.

        For each cell, determines the steepest downslope neighbor.
        Flow direction is encoded as powers of 2:
        64  128  1
        32   X   2
        16   8   4

        Args:
            dem: 2D array of elevation values.

        Returns:
            2D array of D8 flow direction codes.
        """
        rows, cols = dem.shape
        flow_dir = np.zeros((rows, cols), dtype=int)

        for i in range(rows):
            for j in range(cols):
                max_drop = 0.0
                best_dir = 0

                for idx in range(8):
                    di, dj = D8_OFFSETS[idx]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        drop = (dem[i, j] - dem[ni, nj]) / D8_DISTANCES[idx]
                        if drop > max_drop:
                            max_drop = drop
                            best_dir = D8_CODES[idx]

                flow_dir[i, j] = best_dir

        return flow_dir

    def calculate_flow_accumulation(
        self,
        flow_dir: np.ndarray,
    ) -> np.ndarray:
        """Calculate flow accumulation from D8 flow directions.

        Counts the number of upstream cells draining through each cell.

        Args:
            flow_dir: D8 flow direction array.

        Returns:
            2D array of flow accumulation values (cell counts).
        """
        rows, cols = flow_dir.shape
        accum = np.ones((rows, cols), dtype=float)

        in_degree = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                if flow_dir[i, j] == 0:
                    continue
                idx = (
                    D8_CODES.index(flow_dir[i, j]) if flow_dir[i, j] in D8_CODES else -1
                )
                if idx < 0:
                    continue
                di, dj = D8_OFFSETS[idx]
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    in_degree[ni, nj] += 1

        queue = []
        for i in range(rows):
            for j in range(cols):
                if in_degree[i, j] == 0:
                    queue.append((i, j))

        while queue:
            i, j = queue.pop(0)
            if flow_dir[i, j] == 0:
                continue
            idx = D8_CODES.index(flow_dir[i, j]) if flow_dir[i, j] in D8_CODES else -1
            if idx < 0:
                continue
            di, dj = D8_OFFSETS[idx]
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                accum[ni, nj] += accum[i, j]
                in_degree[ni, nj] -= 1
                if in_degree[ni, nj] == 0:
                    queue.append((ni, nj))

        return accum

    def delineate_basin(
        self,
        flow_dir: np.ndarray,
        outlet_row: int,
        outlet_col: int,
    ) -> np.ndarray:
        """Delineate a basin from an outlet point by tracing upstream.

        Args:
            flow_dir: D8 flow direction array.
            outlet_row: Row index of outlet.
            outlet_col: Column index of outlet.

        Returns:
            Binary mask (1 = in basin, 0 = outside).
        """
        rows, cols = flow_dir.shape
        basin = np.zeros((rows, cols), dtype=int)
        basin[outlet_row, outlet_col] = 1

        reverse_map: dict[tuple[int, int], list] = {}
        for i in range(rows):
            for j in range(cols):
                if flow_dir[i, j] == 0:
                    continue
                idx = (
                    D8_CODES.index(flow_dir[i, j]) if flow_dir[i, j] in D8_CODES else -1
                )
                if idx < 0:
                    continue
                di, dj = D8_OFFSETS[idx]
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    key = (ni, nj)
                    if key not in reverse_map:
                        reverse_map[key] = []
                    reverse_map[key].append((i, j))

        queue = [(outlet_row, outlet_col)]
        while queue:
            r, c = queue.pop(0)
            for ui, uj in reverse_map.get((r, c), []):
                if basin[ui, uj] == 0:
                    basin[ui, uj] = 1
                    queue.append((ui, uj))

        return basin

    def extract_stream_network(
        self,
        flow_accumulation: np.ndarray,
        threshold: float = 100.0,
    ) -> np.ndarray:
        """Extract stream network from flow accumulation.

        Args:
            flow_accumulation: Flow accumulation values.
            threshold: Minimum accumulation to classify as stream.

        Returns:
            Binary stream network mask.
        """
        return (flow_accumulation >= threshold).astype(int)

    def calculate_slope(
        self,
        dem: np.ndarray,
        cell_size: float = 30.0,
    ) -> np.ndarray:
        """Calculate terrain slope from DEM using 3x3 window.

        Uses Horn's method (1981) for slope calculation.

        Args:
            dem: 2D elevation array.
            cell_size: Grid cell size in meters.

        Returns:
            Slope in degrees.
        """
        dy, dx = np.gradient(dem, cell_size)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return cast("np.ndarray", np.degrees(slope_rad))

    def full_delineation(
        self,
        dem: xr.DataArray,
        outlet: tuple[int, int],
        stream_threshold: float = 100.0,
        cell_size: float = 30.0,
    ) -> xr.Dataset:
        """Run full watershed delineation pipeline.

        Args:
            dem: Digital elevation model as DataArray.
            outlet: (row, col) outlet position.
            stream_threshold: Flow accumulation threshold for streams.
            cell_size: Grid cell size in meters.

        Returns:
            Dataset with flow direction, accumulation, basin, streams, slope.
        """
        elev = dem.values if hasattr(dem, "values") else np.asarray(dem)
        if elev.ndim > 2:
            elev = elev.squeeze()

        flow_dir = self.calculate_flow_direction_d8(elev)
        flow_accum = self.calculate_flow_accumulation(flow_dir)
        basin = self.delineate_basin(flow_dir, outlet[0], outlet[1])
        streams = self.extract_stream_network(flow_accum, stream_threshold)
        slope = self.calculate_slope(elev, cell_size)

        basin_area_cells = int(basin.sum())

        dims = dem.dims if hasattr(dem, "dims") else ("y", "x")
        coords = dem.coords if hasattr(dem, "coords") else {}

        return xr.Dataset(
            {
                "flow_direction": xr.DataArray(
                    flow_dir,
                    dims=dims[-2:],
                    coords={k: v for k, v in coords.items() if k in dims[-2:]},
                ),
                "flow_accumulation": xr.DataArray(
                    flow_accum,
                    dims=dims[-2:],
                    coords={k: v for k, v in coords.items() if k in dims[-2:]},
                ),
                "basin_mask": xr.DataArray(
                    basin,
                    dims=dims[-2:],
                    coords={k: v for k, v in coords.items() if k in dims[-2:]},
                ),
                "stream_network": xr.DataArray(
                    streams,
                    dims=dims[-2:],
                    coords={k: v for k, v in coords.items() if k in dims[-2:]},
                ),
                "slope_degrees": xr.DataArray(
                    slope,
                    dims=dims[-2:],
                    coords={k: v for k, v in coords.items() if k in dims[-2:]},
                ),
            },
            attrs={
                "basin_area_cells": basin_area_cells,
                "basin_area_km2": float(basin_area_cells * (cell_size**2) / 1e6),
                "outlet": outlet,
                "stream_threshold": stream_threshold,
            },
        )
