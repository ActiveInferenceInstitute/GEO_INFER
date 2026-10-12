"""Forest inventory and biomass estimation."""

import logging
import xarray as xr

import numpy as np


def _normalized(data: xr.DataArray) -> xr.DataArray:
    """Scale a DataArray to [0, 1] by its finite maximum.

    Returns ``data / max``; uniform 1.0 when the maximum is zero,
    negative, or not finite, so an all-zero tree density cannot
    propagate NaN into biomass estimates.
    """
    max_value = float(data.max())
    if max_value <= 0.0 or not np.isfinite(max_value):
        return xr.ones_like(data)
    return data / max_value


logger = logging.getLogger(__name__)


class ForestInventory:
    """Forest inventory and biomass estimation."""

    def __init__(self, config: dict | None = None):
        """Initialize forest inventory."""
        self.config = config or {}

    def estimate_biomass(
        self, forest_cover: xr.DataArray, tree_density: xr.DataArray | None = None
    ) -> xr.DataArray:
        """
        Estimate forest biomass.

        Args:
            forest_cover: Forest cover percentage
            tree_density: Optional tree density data

        Returns:
            Estimated biomass (tons/ha)
        """
        # Allometric biomass estimation
        # Typical: 50-200 tons/ha depending on forest type
        base_biomass = 100.0  # tons/ha for mature forest

        biomass = forest_cover / 100.0 * base_biomass

        if tree_density is not None:
            # Adjust based on tree density
            biomass = biomass * _normalized(tree_density)

        return biomass

    def calculate_forest_area(
        self, forest_cover: xr.DataArray, cell_area: xr.DataArray | None = None
    ) -> xr.DataArray:
        """
        Calculate forest area.

        Args:
            forest_cover: Forest cover percentage
            cell_area: Optional cell area data

        Returns:
            Forest area
        """
        if cell_area is None:
            # Assume standard cell size (e.g., from H3)
            cell_area = xr.ones_like(forest_cover) * 0.1  # km²

        forest_area = forest_cover / 100.0 * cell_area
        return forest_area
