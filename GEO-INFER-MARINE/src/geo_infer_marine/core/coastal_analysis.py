"""
Coastal zone analysis module.

Handles coastal vulnerability assessment and coastal zone management.
"""

import logging
from typing import Any
from collections.abc import Sequence
import xarray as xr

logger = logging.getLogger(__name__)


class CoastalAnalyzer:
    """
    Analyze coastal zones and assess vulnerability.
    """

    def __init__(self, config: dict | None = None):
        """Initialize coastal analyzer."""
        self.config = config or {}

    def assess_coastal_vulnerability(
        self,
        elevation: xr.DataArray,
        sea_level: xr.DataArray,
        wave_height: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess coastal vulnerability to sea-level rise.

        Args:
            elevation: Coastal elevation data
            sea_level: Sea level data
            wave_height: Optional wave height data

        Returns:
            Vulnerability assessment results
        """
        # Calculate relative elevation
        relative_elevation = elevation - sea_level

        # Vulnerability index (lower elevation = higher vulnerability).
        # Cells at or below sea level (relative elevation <= 0) are the
        # most exposed; clamping the relative elevation to a floor of 0
        # keeps the index finite and bounded in [0, 1] instead of
        # diverging at relative elevation == -1 or turning negative
        # below it.
        vulnerability = 1.0 / (relative_elevation.clip(min=0.0) + 1.0)

        if wave_height is not None:
            # Incorporate wave impacts
            vulnerability = vulnerability * (1 + wave_height / 10.0)

        vulnerability = vulnerability.clip(min=0.0, max=1.0)

        return xr.Dataset(
            {
                "relative_elevation": relative_elevation,
                "vulnerability_index": vulnerability,
            }
        )

    def analyze_coastal_erosion(
        self,
        shoreline_data: xr.DataArray,
        time_periods: Sequence[Any],
    ) -> xr.Dataset:
        """
        Analyze coastal erosion over time.

        Args:
            shoreline_data: Shoreline position data
            time_periods: List of time periods to analyze; erosion
                rates need at least two consecutive periods

        Returns:
            Erosion analysis results

        Raises:
            ValueError: If fewer than two time periods are supplied.
        """
        if len(time_periods) < 2:
            raise ValueError(
                "coastal_analysis.analyze_coastal_erosion requires at least "
                f"2 time periods to compute erosion rates, got {len(time_periods)}"
            )

        erosion_rates = []

        for i in range(len(time_periods) - 1):
            period1 = shoreline_data.sel(time=time_periods[i])
            period2 = shoreline_data.sel(time=time_periods[i + 1])
            erosion = period1 - period2  # Positive = erosion
            erosion_rates.append(erosion)

        return xr.Dataset({"erosion_rates": xr.concat(erosion_rates, dim="period")})
