"""
Sea-level rise analysis module.
"""

import logging
from typing import cast
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class SeaLevelAnalyzer:
    """
    Analyze sea-level rise and impacts.
    """

    def __init__(self, config: dict | None = None):
        """Initialize sea-level analyzer."""
        self.config = config or {}

    def project_sea_level_rise(
        self,
        historical_data: xr.DataArray,
        scenario: str = "rcp45",
        years: list[int] | None = None,
    ) -> xr.DataArray:
        """
        Project future sea-level rise.

        Args:
            historical_data: Historical sea-level data
            scenario: Climate scenario (rcp26, rcp45, rcp85)
            years: List of future years to project

        Returns:
            Projected sea-level data
        """
        years = years or [2050, 2100]

        # Calculate trend
        trend = self._calculate_trend(historical_data)

        # Scenario-based scaling
        scenario_factors = {"rcp26": 0.5, "rcp45": 1.0, "rcp85": 2.0}
        factor = scenario_factors.get(scenario, 1.0)

        # Project
        projections = []
        for year in years:
            years_ahead = year - (
                historical_data.time.max().values.astype("datetime64[Y]").astype(int)
                + 1970
            )
            projected = historical_data.mean(dim="time") + trend * years_ahead * factor
            projected = projected.expand_dims("time").assign_coords(
                time=[np.datetime64(f"{year}-01-01")]
            )
            projections.append(projected)

        return cast(xr.DataArray, xr.concat(projections, dim="time"))

    def _calculate_trend(self, data: xr.DataArray) -> float:
        """Calculate linear trend in units per year.

        Raises:
            ValueError: If ``data`` is not a one-dimensional time series.
        """
        if data.ndim != 1 or "time" not in data.dims:
            raise ValueError(
                "historical_data must be a one-dimensional time series with a "
                "'time' dimension; got dims "
                f"{tuple(str(d) for d in data.dims)} with ndim={data.ndim}"
            )
        time_numeric = data.time.values.astype("datetime64[Y]").astype(int) + 1970
        trend = np.polyfit(time_numeric, data.values, 1)[0]
        return float(trend)

    def assess_inundation(
        self, elevation: xr.DataArray, sea_level: xr.DataArray
    ) -> xr.Dataset:
        """
        Assess coastal inundation under sea-level rise.

        Args:
            elevation: Land elevation
            sea_level: Sea level

        Returns:
            Inundation assessment
        """
        inundated = elevation < sea_level
        inundation_depth = xr.where(inundated, sea_level - elevation, 0)

        return xr.Dataset(
            {"inundated": inundated, "inundation_depth": inundation_depth}
        )
