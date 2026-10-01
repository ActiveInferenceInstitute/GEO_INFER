"""
Climate projections module.

Handles climate change projections and scenario analysis.
"""

import logging
from typing import cast
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ClimateProjections:
    """
    Climate change projections and scenario analysis.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize climate projections.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scenarios = ["ssp126", "ssp245", "ssp370", "ssp585"]

    def project_future_climate(
        self,
        historical_data: xr.DataArray,
        scenario: str = "ssp245",
        years: list[int] | None = None,
    ) -> xr.DataArray:
        """
        Project future climate based on historical data and scenario.

        This is a simplified linear-scaling projection: the linear trend of
        the historical series is extrapolated forward and scaled by a
        scenario factor. It is NOT a climate model emulator, an ensemble
        method, or a pattern-scaling approach based on CMIP6 response
        patterns; treat results as illustrative, not predictive.

        Args:
            historical_data: Historical climate data with a ``time``
                coordinate (datetime or numeric year values)
            scenario: Climate scenario (ssp126, ssp245, ssp370, ssp585)
            years: List of future years to project

        Returns:
            Projected climate data
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        years = years or [2050, 2100]

        # Fit the historical trend on a calendar-year axis so the slope is
        # per year regardless of the sampling frequency (annual, monthly, ...).
        time_years = self._time_elapsed_years(historical_data)
        trend = self._calculate_trend(historical_data, time_years)

        # Apply scenario-based scaling
        scenario_factor = self._get_scenario_factor(scenario)

        # First observation year on the calendar scale (handles datetime or
        # numeric year coordinates).
        t_min = historical_data["time"].min().values
        if np.issubdtype(np.asarray(t_min).dtype, np.datetime64):
            first_year = (
                int(np.asarray(t_min, dtype="datetime64[Y]").astype(int)) + 1970
            )
        else:
            first_year = int(t_min)

        # The fitted line passes through the historical mean at the mean
        # elapsed time, so project from there (mean + slope * years is the
        # least-squares line evaluated at the target year).
        mean_time = float(np.mean(time_years))
        projections = []
        for year in years:
            years_from_first = year - first_year
            projected = (
                historical_data.mean(dim="time")
                + trend * (years_from_first - mean_time) * scenario_factor
            )
            projected = projected.expand_dims("time").assign_coords(
                time=[np.datetime64(f"{year}-01-01")]
            )
            projections.append(projected)

        return cast(xr.DataArray, xr.concat(projections, dim="time"))

    @staticmethod
    def _time_elapsed_years(data: xr.DataArray) -> np.ndarray:
        """Elapsed years since the first observation on a calendar scale.

        Datetime coordinates are converted to fractional 365.25-day years so
        the trend slope is expressed per calendar year instead of per time
        step; numeric coordinates are already calendar years.
        """
        times = np.asarray(data["time"].values)
        if np.issubdtype(times.dtype, np.datetime64):
            days = (times - times[0]) / np.timedelta64(1, "D")
            return days.astype(float) / 365.25
        years = times.astype(float)
        return years - years[0]

    def _calculate_trend(
        self, data: xr.DataArray, time_years: np.ndarray | None = None
    ) -> xr.DataArray:
        """Calculate linear trend (per year) from time series."""
        if time_years is None:
            time_years = self._time_elapsed_years(data)
        trend = np.polyfit(time_years, data.values, 1)[0]
        return xr.DataArray(trend, dims=data.dims[:-1])

    def _get_scenario_factor(self, scenario: str) -> float:
        """Get scenario-specific scaling factor."""
        factors = {
            "ssp126": 0.5,  # Low emissions
            "ssp245": 1.0,  # Medium emissions
            "ssp370": 1.5,  # High emissions
            "ssp585": 2.0,  # Very high emissions
        }
        return factors.get(scenario, 1.0)
