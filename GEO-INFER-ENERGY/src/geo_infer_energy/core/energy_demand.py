"""Energy demand forecasting module."""

import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class EnergyDemandForecaster:
    """Forecast energy demand."""

    def __init__(self, config: dict | None = None):
        """Initialize demand forecaster.

        Documented contract: ``config`` is accepted and stored for API
        stability but its contents are currently ignored by this forecaster
        (reserved argument; only ``WindAnalyzer`` consumes config keys).
        """
        self.config = config or {}

    def forecast_demand(
        self,
        historical_demand: xr.DataArray,
        temperature: xr.DataArray | None = None,
        population: xr.DataArray | None = None,
        forecast_years: int = 10,
    ) -> xr.Dataset:
        """
        Forecast future energy demand.

        Args:
            historical_demand: Historical demand data
            temperature: Optional temperature data (for heating/cooling)
            population: Optional population data
            forecast_years: Number of years to forecast

        Returns:
            Demand forecast
        """
        historical_demand = _validate_time_series(historical_demand)
        # Slope per calendar year, derived from the actual time coordinate
        # (not per index step) so monthly/sub-annual sampling is extrapolated
        # at the correct per-year rate.
        trend = _slope_per_year(historical_demand)

        # Optional per-period temperature adjustment: use the temperature
        # series (not a single scalar) so each forecast year gets its own
        # heating/cooling adjustment. Baseline 20 degC; 1% demand shift per
        # degree of anomaly.
        if temperature is not None:
            tvals = np.asarray(temperature.values, dtype=float).flatten()
            n = len(tvals)
            if n >= forecast_years:
                edges = np.linspace(0, n, forecast_years + 1, dtype=int)
                period_means = np.array(
                    [
                        tvals[edges[i] : edges[i + 1]].mean()
                        for i in range(forecast_years)
                    ]
                )
            else:
                period_means = np.resize(tvals, forecast_years)

        # Population growth rate derived from the passed population data
        # (CAGR over the observed period) instead of a hard-coded assumption.
        annual_pop_growth: float | None = None
        if population is not None:
            pvals = np.asarray(population.values, dtype=float).flatten()
            pvals = pvals[pvals > 0]
            if len(pvals) > 1:
                annual_pop_growth = float(
                    (pvals[-1] / pvals[0]) ** (1.0 / (len(pvals) - 1)) - 1.0
                )

        forecast_values = []
        last_value = historical_demand.isel(time=-1)
        for year in range(1, forecast_years + 1):
            forecasted = last_value + trend * year
            if temperature is not None:
                temp_factor = 1 + (period_means[year - 1] - 20) / 100
                forecasted = forecasted * temp_factor
            if annual_pop_growth is not None:
                forecasted = forecasted * ((1 + annual_pop_growth) ** year)
            forecast_values.append(forecasted)

        return xr.Dataset(
            {"demand_forecast": xr.concat(forecast_values, dim="forecast_year")}
        )

    def identify_peak_demand(self, demand_time_series: xr.DataArray) -> xr.Dataset:
        """
        Identify peak demand periods.

        Args:
            demand_time_series: Time series of demand

        Returns:
            Peak demand analysis
        """
        peak_demand = demand_time_series.max(dim="time")
        peak_time = demand_time_series.idxmax(dim="time")
        average_demand = demand_time_series.mean(dim="time")
        peak_factor = peak_demand / (average_demand + 1e-10)

        return xr.Dataset(
            {
                "peak_demand": peak_demand,
                "peak_time": peak_time,
                "average_demand": average_demand,
                "peak_factor": peak_factor,
            }
        )


def _validate_time_series(da: xr.DataArray) -> xr.DataArray:
    """Require a 1D (time,) demand series with a usable time coordinate."""
    if not isinstance(da, xr.DataArray):
        raise ValueError(
            "historical_demand must be an xr.DataArray with a single 'time' "
            f"dimension, got {type(da).__name__}"
        )
    if "time" not in da.dims:
        raise ValueError(
            f"historical_demand must have a 'time' dimension, got dims {tuple(da.dims)}"
        )
    if len(da.dims) != 1:
        raise ValueError(
            "historical_demand must be 1D along 'time'; aggregate spatial or "
            f"other dimensions first, got dims {tuple(da.dims)}"
        )
    if "time" not in da.coords:
        raise ValueError(
            "historical_demand must carry a 'time' coordinate so the trend "
            "can be expressed per calendar year"
        )
    return da


def _slope_per_year(da: xr.DataArray) -> float:
    """Least-squares slope of ``da`` expressed per calendar year."""
    tvals = np.asarray(da.coords["time"].values)
    if np.issubdtype(tvals.dtype, np.datetime64):
        # Fractional years via month index so monthly/weekly sampling keeps
        # its true spacing (years-since-epoch / 12 is a pure shift+scale of
        # calendar time and preserves the fitted slope's per-year units).
        years = tvals.astype("datetime64[M]").astype(np.int64) / 12.0
    else:
        try:
            years = np.asarray(tvals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "time coordinate must be datetime64 or numeric year values, "
                f"got dtype {tvals.dtype}"
            ) from exc
    if years.size < 2 or np.all(years == years.flat[0]):
        raise ValueError(
            "historical_demand needs at least two distinct time values to "
            "estimate a trend"
        )
    return float(np.polyfit(years, np.asarray(da.values, dtype=float), 1)[0])
