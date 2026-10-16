"""Regression tests for demand-forecast trend derivation (GS-177).

The forecast slope must be expressed per calendar year, derived from the
actual time coordinate, and non-1D (time, space) raster input must be
rejected with a clear ValueError instead of crashing in np.polyfit.
"""

import numpy as np
import pytest
import xarray as xr

from geo_infer_energy.core.energy_demand import EnergyDemandForecaster


@pytest.fixture
def forecaster():
    return EnergyDemandForecaster()


class TestForecastTrend:
    def test_annual_input_output_unchanged(self, forecaster):
        """Annual 1D input keeps its historical forecast values exactly."""
        demand = xr.DataArray(
            np.linspace(100, 120, 10),
            dims=("time",),
            coords={"time": np.arange(2010, 2020)},
        )
        result = forecaster.forecast_demand(demand, forecast_years=3)
        np.testing.assert_allclose(
            np.asarray(result["demand_forecast"].values).ravel(),
            [122.22222222, 124.44444444, 126.66666667],
        )

    def test_monthly_datetime_extrapolates_per_year(self, forecaster):
        """Monthly samples extrapolate at the per-year rate, not per step."""
        t = np.arange("2020-01", "2022-01", dtype="datetime64[M]")
        demand = xr.DataArray(
            100.0 + np.arange(len(t), dtype=float),
            dims=("time",),
            coords={"time": t},
        )
        result = forecaster.forecast_demand(demand, forecast_years=2)
        # Last observed value 123; slope is 12 units per calendar year.
        np.testing.assert_allclose(
            np.asarray(result["demand_forecast"].values).ravel(),
            [135.0, 147.0],
        )

    def test_2d_raster_input_raises_value_error(self, forecaster):
        """(time, space) raster input is rejected with a clear message."""
        demand = xr.DataArray(
            np.random.default_rng(0).uniform(80, 120, size=(10, 4)),
            dims=("time", "space"),
            coords={"time": np.arange(2010, 2020), "space": np.arange(4)},
        )
        with pytest.raises(ValueError, match="1D along 'time'"):
            forecaster.forecast_demand(demand, forecast_years=5)

    def test_missing_time_coordinate_raises_value_error(self, forecaster):
        """A time dimension without a coordinate cannot yield a per-year slope."""
        demand = xr.DataArray(np.arange(5.0), dims=("time",))
        with pytest.raises(ValueError, match="time.*coordinate"):
            forecaster.forecast_demand(demand, forecast_years=3)

    def test_constant_time_raises_value_error(self, forecaster):
        """Fewer than two distinct time values cannot estimate a trend."""
        demand = xr.DataArray(
            np.arange(5.0),
            dims=("time",),
            coords={"time": [2020, 2020, 2020, 2020, 2020]},
        )
        with pytest.raises(ValueError, match="distinct time values"):
            forecaster.forecast_demand(demand, forecast_years=3)

    def test_daily_samples_within_one_month(self, forecaster):
        """Daily observations retain their spacing instead of collapsing to one month."""
        times = np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]")
        demand = xr.DataArray(
            [100.0, 101.0, 102.0, 103.0], dims="time", coords={"time": times}
        )
        result = forecaster.forecast_demand(demand, forecast_years=1)
        # A unit per day in January is 31 units per month, or 372 per year.
        np.testing.assert_allclose(result.demand_forecast.values, [475.0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_demand_is_rejected(self, forecaster, bad):
        """Invalid observed values cannot silently yield nonfinite forecasts."""
        demand = xr.DataArray([100.0, bad], dims="time", coords={"time": [2020, 2021]})
        with pytest.raises(ValueError, match="finite"):
            forecaster.forecast_demand(demand)

    def test_missing_datetime_is_rejected(self, forecaster):
        """A missing timestamp cannot become a large negative integer year."""
        times = np.array(["2024-01-01", "NaT"], dtype="datetime64[ns]")
        demand = xr.DataArray([100.0, 101.0], dims="time", coords={"time": times})
        with pytest.raises(ValueError, match="NaT"):
            forecaster.forecast_demand(demand)
