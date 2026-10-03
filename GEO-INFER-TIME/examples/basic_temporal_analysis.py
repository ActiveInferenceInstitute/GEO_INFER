"""Run a deterministic daily TIME example on an explicit weekly cycle."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from geo_infer_time import (
    EventDetector,
    ForecastingEngine,
    TemporalAnalyzer,
    TimeSeries,
)


def generate_sample_timeseries(
    n_points: int = 56, trend: float = 0.1, seasonality: bool = True, noise: float = 0.1
) -> pd.DataFrame:
    """Create a seeded daily fixture with optional seven-sample seasonality."""
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    positions = np.arange(n_points)
    values = 20 + trend * positions
    if seasonality:
        values = values + 2 * np.sin(2 * np.pi * positions / 7)
    values = values + np.random.default_rng(42).normal(0, noise, n_points)
    return pd.DataFrame(
        {"value": values},
        index=pd.date_range("2024-01-01", periods=n_points, freq="D", tz="UTC"),
    )


def main() -> int:
    """Exercise real TIME interfaces; backend failures propagate as CLI failure."""
    observations = TimeSeries(generate_sample_timeseries())
    analyzer, forecaster = TemporalAnalyzer(), ForecastingEngine()
    trend = analyzer.detect_trend(observations)
    decomposition = analyzer.decompose(observations, period=7)
    forecast = forecaster.forecast_exponential_smoothing(
        observations, horizon=7, trend="add", seasonal="add", seasonal_periods=7
    )
    validation = forecaster.validate_forecast(
        observations, forecast, validation_split=0.25
    )
    events = EventDetector().detect_anomalies(observations)
    print(
        json.dumps(
            {
                "observations": len(observations),
                "utc_start": observations.start_time.isoformat(),
                "seasonal_period_samples": decomposition["period"],
                "slope_per_sample": trend["slope_per_sample"],
                "forecast": forecast["forecast"],
                "timestamps": forecast["timestamps"],
                "holdout_mae": validation["mae"],
                "anomalies": events["count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
