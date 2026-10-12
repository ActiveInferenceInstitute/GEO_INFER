"""Demonstrate selected TIME method families with finite analytical fixtures.

For the complete method inventory and numerical/failure contracts, run the
TIME pytest suite. This CLI does not claim universal method correctness.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from geo_infer_time import (
    ForecastingEngine,
    TemporalAnalyzer,
    TemporalInterpolator,
    TemporalStatistics,
    TimeSeries,
)


def main() -> int:
    """Call the actual public engines; retain their failures and diagnostics."""
    axis = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    observations = TimeSeries(
        pd.Series(
            [2 * i + math.sin(math.pi * i / 2) for i in range(24)],
            index=axis,
            name="value",
        )
    )
    analyzer, forecaster = TemporalAnalyzer(), ForecastingEngine()
    trend = analyzer.detect_trend(observations)
    decomposition = analyzer.decompose(observations, period=4)
    forecast = forecaster.forecast_linear(observations, horizon=3)
    entropy = analyzer.compute_temporal_entropy(observations, bins=4)
    missing = observations.to_dataframe()
    missing.iloc[5, 0] = float("nan")
    interpolated = TemporalInterpolator().interpolate_gap_aware(
        TimeSeries(missing), max_gap_size=1
    )
    diagnostics = TemporalStatistics().durbin_watson_test([1, -2, 3, -4])
    print(
        json.dumps(
            {
                "observations": len(observations),
                "trend": trend["trend_direction"],
                "slope_per_sample": trend["slope_per_sample"],
                "decomposition_period_samples": decomposition["period"],
                "forecast": forecast,
                "shannon_entropy_bits": entropy["shannon_entropy"]["value"],
                "gaps_filled": interpolated.metadata["filled_count"],
                "durbin_watson": diagnostics["dw_statistic"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
