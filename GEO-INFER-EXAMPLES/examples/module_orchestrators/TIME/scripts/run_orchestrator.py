#!/usr/bin/env python3
"""GEO-INFER-TIME module orchestrator.

Runs one documented end-to-end TIME operation on synthetic data: build a
120-day synthetic daily sensor series (linear trend + weekly seasonality +
seeded noise), wrap it in the module's ``TimeSeries`` model, and compute
14-day rolling-window statistics plus a full summary profile through the
real ``geo_infer_time`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from geo_infer_time import TemporalAnalyzer, TemporalStatistics, TimeSeries

    rng = np.random.default_rng(2024)
    n_days = 120
    timestamps = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)
    values = 10.0 + 0.02 * t + 2.0 * np.sin(2.0 * np.pi * t / 7.0)
    values = values + rng.normal(0.0, 0.5, n_days)

    series = TimeSeries(
        pd.Series(values, index=timestamps),
        spatial_location={"lat": 41.7417, "lon": -124.2019},
        metadata={"station": "synthetic-del-norte-sensor"},
    )

    analyzer = TemporalAnalyzer()
    rolling = analyzer.calculate_rolling_statistics(
        series, window=14, statistics=["mean", "std", "min", "max"]
    )
    summary = TemporalStatistics().calculate_summary(
        values.tolist(), timestamps=list(timestamps)
    )

    rolling_stats = rolling["statistics"]
    return {
        "operation": "rolling_window_statistics_on_daily_series",
        "series_length": int(rolling["series_length"]),
        "window": int(rolling["window"]),
        "valid_rolling_observations": int(rolling["summary"]["valid_observations"]),
        "rolling_latest": {
            name: rolling_stats[name]["latest"]
            for name in ("mean", "std", "min", "max")
        },
        "rolling_mean_first": float(rolling_stats["mean"]["values"][0]),
        "summary": {
            "mean": float(summary["central_tendency"]["mean"]),
            "median": float(summary["central_tendency"]["median"]),
            "std": float(summary["dispersion"]["std"]),
            "min": float(summary["quantiles"]["min"]),
            "max": float(summary["quantiles"]["max"]),
            "trend_direction": summary["dynamics"]["trend_direction"],
            "trend_strength": float(summary["dynamics"]["trend_strength"]),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("TIME", _operation))
