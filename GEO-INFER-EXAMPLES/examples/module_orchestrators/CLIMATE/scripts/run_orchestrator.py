#!/usr/bin/env python3
"""GEO-INFER-CLIMATE module orchestrator.

Runs one documented end-to-end CLIMATE operation on synthetic data: build a
deterministic synthetic climate dataset, then validate, preprocess, subset,
and analyze it through the exported convenience API — ``ClimateDataProcessor``
(validate/preprocess/temporal extraction), ``ClimateIndicesCalculator`` (SPI
and extreme indices), ``ExtremeEventAnalyzer`` (heatwave and drought
detection), and ``TemperatureTrendAnalyzer`` (linear and Mann-Kendall
trends). All work goes through the real ``geo_infer_climate`` public API,
mirroring the module's documented basic-analysis flow.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _synthetic_dataset():
    """Build a deterministic synthetic daily climate dataset (no network)."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(42)
    time = pd.date_range("2000-01-01", periods=730, freq="D")
    lat = np.linspace(30.0, 50.0, 8)
    lon = np.linspace(-120.0, -100.0, 8)
    day = np.arange(len(time))
    temperature = (
        15.0
        + 8.0 * np.sin(2 * np.pi * day / 365.0)[:, None, None]
        + rng.normal(0.0, 1.5, size=(len(time), len(lat), len(lon)))
    )
    precipitation = np.maximum(
        0.0,
        2.0
        + np.sin(2 * np.pi * day / 365.0)[:, None, None]
        + rng.normal(0.0, 0.5, size=(len(time), len(lat), len(lon))),
    )
    return xr.Dataset(
        {
            "temperature": (("time", "latitude", "longitude"), temperature),
            "precipitation": (("time", "latitude", "longitude"), precipitation),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def _operation() -> dict[str, Any]:
    from geo_infer_climate import (
        ClimateDataProcessor,
        ClimateIndicesCalculator,
        ExtremeEventAnalyzer,
        TemperatureTrendAnalyzer,
    )

    dataset = _synthetic_dataset()

    processor = ClimateDataProcessor()
    validation = processor.validate_dataset(dataset)
    processed = processor.preprocess_dataset(
        dataset, operations=["standardize_coords", "sort_time"]
    )
    # Detrend only the series used for trend statistics: detrended
    # precipitation would go negative and break the SPI gamma fit.
    detrended = processor.preprocess_dataset(processed, operations=["detrend"])
    subset = processor.extract_temporal_subset(processed, "2001-01-01", "2001-12-31")

    temp_point = processed["temperature"].isel(lat=4, lon=4)
    precip_point = processed["precipitation"].isel(lat=4, lon=4)

    indices_calc = ClimateIndicesCalculator()
    spi = indices_calc.calculate_spi(precip_point, timescale=3)
    extremes = indices_calc.calculate_extreme_indices(temp_point, precip_point)

    event_analyzer = ExtremeEventAnalyzer()
    heatwaves = event_analyzer.detect_heatwaves(
        temp_point, threshold_percentile=90.0, min_duration=3
    )
    droughts = event_analyzer.detect_droughts(
        precip_point, threshold_percentile=10.0, min_duration=30
    )

    trend_analyzer = TemperatureTrendAnalyzer()
    annual = detrended["temperature"].isel(lat=4, lon=4).resample(time="YS").mean()
    years = annual["time"].dt.year.values.astype(float)
    linear = trend_analyzer.linear_trend(annual.values, years)
    mk = trend_analyzer.mann_kendall_test(annual.values)

    return {
        "operation": "validate_preprocess_indices_extremes_trends",
        "validation": {k: bool(v) for k, v in validation.items()},
        "preprocessing": {
            "time_steps": int(processed.sizes["time"]),
            "spatial_dims": ["lat", "lon"],
        },
        "temporal_subset": {
            "start": "2001-01-01",
            "end": "2001-12-31",
            "time_steps": int(subset.sizes["time"]),
        },
        "spi": {
            "mean": round(float(spi.mean().values), 6),
            "std": round(float(spi.std().values), 6),
        },
        "extreme_indices": {
            "hot_days": int(extremes["hot_days"].values),
            "cold_days": int(extremes["cold_days"].values),
            "max_temp": round(float(extremes["max_temp"].values), 6),
            "min_temp": round(float(extremes["min_temp"].values), 6),
            "heavy_precip_days": int(extremes["heavy_precip_days"].values),
            "total_precip": round(float(extremes["total_precip"].values), 6),
        },
        "heatwaves": {
            "events_detected": int(heatwaves["events_detected"]),
            "total_hot_days": int(heatwaves["total_hot_days"]),
            "threshold_temp": round(float(heatwaves["threshold_temp"]), 6),
            "max_event_duration_days": int(
                max((e["duration_days"] for e in heatwaves["events"]), default=0)
            ),
        },
        "droughts": {
            "events_detected": int(droughts["events_detected"]),
            "max_event_duration_days": int(
                max((e["duration_days"] for e in droughts["events"]), default=0)
            ),
        },
        "trends": {
            "slope_per_decade": round(float(linear["slope_per_decade"]), 6),
            "r_squared": round(float(linear["r_squared"]), 6),
            "p_value": round(float(linear["p_value"]), 6),
            "mann_kendall_trend": str(mk["trend"]),
            "mann_kendall_significant": bool(mk["significant"]),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("CLIMATE", _operation))
