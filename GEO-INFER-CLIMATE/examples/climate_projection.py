#!/usr/bin/env python3
"""
GEO-INFER-CLIMATE Example: Bias Correction, Downscaling, and Projection

Demonstrates the real statistical workflow of the module on synthetic data:
bias-correcting a biased model field, downscaling to a finer grid, and
projecting future conditions under SSP scenarios.
"""

import numpy as np
import pandas as pd
import xarray as xr

from geo_infer_climate import (
    DownscalingMethods,
    ExtremeEventAnalyzer,
    ClimateProjections,
)


def create_synthetic_data(n_years: int = 30) -> tuple[xr.DataArray, xr.DataArray]:
    """Create synthetic 'observations' and a biased 'model' on a coarse grid."""
    rng = np.random.default_rng(42)
    n_days = 365 * n_years
    time = pd.date_range("1990-01-01", periods=n_days, freq="D")
    lat = np.linspace(35.0, 45.0, 6)
    lon = np.linspace(-115.0, -105.0, 6)

    day_of_year = np.arange(n_days) % 365
    # Observed temperature: seasonal cycle plus warming trend.
    observed = (
        15.0
        + 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)[:, None, None]
        + 0.0002 * np.arange(n_days)[:, None, None]
        + rng.normal(0, 2.0, (n_days, len(lat), len(lon)))
    )
    # Model: same climate but with a +3 deg C warm bias and inflated variance.
    model = observed + 3.0 + 1.5 * (observed - observed.mean())

    obs_da = xr.DataArray(
        observed,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    model_da = xr.DataArray(
        model,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return obs_da, model_da


def main() -> None:
    print("=" * 60)
    print("GEO-INFER-CLIMATE: Bias Correction, Downscaling, Projection")
    print("=" * 60)

    obs, model = create_synthetic_data()
    print(
        f"\nObservations: mean={float(obs.mean()):.2f} deg C, std={float(obs.std()):.2f}"
    )
    print(
        f"Raw model:    mean={float(model.mean()):.2f} deg C, std={float(model.std()):.2f}"
    )

    # 1. Bias correction (linear): recover the observed mean/std.
    print("\n1. Linear bias correction...")
    downscaler = DownscalingMethods()
    corrected = downscaler.bias_correction(model, obs, method="linear")
    print(
        f"   Corrected:    mean={float(corrected.mean()):.2f} deg C, std={float(corrected.std()):.2f}"
    )

    # 2. Quantile mapping bias correction on a single cell.
    print("\n2. Quantile mapping bias correction (central cell)...")
    obs_point = obs.isel(lat=3, lon=3)
    model_point = model.isel(lat=3, lon=3)
    corrected_qm = downscaler.bias_correction(model_point, obs_point, method="quantile")
    print(
        f"   Corrected:    mean={float(corrected_qm.mean()):.2f} deg C (observed {float(obs_point.mean()):.2f})"
    )

    # 3. Statistical downscaling (interpolation to a 2x finer grid).
    print("\n3. Interpolation-based downscaling...")
    fine = downscaler.statistical_downscaling(corrected.isel(time=0), method="linear")
    print(
        f"   Coarse grid: {corrected.sizes['lat']}x{corrected.sizes['lon']}"
        f" -> Fine grid: {fine.sizes['lat']}x{fine.sizes['lon']}"
    )

    # 4. SSP scenario projection of the corrected annual means.
    print("\n4. SSP scenario projection (linear-scaling method)...")
    projector = ClimateProjections()
    annual = corrected.isel(lat=3, lon=3).resample(time="YS").mean()
    projected = projector.project_future_climate(
        annual, scenario="ssp245", years=[2050, 2100]
    )
    for year, value in zip(projected.time.dt.year.values, projected.values):
        print(f"   ssp245 @ {int(year)}: {float(value):.2f} deg C")

    # 5. Drought detection on projected precipitation-like series.
    print("\n5. Drought detection on a synthetic precipitation series...")
    n = 3650
    wet = np.maximum(
        0.0, 2.0 + np.sin(2 * np.pi * np.arange(n) / 365)
    ) + np.random.default_rng(1).exponential(1.0, n)
    wet[1000:1050] = 0.0  # a 50-step dry spell
    precip = xr.DataArray(wet, dims=["time"])
    analyzer = ExtremeEventAnalyzer()
    droughts = analyzer.detect_droughts(
        precip, threshold_percentile=10.0, min_duration=30
    )
    print(f"   Droughts detected: {droughts['events_detected']}")
    for event in droughts["events"]:
        print(
            f"   - days {event['start_index']}-{event['end_index']} "
            f"({event['duration_days']} steps, mean {event['mean_precip']:.2f} mm)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
