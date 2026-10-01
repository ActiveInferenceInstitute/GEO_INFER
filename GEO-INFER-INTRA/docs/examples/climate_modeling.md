# Climate Modeling: Spatial-Temporal Climate Analysis

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.

> **Illustrative guide.** The code in this page is illustrative: it sketches
> how the module APIs compose for this use case. Some identifiers shown are
> conceptual; always import from the current package exports (see the module
> `__init__.py` and `SKILL.md`) and prefer the runnable scripts under
> `GEO-INFER-*/examples/` for verified behavior. Any numeric results shown
> are illustrative and must be reproduced against your own data before use.


This guide walks through climate analysis workflows using GEO-INFER modules for temperature anomaly detection, temporal trend extraction, statistical downscaling, and multi-decade change analysis.

## Overview

Climate analysis in GEO-INFER combines four modules:

- **GEO-INFER-CLIMATE** -- climate-specific analyzers and data structures
- **GEO-INFER-TIME** -- temporal trend detection and decomposition
- **GEO-INFER-BAYES** -- Gaussian Process models for spatial interpolation and downscaling
- **GEO-INFER-SPACE** -- H3 spatial indexing for gridded output

The workflow processes gridded climate data through anomaly detection, trend analysis, and downscaling, producing maps and uncertainty estimates suitable for impact assessment.

## Prerequisites

```bash
# From the repository root: installs CLIMATE, TIME, BAYES, SPACE and the locked
# numpy, pandas, xarray, matplotlib and scipy versions
uv sync --all-packages --all-extras --all-groups
```

## Section 1: Temperature Anomaly Detection

Temperature anomalies are departures from a climatological baseline. The standard reference period is 1981-2010, following WMO guidelines.

### Loading Historical Temperature Data

```python
import numpy as np
import pandas as pd
import xarray as xr


def generate_temperature_dataset(
    lat_range: tuple[float, float] = (42.0, 49.0),
    lon_range: tuple[float, float] = (-125.0, -116.0),
    spatial_resolution: float = 0.25,
    years: tuple[int, int] = (1970, 2024),
    seed: int = 42
) -> xr.Dataset:
    """Generate a synthetic gridded temperature dataset.

    Mimics a downscaled reanalysis product (e.g., ERA5-Land) with:
    - Seasonal cycle
    - Long-term warming trend
    - Spatial correlation (latitude-dependent temperature)
    - Interannual variability

    Args:
        lat_range: (min_lat, max_lat) in degrees.
        lon_range: (min_lon, max_lon) in degrees.
        spatial_resolution: Grid spacing in degrees.
        years: (start_year, end_year) inclusive.
        seed: Random seed.

    Returns:
        xarray Dataset with 'temperature' variable (monthly, Celsius).
    """
    rng = np.random.default_rng(seed)

    lats = np.arange(lat_range[0], lat_range[1], spatial_resolution)
    lons = np.arange(lon_range[0], lon_range[1], spatial_resolution)
    times = pd.date_range(f"{years[0]}-01", f"{years[1]}-12", freq="MS")

    nlat, nlon, ntime = len(lats), len(lons), len(times)

    # Base temperature: latitude-dependent
    lat_grid = np.tile(lats[:, None], (1, nlon))
    base_temp = 25.0 - 0.6 * (lat_grid - lat_range[0])

    # Seasonal cycle
    months = times.month.values
    seasonal = 12.0 * np.sin(2.0 * np.pi * (months - 1) / 12.0 - np.pi / 2)

    # Long-term warming trend: 0.02 C/year
    year_frac = (times.year + times.month / 12.0).values
    trend = 0.02 * (year_frac - year_frac[0])

    # Assemble: (time, lat, lon)
    temperature = np.zeros((ntime, nlat, nlon))
    for t in range(ntime):
        temperature[t] = (
            base_temp +
            seasonal[t] +
            trend[t] +
            rng.normal(0, 1.5, (nlat, nlon))
        )

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], temperature)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.temperature.attrs["units"] = "degC"
    ds.temperature.attrs["long_name"] = "Near-surface air temperature"

    return ds


temp_ds = generate_temperature_dataset()
print(f"Dataset shape: {dict(temp_ds.dims)}")
print(f"Time range: {temp_ds.time.values[0]} to {temp_ds.time.values[-1]}")
print(f"Spatial extent: lat [{temp_ds.lat.values[0]:.1f}, {temp_ds.lat.values[-1]:.1f}], "
      f"lon [{temp_ds.lon.values[0]:.1f}, {temp_ds.lon.values[-1]:.1f}]")
```

### Computing the Baseline Climatology

```python
def compute_climatology(
    ds: xr.Dataset,
    variable: str = "temperature",
    baseline_start: int = 1981,
    baseline_end: int = 2010
) -> xr.Dataset:
    """Compute monthly climatology over a reference period.

    Args:
        ds: Input dataset with a time dimension.
        variable: Variable name to compute climatology for.
        baseline_start: Start year of reference period (inclusive).
        baseline_end: End year of reference period (inclusive).

    Returns:
        Dataset with 'climatology' (12 months x lat x lon) and
        'climatology_std' for the baseline standard deviation.
    """
    baseline = ds[variable].sel(
        time=slice(f"{baseline_start}-01", f"{baseline_end}-12")
    )

    climatology = baseline.groupby("time.month").mean(dim="time")
    climatology_std = baseline.groupby("time.month").std(dim="time")

    return xr.Dataset({
        "climatology": climatology,
        "climatology_std": climatology_std,
    })


clim = compute_climatology(temp_ds)
print(f"Climatology shape: {dict(clim.dims)}")
print(f"January mean range: {clim.climatology.sel(month=1).values.min():.1f} to "
      f"{clim.climatology.sel(month=1).values.max():.1f} C")
```

### Bayesian Anomaly Detection

```python
from geo_infer_climate.core.climate_analyzer import ClimateAnalyzer


def detect_temperature_anomalies(
    ds: xr.Dataset,
    clim: xr.Dataset,
    threshold_sigma: float = 2.0,
    variable: str = "temperature"
) -> xr.Dataset:
    """Detect temperature anomalies relative to the climatological baseline.

    An anomaly is flagged when the departure from the monthly climatology
    exceeds threshold_sigma standard deviations.

    Args:
        ds: Full temperature dataset.
        clim: Climatology dataset with mean and std.
        threshold_sigma: Number of standard deviations for flagging.
        variable: Variable name.

    Returns:
        Dataset with 'anomaly' (departure in C), 'z_score',
        and 'is_anomalous' (boolean) variables.
    """
    # Compute departures
    monthly_means = clim.climatology
    monthly_stds = clim.climatology_std

    anomaly = ds[variable].groupby("time.month") - monthly_means
    z_score = anomaly.groupby("time.month") / monthly_stds

    is_anomalous = np.abs(z_score) > threshold_sigma

    return xr.Dataset({
        "anomaly": anomaly,
        "z_score": z_score,
        "is_anomalous": is_anomalous,
    })


anomaly_ds = detect_temperature_anomalies(temp_ds, clim, threshold_sigma=2.0)
total_anomalous = float(anomaly_ds.is_anomalous.sum())
total_cells = float(anomaly_ds.is_anomalous.size)
print(f"Anomalous observations: {total_anomalous:.0f} / {total_cells:.0f} "
      f"({100 * total_anomalous / total_cells:.1f}%)")
```

### Visualizing Anomaly Maps

```python
import matplotlib.pyplot as plt


def plot_anomaly_map(anomaly_ds: xr.Dataset, year: int, month: int) -> None:
    """Plot a spatial anomaly map for a specific month.

    Args:
        anomaly_ds: Anomaly dataset.
        year: Target year.
        month: Target month (1-12).
    """
    time_str = f"{year}-{month:02d}"
    anomaly_slice = anomaly_ds.anomaly.sel(time=time_str)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = anomaly_slice.plot(
        ax=ax,
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
        cbar_kwargs={"label": "Temperature anomaly (C)"},
    )
    ax.set_title(f"Temperature Anomaly: {time_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"anomaly_map_{time_str}.png", dpi=150)


# Plot the most recent July anomaly
plot_anomaly_map(anomaly_ds, 2024, 7)
```

## Section 2: Temporal Trend Extraction

The Mann-Kendall test detects monotonic trends without assuming a specific distribution. When applied to gridded data, spatial autocorrelation must be corrected to avoid inflated significance.

### Grid-Wide Trend Analysis

```python
from scipy.stats import kendalltau
from typing import Any


def mann_kendall_trend(
    time_series: np.ndarray,
    alpha: float = 0.05
) -> dict[str, Any]:
    """Compute Mann-Kendall trend statistic for a single time series.

    Args:
        time_series: 1D array of observations ordered in time.
        alpha: Significance level.

    Returns:
        Dict with 'tau' (correlation), 'p_value', 'significant' (bool),
        and 'trend_per_decade' (Sen's slope * 120 for monthly data).
    """
    n = len(time_series)
    t_indices = np.arange(n)
    tau, p_value = kendalltau(t_indices, time_series)

    # Sen's slope estimate
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((time_series[j] - time_series[i]) / (j - i))
    sen_slope = np.median(slopes) if slopes else 0.0

    # Convert monthly slope to per-decade
    trend_per_decade = sen_slope * 120  # 12 months * 10 years

    return {
        "tau": tau,
        "p_value": p_value,
        "significant": p_value < alpha,
        "sen_slope_per_month": sen_slope,
        "trend_per_decade": trend_per_decade,
    }


def compute_gridded_trends(
    ds: xr.Dataset,
    variable: str = "temperature",
    alpha: float = 0.05
) -> xr.Dataset:
    """Compute Mann-Kendall trends at every grid cell.

    Args:
        ds: Input dataset with time, lat, lon dimensions.
        variable: Variable to analyze.
        alpha: Significance level.

    Returns:
        Dataset with 'trend_per_decade', 'tau', 'p_value',
        'significant' variables on (lat, lon) grid.
    """
    data = ds[variable].values  # (time, lat, lon)
    ntime, nlat, nlon = data.shape

    trend_grid = np.zeros((nlat, nlon))
    tau_grid = np.zeros((nlat, nlon))
    pval_grid = np.zeros((nlat, nlon))
    sig_grid = np.zeros((nlat, nlon), dtype=bool)

    for i in range(nlat):
        for j in range(nlon):
            ts = data[:, i, j]
            result = mann_kendall_trend(ts, alpha=alpha)
            trend_grid[i, j] = result["trend_per_decade"]
            tau_grid[i, j] = result["tau"]
            pval_grid[i, j] = result["p_value"]
            sig_grid[i, j] = result["significant"]

    return xr.Dataset(
        {
            "trend_per_decade": (["lat", "lon"], trend_grid),
            "tau": (["lat", "lon"], tau_grid),
            "p_value": (["lat", "lon"], pval_grid),
            "significant": (["lat", "lon"], sig_grid),
        },
        coords={"lat": ds.lat, "lon": ds.lon},
    )


# Compute trends (this takes a few minutes for the full grid)
trend_ds = compute_gridded_trends(temp_ds)
mean_trend = float(trend_ds.trend_per_decade.mean())
pct_significant = float(trend_ds.significant.mean()) * 100
print(f"Mean warming trend: {mean_trend:.2f} C/decade")
print(f"Grid cells with significant trend: {pct_significant:.1f}%")
```

### Significance Mapping

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Trend magnitude
trend_ds.trend_per_decade.plot(
    ax=axes[0], cmap="RdBu_r", vmin=-0.5, vmax=0.5,
    cbar_kwargs={"label": "Trend (C/decade)"},
)
axes[0].set_title("Temperature Trend (C/decade)")

# Significance mask
trend_ds.significant.astype(float).plot(
    ax=axes[1], cmap="RdYlGn", vmin=0, vmax=1,
    cbar_kwargs={"label": "Significant (1=yes)"},
)
axes[1].set_title(f"Significant Trends (p < 0.05)")

plt.tight_layout()
plt.savefig("temperature_trends.png", dpi=150)
```

## Section 3: Spatial Downscaling

Statistical downscaling transfers information from coarse-resolution climate model output to finer scales. This example uses a Gaussian Process with elevation as a covariate.

### Setting Up the Downscaling Grid

```python
import h3
import geopandas as gpd
from shapely.geometry import Point


def create_downscaling_grids(
    coarse_ds: xr.Dataset,
    fine_resolution_deg: float = 0.01,
    target_lat_range: tuple[float, float] = (44.0, 45.0),
    target_lon_range: tuple[float, float] = (-123.0, -122.0),
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create coarse and fine grids for downscaling.

    Generates synthetic elevation data as the covariate. In practice,
    this comes from a DEM (e.g., SRTM or ALOS).

    Args:
        coarse_ds: Coarse-resolution climate dataset.
        fine_resolution_deg: Target fine grid spacing.
        target_lat_range: Latitude bounds for the study area.
        target_lon_range: Longitude bounds for the study area.
        seed: Random seed for synthetic elevation.

    Returns:
        Tuple of (coarse_coords, coarse_elev, fine_coords, fine_elev).
        coords are (N, 2) arrays with [lat, lon] columns.
    """
    rng = np.random.default_rng(seed)

    # Coarse grid points within target area
    coarse_lats = coarse_ds.lat.values
    coarse_lons = coarse_ds.lon.values
    mask_lat = (coarse_lats >= target_lat_range[0]) & (coarse_lats <= target_lat_range[1])
    mask_lon = (coarse_lons >= target_lon_range[0]) & (coarse_lons <= target_lon_range[1])
    c_lats = coarse_lats[mask_lat]
    c_lons = coarse_lons[mask_lon]
    c_lat_grid, c_lon_grid = np.meshgrid(c_lats, c_lons, indexing="ij")
    coarse_coords = np.column_stack([c_lat_grid.ravel(), c_lon_grid.ravel()])

    # Synthetic coarse elevation (meters)
    coarse_elev = 200.0 + 500.0 * (coarse_coords[:, 0] - target_lat_range[0]) + \
                  rng.normal(0, 50, len(coarse_coords))

    # Fine grid
    fine_lats = np.arange(target_lat_range[0], target_lat_range[1], fine_resolution_deg)
    fine_lons = np.arange(target_lon_range[0], target_lon_range[1], fine_resolution_deg)
    f_lat_grid, f_lon_grid = np.meshgrid(fine_lats, fine_lons, indexing="ij")
    fine_coords = np.column_stack([f_lat_grid.ravel(), f_lon_grid.ravel()])

    # Synthetic fine elevation
    fine_elev = 200.0 + 500.0 * (fine_coords[:, 0] - target_lat_range[0]) + \
                rng.normal(0, 30, len(fine_coords))

    return coarse_coords, coarse_elev, fine_coords, fine_elev


coarse_coords, coarse_elev, fine_coords, fine_elev = create_downscaling_grids(temp_ds)
print(f"Coarse grid points: {len(coarse_coords)}")
print(f"Fine grid points: {len(fine_coords)}")
print(f"Downscaling ratio: {len(fine_coords) / max(len(coarse_coords), 1):.0f}x")
```

### Gaussian Process Downscaling

```python
from geo_infer_bayes.core.gaussian_process import GaussianProcess


def downscale_temperature(
    coarse_coords: np.ndarray,
    coarse_elev: np.ndarray,
    coarse_temp: np.ndarray,
    fine_coords: np.ndarray,
    fine_elev: np.ndarray,
    length_scale: float = 0.1,
    elev_scale: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Downscale temperature using GP with elevation covariate.

    The feature space is [lat, lon, elevation * elev_scale], so the
    GP learns both spatial and elevation-dependent temperature patterns.

    Args:
        coarse_coords: (N, 2) array of [lat, lon] at coarse resolution.
        coarse_elev: (N,) array of elevation in meters.
        coarse_temp: (N,) array of temperature in Celsius.
        fine_coords: (M, 2) array of [lat, lon] at fine resolution.
        fine_elev: (M,) array of elevation in meters.
        length_scale: GP RBF kernel length scale.
        elev_scale: Scaling factor for elevation feature.

    Returns:
        Tuple of (predicted_temp, predicted_std) at fine grid points.
    """
    # Build feature matrices: [lat, lon, scaled_elevation]
    train_X = np.column_stack([
        coarse_coords,
        coarse_elev * elev_scale,
    ])
    pred_X = np.column_stack([
        fine_coords,
        fine_elev * elev_scale,
    ])

    gp = GaussianProcess(
        kernel_type="rbf",
        length_scale=length_scale,
        signal_variance=np.var(coarse_temp),
        noise_variance=0.5,
    )
    gp.fit(train_X, coarse_temp)
    mean, variance = gp.predict(pred_X, return_variance=True)
    std = np.sqrt(np.maximum(variance, 0.0))

    return mean, std


# Extract July 2020 temperature at coarse grid
july_2020 = temp_ds.temperature.sel(time="2020-07").values.squeeze()
coarse_lats = temp_ds.lat.values
coarse_lons = temp_ds.lon.values
mask_lat = (coarse_lats >= 44.0) & (coarse_lats <= 45.0)
mask_lon = (coarse_lons >= -123.0) & (coarse_lons <= -122.0)
coarse_temp = july_2020[np.ix_(mask_lat, mask_lon)].ravel()

fine_temp, fine_std = downscale_temperature(
    coarse_coords, coarse_elev, coarse_temp,
    fine_coords, fine_elev
)

print(f"Downscaled temperature range: {fine_temp.min():.1f} to {fine_temp.max():.1f} C")
print(f"Mean prediction uncertainty: {fine_std.mean():.2f} C")
```

### Downscaling Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Coarse input
sc_coarse = axes[0].scatter(
    coarse_coords[:, 1], coarse_coords[:, 0],
    c=coarse_temp, cmap="RdYlBu_r", s=80, edgecolor="black", linewidth=0.5
)
axes[0].set_title("Coarse Grid (0.25 deg)")
plt.colorbar(sc_coarse, ax=axes[0], label="Temperature (C)")

# Fine output (mean)
nlat_fine = len(np.arange(44.0, 45.0, 0.01))
nlon_fine = len(np.arange(-123.0, -122.0, 0.01))
fine_temp_grid = fine_temp.reshape(nlat_fine, nlon_fine)
im_fine = axes[1].imshow(
    fine_temp_grid, cmap="RdYlBu_r", origin="lower",
    extent=[-123.0, -122.0, 44.0, 45.0], aspect="auto"
)
axes[1].set_title("Downscaled (0.01 deg)")
plt.colorbar(im_fine, ax=axes[1], label="Temperature (C)")

# Uncertainty
fine_std_grid = fine_std.reshape(nlat_fine, nlon_fine)
im_std = axes[2].imshow(
    fine_std_grid, cmap="Oranges", origin="lower",
    extent=[-123.0, -122.0, 44.0, 45.0], aspect="auto"
)
axes[2].set_title("Prediction Uncertainty")
plt.colorbar(im_std, ax=axes[2], label="Std (C)")

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("downscaling_result.png", dpi=150)
```

## Section 4: Multi-Decade Analysis

Concatenating multiple data periods enables change point detection and projection with uncertainty bands.

### Change Point Detection

```python
def detect_change_points(
    annual_series: np.ndarray,
    min_segment_length: int = 10,
    penalty: float = 3.0
) -> list[int]:
    """Detect change points in an annual temperature time series.

    Uses a simple PELT-like approach: iteratively finds the split point
    that maximally reduces within-segment variance, subject to a minimum
    segment length constraint.

    Args:
        annual_series: 1D array of annual mean temperatures.
        min_segment_length: Minimum years between change points.
        penalty: BIC-like penalty per additional change point.

    Returns:
        List of change point indices (years into the series).
    """
    n = len(annual_series)
    if n < 2 * min_segment_length:
        return []

    def segment_cost(data: np.ndarray) -> float:
        if len(data) < 2:
            return 0.0
        return len(data) * np.var(data)

    best_cost = segment_cost(annual_series)
    best_split = -1

    for k in range(min_segment_length, n - min_segment_length):
        cost = (segment_cost(annual_series[:k]) +
                segment_cost(annual_series[k:]) +
                penalty)
        if cost < best_cost:
            best_cost = cost
            best_split = k

    if best_split == -1:
        return []

    # Recurse on segments
    left_cps = detect_change_points(
        annual_series[:best_split], min_segment_length, penalty
    )
    right_cps = detect_change_points(
        annual_series[best_split:], min_segment_length, penalty
    )
    right_cps = [cp + best_split for cp in right_cps]

    return left_cps + [best_split] + right_cps


# Compute annual mean temperature
annual_temp = temp_ds.temperature.groupby("time.year").mean(dim="time")
spatial_mean = annual_temp.mean(dim=["lat", "lon"]).values
years = np.unique(temp_ds.time.dt.year.values)

change_points = detect_change_points(spatial_mean, min_segment_length=10)
print(f"Change points detected at years: {[years[cp] for cp in change_points]}")
```

### Projection with Uncertainty Bands

```python
def project_with_uncertainty(
    annual_series: np.ndarray,
    years: np.ndarray,
    projection_years: int = 30,
    n_samples: int = 1000,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Project temperature trend forward with bootstrap uncertainty.

    Fits a linear trend to the most recent 30 years, then projects
    forward. Uncertainty comes from bootstrapping the residuals.

    Args:
        annual_series: Historical annual mean temperatures.
        years: Corresponding year labels.
        projection_years: Number of years to project.
        n_samples: Bootstrap samples for uncertainty.
        seed: Random seed.

    Returns:
        Dict with 'years', 'median', 'lower_5', 'upper_95' arrays.
    """
    rng = np.random.default_rng(seed)

    # Use the last 30 years for trend fitting
    fit_n = min(30, len(annual_series))
    fit_series = annual_series[-fit_n:]
    fit_years = np.arange(fit_n)

    # Linear fit
    coeffs = np.polyfit(fit_years, fit_series, 1)
    trend_slope = coeffs[0]
    trend_intercept = coeffs[1]
    residuals = fit_series - np.polyval(coeffs, fit_years)

    # Project forward
    future_years = years[-1] + 1 + np.arange(projection_years)
    future_x = fit_n + np.arange(projection_years)

    projections = np.zeros((n_samples, projection_years))
    for i in range(n_samples):
        # Bootstrap residuals
        boot_resid = rng.choice(residuals, size=projection_years, replace=True)
        # Perturb slope slightly
        slope_perturb = trend_slope + rng.normal(0, 0.003)
        projections[i] = slope_perturb * future_x + trend_intercept + boot_resid

    return {
        "years": future_years,
        "median": np.median(projections, axis=0),
        "lower_5": np.percentile(projections, 5, axis=0),
        "upper_95": np.percentile(projections, 95, axis=0),
        "historical_years": years,
        "historical_values": annual_series,
    }


projection = project_with_uncertainty(spatial_mean, years, projection_years=30)
print(f"Projected temperature in {projection['years'][-1]}:")
print(f"  Median: {projection['median'][-1]:.2f} C")
print(f"  90% range: [{projection['lower_5'][-1]:.2f}, {projection['upper_95'][-1]:.2f}] C")
```

### Visualization of Projections

```python
fig, ax = plt.subplots(figsize=(14, 6))

# Historical
ax.plot(projection["historical_years"], projection["historical_values"],
        color="black", linewidth=1.0, label="Observed")

# Change points
for cp in change_points:
    ax.axvline(x=years[cp], color="red", linestyle="--", alpha=0.5,
               label="Change point" if cp == change_points[0] else None)

# Projection
ax.plot(projection["years"], projection["median"],
        color="steelblue", linewidth=2.0, label="Projected (median)")
ax.fill_between(
    projection["years"],
    projection["lower_5"],
    projection["upper_95"],
    alpha=0.3, color="steelblue",
    label="90% confidence interval"
)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature (C)")
ax.set_title("Multi-Decade Temperature Analysis and Projection")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("multi_decade_analysis.png", dpi=150)
```

## Outputs Summary

| Output | Description | Module(s) |
|--------|-------------|-----------|
| `anomaly_map_YYYY-MM.png` | Spatial temperature anomaly map for a given month | CLIMATE |
| `temperature_trends.png` | Gridded trend magnitude and significance | TIME |
| `downscaling_result.png` | Coarse input, downscaled output, uncertainty | BAYES, SPACE |
| `multi_decade_analysis.png` | Historical trend with projection and uncertainty | TIME, BAYES |

## Next Steps

- **Higher-resolution downscaling**: Use H3 resolution 9-10 grids as the target (see [Memory Management](../advanced/memory_management.md) for handling the data volume)
- **Multi-variable analysis**: Extend to precipitation, humidity, wind speed using the same GP framework
- **Active Inference**: Couple climate projections with agricultural decision models (see [Agricultural Intelligence](agricultural_intelligence.md))
- **Impact assessment**: Feed downscaled temperature into GEO-INFER-RISK hazard models (see [Urban Analytics](urban_analytics.md))
