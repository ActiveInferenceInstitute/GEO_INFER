"""Deforestation detection using change detection algorithms.

Implements time-series based change detection for identifying forest loss
including BFAST-style breakpoint detection and magnitude-based thresholding.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class DeforestationDetector:
    """Detect deforestation events from multi-temporal satellite imagery.

    Uses NDVI or other vegetation index time series to identify
    statistically significant vegetation loss between observation periods.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize deforestation detector.

        Args:
            config: Configuration with detection parameters.
        """
        self.config = config or {}
        self.change_threshold: float = self.config.get("change_threshold", 0.15)
        self.confidence_level: float = self.config.get("confidence_level", 0.95)

    def detect_change_two_date(
        self,
        before: xr.DataArray,
        after: xr.DataArray,
        threshold: float | None = None,
    ) -> xr.Dataset:
        """Detect deforestation using two-date change detection.

        Compares vegetation index values between two dates and flags
        pixels with loss exceeding the threshold.

        Args:
            before: Vegetation index from earlier date.
            after: Vegetation index from later date.
            threshold: Minimum NDVI decrease to flag as deforestation.
                Defaults to self.change_threshold.

        Returns:
            Dataset with change magnitude, deforestation mask, and statistics.
        """
        thresh = threshold if threshold is not None else self.change_threshold

        change = before - after
        relative_change = xr.where(
            before != 0,
            change / (np.abs(before) + 1e-10),
            0.0,
        )

        deforested = (change > thresh) & (before > 0.3)

        total_pixels = float(before.size)
        deforested_pixels = float(deforested.sum())
        deforestation_rate = (
            deforested_pixels / total_pixels if total_pixels > 0 else 0.0
        )

        return xr.Dataset(
            {
                "change_magnitude": change,
                "relative_change": relative_change,
                "deforestation_mask": deforested,
            },
            attrs={
                "threshold": thresh,
                "deforestation_rate": deforestation_rate,
                "deforested_pixel_count": int(deforested_pixels),
                "total_pixel_count": int(total_pixels),
            },
        )

    def detect_change_time_series(
        self,
        ndvi_series: xr.DataArray,
        window_size: int = 3,
    ) -> xr.Dataset:
        """Detect deforestation from NDVI time series using moving average.

        Compares recent NDVI values against historical baseline using
        a moving window approach. Flags anomalous drops as potential
        deforestation events.

        Args:
            ndvi_series: NDVI time series with 'time' dimension.
            window_size: Number of time steps for baseline window.

        Returns:
            Dataset with anomaly scores and change detection results.
        """
        # Historical-only baseline: shift one step so each observation is
        # compared against the window *before* it, never against itself.
        shifted = ndvi_series.shift(time=1)
        baseline_mean = shifted.rolling(time=window_size, min_periods=1).mean()
        baseline_std = shifted.rolling(time=window_size, min_periods=1).std()
        # NaN std (too few distinct historical values) falls back to the
        # 0.01 floor; a NaN baseline mean falls back to the current value
        # so the first step gets a zero anomaly instead of NaN.
        baseline_std = xr.where(baseline_std < 0.01, 0.01, baseline_std)
        baseline_std = baseline_std.fillna(0.01)
        baseline_mean = baseline_mean.fillna(ndvi_series)

        z_score = (ndvi_series - baseline_mean) / baseline_std

        from scipy.stats import norm

        critical_z = norm.ppf(1 - (1 - self.confidence_level) / 2)
        significant_decrease = z_score < -critical_z

        step_change = ndvi_series.diff(dim="time")
        step_change = xr.where(step_change > 0, 0, step_change)

        # Pad with zero at the start so cumulative_loss aligns with original time axis
        first_time = ndvi_series.isel(time=0)
        zero_start = xr.zeros_like(first_time).expand_dims(dim="time")
        step_change_padded = xr.concat([zero_start, step_change], dim="time")
        step_change_padded["time"] = ndvi_series["time"]
        cumulative_loss = step_change_padded.cumsum(dim="time")

        return xr.Dataset(
            {
                "z_score": z_score,
                "significant_decrease": significant_decrease,
                "cumulative_loss": cumulative_loss,
                "baseline_mean": baseline_mean,
            },
        )

    def calculate_annual_deforestation_rate(
        self,
        forest_cover_series: xr.DataArray,
    ) -> dict[str, float]:
        """Calculate annual deforestation rate from forest cover time series.

        Uses the compound rate formula:
        annual_rate = 1 - (cover_end / cover_start)^(1/years)

        Args:
            forest_cover_series: Forest cover percentage over time.

        Returns:
            Dictionary with annual rate and related statistics.
        """
        n_times = len(forest_cover_series.time)
        if n_times < 2:
            return {
                "annual_rate_pct": 0.0,
                "total_loss_pct": 0.0,
                "years_covered": 0,
            }

        cover_start = float(forest_cover_series.isel(time=0).mean())
        cover_end = float(forest_cover_series.isel(time=-1).mean())
        years = n_times - 1

        total_loss_pct = cover_start - cover_end
        if cover_start > 0 and cover_end > 0:
            annual_rate = 1.0 - (cover_end / cover_start) ** (1.0 / years)
        elif cover_start > 0:
            annual_rate = 1.0
        else:
            annual_rate = 0.0

        return {
            "annual_rate_pct": float(annual_rate * 100),
            "total_loss_pct": float(total_loss_pct),
            "cover_start_pct": float(cover_start),
            "cover_end_pct": float(cover_end),
            "years_covered": years,
        }

    def calculate_fragmentation_index(
        self,
        forest_mask: xr.DataArray,
    ) -> dict[str, float]:
        """Calculate forest fragmentation indices.

        Computes edge density and core-to-edge ratio as indicators
        of forest fragmentation.

        Args:
            forest_mask: Binary forest mask (1=forest, 0=non-forest).

        Returns:
            Fragmentation metrics dictionary.
        """
        data = forest_mask.values.astype(float)
        total_pixels = float(data.size)
        forest_pixels = float(np.sum(data > 0))

        if forest_pixels == 0:
            return {
                "forest_fraction": 0.0,
                "edge_density": 0.0,
                "core_fraction": 0.0,
                "edge_pixel_count": 0,
                "core_pixel_count": 0,
                "fragmentation_index": 1.0,
            }

        # An edge pixel is a forest pixel with at least one in-bounds
        # 4-neighbor that is non-forest; core pixels have only forest
        # neighbors. Vectorized via shifted-array comparisons.
        forest = data > 0
        rows, cols = forest.shape[-2], forest.shape[-1]
        layers = (
            forest.reshape(-1, rows, cols)
            if forest.ndim > 2
            else forest[np.newaxis, :, :]
        )

        edge_count = 0
        for layer in layers:
            core = layer.copy()
            core[1:, :] &= layer[:-1, :]  # up neighbor must be forest
            core[:-1, :] &= layer[1:, :]  # down neighbor must be forest
            core[:, 1:] &= layer[:, :-1]  # left neighbor must be forest
            core[:, :-1] &= layer[:, 1:]  # right neighbor must be forest
            edge_count += int(np.sum(layer & ~core))

        forest_fraction = forest_pixels / total_pixels
        edge_density = edge_count / forest_pixels if forest_pixels > 0 else 0.0
        core_pixels = forest_pixels - edge_count
        core_fraction = core_pixels / forest_pixels if forest_pixels > 0 else 0.0

        fragmentation_index = edge_density

        return {
            "forest_fraction": float(forest_fraction),
            "edge_density": float(edge_density),
            "core_fraction": float(core_fraction),
            "edge_pixel_count": int(edge_count),
            "core_pixel_count": int(core_pixels),
            "fragmentation_index": float(fragmentation_index),
        }
