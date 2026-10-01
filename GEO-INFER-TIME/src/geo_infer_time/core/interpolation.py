"""
Temporal interpolation for GEO-INFER-TIME.

This module provides temporal interpolation and imputation methods
for filling gaps in time series data, including nearest-neighbor,
seasonal-aware, and gap-aware interpolation strategies.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from ..models.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class TemporalInterpolator:
    """
    Temporal interpolator for time series data.

    Provides various interpolation methods for filling missing values
    and resampling temporal data, including nearest-neighbor, seasonal,
    and gap-aware strategies with quality metrics.
    """

    def __init__(self) -> None:
        """Initialize the temporal interpolator."""
        self._interpolation_log: list[dict[str, Any]] = []

    def interpolate(
        self,
        timeseries: TimeSeries,
        method: str = "linear",
        limit: int | None = None,
    ) -> TimeSeries:
        """
        Interpolate missing values in time series.

        Args:
            timeseries: TimeSeries object
            method: Interpolation method ('linear', 'time', 'polynomial',
                'spline', 'nearest', 'zero', 'cubic')
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            Interpolated TimeSeries
        """
        data = timeseries.to_dataframe()

        method_map = {
            "linear": lambda df: df.interpolate(method="linear", limit=limit),
            "time": lambda df: df.interpolate(method="time", limit=limit),
            "polynomial": lambda df: df.interpolate(
                method="polynomial", order=2, limit=limit
            ),
            "spline": lambda df: df.interpolate(method="spline", order=2, limit=limit),
            "nearest": lambda df: df.interpolate(method="nearest", limit=limit),
            "zero": lambda df: df.interpolate(method="zero", limit=limit),
            "cubic": lambda df: df.interpolate(method="cubic", limit=limit),
        }

        if method not in method_map:
            raise ValueError(
                f"Unknown interpolation method: {method}. "
                f"Available: {list(method_map.keys())}"
            )

        missing_before = int(data.isna().sum().sum())
        interpolated = method_map[method](data)
        missing_after = int(interpolated.isna().sum().sum())
        filled_count = missing_before - missing_after

        logger.info(
            "Interpolated %d missing values using method='%s'",
            filled_count,
            method,
        )

        self._interpolation_log.append(
            {
                "method": method,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "filled": filled_count,
            }
        )

        return TimeSeries(
            data=interpolated,
            spatial_location=timeseries.spatial_location,
            metadata={
                **timeseries.metadata,
                "interpolated": True,
                "method": method,
                "filled_count": filled_count,
            },
        )

    def impute(
        self,
        timeseries: TimeSeries,
        method: str = "forward_fill",
    ) -> TimeSeries:
        """
        Impute missing values using various strategies.

        Args:
            timeseries: TimeSeries object
            method: Imputation method ('forward_fill', 'backward_fill',
                'mean', 'median', 'mode', 'constant')

        Returns:
            Imputed TimeSeries
        """
        data = timeseries.to_dataframe()
        missing_before = int(data.isna().sum().sum())

        if method == "forward_fill":
            imputed = data.ffill()
        elif method == "backward_fill":
            imputed = data.bfill()
        elif method == "mean":
            imputed = data.fillna(data.mean())
        elif method == "median":
            imputed = data.fillna(data.median())
        elif method == "mode":
            modes = data.mode().iloc[0] if not data.mode().empty else data.mean()
            imputed = data.fillna(modes)
        elif method == "constant":
            imputed = data.fillna(0.0)
        else:
            raise ValueError(
                f"Unknown imputation method: {method}. "
                f"Available: forward_fill, backward_fill, mean, median, mode, constant"
            )

        missing_after = int(imputed.isna().sum().sum())

        return TimeSeries(
            data=imputed,
            spatial_location=timeseries.spatial_location,
            metadata={
                **timeseries.metadata,
                "imputed": True,
                "method": method,
                "filled_count": missing_before - missing_after,
            },
        )

    def interpolate_seasonal(
        self,
        timeseries: TimeSeries,
        period: int,
        limit: int | None = None,
    ) -> TimeSeries:
        """
        Interpolate missing values using seasonal patterns.

        Uses values from the same seasonal position in adjacent cycles
        to fill gaps, preserving seasonal structure.

        Args:
            timeseries: TimeSeries object
            period: Seasonal period length (e.g. 12 for monthly, 7 for weekly)
            limit: Maximum consecutive NaNs to fill

        Returns:
            Interpolated TimeSeries with seasonal awareness
        """
        data = timeseries.to_dataframe()
        result = data.copy()
        missing_before = int(data.isna().sum().sum())

        for col in result.columns:
            series = result[col].copy()
            nan_mask = series.isna()

            if not nan_mask.any():
                continue

            # Build seasonal lookup: average value at each seasonal position
            seasonal_means = np.full(period, np.nan)
            for pos in range(period):
                position_values = series.iloc[pos::period].dropna()
                if len(position_values) > 0:
                    seasonal_means[pos] = position_values.mean()

            # Fill gaps using seasonal means
            consecutive_count = 0
            for i in range(len(series)):
                if nan_mask.iloc[i]:
                    consecutive_count += 1
                    if limit is not None and consecutive_count > limit:
                        continue
                    seasonal_pos = i % period
                    if not np.isnan(seasonal_means[seasonal_pos]):
                        result.iloc[i, result.columns.get_loc(col)] = seasonal_means[
                            seasonal_pos
                        ]
                else:
                    consecutive_count = 0

        missing_after = int(result.isna().sum().sum())

        logger.info(
            "Seasonal interpolation (period=%d) filled %d values",
            period,
            missing_before - missing_after,
        )

        return TimeSeries(
            data=result,
            spatial_location=timeseries.spatial_location,
            metadata={
                **timeseries.metadata,
                "interpolated": True,
                "method": "seasonal",
                "period": period,
                "filled_count": missing_before - missing_after,
            },
        )

    def interpolate_gap_aware(
        self,
        timeseries: TimeSeries,
        max_gap_size: int,
        method: str = "linear",
    ) -> TimeSeries:
        """
        Interpolate only gaps smaller than a given threshold.

        Large gaps are left as NaN since interpolation across them
        may produce unreliable values.

        Args:
            timeseries: TimeSeries object
            max_gap_size: Maximum gap length (in data points) to interpolate
            method: Interpolation method for small gaps

        Returns:
            TimeSeries with only small gaps filled
        """
        data = timeseries.to_dataframe()
        result = data.copy()
        filled_total = 0

        for col in result.columns:
            series = result[col].copy()
            nan_mask = series.isna()

            if not nan_mask.any():
                continue

            # Identify contiguous gap regions
            gap_groups = nan_mask.astype(int).diff().ne(0).cumsum()
            gap_groups = gap_groups[nan_mask]

            for gap_id in gap_groups.unique():
                gap_indices = gap_groups[gap_groups == gap_id].index
                gap_size = len(gap_indices)

                if gap_size <= max_gap_size:
                    # Small gap: interpolate
                    start_idx = series.index.get_loc(gap_indices[0])
                    end_idx = series.index.get_loc(gap_indices[-1])

                    # Get context window around the gap
                    context_start = max(0, start_idx - 1)
                    context_end = min(len(series), end_idx + 2)
                    segment = series.iloc[context_start:context_end]

                    if method == "linear":
                        filled_segment = segment.interpolate(method="linear")
                    elif method == "nearest":
                        filled_segment = segment.interpolate(method="nearest")
                    elif method == "cubic":
                        filled_segment = segment.interpolate(method="cubic")
                    else:
                        filled_segment = segment.interpolate(method="linear")

                    for idx in gap_indices:
                        loc = series.index.get_loc(idx)
                        if context_start <= loc < context_end:
                            relative_loc = loc - context_start
                            new_val = filled_segment.iloc[relative_loc]
                            result.iloc[loc, result.columns.get_loc(col)] = new_val
                            filled_total += 1

        logger.info(
            "Gap-aware interpolation (max_gap=%d) filled %d values",
            max_gap_size,
            filled_total,
        )

        return TimeSeries(
            data=result,
            spatial_location=timeseries.spatial_location,
            metadata={
                **timeseries.metadata,
                "interpolated": True,
                "method": f"gap_aware_{method}",
                "max_gap_size": max_gap_size,
                "filled_count": filled_total,
            },
        )

    def resample_interpolate(
        self,
        timeseries: TimeSeries,
        target_freq: str,
        method: str = "linear",
    ) -> TimeSeries:
        """
        Resample to a target frequency and interpolate new points.

        Args:
            timeseries: TimeSeries object
            target_freq: Target pandas frequency string (e.g. 'h', '30min', 'D')
            method: Interpolation method for new data points

        Returns:
            Resampled and interpolated TimeSeries
        """
        data = timeseries.to_dataframe()
        original_len = len(data)

        # Create target index spanning the original range
        target_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=target_freq,
        )

        # Reindex to target frequency (introduces NaN at new positions)
        resampled = data.reindex(target_index)

        # Interpolate the new NaN positions
        if method == "linear":
            resampled = resampled.interpolate(method="linear")
        elif method == "time":
            resampled = resampled.interpolate(method="time")
        elif method == "cubic":
            resampled = resampled.interpolate(method="cubic")
        elif method == "nearest":
            resampled = resampled.interpolate(method="nearest")
        else:
            resampled = resampled.interpolate(method=method)

        logger.info(
            "Resampled from %d to %d points at freq='%s'",
            original_len,
            len(resampled),
            target_freq,
        )

        return TimeSeries(
            data=resampled,
            spatial_location=timeseries.spatial_location,
            metadata={
                **timeseries.metadata,
                "resampled": True,
                "target_freq": target_freq,
                "interpolation_method": method,
                "original_length": original_len,
            },
        )

    def interpolation_quality(
        self,
        original: TimeSeries,
        interpolated: TimeSeries,
    ) -> dict[str, Any]:
        """
        Assess interpolation quality by comparing statistics.

        Compares the original and interpolated series to provide
        quality metrics indicating how well the interpolation preserved
        the statistical properties.

        Args:
            original: Original TimeSeries (with gaps)
            interpolated: Interpolated TimeSeries (gaps filled)

        Returns:
            Dictionary of quality metrics including correlation,
            mean error, and distribution similarity.
        """
        orig_df = original.to_dataframe()
        interp_df = interpolated.to_dataframe()

        metrics: dict[str, Any] = {
            "columns": {},
            "overall_quality": 0.0,
        }

        quality_scores = []

        for col in orig_df.columns:
            if col not in interp_df.columns:
                continue

            orig_col = orig_df[col].dropna()
            interp_col = interp_df[col].dropna()

            if len(orig_col) == 0 or len(interp_col) == 0:
                continue

            # Mean preservation
            orig_mean = float(orig_col.mean())
            interp_mean = float(interp_col.mean())
            mean_diff = abs(interp_mean - orig_mean)
            mean_scale = abs(orig_mean) + 1e-10
            mean_similarity = max(0.0, 1.0 - mean_diff / mean_scale)

            # Standard deviation preservation
            orig_std = float(orig_col.std())
            interp_std = float(interp_col.std())
            std_diff = abs(interp_std - orig_std)
            std_scale = orig_std + 1e-10
            std_similarity = max(0.0, 1.0 - std_diff / std_scale)

            # Correlation on overlapping non-NaN values
            common_idx = (
                orig_df[col].dropna().index.intersection(interp_df[col].dropna().index)
            )
            if len(common_idx) > 2:
                correlation = float(
                    np.corrcoef(
                        orig_df.loc[common_idx, col].values,
                        interp_df.loc[common_idx, col].values,
                    )[0, 1]
                )
            else:
                correlation = np.nan

            # Missing value reduction
            orig_missing = int(orig_df[col].isna().sum())
            interp_missing = int(interp_df[col].isna().sum())
            gap_fill_rate = (orig_missing - interp_missing) / max(orig_missing, 1)

            col_quality = (mean_similarity + std_similarity) / 2.0
            quality_scores.append(col_quality)

            metrics["columns"][col] = {
                "mean_original": orig_mean,
                "mean_interpolated": interp_mean,
                "mean_similarity": round(mean_similarity, 4),
                "std_original": orig_std,
                "std_interpolated": interp_std,
                "std_similarity": round(std_similarity, 4),
                "correlation": round(correlation, 4)
                if not np.isnan(correlation)
                else None,
                "missing_original": orig_missing,
                "missing_interpolated": interp_missing,
                "gap_fill_rate": round(gap_fill_rate, 4),
                "quality_score": round(col_quality, 4),
            }

        if quality_scores:
            metrics["overall_quality"] = round(float(np.mean(quality_scores)), 4)

        return metrics

    def get_interpolation_log(self) -> list[dict[str, Any]]:
        """
        Get the log of all interpolation operations performed.

        Returns:
            List of interpolation operation records.
        """
        return list(self._interpolation_log)
