"""
TimeSeries data model for GEO-INFER-TIME.

This module provides a comprehensive TimeSeries class for managing
temporal geospatial data with metadata and analysis capabilities.
"""

import logging
from typing import Any, cast
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeSeries:
    """
    TimeSeries data model for temporal geospatial data.

    Provides a structured representation of time-series data with
    temporal indexing, metadata, and spatial context.
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame | np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
        spatial_location: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a TimeSeries object.

        Args:
            data: Time series data (Series, DataFrame, or array)
            timestamps: Optional timestamps (if None, uses data index)
            spatial_location: Optional spatial location {"lat": float, "lon": float}
            metadata: Optional metadata dictionary
        """
        if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("data must be a pandas Series, DataFrame, or numpy array")

        normalized_timestamps = None
        if timestamps is not None:
            try:
                normalized_timestamps = pd.DatetimeIndex(timestamps)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not convert timestamps to datetime: {exc}"
                ) from exc
            if len(normalized_timestamps) != len(data):
                raise ValueError("timestamps length must match data length")

        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if normalized_timestamps is None:
                raise ValueError("timestamps required when data is numpy array")
            self.data = pd.DataFrame(data, index=normalized_timestamps)
        elif isinstance(data, pd.Series):
            self.data = data.to_frame()
            if normalized_timestamps is not None:
                self.data.index = normalized_timestamps
        else:
            self.data = data.copy()
            if normalized_timestamps is not None:
                self.data.index = normalized_timestamps

        self.spatial_location = (
            dict(spatial_location) if spatial_location is not None else None
        )
        self.metadata = dict(metadata) if metadata is not None else {}

        # Validate temporal index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Could not convert index to datetime: {exc}") from exc

        logger.debug(f"Created TimeSeries with {len(self.data)} observations")

    def __len__(self) -> int:
        """Get length of time series."""
        return len(self.data)

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Get timestamps."""
        return self.data.index

    @property
    def start_time(self) -> datetime:
        """Get start time."""
        if self.data.empty:
            raise ValueError("TimeSeries is empty")
        return cast(datetime, pd.Timestamp(self.data.index[0]).to_pydatetime())

    @property
    def end_time(self) -> datetime:
        """Get end time."""
        if self.data.empty:
            raise ValueError("TimeSeries is empty")
        return cast(datetime, pd.Timestamp(self.data.index[-1]).to_pydatetime())

    @property
    def duration(self) -> timedelta:
        """Get time series duration."""
        return self.end_time - self.start_time

    @property
    def frequency(self) -> str | None:
        """Get inferred frequency."""
        try:
            freq = pd.infer_freq(self.data.index)
            return str(freq) if freq is not None else None
        except Exception:
            return None

    def resample(self, frequency: str, method: str = "mean") -> "TimeSeries":
        """
        Resample the time series to a different frequency.

        Args:
            frequency: Target frequency (e.g., '1H', '1D', '1W')
            method: Aggregation method ('mean', 'sum', 'max', 'min', 'first', 'last')

        Returns:
            Resampled TimeSeries
        """
        if method == "mean":
            resampled_data = self.data.resample(frequency).mean()
        elif method == "sum":
            resampled_data = self.data.resample(frequency).sum()
        elif method == "max":
            resampled_data = self.data.resample(frequency).max()
        elif method == "min":
            resampled_data = self.data.resample(frequency).min()
        elif method == "first":
            resampled_data = self.data.resample(frequency).first()
        elif method == "last":
            resampled_data = self.data.resample(frequency).last()
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        return TimeSeries(
            data=resampled_data,
            spatial_location=self.spatial_location,
            metadata={**self.metadata, "resampled_from": self.frequency},
        )

    def interpolate(self, method: str = "linear") -> "TimeSeries":
        """
        Interpolate missing values.

        Args:
            method: Interpolation method ('linear', 'time', 'polynomial', 'spline')

        Returns:
            Interpolated TimeSeries
        """
        interpolated_data = self.data.interpolate(method=method)

        return TimeSeries(
            data=interpolated_data,
            spatial_location=self.spatial_location,
            metadata={**self.metadata, "interpolated": True},
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistical summary of the time series.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "count": len(self.data),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_days": self.duration.total_seconds() / 86400,
            "frequency": self.frequency,
        }

        # Add column-specific statistics
        for col in self.data.columns:
            stats[col] = {
                "mean": float(self.data[col].mean()),
                "std": float(self.data[col].std()),
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max()),
                "missing_count": int(self.data[col].isna().sum()),
            }

        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame representation
        """
        return self.data.copy()

    def slice(self, start: datetime, end: datetime) -> "TimeSeries":
        """
        Slice the time series to a time range.

        Args:
            start: Start time
            end: End time

        Returns:
            Sliced TimeSeries
        """
        if start > end:
            raise ValueError("slice start must not be after end")

        sliced_data = self.data.loc[start:end]  # type: ignore[misc]

        return TimeSeries(
            data=sliced_data,
            spatial_location=self.spatial_location,
            metadata={**self.metadata, "sliced": True},
        )
