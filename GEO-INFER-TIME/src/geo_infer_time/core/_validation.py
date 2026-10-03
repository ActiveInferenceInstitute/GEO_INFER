"""Shared numerical and cadence validation for sample-based temporal methods."""

from typing import Any

import numpy as np
import pandas as pd

from geo_infer_time.core.timestamps import normalize_datetime_index


def positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    """Reject boolean, fractional, and nonpositive sample counts."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite_vector(values: Any, name: str = "values") -> np.ndarray:
    """Own a one-dimensional finite numeric vector, without dropping samples."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a one-dimensional finite numeric series")
    return array.copy()


def root_mean_square(values: np.ndarray) -> float:
    """Compute finite-vector RMS without squaring its physical scale."""
    scale = float(np.max(np.abs(values)))
    return scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def bounded_date_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str | pd.DateOffset,
    max_points: int,
) -> pd.DatetimeIndex:
    """Bound fixed-step grid allocation before constructing the pandas index."""
    positive_integer(max_points, "max_points")
    offset = pd.tseries.frequencies.to_offset(frequency)
    if offset.n <= 0:
        raise ValueError("frequency must advance time")
    if isinstance(offset, pd.tseries.offsets.Tick):
        count = (end - start) // pd.Timedelta(offset) + 1
        if count > max_points:
            raise ValueError("Requested time grid exceeds max_points allocation budget")
    result = pd.date_range(start=start, end=end, freq=offset)
    if len(result) > max_points:
        raise ValueError("Requested time grid exceeds max_points allocation budget")
    return result


def univariate_series(timeseries: Any, *, minimum: int = 1) -> pd.Series:
    """Select an explicitly univariate series and validate its UTC axis."""
    frame = timeseries.to_dataframe()
    if frame.shape[1] != 1:
        raise ValueError("Select exactly one value column before temporal analysis")
    index = normalize_datetime_index(frame.index)
    values = finite_vector(frame.iloc[:, 0].to_numpy())
    if len(values) < minimum:
        raise ValueError(f"Temporal analysis requires at least {minimum} observations")
    return pd.Series(values, index=index, name=frame.columns[0])


def regular_frequency(index: pd.DatetimeIndex) -> str | pd.DateOffset:
    """Require an exact cadence and preserve explicit calendar parameters."""
    if len(index) < 2:
        raise ValueError("At least two timestamps are required to determine cadence")
    if index.freq is not None:
        # A two-point monthly axis cannot be inferred from elapsed seconds.
        # Keep an explicitly declared offset, including custom calendars, but
        # verify every timestamp rather than trusting potentially stale metadata.
        if not pd.date_range(index[0], periods=len(index), freq=index.freq).equals(
            index
        ):
            raise ValueError("Explicit frequency does not match the timestamp axis")
        return index.freq
    frequency = pd.infer_freq(index) if len(index) >= 3 else None
    if frequency is None:
        offset = pd.tseries.frequencies.to_offset(index[1] - index[0])
        if not pd.date_range(index[0], periods=len(index), freq=offset).equals(index):
            raise ValueError(
                "Timestamps must have a regular cadence; resample explicitly"
            )
        frequency = offset.freqstr
    return str(frequency)


def paired_series(first: Any, second: Any) -> tuple[pd.Series, pd.Series]:
    """Require identical temporal identities before computing sample lags."""
    left = univariate_series(first, minimum=2)
    right = univariate_series(second, minimum=2)
    if not left.index.equals(right.index):
        raise ValueError("Paired series must have identical UTC timestamp axes")
    regular_frequency(left.index)
    return left, right
