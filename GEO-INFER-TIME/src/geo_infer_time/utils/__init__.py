"""Utility functions for temporal analysis.

Provides validation, frequency detection, alignment, gap-filling,
and convenience factory functions for working with TimeSeries objects.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from geo_infer_time.models.timeseries import TimeSeries
from geo_infer_time.core.timestamps import normalize_timestamp

logger = logging.getLogger(__name__)

__all__ = [
    "validate_timeseries",
    "detect_frequency",
    "align_timeseries",
    "create_timeseries",
    "fill_gaps",
]


def validate_timeseries(ts: TimeSeries) -> dict[str, Any]:
    """Validate a TimeSeries for completeness, gaps, and data quality.

    Checks performed:
    - Non-empty data
    - Monotonically increasing timestamps
    - Duplicate timestamp detection
    - Missing value counts per column
    - Gap detection (intervals > 1.5x the median interval)
    - Frequency regularity

    Args:
        ts: The TimeSeries to validate.

    Returns:
        A dictionary with validation results::

            {
                "valid": bool,
                "errors": [...],
                "warnings": [...],
                "row_count": int,
                "column_count": int,
                "missing_values": {col: int, ...},
                "missing_pct": {col: float, ...},
                "duplicate_timestamps": int,
                "is_monotonic": bool,
                "detected_frequency": str | None,
                "gap_count": int,
                "gaps": [{"start": ..., "end": ..., "size": ...}, ...],
            }
    """
    errors: list[str] = []
    warnings: list[str] = []
    df = ts.to_dataframe()
    index = ts.timestamps

    # Basic shape
    row_count = len(df)
    col_count = len(df.columns)

    if row_count == 0:
        errors.append("TimeSeries is empty (0 rows).")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "row_count": 0,
            "column_count": col_count,
            "missing_values": {},
            "missing_pct": {},
            "duplicate_timestamps": 0,
            "is_monotonic": True,
            "detected_frequency": None,
            "gap_count": 0,
            "gaps": [],
        }

    # Monotonicity
    is_monotonic = bool(index.is_monotonic_increasing)
    if not is_monotonic:
        warnings.append("Timestamps are not monotonically increasing.")

    # Duplicates
    dup_count = int(index.duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Found {dup_count} duplicate timestamp(s).")

    # Missing values
    missing_values: dict[str, int] = {}
    missing_pct: dict[str, float] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        missing_values[col] = n_missing
        missing_pct[col] = round(n_missing / row_count * 100, 2) if row_count else 0.0
        if n_missing > 0:
            warnings.append(
                f"Column '{col}' has {n_missing} missing value(s) "
                f"({missing_pct[col]}%)."
            )

    # Frequency detection
    detected_freq = detect_frequency(ts)

    # Gap detection
    gaps: list[dict[str, Any]] = []
    if row_count > 1 and is_monotonic:
        diffs = pd.Series(index[1:] - index[:-1])
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        if not positive_diffs.empty:
            threshold = positive_diffs.median() * 1.5
            for position, gap_size in enumerate(diffs.to_numpy(), start=1):
                if gap_size > threshold:
                    gaps.append(
                        {
                            "start": index[position - 1].isoformat(),
                            "end": index[position].isoformat(),
                            "size": str(gap_size),
                        }
                    )

    if gaps:
        warnings.append(f"Detected {len(gaps)} gap(s) in time series.")

    valid = len(errors) == 0

    result = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": row_count,
        "column_count": col_count,
        "missing_values": missing_values,
        "missing_pct": missing_pct,
        "duplicate_timestamps": dup_count,
        "is_monotonic": is_monotonic,
        "detected_frequency": detected_freq,
        "gap_count": len(gaps),
        "gaps": gaps,
    }

    logger.info(
        "Validation complete: valid=%s, %d error(s), %d warning(s)",
        valid,
        len(errors),
        len(warnings),
    )
    return result


def detect_frequency(ts: TimeSeries) -> str | None:
    """Detect the frequency of a TimeSeries.

    Uses ``pandas.infer_freq`` on the DatetimeIndex. If that fails (e.g.
    because of gaps or irregular spacing), falls back to the median
    difference between consecutive timestamps and maps it to common
    pandas offset aliases.

    Args:
        ts: The TimeSeries to analyze.

    Returns:
        A pandas frequency alias string (e.g. ``'h'``, ``'D'``, ``'min'``)
        or None if the frequency cannot be determined.
    """
    index = ts.timestamps
    if len(index) < 2:
        return None

    # Try pandas built-in first
    try:
        freq = pd.infer_freq(index)
        if freq is not None:
            logger.debug("Inferred frequency via pandas: %s", freq)
            return str(freq)
    except Exception:
        pass

    # Fallback: median interval mapping
    diffs = pd.Series(index).diff().dropna()
    if diffs.empty:
        return None

    median_seconds = diffs.median().total_seconds()

    frequency_map = [
        (1, "s"),
        (60, "min"),
        (3600, "h"),
        (86400, "D"),
        (604800, "W"),
        (2592000, "ME"),  # ~30 days
        (31536000, "YE"),  # ~365 days
    ]

    best_alias: str | None = None
    best_ratio = float("inf")
    for seconds, alias in frequency_map:
        ratio = max(median_seconds / seconds, seconds / median_seconds)
        if ratio < best_ratio:
            best_ratio = ratio
            best_alias = alias

    # Only accept if within 2x tolerance
    if best_ratio <= 2.0:
        logger.debug(
            "Estimated frequency from median interval (%.1fs): %s",
            median_seconds,
            best_alias,
        )
        return best_alias

    logger.debug(
        "Could not determine frequency (median interval: %.1fs)", median_seconds
    )
    return None


def align_timeseries(
    ts_list: list[TimeSeries],
    method: str = "outer",
    fill_method: str | None = "ffill",
) -> list[TimeSeries]:
    """Align multiple TimeSeries to a common time index.

    Computes the union (outer) or intersection (inner) of all timestamp
    indices and reindexes each series accordingly.

    Args:
        ts_list: List of TimeSeries to align.
        method: ``'outer'`` to use the union of all timestamps (default),
            or ``'inner'`` to use only timestamps present in all series.
        fill_method: Method to fill NaN values introduced by reindexing.
            Accepts ``'ffill'``, ``'bfill'``, ``'linear'``, or None.

    Returns:
        A new list of TimeSeries aligned to the common index.

    Raises:
        ValueError: If *ts_list* is empty or *method* is invalid.
    """
    if not ts_list:
        raise ValueError("ts_list must not be empty.")
    if method not in ("outer", "inner"):
        raise ValueError(f"method must be 'outer' or 'inner', got '{method}'.")

    logger.info("Aligning %d time series using '%s' join", len(ts_list), method)

    indices = [ts.timestamps for ts in ts_list]

    if method == "outer":
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.union(idx)
    else:
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)

    common_index = common_index.sort_values()

    aligned: list[TimeSeries] = []
    for ts in ts_list:
        df = ts.to_dataframe().reindex(common_index)

        if fill_method == "linear":
            df = df.interpolate(method="time")
        elif fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        # fill_method=None leaves NaNs in place

        aligned.append(
            TimeSeries(
                data=df,
                spatial_location=ts.spatial_location,
                metadata={**ts.metadata, "aligned": True},
            )
        )

    logger.info("Aligned to common index with %d timestamps", len(common_index))
    return aligned


def create_timeseries(
    values: list[float] | np.ndarray | dict[str, list[float]],
    start: str | datetime,
    freq: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    spatial_location: dict[str, float | str] | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """Create a TimeSeries from raw values and a start time / frequency.

    A convenience factory that builds the DatetimeIndex for you.

    Args:
        values: Data values. A flat list or array produces a single-column
            DataFrame. A dict of ``{column_name: list}`` produces multiple
            columns.
        start: Start datetime (string or datetime object).
        freq: Pandas frequency alias (e.g. ``'h'``, ``'D'``, ``'5min'``).
        name: Optional column name when *values* is a flat sequence.
            Defaults to ``"value"``.
        metadata: Optional metadata dict.
        spatial_location: Optional spatial location dict.
        **kwargs: Additional keyword arguments forwarded to the TimeSeries
            constructor.

    Returns:
        A new TimeSeries.
    """
    start = normalize_timestamp(start)
    if isinstance(values, dict):
        length = len(next(iter(values.values())))
        index = pd.date_range(start=start, periods=length, freq=freq)
        df = pd.DataFrame(values, index=index)
    else:
        arr = np.asarray(values)
        index = pd.date_range(start=start, periods=len(arr), freq=freq)
        col_name = name or "value"
        df = pd.DataFrame({col_name: arr}, index=index)

    logger.debug(
        "Created TimeSeries: %d rows, start=%s, freq=%s",
        len(df),
        index[0],
        freq,
    )

    return TimeSeries(
        data=df,
        spatial_location=spatial_location,
        metadata=metadata or {},
        **kwargs,
    )


def fill_gaps(
    ts: TimeSeries,
    method: str = "linear",
    freq: str | None = None,
    limit: int | None = None,
) -> TimeSeries:
    """Fill temporal gaps in a TimeSeries.

    First reindexes the series to a regular frequency (detected or
    specified), then fills newly introduced NaN values using the
    chosen interpolation method.

    Args:
        ts: The TimeSeries with potential gaps.
        method: Fill method -- ``'linear'``, ``'ffill'``, ``'bfill'``,
            ``'time'``, or any method accepted by
            :meth:`pandas.DataFrame.interpolate`.
        freq: Target frequency for the regular index. If None, the
            frequency is auto-detected via :func:`detect_frequency`.
        limit: Maximum number of consecutive NaN values to fill. None
            means no limit.

    Returns:
        A new TimeSeries with gaps filled.

    Raises:
        ValueError: If frequency cannot be determined.
    """
    if freq is None:
        freq = detect_frequency(ts)
        if freq is None:
            raise ValueError(
                "Cannot determine frequency automatically. Pass freq explicitly."
            )

    logger.info("Filling gaps with method='%s', freq='%s'", method, freq)

    df = ts.to_dataframe()
    regular_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(regular_index)

    if method == "ffill":
        df = df.ffill(limit=limit)
    elif method == "bfill":
        df = df.bfill(limit=limit)
    else:
        df = df.interpolate(method=method, limit=limit)

    return TimeSeries(
        data=df,
        spatial_location=ts.spatial_location,
        metadata={**ts.metadata, "gaps_filled": True, "fill_method": method},
    )
