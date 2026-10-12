"""TIME-backed validation for unordered, repeated observation instants."""

from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd

from geo_infer_time import normalize_timestamp


def normalize_observation_timestamps(
    values: Iterable[datetime | str],
) -> pd.DatetimeIndex:
    """Keep observation order and duplicates while preserving UTC precision.

    Unlike a model time axis, observation rows can repeat an instant across
    different identities and arrive out of order. Validate original values
    before pandas can interpret a number as an implicit epoch.
    """
    normalized = []
    for value in values:
        normalize_timestamp(value)
        normalized.append(pd.Timestamp(value).tz_convert(UTC))
    return pd.DatetimeIndex(normalized, tz=UTC)


def normalize_temporal_range(
    interval: tuple[datetime | str, datetime | str] | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Validate an inclusive query interval before querying or caching."""
    if interval is None:
        return None
    if len(interval) != 2:
        raise ValueError("temporal_range requires exactly two timestamps")
    start, end = normalize_observation_timestamps(interval)
    if start > end:
        raise ValueError("temporal_range start must be before or equal to end")
    return start, end
