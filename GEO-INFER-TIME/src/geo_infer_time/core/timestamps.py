"""Canonical aware UTC timestamps and ordered, precision-preserving indexes."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["normalize_timestamp", "normalize_datetime_index"]


def normalize_timestamp(value: datetime | str) -> datetime:
    """Require an aware datetime or ISO-8601 string and normalize it to UTC.

    Numeric epochs require an explicit unit conversion at the owning transport
    boundary. Naive wall-clock values are never assumed to mean UTC.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 string") from exc
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime or an ISO-8601 string")
    if value != value:
        raise ValueError("timestamp must not be NaT")
    if value.utcoffset() is None:
        raise ValueError(
            "timestamp must be timezone-aware; attach an explicit UTC offset"
        )
    return value.astimezone(UTC)


def normalize_datetime_index(
    values: Iterable[datetime | str] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Return a unique, increasing UTC index without losing nanoseconds.

    Input order is retained. Duplicates, reversed instants, NaT, numeric epochs
    and naive timestamps raise rather than being sorted, repaired or discarded.
    Empty indexes are permitted for empty analytical results.
    """
    import pandas as pd

    if isinstance(values, pd.DatetimeIndex):
        if values.hasnans:
            raise ValueError("timestamps must not contain NaT")
        if len(values) and values.tz is None:
            raise ValueError("timestamps must be timezone-aware")
        index = (
            values.tz_convert(UTC) if values.tz is not None else values.tz_localize(UTC)
        )
        index = index.copy()
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError("timestamps must be an iterable of aware timestamps")
        # Validate through the scalar contract, then parse the original value
        # with pandas to preserve fractional precision beyond microseconds.
        normalized = []
        for value in values:
            normalize_timestamp(value)
            normalized.append(pd.Timestamp(value).tz_convert(UTC))
        index = pd.DatetimeIndex(normalized, tz=UTC)
    if not index.is_unique:
        raise ValueError("timestamps must be unique after UTC normalization")
    if not index.is_monotonic_increasing:
        raise ValueError("timestamps must be strictly increasing")
    return index
