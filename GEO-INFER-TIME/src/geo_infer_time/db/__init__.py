"""Database utilities for temporal data.

Provides an abstract storage interface and an in-memory implementation
for persisting and querying TimeSeries objects by name and time range.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

from geo_infer_time.models.timeseries import TimeSeries

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesStore",
    "InMemoryStore",
]


class TimeSeriesStore(ABC):
    """Abstract base class for time series storage backends.

    All storage implementations must provide the five core operations:
    ``store``, ``retrieve``, ``list_series``, ``delete``, and ``query``.

    Example subclasses might back onto PostgreSQL/TimescaleDB, InfluxDB,
    Redis, or a simple file system -- but the consumer code only depends
    on this interface.
    """

    @abstractmethod
    def store(self, name: str, ts: TimeSeries) -> None:
        """Store a TimeSeries under the given name.

        If a series with the same name already exists, it is overwritten.

        Args:
            name: Unique identifier for the time series.
            ts: The TimeSeries to store.
        """

    @abstractmethod
    def retrieve(self, name: str) -> TimeSeries:
        """Retrieve a TimeSeries by name.

        Args:
            name: Identifier of the time series.

        Returns:
            The stored TimeSeries.

        Raises:
            KeyError: If no series with *name* exists.
        """

    @abstractmethod
    def list_series(self) -> list[str]:
        """List the names of all stored time series.

        Returns:
            A sorted list of series names.
        """

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete a stored time series.

        Args:
            name: Identifier of the time series to remove.

        Raises:
            KeyError: If no series with *name* exists.
        """

    @abstractmethod
    def query(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> TimeSeries:
        """Query a stored time series for a specific time range.

        Args:
            name: Identifier of the time series.
            start: Start of the time range (inclusive). None means the
                beginning of the series.
            end: End of the time range (inclusive). None means the end
                of the series.

        Returns:
            A TimeSeries sliced to the requested range.

        Raises:
            KeyError: If no series with *name* exists.
        """


class InMemoryStore(TimeSeriesStore):
    """In-memory dictionary-backed time series store.

    Suitable for development, testing, and lightweight applications.
    Data does **not** persist across process restarts.

    Example::

        store = InMemoryStore()
        store.store("temperature", ts)
        names = store.list_series()          # ["temperature"]
        result = store.query("temperature",
                             start=datetime(2024, 1, 1),
                             end=datetime(2024, 6, 30))
    """

    def __init__(self) -> None:
        self._data: dict[str, TimeSeries] = {}
        logger.debug("Initialized InMemoryStore")

    def store(self, name: str, ts: TimeSeries) -> None:
        """Store a TimeSeries, making a defensive copy of the data."""
        self._data[name] = TimeSeries(
            data=ts.to_dataframe(),
            spatial_location=deepcopy(ts.spatial_location),
            metadata=deepcopy(ts.metadata),
        )
        logger.info("Stored time series '%s' (%d rows)", name, len(ts))

    def retrieve(self, name: str) -> TimeSeries:
        """Retrieve a full TimeSeries by name."""
        if name not in self._data:
            raise KeyError(f"No time series found with name '{name}'.")
        ts = self._data[name]
        logger.debug("Retrieved time series '%s' (%d rows)", name, len(ts))
        return TimeSeries(
            data=ts.to_dataframe(),
            spatial_location=deepcopy(ts.spatial_location),
            metadata=deepcopy(ts.metadata),
        )

    def list_series(self) -> list[str]:
        """Return a sorted list of all stored series names."""
        return sorted(self._data.keys())

    def delete(self, name: str) -> None:
        """Remove a stored time series."""
        if name not in self._data:
            raise KeyError(f"No time series found with name '{name}'.")
        del self._data[name]
        logger.info("Deleted time series '%s'", name)

    def query(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> TimeSeries:
        """Query a stored time series by time range.

        Returns a new TimeSeries containing only the rows within the
        [start, end] window. If start or end is None, the corresponding
        bound of the stored series is used.
        """
        if name not in self._data:
            raise KeyError(f"No time series found with name '{name}'.")

        ts = self._data[name]
        df = ts.to_dataframe()

        if start is not None:
            df = df.loc[df.index >= start]
        if end is not None:
            df = df.loc[df.index <= end]

        logger.debug(
            "Query '%s' [%s, %s] returned %d rows",
            name,
            start,
            end,
            len(df),
        )

        return TimeSeries(
            data=df,
            spatial_location=deepcopy(ts.spatial_location),
            metadata=deepcopy(ts.metadata),
        )

    def __len__(self) -> int:
        """Return the number of stored series."""
        return len(self._data)

    def __contains__(self, name: str) -> bool:
        """Check if a series name exists in the store."""
        return name in self._data

    def __repr__(self) -> str:
        return f"InMemoryStore(series_count={len(self._data)})"
