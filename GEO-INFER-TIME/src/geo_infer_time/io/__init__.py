"""Input/output utilities for temporal data.

Provides readers and writers for common time series file formats
(CSV, JSON, Parquet) with conversion to/from the TimeSeries model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from geo_infer_time.models.timeseries import TimeSeries

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesReader",
    "TimeSeriesWriter",
    "read_timeseries",
    "write_timeseries",
]


class TimeSeriesReader:
    """Read time series data from various file formats.

    Supports CSV, JSON, and Parquet files. Each format reader converts the
    file contents into a TimeSeries object with a DatetimeIndex.

    Example::

        reader = TimeSeriesReader()
        ts = reader.read("data/temperature.csv", time_column="date")
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".parquet", ".pq"}

    def read(
        self,
        path: str | Path,
        time_column: str | None = None,
        metadata: dict[str, Any] | None = None,
        spatial_location: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TimeSeries:
        """Read a time series file into a TimeSeries object.

        Args:
            path: Path to the data file (CSV, JSON, or Parquet).
            time_column: Name of the column containing timestamps. If None,
                the reader looks for common names ('timestamp', 'time',
                'date', 'datetime') or uses the DataFrame index.
            metadata: Optional metadata dict to attach to the TimeSeries.
            spatial_location: Optional spatial location dict, e.g.
                ``{"lat": 45.0, "lon": -122.0}``.
            **kwargs: Additional keyword arguments forwarded to the
                underlying pandas read function.

        Returns:
            A TimeSeries object.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )

        logger.info("Reading time series from %s", path)

        if suffix == ".csv":
            df = self._read_csv(path, **kwargs)
        elif suffix == ".json":
            df = self._read_json(path, **kwargs)
        elif suffix in (".parquet", ".pq"):
            df = self._read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        df = self._set_time_index(df, time_column)

        logger.info(
            "Loaded %d rows x %d columns from %s",
            len(df),
            len(df.columns),
            path.name,
        )

        return TimeSeries(
            data=df,
            spatial_location=spatial_location,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
        """Read CSV with sensible defaults."""
        kwargs.setdefault("parse_dates", True)
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def _read_json(path: Path, **kwargs: Any) -> pd.DataFrame:
        """Read JSON (records or table orientation)."""
        kwargs.setdefault("convert_dates", True)
        return pd.read_json(path, **kwargs)

    @staticmethod
    def _read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
        """Read Parquet file."""
        return pd.read_parquet(path, **kwargs)

    @staticmethod
    def _set_time_index(df: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
        """Ensure the DataFrame has a DatetimeIndex.

        If *time_column* is given, that column is converted to datetime and
        set as the index. Otherwise the method looks for common column names
        or falls back to the existing index.
        """
        common_names = ["timestamp", "time", "date", "datetime"]

        if time_column is not None:
            if time_column not in df.columns:
                raise ValueError(
                    f"Column '{time_column}' not found in data. "
                    f"Available: {list(df.columns)}"
                )
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.set_index(time_column)
            return df

        # Already a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # Auto-detect a time column
        for name in common_names:
            if name in df.columns:
                df[name] = pd.to_datetime(df[name])
                df = df.set_index(name)
                logger.debug("Auto-detected time column: %s", name)
                return df

        # Last resort: try to parse the existing index
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(
                "Could not identify a time column or convert the index "
                "to datetime. Specify time_column explicitly."
            )

        return df


class TimeSeriesWriter:
    """Write TimeSeries objects to various file formats.

    Supports CSV, JSON, and Parquet output. Metadata and spatial location
    are stored as a sidecar JSON file (``<name>.meta.json``) when present.

    Example::

        writer = TimeSeriesWriter()
        writer.write(ts, "output/temperature.parquet")
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".parquet", ".pq"}

    def write(
        self,
        ts: TimeSeries,
        path: str | Path,
        write_metadata: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Write a TimeSeries to a file.

        Args:
            ts: The TimeSeries to write.
            path: Destination file path.
            write_metadata: If True and the TimeSeries has metadata or
                spatial_location, write a sidecar ``.meta.json`` file.
            **kwargs: Additional keyword arguments forwarded to the
                underlying pandas write function.

        Returns:
            The resolved Path that was written.

        Raises:
            ValueError: If the file format is not supported.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        df = ts.to_dataframe()

        logger.info("Writing time series (%d rows) to %s", len(df), path)

        if suffix == ".csv":
            df.to_csv(path, **kwargs)
        elif suffix == ".json":
            kwargs.setdefault("date_format", "iso")
            df.to_json(path, **kwargs)
        elif suffix in (".parquet", ".pq"):
            df.to_parquet(path, **kwargs)

        # Write sidecar metadata
        if write_metadata and (ts.metadata or ts.spatial_location):
            meta_path = path.with_suffix(".meta.json")
            meta = {}
            if ts.metadata:
                meta["metadata"] = ts.metadata
            if ts.spatial_location:
                meta["spatial_location"] = ts.spatial_location
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            logger.debug("Wrote metadata sidecar to %s", meta_path)

        return path


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------


def read_timeseries(
    path: str | Path,
    time_column: str | None = None,
    metadata: dict[str, Any] | None = None,
    spatial_location: dict[str, float] | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """Read a time series file into a TimeSeries object.

    This is a convenience wrapper around :class:`TimeSeriesReader`.

    Args:
        path: Path to the data file (CSV, JSON, or Parquet).
        time_column: Name of the column containing timestamps.
        metadata: Optional metadata dict.
        spatial_location: Optional spatial location dict.
        **kwargs: Forwarded to the pandas reader.

    Returns:
        A TimeSeries object.
    """
    reader = TimeSeriesReader()
    return reader.read(
        path,
        time_column=time_column,
        metadata=metadata,
        spatial_location=spatial_location,
        **kwargs,
    )


def write_timeseries(
    ts: TimeSeries,
    path: str | Path,
    write_metadata: bool = True,
    **kwargs: Any,
) -> Path:
    """Write a TimeSeries to a file.

    This is a convenience wrapper around :class:`TimeSeriesWriter`.

    Args:
        ts: The TimeSeries to write.
        path: Destination file path.
        write_metadata: Whether to write a sidecar metadata file.
        **kwargs: Forwarded to the pandas writer.

    Returns:
        The resolved Path that was written.
    """
    writer = TimeSeriesWriter()
    return writer.write(ts, path, write_metadata=write_metadata, **kwargs)
