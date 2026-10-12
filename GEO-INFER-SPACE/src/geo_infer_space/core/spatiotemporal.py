"""Explicit H3 state and UTC time axes for observed geospatial series."""

from collections.abc import Iterable
from datetime import datetime
from itertools import islice

import numpy as np
import pandas as pd

from geo_infer_space.core.state_space import H3StateSpace
from geo_infer_time import TimeSeries, normalize_datetime_index

__all__ = ["align_h3_observations"]


def align_h3_observations(
    data: pd.DataFrame,
    *,
    state_space: H3StateSpace,
    timestamps: Iterable[datetime | str],
    cell_column: str = "cell",
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    max_entries: int = 1_000_000,
) -> TimeSeries:
    """Align finite observations to explicit ordered H3 and UTC axes.

    Columns follow ``state_space.cells`` and rows follow ``timestamps``.
    Absent cell/time pairs remain NaN. Duplicate normalized pairs, unknown
    cells/times, and nonfinite observations raise; no filling or averaging is
    performed. Check the output allocation budget before constructing its array.
    Frame attributes are retained as owned metadata. The structural
    ``spatial_index`` and ``crs`` fields describe the aligned H3 axis.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(state_space, H3StateSpace):
        raise TypeError("state_space must be an H3StateSpace")
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries < 1
    ):
        raise ValueError("max_entries must be a positive integer")
    columns = (cell_column, timestamp_column, value_column)
    if len(set(columns)) != 3 or any(column not in data.columns for column in columns):
        raise ValueError(
            "Provide three distinct existing cell, timestamp and value columns"
        )
    if not data.columns.is_unique:
        raise ValueError("data column names must be unique")
    if isinstance(timestamps, (str, bytes)):
        raise TypeError("timestamps must be an iterable of aware timestamps")
    max_times = max_entries // len(state_space.cells)
    supplied = list(islice(timestamps, max_times + 1))
    if len(supplied) > max_times:
        raise ValueError("Aligned observations exceed max_entries")
    time_axis = normalize_datetime_index(supplied)
    if len(time_axis) == 0:
        raise ValueError("timestamps must not be empty")
    cell_positions = {cell: i for i, cell in enumerate(state_space.cells)}
    time_positions = {timestamp: i for i, timestamp in enumerate(time_axis)}
    observed = []
    seen = set()
    for cell, raw_timestamp, value in data.loc[:, list(columns)].itertuples(
        index=False, name=None
    ):
        if not isinstance(cell, str) or cell not in cell_positions:
            raise ValueError("Observation cell is outside the H3 state space")
        instant = normalize_datetime_index([raw_timestamp])[0]
        if instant not in time_positions:
            raise ValueError("Observation timestamp is outside the supplied time axis")
        pair = (cell, instant)
        if pair in seen:
            raise ValueError(
                "Duplicate cell/timestamp observation after UTC normalization"
            )
        if isinstance(value, (bool, str, bytes)):
            raise TypeError("Observed values must be finite numbers")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("Observed values must be finite numbers") from exc
        if not np.isfinite(numeric_value):
            raise ValueError("Observed values must be finite")
        seen.add(pair)
        observed.append((time_positions[instant], cell_positions[cell], numeric_value))
    values = np.full((len(time_axis), len(state_space.cells)), np.nan)
    for row, column, value in observed:
        values[row, column] = value
    return TimeSeries(
        pd.DataFrame(values, index=time_axis, columns=state_space.cells),
        metadata={**data.attrs, "spatial_index": "h3", "crs": "EPSG:4326"},
    )
