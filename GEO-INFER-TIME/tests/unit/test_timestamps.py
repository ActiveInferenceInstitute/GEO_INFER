"""UTC identity, precision and fail-closed temporal axis contracts."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from geo_infer_time import TimeSeries, normalize_datetime_index, normalize_timestamp
from geo_infer_time.core.inference_schedule import inference_schedule
from geo_infer_time.io import TimeSeriesReader, TimeSeriesWriter


def test_equivalent_offsets_have_one_utc_identity():
    assert (
        normalize_timestamp("2024-01-01T00:00:00Z")
        == normalize_timestamp("2023-12-31T16:00:00-08:00")
        == datetime(2024, 1, 1, tzinfo=UTC)
    )
    with pytest.raises(ValueError, match="unique"):
        normalize_datetime_index(["2024-01-01T00:00:00Z", "2023-12-31T16:00:00-08:00"])


@pytest.mark.parametrize("value", [datetime(2024, 1, 1), "2024-01-01", pd.NaT])
def test_ambiguous_or_missing_instants_rejected(value):
    with pytest.raises(ValueError):
        normalize_timestamp(value)
    with pytest.raises(ValueError):
        normalize_datetime_index([value])


@pytest.mark.parametrize("value", [0, 1.0, True, None, np.datetime64("2024-01-01")])
def test_implicit_epoch_or_unsupported_values_rejected(value):
    with pytest.raises(TypeError):
        normalize_timestamp(value)
    with pytest.raises(TypeError):
        normalize_datetime_index([value])


def test_dst_fold_preserves_two_distinct_instants():
    zone = ZoneInfo("America/Los_Angeles")
    first = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=0)
    second = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=1)
    index = normalize_datetime_index([first, second])
    assert index[1] - index[0] == pd.Timedelta(hours=1)


def test_nanosecond_index_and_duration_survive_construction():
    stamps = ["2024-01-01T00:00:00.000000001Z", "2024-01-01T00:00:00.000000009Z"]
    series = TimeSeries(np.array([2.0, 3.0]), timestamps=stamps)
    assert series.start_time.nanosecond == 1
    assert series.end_time.nanosecond == 9
    assert series.duration == pd.Timedelta(nanoseconds=8)
    assert series.timestamps.equals(normalize_datetime_index(stamps))


def test_time_series_does_not_invent_time_or_reorder_observations():
    with pytest.raises(TypeError, match="timestamp"):
        TimeSeries(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="increasing"):
        TimeSeries(
            pd.Series(
                [1.0, 2.0],
                index=pd.DatetimeIndex(
                    ["2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z"]
                ),
            )
        )


def test_time_series_owns_input_values():
    source = pd.Series(
        [1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
    )
    series = TimeSeries(source)
    source.iloc[0] = 99
    assert series.to_dataframe().iloc[0, 0] == 1


def test_csv_preserves_utc_identity_and_rejects_naive_data(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text(
        "timestamp,value\n2023-12-31T16:00:00-08:00,2\n2024-01-01T01:00:00Z,4\n"
    )
    series = TimeSeriesReader().read(path)
    assert series.timestamps[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert series.to_dataframe()["value"].tolist() == [2, 4]
    path.write_text("timestamp,value\n2024-01-01,2\n")
    with pytest.raises(ValueError, match="timezone-aware"):
        TimeSeriesReader().read(path)


def test_csv_nanoseconds_round_trip(tmp_path):
    series = TimeSeries(
        pd.DataFrame(
            {"value": [2.0, 3.0]},
            index=pd.DatetimeIndex(
                ["2024-01-01T00:00:00.000000001Z", "2024-01-01T00:00:00.000000009Z"],
                name="timestamp",
            ),
        )
    )
    path = TimeSeriesWriter().write(series, tmp_path / "precise.csv")
    pd.testing.assert_frame_equal(
        TimeSeriesReader().read(path).to_dataframe(), series.to_dataframe()
    )


def test_schedule_keeps_existing_explicit_gap_contract():
    with pytest.raises(ValueError, match="step_seconds"):
        inference_schedule(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:02:00Z"], step_seconds=60
        )


def test_schedule_does_not_truncate_submicrosecond_gap():
    with pytest.raises(ValueError, match="step_seconds"):
        inference_schedule(
            ["2024-01-01T00:00:00.000000001Z", "2024-01-01T00:01:00.000000002Z"],
            step_seconds=60,
        )
