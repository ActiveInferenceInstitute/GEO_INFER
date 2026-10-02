"""Behavioral contracts for the public TimeSeries data model."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import geo_infer_time
from geo_infer_time.models.timeseries import TimeSeries


@pytest.fixture
def daily_series() -> TimeSeries:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame({"value": [1.0, np.nan, 3.0, 4.0]}, index=index)
    return TimeSeries(
        data,
        spatial_location={"lat": 45.5, "lon": -122.6},
        metadata={"source": "sensor"},
    )


def test_package_version_matches_distribution_metadata() -> None:
    assert geo_infer_time.__version__ == version("geo-infer-time")


def test_constructor_normalizes_supported_inputs_and_copies_metadata() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    metadata = {"source": "array"}
    location = {"lat": 1.0, "lon": 2.0}

    timeseries = TimeSeries(
        np.array([1.0, 2.0, 3.0]),
        timestamps=timestamps,
        metadata=metadata,
        spatial_location=location,
    )
    metadata["source"] = "changed"
    location["lat"] = 99.0

    assert list(timeseries.timestamps) == list(timestamps)
    assert timeseries.to_dataframe().iloc[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert timeseries.metadata == {"source": "array"}
    assert timeseries.spatial_location == {"lat": 1.0, "lon": 2.0}


@pytest.mark.parametrize(
    ("data", "timestamps", "exception"),
    [
        ([1, 2], None, TypeError),
        (np.array([1, 2]), None, ValueError),
        (
            np.array([1, 2]),
            pd.date_range("2024-01-01", periods=1, tz="UTC"),
            ValueError,
        ),
    ],
)
def test_constructor_rejects_invalid_input_contracts(data, timestamps, exception):
    with pytest.raises(exception):
        TimeSeries(data, timestamps=timestamps)


def test_constructor_converts_datetime_like_index() -> None:
    timeseries = TimeSeries(
        pd.Series([1, 2], index=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])
    )

    assert isinstance(timeseries.timestamps, pd.DatetimeIndex)
    assert timeseries.start_time == pd.Timestamp("2024-01-01", tz="UTC")
    assert timeseries.end_time == pd.Timestamp("2024-01-02", tz="UTC")
    assert timeseries.duration == pd.Timedelta(days=1)
    assert timeseries.frequency is None


def test_empty_timeseries_has_explicit_temporal_boundary_error() -> None:
    timeseries = TimeSeries(pd.DataFrame(index=pd.DatetimeIndex([])))

    with pytest.raises(ValueError, match="empty"):
        _ = timeseries.start_time
    with pytest.raises(ValueError, match="empty"):
        _ = timeseries.end_time


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", [1.5, 3.5]),
        ("sum", [3.0, 7.0]),
        ("max", [2.0, 4.0]),
        ("min", [1.0, 3.0]),
        ("first", [1.0, 3.0]),
        ("last", [2.0, 4.0]),
    ],
)
def test_resample_supports_each_documented_method(method: str, expected: list[float]):
    index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    timeseries = TimeSeries(pd.Series([1.0, 2.0, 3.0, 4.0], index=index))

    resampled = timeseries.resample("2h", method=method)

    assert resampled.to_dataframe().iloc[:, 0].tolist() == expected
    assert resampled.metadata["resampled_from"] == "h"


def test_resample_rejects_unknown_method(daily_series: TimeSeries) -> None:
    with pytest.raises(ValueError, match="Unknown resampling method"):
        daily_series.resample("2D", method="median")


def test_interpolation_statistics_and_dataframe_copy(daily_series: TimeSeries) -> None:
    interpolated = daily_series.interpolate(method="linear")
    stats = interpolated.get_statistics()
    exported = interpolated.to_dataframe()
    exported.iloc[0, 0] = 999

    assert interpolated.to_dataframe().iloc[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert interpolated.metadata == {"source": "sensor", "interpolated": True}
    assert stats["count"] == 4
    assert stats["duration_days"] == 3.0
    assert stats["frequency"] == "D"
    assert stats["value"]["missing_count"] == 0
    assert stats["value"]["mean"] == pytest.approx(2.5)


def test_slice_preserves_context_and_validates_bounds(daily_series: TimeSeries) -> None:
    sliced = daily_series.slice(
        pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
    )

    assert len(sliced) == 2
    assert sliced.metadata == {"source": "sensor", "sliced": True}
    assert sliced.spatial_location == {"lat": 45.5, "lon": -122.6}

    with pytest.raises(ValueError, match="must not be after"):
        daily_series.slice(
            pd.Timestamp("2024-01-03", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )
