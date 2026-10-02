"""Retain the 100,000-record composition workload in the performance lane."""

from time import monotonic
import math

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point


@pytest.mark.performance
@pytest.mark.timeout(120)
def test_large_scale_spatial_temporal_processing(record_testsuite_property):
    rng = np.random.default_rng(42)
    n_points, n_timestamps = 1000, 100
    coordinates = np.column_stack(
        (rng.uniform(37.7, 37.9, n_points), rng.uniform(-122.5, -122.3, n_points))
    )
    points = [Point(longitude, latitude) for latitude, longitude in coordinates]
    dates = pd.date_range("2026-01-01", periods=n_timestamps, freq="D", tz="UTC")
    frame = gpd.GeoDataFrame(
        {
            "timestamp": np.repeat(dates, n_points),
            "geometry": points * n_timestamps,
            "sensor_id": np.tile(np.arange(n_points), n_timestamps),
            "value": rng.normal(size=n_points * n_timestamps),
        },
        crs="EPSG:4326",
    )
    started = monotonic()
    index = frame.sindex
    timings = {"spatial_index_seconds": monotonic() - started}
    assert index.size == len(frame)
    started = monotonic()
    frame["h3_index"] = frame.geometry.map(
        lambda point: h3.latlng_to_cell(point.y, point.x, 10)
    )
    timings["h3_index_seconds"] = monotonic() - started
    expected = [
        h3.latlng_to_cell(latitude, longitude, 10)
        for latitude, longitude in coordinates
    ]
    assert frame["h3_index"].tolist() == expected * n_timestamps
    started = monotonic()
    aggregated = frame.groupby(["sensor_id", "timestamp"])["value"].mean()
    timings["aggregation_seconds"] = monotonic() - started
    assert len(aggregated) == n_points * n_timestamps
    np.testing.assert_array_equal(
        aggregated.to_numpy(),
        frame.sort_values(["sensor_id", "timestamp"])["value"].to_numpy(),
    )
    for name, seconds in timings.items():
        assert math.isfinite(seconds) and seconds >= 0
        record_testsuite_property(name, seconds)
