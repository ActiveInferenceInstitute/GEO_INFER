"""Numerical SPACE/TIME analysis on data persisted through the DATA backend."""

from __future__ import annotations

import asyncio

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geo_infer_data.core.storage import LocalFileBackend
from geo_infer_data.models.schemas import DataLineage, DatasetMetadata
from geo_infer_space.core.analytics import SpatialAnalyticsInterface
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries


@pytest.fixture
def sample_spatial_temporal_data():
    """Twenty distinct sensors with aware instants and reproducible values."""
    rng = np.random.default_rng(42)
    lats = rng.uniform(37.7, 37.9, 20)
    lngs = rng.uniform(-122.5, -122.3, 20)
    records = []
    for day, timestamp in enumerate(
        pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    ):
        for sensor in range(20):
            records.append(
                {
                    "timestamp": timestamp,
                    "geometry": Point(lngs[sensor], lats[sensor]),
                    "sensor_id": f"sensor_{sensor:03d}",
                    "temperature": 20 + day * 0.25 + rng.normal(0, 0.1),
                }
            )
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


@pytest.mark.integration
class TestSpaceTimeDataIntegration:
    def test_spatial_indexing_with_temporal_data(self, sample_spatial_temporal_data):
        frame = sample_spatial_temporal_data
        indexer = SpatialIndexingInterface(backend="h3")
        actual = [
            indexer.latlng_to_cell(point.y, point.x, resolution=9)
            for point in frame.geometry
        ]
        assert actual == [
            h3.latlng_to_cell(point.y, point.x, 9) for point in frame.geometry
        ]
        frame["cell"] = actual
        grouped = frame.groupby(["cell", "timestamp"])["temperature"].mean()
        assert grouped.index.get_level_values("timestamp").nunique() == 30
        for (cell, timestamp), value in grouped.items():
            expected = frame.loc[
                (frame.cell == cell) & (frame.timestamp == timestamp), "temperature"
            ].mean()
            assert value == pytest.approx(expected)

    def test_temporal_analysis_with_spatial_context(self, sample_spatial_temporal_data):
        frame = sample_spatial_temporal_data
        indexer = SpatialIndexingInterface(backend="h3")
        frame["cell"] = [
            indexer.latlng_to_cell(point.y, point.x, resolution=9)
            for point in frame.geometry
        ]
        analyzer = TemporalAnalyzer()
        for cell in sorted(frame.cell.unique())[:5]:
            # Multiple sensors in a cell combine before constructing the unique time axis.
            values = (
                frame.loc[frame.cell == cell].groupby("timestamp")["temperature"].mean()
            )
            series = TimeSeries(data=values.to_numpy(), timestamps=values.index)
            trend = analyzer.detect_trend(series, method="linear")
            slope, intercept = np.polyfit(np.arange(len(values)), values.to_numpy(), 1)
            assert trend["slope_per_sample"] == pytest.approx(slope, abs=1e-12)
            np.testing.assert_allclose(
                trend["trend_values"],
                intercept + slope * np.arange(len(values)),
                atol=1e-12,
                rtol=0,
            )
            assert trend["trend_direction"] == "increasing"
            assert str(series.timestamps.tz) == "UTC"

    def test_data_storage_and_retrieval_workflow(
        self, sample_spatial_temporal_data, tmp_path
    ):
        frame = sample_spatial_temporal_data
        backend = LocalFileBackend({"base_path": str(tmp_path)})
        metadata = DatasetMetadata(
            title="Spatial temporal fixture",
            lineage=DataLineage(
                source="deterministic sensors",
                process="analysis fixture",
                created_by="contract-test",
            ),
        )
        identifier = asyncio.run(backend.store(frame, metadata))
        restored = asyncio.run(
            LocalFileBackend({"base_path": str(tmp_path)}).retrieve(identifier, {})
        )
        pd.testing.assert_frame_equal(restored, frame)
        assert restored.crs == frame.crs
        assert str(restored.timestamp.dt.tz) == "UTC"

    def test_spatial_temporal_interpolation_workflow(
        self, sample_spatial_temporal_data
    ):
        frame = sample_spatial_temporal_data
        latest = frame.loc[frame.timestamp == frame.timestamp.max()]
        points = [
            (point.y, point.x, value)
            for point, value in zip(latest.geometry, latest.temperature)
        ]
        analytics = SpatialAnalyticsInterface(backend="h3")
        result = analytics.interpolate_values(points, resolution=9)
        expected = {h3.latlng_to_cell(lat, lng, 9): value for lat, lng, value in points}
        assert result["interpolated"] == expected
        assert result["target_count"] == len(expected)
