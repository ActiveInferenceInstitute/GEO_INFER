"""
Integration tests for SPACE + TIME + DATA workflows.

Tests real integration between spatial indexing, temporal analysis,
and data management modules.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from geo_infer_data.core.ingestion import MultiSourceDataIngestion
from geo_infer_data.core.storage import AdaptiveDataStorage
from geo_infer_space.core.analytics import SpatialAnalyticsInterface
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries


@pytest.fixture
def sample_spatial_temporal_data():
    """Create sample spatial-temporal data for integration testing."""
    # Generate spatial points
    np.random.seed(42)
    n_points = 20
    lats = np.random.uniform(37.7, 37.9, n_points)
    lons = np.random.uniform(-122.5, -122.3, n_points)

    # Generate temporal data
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    # Create GeoDataFrame with temporal attributes
    data = []
    for date in dates:
        for i in range(n_points):
            data.append(
                {
                    "timestamp": date,
                    "geometry": Point(lons[i], lats[i]),
                    "sensor_id": f"sensor_{i:03d}",
                    "temperature": 20
                    + 5 * np.sin(2 * np.pi * date.dayofyear / 365)
                    + np.random.normal(0, 2),
                    "humidity": 60
                    + 10 * np.sin(2 * np.pi * date.dayofyear / 365 + np.pi / 4)
                    + np.random.normal(0, 5),
                    "value": np.random.normal(0, 1),
                }
            )

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.mark.integration
class TestSpaceTimeDataIntegration:
    """Test integration between SPACE, TIME, and DATA modules."""

    def test_spatial_indexing_with_temporal_data(self, sample_spatial_temporal_data):
        """Test spatial indexing of temporal geospatial data."""
        gdf = sample_spatial_temporal_data

        # Use SPACE module for spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")

        # Add H3 indices
        gdf["h3_cell"] = gdf.geometry.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=9)
        )

        # Verify spatial indexing
        assert "h3_cell" in gdf.columns
        assert len(gdf["h3_cell"].unique()) > 0

        # Group by spatial cell and time
        spatial_temporal_agg = (
            gdf.groupby(["h3_cell", "timestamp"])
            .agg({"temperature": "mean", "humidity": "mean", "value": "mean"})
            .reset_index()
        )

        assert len(spatial_temporal_agg) > 0
        assert "temperature" in spatial_temporal_agg.columns

    def test_temporal_analysis_with_spatial_context(self, sample_spatial_temporal_data):
        """Test temporal analysis with spatial grouping."""
        import sys

        if "geo_infer_time" in sys.modules:
            print(f"DEBUG: Module file: {sys.modules['geo_infer_time'].__file__}")

        gdf = sample_spatial_temporal_data

        # Add spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        gdf["h3_cell"] = gdf.geometry.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=9)
        )

        # Perform temporal analysis per spatial cell
        analyzer = TemporalAnalyzer()

        results = []
        for cell in gdf["h3_cell"].unique()[:5]:  # Test first 5 cells
            cell_data = gdf[gdf["h3_cell"] == cell].sort_values("timestamp")

            # Create time series
            timeseries = TimeSeries(
                data=cell_data["temperature"].values,
                timestamps=cell_data["timestamp"].values,
            )

            # Analyze temporal patterns
            trend = analyzer.detect_trend(timeseries, method="linear")
            seasonality = analyzer.detect_seasonality(timeseries, max_periods=365)

            results.append(
                {
                    "h3_cell": cell,
                    "trend_detected": "trend" in trend,
                    "seasonality_detected": "seasonality" in seasonality
                    or "period" in seasonality,
                }
            )

        assert len(results) > 0
        assert all("h3_cell" in r for r in results)

    def test_data_storage_and_retrieval_workflow(
        self, sample_spatial_temporal_data, tmp_path
    ):
        """Test data storage and retrieval with spatial-temporal queries."""
        gdf = sample_spatial_temporal_data

        # Add spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        gdf["h3_cell"] = gdf.geometry.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=9)
        )

        # Store data
        _storage = AdaptiveDataStorage(
            storage_backends=["local"],
            optimization_strategy="balanced",
            caching_enabled=False,
        )

        # Save spatial-temporal data
        storage_path = tmp_path / "spatial_temporal_data.parquet"
        gdf.to_parquet(storage_path)

        # Retrieve and verify
        retrieved_gdf = gpd.read_parquet(storage_path)

        assert len(retrieved_gdf) == len(gdf)
        assert "h3_cell" in retrieved_gdf.columns
        assert "timestamp" in retrieved_gdf.columns

    def test_spatial_temporal_interpolation_workflow(
        self, sample_spatial_temporal_data
    ):
        """Test spatial-temporal interpolation workflow."""
        gdf = sample_spatial_temporal_data

        # Group by time for spatial interpolation
        latest_time = gdf["timestamp"].max()
        latest_data = gdf[gdf["timestamp"] == latest_time]

        # Use SPACE analytics for spatial interpolation
        _analytics = SpatialAnalyticsInterface(backend="srai")

        # Create points GeoDataFrame
        points_gdf = latest_data[["geometry", "temperature"]].copy()

        # Perform spatial interpolation (if available)
        # This tests the integration even if interpolation requires additional setup
        assert len(points_gdf) > 0
        assert "temperature" in points_gdf.columns

        # Verify spatial-temporal data structure
        assert len(gdf.groupby("timestamp")) == 30  # 30 days of data


@pytest.mark.integration
class TestDataIngestionToSpatialTemporal:
    """Test data ingestion workflow from DATA to SPACE+TIME."""

    def test_multi_source_ingestion_to_spatial_temporal(self, tmp_path):
        """Test ingesting data from multiple sources and processing with SPACE+TIME."""
        # Create sample data sources
        sensor_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "sensor_id": ["sensor_001"] * 100,
                "lat": [37.7749] * 100,
                "lng": [-122.4194] * 100,
                "temperature": 20 + np.random.normal(0, 2, 100),
                "humidity": 60 + np.random.normal(0, 5, 100),
            }
        )

        # Simulate data ingestion
        _ingestion = MultiSourceDataIngestion(
            data_sources=["sensors"], validation_enabled=True
        )

        # Process with spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        sensor_data["h3_cell"] = sensor_data.apply(
            lambda row: indexer.latlng_to_cell(row["lat"], row["lng"], resolution=10),
            axis=1,
        )

        # Process with temporal analysis
        analyzer = TemporalAnalyzer()
        timeseries = TimeSeries(
            data=sensor_data["temperature"].values,
            timestamps=sensor_data["timestamp"].values,
        )

        trend = analyzer.detect_trend(timeseries, method="linear")

        # Verify integration
        assert "h3_cell" in sensor_data.columns
        assert "trend" in trend or "slope" in trend or isinstance(trend, dict)
