#!/usr/bin/env python3
"""
Adaptive data storage example for GEO-INFER-DATA.

This example demonstrates how to use the AdaptiveDataStorage class
to store and retrieve geospatial data with automatic optimization.

Usage:
    python storage_example.py

Requirements:
    - GEO-INFER-DATA package installed
    - Required dependencies (geopandas, pandas, numpy, etc.)
"""

import asyncio
import logging
from datetime import datetime, UTC
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from geo_infer_data.core.storage import AdaptiveDataStorage
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_geodataframe():
    """Create sample geospatial data for the example."""
    # Create sample points in San Francisco area
    n_points = 1000
    latitudes = np.random.normal(37.7749, 0.1, n_points)
    longitudes = np.random.normal(-122.4194, 0.1, n_points)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "id": range(n_points),
            "timestamp": pd.date_range("2023-01-01", periods=n_points, freq="h"),
            "temperature": np.random.normal(18, 5, n_points),
            "humidity": np.random.normal(65, 10, n_points),
            "air_quality": np.random.normal(50, 15, n_points),
            "noise_level": np.random.normal(60, 8, n_points),
        },
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )

    return gdf


def create_sample_dataframe():
    """Create sample time series data for the example."""
    # Create sample time series data
    n_records = 10000
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="15min"),
            "sensor_id": np.random.choice(
                ["sensor_001", "sensor_002", "sensor_003"], n_records
            ),
            "temperature": np.random.normal(20, 3, n_records),
            "humidity": np.random.normal(60, 8, n_records),
            "pressure": np.random.normal(1013, 5, n_records),
        }
    )

    return df


async def main():
    """Main example function."""
    logger.info("Starting adaptive storage example")

    # Initialize storage system
    storage = AdaptiveDataStorage(
        storage_backends=["postgresql", "minio", "redis"],
        optimization_strategy="access_pattern_based",
        compression_enabled=True,
        indexing_strategy="h3",
        caching_enabled=True,
    )

    logger.info("Initialized storage system")

    # Create sample data
    logger.info("Creating sample datasets")

    # Geospatial point data
    geodataframe = create_sample_geodataframe()
    logger.info(f"Created GeoDataFrame with {len(geodataframe)} records")

    # Time series data
    time_series_df = create_sample_dataframe()
    logger.info(f"Created time series DataFrame with {len(time_series_df)} records")

    # Create metadata
    geospatial_metadata = DatasetMetadata(
        title="San Francisco Environmental Monitoring Points",
        description="Environmental monitoring data from sensors across San Francisco",
        spatial=SpatialExtent(bbox=[-122.6, 37.6, -122.2, 38.0], crs="EPSG:4326"),
        temporal=TemporalExtent(
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2023, 12, 31, tzinfo=UTC),
            resolution="PT1H",
        ),
        lineage=DataLineage(
            source="environmental_sensors",
            process="automated_collection",
            created_by="geo_infer_data_example",
        ),
        keywords=["environment", "monitoring", "san_francisco", "sensors"],
        contact={"organization": "Example Organization", "email": "data@example.com"},
    )

    time_series_metadata = DatasetMetadata(
        title="Environmental Time Series Data",
        description="High-frequency environmental measurements from weather stations",
        spatial=SpatialExtent(bbox=[-122.6, 37.6, -122.2, 38.0], crs="EPSG:4326"),
        temporal=TemporalExtent(
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2023, 12, 31, tzinfo=UTC),
            resolution="PT15M",
        ),
        lineage=DataLineage(
            source="weather_stations",
            process="real_time_collection",
            created_by="geo_infer_data_example",
        ),
        keywords=["time_series", "weather", "environmental", "real_time"],
        contact={"organization": "Example Organization", "email": "data@example.com"},
    )

    # Store datasets
    logger.info("Storing geospatial dataset")

    geospatial_id = await storage.store_geospatial_data(
        geodataframe,
        geospatial_metadata,
        access_patterns={
            "spatial_queries": [
                {"bbox": [-122.5, 37.7, -122.3, 37.9], "frequency": "high"},
                {"bbox": [-122.4, 37.6, -122.2, 37.8], "frequency": "medium"},
            ],
            "temporal_queries": [
                {
                    "start": datetime(2023, 6, 1, tzinfo=UTC),
                    "end": datetime(2023, 8, 31, tzinfo=UTC),
                    "frequency": "high",
                }
            ],
            "query_frequency": "high",
        },
    )

    logger.info(f"Stored geospatial dataset with ID: {geospatial_id}")

    logger.info("Storing time series dataset")

    time_series_id = await storage.store_geospatial_data(
        time_series_df,
        time_series_metadata,
        access_patterns={
            "temporal_queries": [
                {
                    "start": datetime(2023, 1, 1, tzinfo=UTC),
                    "end": datetime(2023, 12, 31, tzinfo=UTC),
                    "frequency": "daily",
                }
            ],
            "query_frequency": "medium",
            "batch_processing": True,
        },
    )

    logger.info(f"Stored time series dataset with ID: {time_series_id}")

    # Query data
    logger.info("Querying stored data")

    # Spatial query
    spatial_results = await storage.adaptive_query(
        spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
        temporal_range=(
            datetime(2023, 6, 1, tzinfo=UTC),
            datetime(2023, 6, 30, tzinfo=UTC),
        ),
        optimization_hints={"frequent_queries": True},
    )

    logger.info(
        f"Spatial query returned {len(spatial_results) if hasattr(spatial_results, '__len__') else 1} results"
    )

    # Temporal query
    temporal_results = await storage.adaptive_query(
        temporal_range=(
            datetime(2023, 7, 1, tzinfo=UTC),
            datetime(2023, 7, 31, tzinfo=UTC),
        ),
        optimization_hints={"real_time": False},
    )

    logger.info(
        f"Temporal query returned {len(temporal_results) if hasattr(temporal_results, '__len__') else 1} results"
    )

    # Optimize storage based on patterns
    logger.info("Optimizing storage for access patterns")

    patterns = {
        geospatial_id: {
            "frequent_queries": True,
            "spatial_bounds": [[-122.5, 37.7, -122.3, 37.9]],
            "peak_hours": [9, 10, 11, 14, 15, 16],
        },
        time_series_id: {
            "batch_processing": True,
            "temporal_ranges": [
                (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC))
            ],
            "query_frequency": "weekly",
        },
    }

    optimizations = storage.optimize_for_patterns(patterns, time_window="30d")

    logger.info("Storage optimizations applied:")
    for action in optimizations["actions"]:
        logger.info(f"  - {action}")

    # Get storage statistics
    logger.info("Getting storage statistics")

    stats = storage.get_storage_stats()

    logger.info("Storage Statistics:")
    logger.info(f"  Backends: {', '.join(stats['backends'])}")
    logger.info(f"  Total datasets: {stats['datasets']}")
    logger.info(f"  Total size: {stats['total_size'] / (1024 * 1024):.2f} MB")
    logger.info(f"  Optimization strategy: {stats['optimization_strategy']}")
    logger.info(f"  Compression enabled: {stats['compression_enabled']}")
    logger.info(f"  Caching enabled: {stats['caching_enabled']}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save sample data
    geodataframe.to_file(output_dir / "sample_geodata.geojson", driver="GeoJSON")
    time_series_df.to_csv(output_dir / "sample_timeseries.csv", index=False)

    # Save metadata
    with open(output_dir / "geospatial_metadata.json", "w") as f:
        import json

        json.dump(geospatial_metadata.dict(), f, indent=2, default=str)

    with open(output_dir / "timeseries_metadata.json", "w") as f:
        json.dump(time_series_metadata.dict(), f, indent=2, default=str)

    logger.info(f"Example results saved to {output_dir}")

    logger.info("Adaptive storage example completed successfully")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
