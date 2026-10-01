#!/usr/bin/env python3
"""
Data validation example for GEO-INFER-DATA.

This example demonstrates how to use the DataQualityManager and
GeospatialValidator classes for comprehensive data validation.

Usage:
    python validation_example.py

Requirements:
    - GEO-INFER-DATA package installed
    - Required dependencies (geopandas, pandas, numpy, etc.)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from geo_infer_data.core.validation import DataQualityManager
from geo_infer_data.utils.validation import GeospatialValidator
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
    QualityStatus,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_valid_geodataframe():
    """Create valid geospatial data for testing."""
    # Create clean geospatial data
    n_points = 1000
    latitudes = np.random.normal(37.7749, 0.05, n_points)  # San Francisco area
    longitudes = np.random.normal(-122.4194, 0.05, n_points)

    return gpd.GeoDataFrame(
        {
            "id": range(n_points),
            "timestamp": pd.date_range("2023-01-01", periods=n_points, freq="h"),
            "temperature": np.random.normal(
                18, 3, n_points
            ),  # Realistic temperature range
            "humidity": np.random.normal(65, 8, n_points),  # Realistic humidity range
            "air_quality": np.random.normal(45, 10, n_points),  # AQI values
        },
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


def create_invalid_geodataframe():
    """Create invalid geospatial data for testing."""
    n_points = 500

    # Create some invalid geometries and coordinates
    latitudes = []
    longitudes = []
    temperatures = []

    for i in range(n_points):
        if i < 50:  # Invalid coordinates (outside valid range)
            latitudes.append(np.random.choice([100, -100]))  # Invalid latitude
            longitudes.append(np.random.choice([200, -200]))  # Invalid longitude
            temperatures.append(np.random.normal(20, 5))
        elif i < 100:  # Null geometries
            latitudes.append(37.7749)
            longitudes.append(-122.4194)
            temperatures.append(None)
        elif i < 150:  # Outlier temperatures
            latitudes.append(37.7749 + np.random.normal(0, 0.01))
            longitudes.append(-122.4194 + np.random.normal(0, 0.01))
            temperatures.append(
                np.random.choice([100, -50])
            )  # Unrealistic temperatures
        else:  # Valid data
            latitudes.append(37.7749 + np.random.normal(0, 0.01))
            longitudes.append(-122.4194 + np.random.normal(0, 0.01))
            temperatures.append(np.random.normal(18, 3))

    return gpd.GeoDataFrame(
        {
            "id": range(n_points),
            "temperature": temperatures,
            "humidity": np.random.normal(65, 8, n_points),
        },
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


def create_incomplete_dataframe():
    """Create incomplete data with missing values."""
    n_records = 1000

    # Create data with various missing patterns
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="h"),
        "temperature": np.random.normal(20, 5, n_records),
        "humidity": np.random.normal(60, 10, n_records),
        "pressure": np.random.normal(1013, 5, n_records),
        "wind_speed": np.random.normal(5, 2, n_records),
    }

    # Introduce missing values
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        missing_indices = np.random.choice(
            n_records, size=n_records // 5, replace=False
        )
        data[col].iloc[missing_indices] = None

    return pd.DataFrame(data)


async def main():
    """Main example function."""
    logger.info("Starting data validation example")

    # Initialize validation systems
    quality_manager = DataQualityManager(
        validation_rules="comprehensive",
        quality_threshold=0.8,
        real_time_monitoring=True,
    )

    validator = GeospatialValidator()

    logger.info("Initialized validation systems")

    # Test with valid data
    logger.info("Testing with valid geospatial data")

    valid_gdf = create_valid_geodataframe()

    # Validate using GeospatialValidator
    validation_result = await validator.build_quality_report(valid_gdf)

    logger.info("Valid Data Validation Results:")
    logger.info(f"  Overall Score: {validation_result.overall_score:.2f}")
    logger.info(f"  Status: {validation_result.status}")
    logger.info(f"  Issues Found: {len(validation_result.issues)}")

    if validation_result.issues:
        logger.info("  Issues:")
        for issue in validation_result.issues[:5]:  # Show first 5 issues
            logger.info(f"    - {issue['type']}: {issue['message']}")

    # Validate using DataQualityManager
    metadata = DatasetMetadata(
        title="Valid Environmental Data",
        description="Clean environmental monitoring data",
        spatial=SpatialExtent(bbox=[-122.6, 37.6, -122.2, 38.0], crs="EPSG:4326"),
        temporal=TemporalExtent(start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)),
        lineage=DataLineage(
            source="sensors",
            process="automated_collection",
            created_by="validation_example",
        ),
    )

    quality_report = await quality_manager.validator.build_quality_report(
        valid_gdf, metadata
    )

    logger.info("Quality Manager Results:")
    logger.info(f"  Overall Score: {quality_report.overall_score:.2f}")
    logger.info(f"  Assessment Method: {quality_report.assessment_method}")

    # Test with invalid data
    logger.info("Testing with invalid geospatial data")

    invalid_gdf = create_invalid_geodataframe()

    invalid_result = await validator.build_quality_report(invalid_gdf)

    logger.info("Invalid Data Validation Results:")
    logger.info(f"  Overall Score: {invalid_result.overall_score:.2f}")
    logger.info(f"  Status: {invalid_result.status}")
    logger.info(f"  Issues Found: {len(invalid_result.issues)}")

    if invalid_result.issues:
        logger.info("  Issues:")
        for issue in invalid_result.issues[:5]:  # Show first 5 issues
            logger.info(
                f"    - {issue['type']}: {issue['message']} ({issue.get('severity', 'unknown')})"
            )

    # Test with incomplete data
    logger.info("Testing with incomplete data")

    incomplete_df = create_incomplete_dataframe()

    incomplete_result = await validator.build_quality_report(incomplete_df)

    logger.info("Incomplete Data Validation Results:")
    logger.info(f"  Overall Score: {incomplete_result.overall_score:.2f}")
    logger.info(f"  Status: {incomplete_result.status}")
    logger.info(f"  Issues Found: {len(incomplete_result.issues)}")

    # Test specific validation types
    logger.info("Testing specific validation types")

    # Geometry validation
    geometry_check = validator.validate_geometries(valid_gdf)
    logger.info(
        f"Geometry Validation: {geometry_check.score:.2f} ({geometry_check.status})"
    )

    # Coordinate validation
    coord_check = validator.validate_coordinates(valid_gdf)
    logger.info(
        f"Coordinate Validation: {coord_check.score:.2f} ({coord_check.status})"
    )

    # Temporal validation
    temporal_check = validator.validate_temporal_data(valid_gdf)
    logger.info(
        f"Temporal Validation: {temporal_check.score:.2f} ({temporal_check.status})"
    )

    # Generate improvement recommendations
    logger.info("Generating improvement recommendations")

    if invalid_result.status != QualityStatus.PASS:
        recommendations = quality_manager.get_improvement_recommendations(
            quality_report
        )
        logger.info("Improvement Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")

    # Test quality trends
    logger.info("Analyzing quality trends")

    # Analyze the reports produced by the validation runs above.
    quality_manager.quality_history.extend(
        [quality_report, invalid_result, incomplete_result]
    )

    trends = quality_manager.get_quality_trends(days=30)

    logger.info("Quality Trends:")
    logger.info(f"  Average Score: {trends.get('average_score', 0):.2f}")
    logger.info(f"  Reports Count: {trends.get('reports_count', 0)}")
    logger.info(f"  Trend: {trends.get('score_trend', 'unknown')}")

    # Save validation results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save validation results as JSON
    validation_summary = {
        "valid_data_score": validation_result.score,
        "valid_data_status": validation_result.status.value,
        "valid_data_issues": len(validation_result.issues),
        "invalid_data_score": invalid_result.score,
        "invalid_data_status": invalid_result.status.value,
        "invalid_data_issues": len(invalid_result.issues),
        "incomplete_data_score": incomplete_result.score,
        "incomplete_data_status": incomplete_result.status.value,
        "incomplete_data_issues": len(incomplete_result.issues),
        "quality_trends": trends,
    }

    with open(output_dir / "validation_results.json", "w") as f:
        import json

        json.dump(validation_summary, f, indent=2, default=str)

    # Save sample datasets
    valid_gdf.to_file(output_dir / "valid_sample.geojson", driver="GeoJSON")
    invalid_gdf.to_file(output_dir / "invalid_sample.geojson", driver="GeoJSON")
    incomplete_df.to_csv(output_dir / "incomplete_sample.csv", index=False)

    logger.info(f"Validation results saved to {output_dir}")

    logger.info("Data validation example completed successfully")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
