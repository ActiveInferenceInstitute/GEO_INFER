#!/usr/bin/env python3
"""
ETL pipeline example for GEO-INFER-DATA.

This example demonstrates how to use the IntelligentETLPipeline class
for complex data transformation workflows with automatic optimization.

Usage:
    python etl_pipeline_example.py

Requirements:
    - GEO-INFER-DATA package installed
    - Required dependencies (geopandas, pandas, numpy, etc.)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from geo_infer_data.core.pipeline import IntelligentETLPipeline
from geo_infer_data.models.schemas import (
    DataSource,
    DataDestination,
    Transformation,
    ETLPipeline,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_raw_environmental_data():
    """Create raw environmental monitoring data."""
    n_records = 5000

    # Create raw sensor data with some issues
    data = {
        "sensor_id": np.random.choice(
            ["sensor_001", "sensor_002", "sensor_003", "sensor_004"], n_records
        ),
        "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="30min"),
        "raw_temperature": np.random.normal(20, 15, n_records),  # Some outliers
        "raw_humidity": np.random.normal(60, 25, n_records),  # Some outliers
        "raw_pressure": np.random.normal(1013, 50, n_records),  # Some outliers
        "latitude": np.random.normal(37.7749, 0.1, n_records),
        "longitude": np.random.normal(-122.4194, 0.1, n_records),
        "battery_level": np.random.normal(85, 10, n_records),
    }

    # Introduce some data quality issues
    # Missing values
    missing_indices = np.random.choice(n_records, size=n_records // 10, replace=False)
    for col in ["raw_temperature", "raw_humidity", "raw_pressure"]:
        data[col].iloc[missing_indices] = None

    # Invalid coordinates
    invalid_indices = np.random.choice(n_records, size=n_records // 20, replace=False)
    data["latitude"].iloc[invalid_indices] = np.random.choice([100, -100])
    data["longitude"].iloc[invalid_indices] = np.random.choice([200, -200])

    # Unrealistic values
    outlier_indices = np.random.choice(n_records, size=n_records // 15, replace=False)
    data["raw_temperature"].iloc[outlier_indices] = np.random.choice([100, -50])

    return pd.DataFrame(data)


async def main():
    """Main example function."""
    logger.info("Starting ETL pipeline example")

    # Create raw data
    logger.info("Creating raw environmental data")

    raw_data = create_raw_environmental_data()
    logger.info(f"Created raw data with {len(raw_data)} records")

    # Save raw data
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    raw_data.to_csv(output_dir / "raw_environmental_data.csv", index=False)

    # Initialize ETL pipeline
    logger.info("Initializing ETL pipeline")

    pipeline = IntelligentETLPipeline(
        workflow_config=None,  # Will create programmatically
        dependency_resolution="automatic",
        error_recovery="intelligent_retry",
        monitoring_enabled=True,
        parallel_execution=True,
    )

    logger.info("ETL pipeline initialized")

    # Define ETL pipeline configuration
    logger.info("Defining ETL pipeline configuration")

    # Data source
    source = DataSource(
        type="file",
        configuration={
            "file_path": str(output_dir / "raw_environmental_data.csv"),
            "format": "csv",
            "encoding": "utf-8",
        },
    )

    # Data destination
    destination = DataDestination(
        type="file",
        configuration={
            "output_path": str(output_dir / "processed_environmental_data.geojson"),
            "format": "geojson",
        },
    )

    # Transformation pipeline
    transformations = [
        Transformation(
            type="filter",
            parameters={
                "conditions": {
                    "raw_temperature": {"min": -50, "max": 100},
                    "raw_humidity": {"min": 0, "max": 100},
                    "raw_pressure": {"min": 900, "max": 1100},
                }
            },
            order=1,
            enabled=True,
        ),
        Transformation(
            type="clean",
            parameters={
                "remove_nulls": True,
                "fill_method": "interpolate",
                "outlier_method": "iqr",
            },
            order=2,
            enabled=True,
        ),
        Transformation(
            type="transform",
            parameters={
                "transformations": {
                    "temperature": {
                        "type": "scale",
                        "factor": 1.0,  # Convert to Celsius if needed
                    },
                    "humidity": {"type": "normalize", "min": 0, "max": 100},
                }
            },
            order=3,
            enabled=True,
        ),
        Transformation(
            type="validate",
            parameters={
                "rules": ["coordinate_validity", "range_checks", "completeness"]
            },
            order=4,
            enabled=True,
        ),
        Transformation(
            type="spatial_join",
            parameters={
                "join_type": "nearest",
                "max_distance": 0.01,
                "target_layer": "sf_neighborhoods",  # Would be actual neighborhood data
            },
            order=5,
            enabled=True,
        ),
        Transformation(
            type="temporal_aggregate",
            parameters={
                "time_column": "timestamp",
                "frequency": "1H",
                "aggregation": {
                    "temperature": "mean",
                    "humidity": "mean",
                    "pressure": "mean",
                },
            },
            order=6,
            enabled=True,
        ),
    ]

    # Create pipeline
    etl_pipeline = ETLPipeline(
        name="Environmental Data Processing Pipeline",
        description="ETL pipeline for processing environmental monitoring data",
        source=source,
        destination=destination,
        transformations=transformations,
        status="active",
    )

    pipeline.pipeline = etl_pipeline

    logger.info(f"Created ETL pipeline: {etl_pipeline.name}")
    logger.info(f"Pipeline has {len(transformations)} transformation steps")

    # Execute ETL workflow
    logger.info("Executing ETL workflow")

    try:
        execution_result = await pipeline.execute_workflow(
            source_data=raw_data,
            target_storage=destination,
            transformation_rules=None,  # Use pipeline configuration
        )

        logger.info("ETL workflow completed successfully")
        logger.info(f"Execution time: {execution_result['execution_time']}")
        logger.info(f"Records processed: {execution_result['transformed_records']}")

        # Get performance metrics
        performance_metrics = pipeline.get_performance_metrics()

        logger.info("Performance Metrics:")
        for operation, metrics in performance_metrics.items():
            logger.info(f"  {operation}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value}")

        # Identify bottlenecks
        bottlenecks = pipeline.identify_bottlenecks(performance_metrics)

        if bottlenecks:
            logger.info("Identified Bottlenecks:")
            for bottleneck in bottlenecks:
                logger.info(
                    f"  - {bottleneck['type']}: {bottleneck.get('operation', 'system')} ({bottleneck.get('severity', 'unknown')})"
                )
        else:
            logger.info("No significant bottlenecks identified")

        # Save processed data
        if "load_result" in execution_result:
            logger.info(f"Data saved to: {execution_result['load_result']}")

        # Save pipeline results
        pipeline_results = {
            "execution_result": execution_result,
            "performance_metrics": performance_metrics,
            "bottlenecks": bottlenecks,
            "pipeline_config": {
                "name": etl_pipeline.name,
                "transformations_count": len(transformations),
                "source_type": source.type,
                "destination_type": destination.type,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(output_dir / "etl_pipeline_results.json", "w") as f:
            import json

            json.dump(pipeline_results, f, indent=2, default=str)

        logger.info(
            f"Pipeline results saved to {output_dir / 'etl_pipeline_results.json'}"
        )

    except Exception as e:
        logger.error(f"ETL pipeline execution failed: {e}")

        # Handle error based on recovery strategy
        logger.info(f"Error recovery strategy: {pipeline.error_recovery}")

        if pipeline.error_recovery == "intelligent_retry":
            logger.info("Attempting intelligent error recovery...")
            # In a real scenario, would attempt recovery
            logger.info("Recovery attempt completed")

        raise

    # Demonstrate pipeline monitoring
    logger.info("Pipeline monitoring and optimization")

    # Get current execution metrics
    current_metrics = pipeline._get_performance_metrics()

    logger.info("Current Execution Metrics:")
    for metric, value in current_metrics.items():
        logger.info(f"  {metric}: {value}")

    # Optimize pipeline based on performance
    if "execution_time_seconds" in current_metrics:
        exec_time = current_metrics["execution_time_seconds"]
        if exec_time > 300:  # More than 5 minutes
            logger.info("Pipeline execution is slow, consider optimization:")
            logger.info("  - Enable parallel processing")
            logger.info("  - Optimize transformation order")
            logger.info("  - Use more efficient data formats")

    logger.info("ETL pipeline example completed successfully")


def demonstrate_pipeline_config():
    """Demonstrate pipeline configuration options."""
    logger.info("ETL Pipeline Configuration Examples:")

    # Example pipeline configurations
    configurations = [
        {
            "name": "Real-time Data Pipeline",
            "description": "High-frequency data processing with minimal latency",
            "config": {
                "parallel_execution": True,
                "error_recovery": "fail_fast",
                "monitoring_enabled": True,
                "transformations": [
                    {"type": "validate", "order": 1},
                    {"type": "clean", "order": 2},
                    {"type": "transform", "order": 3},
                ],
            },
        },
        {
            "name": "Batch Processing Pipeline",
            "description": "Large-scale data processing with comprehensive validation",
            "config": {
                "parallel_execution": True,
                "error_recovery": "intelligent_retry",
                "monitoring_enabled": True,
                "transformations": [
                    {"type": "filter", "order": 1},
                    {"type": "clean", "order": 2},
                    {"type": "validate", "order": 3},
                    {"type": "aggregate", "order": 4},
                    {"type": "spatial_join", "order": 5},
                ],
            },
        },
        {
            "name": "Quality Assurance Pipeline",
            "description": "Data validation and quality improvement",
            "config": {
                "parallel_execution": False,
                "error_recovery": "rollback",
                "monitoring_enabled": True,
                "transformations": [
                    {"type": "validate", "order": 1},
                    {"type": "clean", "order": 2},
                    {"type": "validate", "order": 3},  # Re-validate after cleaning
                    {"type": "transform", "order": 4},
                ],
            },
        },
    ]

    for config in configurations:
        logger.info(f"\n{config['name']}:")
        logger.info(f"  Description: {config['description']}")
        logger.info(f"  Transformations: {len(config['config']['transformations'])}")
        logger.info(f"  Error Recovery: {config['config']['error_recovery']}")
        logger.info(f"  Parallel Execution: {config['config']['parallel_execution']}")


if __name__ == "__main__":
    # Demonstrate configuration options
    demonstrate_pipeline_config()

    # Run the main example
    asyncio.run(main())
