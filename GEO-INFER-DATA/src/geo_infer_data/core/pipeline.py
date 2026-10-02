"""
Intelligent ETL pipeline management for GEO-INFER-DATA.

This module provides comprehensive ETL pipeline management with automatic
dependency resolution, error recovery, performance optimization, and
monitoring capabilities.
"""

import logging
import asyncio
from typing import Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from enum import StrEnum
import json
import yaml
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, box
from sqlalchemy import create_engine

from ..models.schemas import (
    ETLPipeline,
    ExecutionStatus,
    DataSource,
    DataDestination,
    Transformation,
    ExecutionState,
)
from ..utils.validation import GeospatialValidator
from ..utils.performance import PerformanceMonitor
from ..utils.timestamps import normalize_observation_timestamps


logger = logging.getLogger(__name__)


class PipelineStatus(StrEnum):
    """Pipeline execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorRecoveryStrategy(StrEnum):
    """Error recovery strategies."""

    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    SKIP = "skip"
    ROLLBACK = "rollback"
    INTELLIGENT_RETRY = "intelligent_retry"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""

    execution_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration: float | None = None
    records_processed: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    throughput: float = 0.0
    memory_usage: dict[str, float] = field(default_factory=dict)
    cpu_usage: dict[str, float] = field(default_factory=dict)


class TransformationEngine:
    """Engine for executing data transformations."""

    def __init__(self) -> None:
        self.transformations = {
            "filter": self._filter_data,
            "transform": self._transform_data,
            "aggregate": self._aggregate_data,
            "validate": self._validate_data,
            "clean": self._clean_data,
            "spatial_join": self._spatial_join,
            "temporal_aggregate": self._temporal_aggregate,
            "geocode": self._geocode_data,
            "reproject": self._reproject_data,
            "clip": self._clip_data,
        }

    async def execute_transformation(
        self, transformation: Transformation, data: Any, context: dict[str, Any]
    ) -> Any:
        """Execute a single transformation."""
        transform_type = transformation.type
        parameters = transformation.parameters

        if transform_type not in self.transformations:
            raise ValueError(f"Unknown transformation type: {transform_type}")

        transform_func = self.transformations[transform_type]

        try:
            result = await transform_func(data, parameters, context)
            logger.debug(f"Executed transformation {transform_type}")
            return result
        except Exception as e:
            logger.error(f"Transformation {transform_type} failed: {e}")
            raise

    async def _filter_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Apply filtering transformation."""
        filter_conditions = parameters.get("conditions", {})

        if isinstance(data, pd.DataFrame):
            # Filter pandas DataFrame
            filtered_data = data.copy()
            for column, condition in filter_conditions.items():
                if "min" in condition:
                    filtered_data = filtered_data[
                        filtered_data[column] >= condition["min"]
                    ]
                if "max" in condition:
                    filtered_data = filtered_data[
                        filtered_data[column] <= condition["max"]
                    ]
                if "values" in condition:
                    filtered_data = filtered_data[
                        filtered_data[column].isin(condition["values"])
                    ]
            return filtered_data
        else:
            # Handle other data types
            return data

    async def _transform_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Apply data transformation."""
        transformations = parameters.get("transformations", {})

        if isinstance(data, pd.DataFrame):
            transformed_data = data.copy()
            for column, transform in transformations.items():
                transform_type = transform.get("type")

                if transform_type == "scale":
                    factor = transform.get("factor", 1.0)
                    transformed_data[column] = transformed_data[column] * factor
                elif transform_type == "normalize":
                    min_val = transformed_data[column].min()
                    max_val = transformed_data[column].max()
                    transformed_data[column] = (transformed_data[column] - min_val) / (
                        max_val - min_val
                    )
                elif transform_type == "log":
                    transformed_data[column] = np.log(transformed_data[column] + 1)

            return transformed_data
        else:
            return data

    async def _aggregate_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Apply data aggregation."""
        group_by = parameters.get("group_by", [])
        aggregations = parameters.get("aggregations", {})

        if isinstance(data, pd.DataFrame):
            aggregated_data = data.groupby(group_by).agg(aggregations).reset_index()
            return aggregated_data
        else:
            return data

    async def _validate_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Validate data."""
        validator = GeospatialValidator()
        validation_result = await validator.validate_data(data)

        if validation_result.status == "fail":
            raise ValueError(f"Data validation failed: {validation_result.issues}")

        return data

    async def _clean_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Clean tabular data using explicit, configurable operations."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("clean transformation requires a pandas DataFrame")

        cleaned = data.copy()
        if parameters.get("remove_duplicates", True):
            cleaned = cleaned.drop_duplicates()

        if parameters.get("remove_nulls", False):
            cleaned = cleaned.dropna()
        else:
            fill_method = parameters.get("fill_method")
            if fill_method == "interpolate":
                numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
                cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(
                    limit_direction="both"
                )
            elif fill_method in {"forward", "ffill"}:
                cleaned = cleaned.ffill()
            elif fill_method in {"backward", "bfill"}:
                cleaned = cleaned.bfill()
            elif "fill_value" in parameters:
                cleaned = cleaned.fillna(parameters["fill_value"])

        if parameters.get("outlier_method") == "iqr":
            numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                values = cleaned[column]
                q1, q3 = values.quantile([0.25, 0.75])
                spread = q3 - q1
                lower = q1 - 1.5 * spread
                upper = q3 + 1.5 * spread
                cleaned = cleaned[cleaned[column].between(lower, upper)]

        return cleaned.reset_index(drop=True)

    async def _spatial_join(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Join a GeoDataFrame with a configured or contextual spatial layer."""

        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError("spatial_join transformation requires a GeoDataFrame")

        right = context.get("layers", {}).get(parameters.get("target_layer"))
        if right is None:
            target_path = parameters.get("target_path")
            if target_path:
                right = gpd.read_file(target_path)
        if not isinstance(right, gpd.GeoDataFrame):
            raise ValueError(
                "spatial_join requires a target GeoDataFrame or target_path"
            )

        join_kwargs = {
            "how": parameters.get("how", "left"),
            "predicate": parameters.get("predicate", "intersects"),
        }
        if parameters.get("join_type") == "nearest":
            join_kwargs.pop("predicate")
            join_kwargs["max_distance"] = parameters.get("max_distance")
            if join_kwargs["max_distance"] is None:
                join_kwargs.pop("max_distance")
            return gpd.sjoin_nearest(data, right, **join_kwargs)
        return gpd.sjoin(data, right, **join_kwargs)

    async def _temporal_aggregate(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Aggregate records over a configured pandas time frequency."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("temporal_aggregate transformation requires a DataFrame")

        time_column = parameters.get("time_column", "timestamp")
        if time_column not in data.columns:
            raise ValueError(f"Temporal column not found: {time_column}")
        frequency = parameters.get("frequency", "1D")
        aggregation = parameters.get(
            "aggregation", parameters.get("aggregations", "mean")
        )
        frame = data.copy()
        frame[time_column] = normalize_observation_timestamps(frame[time_column])
        group_by = parameters.get("group_by", [])
        grouped = frame.set_index(time_column)
        if group_by:
            return (
                grouped.groupby(group_by)
                .resample(frequency)
                .agg(aggregation)
                .reset_index()
            )
        return grouped.resample(frequency).agg(aggregation).reset_index()

    async def _geocode_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Create point geometries from configured latitude/longitude columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("geocode transformation requires a DataFrame")
        latitude_column = parameters.get("latitude_column", "latitude")
        longitude_column = parameters.get("longitude_column", "longitude")
        missing = [
            column
            for column in (latitude_column, longitude_column)
            if column not in data
        ]
        if missing:
            raise ValueError(f"Geocoding columns not found: {missing}")
        result = data.copy()
        result["geometry"] = [
            Point(float(lon), float(lat))
            for lat, lon in zip(result[latitude_column], result[longitude_column])
        ]
        return gpd.GeoDataFrame(
            result,
            geometry="geometry",
            crs=parameters.get("crs", "EPSG:4326"),
        )

    async def _reproject_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Reproject a GeoDataFrame to the requested CRS."""
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError("reproject transformation requires a GeoDataFrame")
        target_crs = parameters.get("target_crs") or parameters.get("crs")
        if not target_crs:
            raise ValueError("reproject requires target_crs")
        return data.to_crs(target_crs)

    async def _clip_data(
        self, data: Any, parameters: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Clip a GeoDataFrame to a configured geometry or bounding box."""
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError("clip transformation requires a GeoDataFrame")

        mask = context.get("layers", {}).get(parameters.get("mask_layer"))
        if mask is None and parameters.get("mask_path"):
            mask = gpd.read_file(parameters["mask_path"])
        if isinstance(mask, gpd.GeoDataFrame):
            return gpd.clip(data, mask)
        bounds = parameters.get("bounds")
        if bounds is None:
            raise ValueError("clip requires mask_layer, mask_path, or bounds")
        if len(bounds) != 4:
            raise ValueError("clip bounds must be [minx, miny, maxx, maxy]")

        return gpd.clip(data, gpd.GeoDataFrame(geometry=[box(*bounds)], crs=data.crs))


class IntelligentETLPipeline:
    """
    Intelligent ETL pipeline with automatic dependency resolution and error recovery.

    This class provides comprehensive ETL pipeline management with intelligent
    features including automatic dependency resolution, adaptive error recovery,
    performance optimization, and real-time monitoring. It supports complex
    geospatial data transformations with automatic optimization and parallel processing.

    The pipeline supports multiple transformation types:
    - filter: Data filtering based on conditions
    - transform: Data transformation and feature engineering
    - aggregate: Data aggregation and grouping operations
    - validate: Data validation and quality checks
    - clean: Data cleaning and outlier removal
    - spatial_join: Spatial joining of geospatial datasets
    - temporal_aggregate: Temporal aggregation for time-series data
    - geocode: Geocoding operations
    - reproject: Coordinate system reprojection
    - clip: Spatial clipping operations

    Features:
    - Automatic dependency resolution between transformation steps
    - Multiple error recovery strategies (fail_fast, retry, skip, intelligent_retry)
    - Performance monitoring and bottleneck identification
    - Parallel execution of independent transformations
    - Real-time progress tracking and logging
    - Comprehensive error handling and recovery
    - Integration with various data sources and destinations

    Attributes:
        workflow_config: Pipeline configuration (file path or dictionary)
        dependency_resolution: Strategy for resolving transformation dependencies
        error_recovery: Error recovery strategy ('fail_fast', 'retry', 'skip', 'intelligent_retry')
        monitoring_enabled: Whether performance monitoring is enabled
        parallel_execution: Whether to execute transformations in parallel
        pipeline: Configured ETLPipeline object
        execution_history: List of past execution results
        current_execution: Currently running execution status
        transformation_engine: Engine for executing transformations
        performance_monitor: Performance monitoring instance
        max_retries: Maximum retry attempts for retry-capable recovery strategies
        retry_delay: Base delay in seconds for exponential retry backoff

    Methods:
        execute_workflow(): Execute the complete ETL workflow
        _extract_data(): Extract data from source systems
        _transform_data(): Apply transformation pipeline
        _load_data(): Load data to target systems
        _recovery_delay(): Backoff delay before a retry, adapted to error type
        get_performance_metrics(): Get pipeline performance metrics
        identify_bottlenecks(): Identify performance bottlenecks
        _calculate_transformation_progress(): Calculate transformation progress

    Args:
        workflow_config: Path to workflow configuration file (YAML/JSON) or configuration
            dictionary. If None, pipeline must be set programmatically.
        dependency_resolution: Dependency resolution strategy ('automatic', 'manual', 'topological').
            Automatic resolution analyzes transformation dependencies and creates optimal execution order.
        error_recovery: Error recovery strategy for handling failures:
            - 'fail_fast': Stop immediately on first error
            - 'retry': Retry the failed workflow up to max_retries times with
              exponential backoff; return the successful retry's result
            - 'skip': Skip failed operations and continue
            - 'rollback': NOT SUPPORTED - constructing a pipeline with this
              strategy raises ValueError naming the supported strategies
            - 'intelligent_retry': Adaptive retry with error-type-specific
              backoff, capped at max_retries attempts
        max_retries: Maximum number of retry attempts after the initial one
            (default 3). Exhaustion raises with the accumulated error.
        retry_delay: Base delay in seconds for exponential backoff between
            retries (default 1.0).

    Raises:
        ValueError: If error_recovery='rollback' (unsupported) is requested
        ValueError: If error_recovery is not a known strategy or max_retries/retry_delay is negative

    Examples:
        >>> # Initialize with configuration file
        >>> pipeline = IntelligentETLPipeline(
        ...     workflow_config='config/etl_pipeline.yaml',
        ...     dependency_resolution='automatic',
        ...     error_recovery='intelligent_retry',
        ...     monitoring_enabled=True,
        ...     parallel_execution=True
        ... )
        >>>
        >>> # Execute pipeline with raw data
        >>> result = await pipeline.execute_workflow(
        ...     source_data=raw_environmental_data,
        ...     target_storage=processed_data_storage,
        ...     transformation_rules={
        ...         'spatial_reprojection': 'EPSG:4326',
        ...         'temporal_aggregation': '1H',
        ...         'quality_filtering': True
        ...     }
        ... )
        >>>
        >>> # Monitor execution
        >>> print(f"Execution time: {result['execution_time']}")
        >>> print(f"Records processed: {result['transformed_records']}")
        >>>
        >>> # Get performance metrics
        >>> metrics = pipeline.get_performance_metrics()
        >>> print(f"Throughput: {metrics.get('throughput', 0):.2f} records/second")
        >>>
        >>> # Identify bottlenecks
        >>> bottlenecks = pipeline.identify_bottlenecks(metrics)
        >>> for bottleneck in bottlenecks:
        ...     print(f"Bottleneck: {bottleneck['type']} - {bottleneck['severity']}")
    """

    def __init__(
        self,
        workflow_config: str | dict[str, Any] | None = None,
        dependency_resolution: str = "automatic",
        error_recovery: str = "intelligent_retry",
        monitoring_enabled: bool = True,
        parallel_execution: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if max_retries < 0:
            raise ValueError(
                f"GEO-INFER-DATA pipeline: max_retries must be >= 0, got {max_retries}"
            )
        if retry_delay < 0:
            raise ValueError(
                f"GEO-INFER-DATA pipeline: retry_delay must be >= 0, got {retry_delay}"
            )
        try:
            recovery_strategy = ErrorRecoveryStrategy(error_recovery)
        except ValueError:
            raise ValueError(
                "GEO-INFER-DATA pipeline: unknown error_recovery strategy "
                f"{error_recovery!r}; expected one of "
                f"{[s.value for s in ErrorRecoveryStrategy]}"
            ) from None
        if recovery_strategy is ErrorRecoveryStrategy.ROLLBACK:
            raise ValueError(
                "GEO-INFER-DATA pipeline: error_recovery='rollback' is not "
                "supported; no rollback of committed pipeline state is "
                "available. Use 'fail_fast', 'retry', 'skip', or "
                "'intelligent_retry' instead."
            )

        self.workflow_config = workflow_config
        self.dependency_resolution = dependency_resolution
        self.error_recovery = error_recovery
        self.error_recovery_strategy = recovery_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.monitoring_enabled = monitoring_enabled
        self.parallel_execution = parallel_execution

        self.pipeline: ETLPipeline | None = None
        self.execution_history: list[ExecutionStatus] = []
        self.current_execution: ExecutionStatus | None = None
        self.transformation_engine = TransformationEngine()
        self.performance_monitor = PerformanceMonitor() if monitoring_enabled else None

        self._load_configuration()

        logger.info(
            f"Initialized IntelligentETLPipeline with {error_recovery} error recovery"
        )

    def _load_configuration(self) -> None:
        """Load pipeline configuration from file or dictionary."""
        if isinstance(self.workflow_config, str):
            config_path = Path(self.workflow_config)
            if config_path.exists():
                with open(config_path) as f:
                    if config_path.suffix in [".yaml", ".yml"]:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                self.workflow_config = config

        if isinstance(self.workflow_config, dict):
            self.pipeline = ETLPipeline(**self.workflow_config)

    async def execute_workflow(
        self,
        source_data: Any,
        target_storage: Any,
        transformation_rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete ETL workflow with monitoring and error recovery.

        This method orchestrates the entire ETL process including data extraction,
        transformation, loading, and validation. It provides comprehensive monitoring,
        error handling, and performance optimization throughout the execution.

        The workflow follows these phases:
        1. **Extraction**: Extract data from configured sources
        2. **Transformation**: Apply configured transformation pipeline
        3. **Loading**: Load transformed data to target storage
        4. **Validation**: Perform quality checks on final results
        5. **Monitoring**: Track performance and generate metrics

        Each phase includes:
        - Progress tracking and logging
        - Error handling and recovery based on configured strategy
        - Performance monitoring and optimization
        - Data validation and quality checks
        - Metadata tracking and lineage preservation

        Args:
            source_data: Raw data from source systems. Can be:
                - File path (string) for file-based sources
                - Database connection parameters for database sources
                - API configuration for API-based sources
                - Pre-loaded data object (DataFrame, GeoDataFrame, etc.)
                - Streaming data source configuration
            target_storage: Target storage configuration specifying where to store
                processed data. Should include backend type, connection parameters,
                and storage options.
            transformation_rules: Optional transformation rules to override or supplement
                the configured pipeline. Useful for dynamic transformations or testing.
                Should be a dictionary with transformation parameters.

        Returns:
            Comprehensive execution results including:
            {
                'execution_id': str,  # Unique execution identifier
                'status': str,  # Execution status ('completed', 'failed', 'cancelled')
                'extracted_records': int,  # Number of records extracted
                'transformed_records': int,  # Number of records after transformation
                'load_result': dict,  # Loading operation results
                'execution_time': timedelta,  # Total execution time
                'performance_metrics': dict,  # Detailed performance metrics
                'errors': list,  # Any errors encountered
                'warnings': list,  # Any warnings generated
                'quality_score': float,  # Final data quality score
                'metadata': dict  # Execution metadata and lineage
            }

        Raises:
            RuntimeError: If pipeline execution fails and cannot be recovered
                (retries exhausted under 'retry'/'intelligent_retry')
            ConfigurationError: If workflow configuration is invalid
            DataExtractionError: If data extraction fails
            TransformationError: If transformation pipeline fails
            DataLoadingError: If data loading fails
            ValidationError: If final validation fails and strict mode is enabled
            TimeoutError: If execution exceeds configured timeout

        Examples:
            >>> # Execute with file-based data
            >>> result = await pipeline.execute_workflow(
            ...     source_data='data/raw/sensors_2023.csv',
            ...     target_storage={
            ...         'backend': 'postgresql',
            ...         'table_name': 'processed_sensors',
            ...         'if_exists': 'replace'
            ...     },
            ...     transformation_rules={
            ...         'clean_missing': True,
            ...         'outlier_removal': 'iqr',
            ...         'temporal_aggregation': '1H'
            ...     }
            ... )
            >>>
            >>> # Monitor execution progress
            >>> print(f"Status: {result['status']}")
            >>> print(f"Records processed: {result['transformed_records']}")
            >>> print(f"Execution time: {result['execution_time']}")
            >>>
            >>> # Check for issues
            >>> if result['errors']:
            ...     print(f"Errors encountered: {len(result['errors'])}")
            >>> if result['warnings']:
            ...     print(f"Warnings: {result['warnings']}")
        """
        logger.info("Starting ETL workflow execution")

        max_attempts = (
            self.max_retries + 1
            if self.error_recovery_strategy
            in (ErrorRecoveryStrategy.RETRY, ErrorRecoveryStrategy.INTELLIGENT_RETRY)
            else 1
        )
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            if attempt > 0:
                delay = self._recovery_delay(last_error, attempt - 1)
                logger.warning(
                    "ETL workflow retry %d/%d after error: %s",
                    attempt,
                    self.max_retries,
                    last_error,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

            # Create execution status for this attempt
            execution_id = f"exec_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            self.current_execution = ExecutionStatus(
                id=execution_id,
                pipeline_id=self.pipeline.id if self.pipeline else "unknown",
                status=ExecutionState.RUNNING,
                started_at=datetime.now(UTC),
            )

            try:
                # Extract phase
                extracted_data = await self._extract_data(source_data)

                # Transform phase
                transformed_data = await self._transform_data(
                    extracted_data, transformation_rules
                )

                # Load phase
                load_result = await self._load_data(transformed_data, target_storage)

                # Update execution status
                self.current_execution.status = ExecutionState.COMPLETED
                self.current_execution.completed_at = datetime.now(UTC)
                self.current_execution.progress = 100.0

                # Record execution
                self.execution_history.append(self.current_execution)

                execution_result = {
                    "execution_id": execution_id,
                    "status": "completed",
                    "extracted_records": (
                        len(extracted_data) if hasattr(extracted_data, "__len__") else 1
                    ),
                    "transformed_records": (
                        len(transformed_data)
                        if hasattr(transformed_data, "__len__")
                        else 1
                    ),
                    "load_result": load_result,
                    "execution_time": (
                        (
                            self.current_execution.completed_at
                            - self.current_execution.started_at
                        )
                        if (
                            self.current_execution.completed_at
                            and self.current_execution.started_at
                        )
                        else timedelta()
                    ),
                    "performance_metrics": self._get_performance_metrics(),
                }

                logger.info(
                    f"ETL workflow completed successfully in {execution_result['execution_time']}"
                )
                return execution_result

            except Exception as e:
                logger.error(
                    f"ETL workflow attempt {attempt + 1}/{max_attempts} failed: {e}"
                )

                # Update execution status and record the failed attempt
                self.current_execution.status = ExecutionState.FAILED
                self.current_execution.completed_at = datetime.now(UTC)
                self.current_execution.message = str(e)
                self.execution_history.append(self.current_execution)
                last_error = e

        raise RuntimeError(
            "GEO-INFER-DATA pipeline: ETL workflow failed after "
            f"{max_attempts} attempt(s) (error_recovery={self.error_recovery!r}, "
            f"max_retries={self.max_retries}): {last_error}"
        ) from last_error

    async def _extract_data(self, source_data: Any) -> Any:
        """Extract data from source systems."""
        logger.debug("Starting data extraction")

        if self.pipeline and self.pipeline.source:
            # Use configured source
            source_config = self.pipeline.source
            extracted_data = await self._extract_from_configured_source(
                source_config, source_data
            )
        else:
            # Use provided data directly
            extracted_data = source_data

        logger.debug(
            f"Extracted {len(extracted_data) if hasattr(extracted_data, '__len__') else 1} records"
        )
        return extracted_data

    async def _extract_from_configured_source(
        self, source_config: DataSource, source_data: Any
    ) -> Any:
        """Extract data from configured source."""
        source_type = source_config.type

        if source_type == "file":
            # Extract from files
            return await self._extract_from_files(
                source_config.configuration, source_data
            )
        elif source_type == "database":
            # Extract from databases
            return await self._extract_from_database(
                source_config.configuration, source_data
            )
        elif source_type == "api":
            # Extract from APIs
            return await self._extract_from_api(
                source_config.configuration, source_data
            )
        else:
            return source_data

    async def _extract_from_files(
        self, config: dict[str, Any], source_data: Any
    ) -> Any:
        """Read a configured file, using the supplied source data as an override."""
        path = (
            source_data if isinstance(source_data, (str, Path)) else config.get("path")
        )
        if path is None:
            return source_data
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        fmt = str(config.get("format", path.suffix.lstrip("."))).lower()
        if fmt in {"csv", "tsv"}:
            return pd.read_csv(path, sep="\t" if fmt == "tsv" else ",")
        if fmt in {"json", "jsonl"}:
            return pd.read_json(path, lines=fmt == "jsonl")
        if fmt in {"parquet", "pq"}:
            return pd.read_parquet(path)
        if fmt in {"geojson", "gpkg", "shp", "geopackage"}:
            return gpd.read_file(path)
        raise ValueError(f"Unsupported file format: {fmt}")

    async def _extract_from_database(
        self, config: dict[str, Any], source_data: Any
    ) -> Any:
        """Read a configured database query with SQLAlchemy."""
        if source_data is not None and not isinstance(source_data, (str, Path, dict)):
            return source_data
        connection = config.get("connection_string") or config.get("url")
        query = config.get("query")
        table = config.get("table")
        if not connection or not (query or table):
            raise ValueError(
                "database source requires connection_string and query or table"
            )
        engine = create_engine(connection)
        try:
            return (
                pd.read_sql_query(query, engine)
                if query
                else pd.read_sql_table(table, engine)
            )
        finally:
            engine.dispose()

    async def _extract_from_api(self, config: dict[str, Any], source_data: Any) -> Any:
        """Fetch JSON data from a configured HTTP API."""
        if source_data is not None and not isinstance(source_data, (str, Path, dict)):
            return source_data
        endpoint = config.get("url") or config.get("endpoint")
        if not endpoint:
            raise ValueError("api source requires url or endpoint")
        headers = dict(config.get("headers", {}))
        token = config.get("api_key") or config.get("token")
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
        response = requests.get(
            endpoint,
            params=config.get("params"),
            headers=headers,
            timeout=float(config.get("timeout", 30)),
        )
        response.raise_for_status()
        payload = response.json()
        return (
            pd.json_normalize(payload) if isinstance(payload, (list, dict)) else payload
        )

    async def _transform_data(
        self, extracted_data: Any, transformation_rules: dict[str, Any] | None = None
    ) -> Any:
        """Transform extracted data."""
        logger.debug("Starting data transformation")

        if not self.pipeline or not self.pipeline.transformations:
            logger.debug("No transformations configured, returning extracted data")
            return extracted_data

        transformed_data = extracted_data
        execution_context = {
            "pipeline_id": self.pipeline.id,
            "execution_id": (
                self.current_execution.id if self.current_execution else None
            ),
            "start_time": datetime.now(UTC),
        }

        # Execute transformations in order
        for transformation in self.pipeline.transformations:
            if not transformation.enabled:
                continue

            try:
                logger.debug(f"Executing transformation: {transformation.type}")
                transformed_data = (
                    await self.transformation_engine.execute_transformation(
                        transformation, transformed_data, execution_context
                    )
                )

                # Update progress
                if self.current_execution:
                    progress = self._calculate_transformation_progress(transformation)
                    self.current_execution.progress = progress

            except Exception as e:
                logger.error(f"Transformation {transformation.type} failed: {e}")

                if self.error_recovery == ErrorRecoveryStrategy.FAIL_FAST:
                    raise
                elif self.error_recovery == ErrorRecoveryStrategy.SKIP:
                    logger.warning(f"Skipping transformation {transformation.type}")
                    continue
                else:
                    # Retry / intelligent_retry: log the failure, then let the
                    # error bubble to execute_workflow, which owns the single
                    # capped retry budget with backoff.
                    await self._handle_transformation_error(
                        e, transformation, transformed_data
                    )
                    raise

        logger.debug("Data transformation completed")
        return transformed_data

    def _calculate_transformation_progress(
        self, transformation: Transformation
    ) -> float:
        """Calculate transformation progress."""
        if not self.pipeline:
            return 0.0

        total_transformations = len(
            [t for t in self.pipeline.transformations if t.enabled]
        )
        current_index = next(
            (
                i
                for i, t in enumerate(self.pipeline.transformations)
                if t.order == transformation.order
            ),
            0,
        )

        if total_transformations == 0:
            return 100.0

        progress = (current_index + 1) / total_transformations * 100.0
        return min(progress, 100.0)

    async def _load_data(self, transformed_data: Any, target_storage: Any) -> Any:
        """Load transformed data to target storage."""
        logger.debug("Starting data loading")

        if self.pipeline and self.pipeline.destination:
            # Use configured destination
            destination_config = self.pipeline.destination
            load_result = await self._load_to_configured_destination(
                destination_config, transformed_data, target_storage
            )
        else:
            # Use provided storage directly
            load_result = await self._load_to_storage(transformed_data, target_storage)

        logger.debug("Data loading completed")
        return load_result

    async def _load_to_configured_destination(
        self,
        destination_config: DataDestination,
        transformed_data: Any,
        target_storage: Any,
    ) -> Any:
        """Load data to a configured file, database, API, or in-memory destination."""
        destination_type = destination_config.type.lower()
        configuration = destination_config.configuration
        if destination_type in {"dataset", "memory"}:
            return await self._load_to_storage(transformed_data, target_storage)
        if destination_type == "file":
            path = configuration.get("path")
            if not path:
                raise ValueError("file destination requires path")
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fmt = str(configuration.get("format", path.suffix.lstrip("."))).lower()
            if fmt == "csv":
                transformed_data.to_csv(path, index=False)
            elif fmt in {"json", "geojson"}:
                if hasattr(transformed_data, "to_file") and fmt == "geojson":
                    transformed_data.to_file(path, driver="GeoJSON")
                else:
                    transformed_data.to_json(path, orient="records", date_format="iso")
            elif fmt in {"parquet", "pq"}:
                transformed_data.to_parquet(path, index=False)
            else:
                raise ValueError(f"Unsupported file destination format: {fmt}")
            return {"records_loaded": len(transformed_data), "destination": str(path)}
        if destination_type == "database":
            connection = configuration.get("connection_string") or configuration.get(
                "url"
            )
            table = configuration.get("table") or configuration.get("table_name")
            if not connection or not table:
                raise ValueError(
                    "database destination requires connection_string and table"
                )
            engine = create_engine(connection)
            try:
                transformed_data.to_sql(
                    table,
                    engine,
                    if_exists=configuration.get("if_exists", "replace"),
                    index=False,
                )
            finally:
                engine.dispose()
            return {"records_loaded": len(transformed_data), "destination": table}
        if destination_type == "api":
            endpoint = configuration.get("url") or configuration.get("endpoint")
            if not endpoint:
                raise ValueError("api destination requires url or endpoint")
            response = requests.post(
                endpoint,
                json=(
                    transformed_data.to_dict(orient="records")
                    if isinstance(transformed_data, pd.DataFrame)
                    else transformed_data
                ),
                headers=configuration.get("headers"),
                timeout=float(configuration.get("timeout", 30)),
            )
            response.raise_for_status()
            return {"records_loaded": len(transformed_data), "destination": endpoint}
        raise ValueError(f"Unsupported destination type: {destination_config.type}")

    async def _load_to_storage(self, transformed_data: Any, target_storage: Any) -> Any:
        """Store data in an injected storage object or report an explicit memory load."""
        records = len(transformed_data) if hasattr(transformed_data, "__len__") else 1
        if target_storage is None:
            return {"records_loaded": records, "destination": "memory"}
        if hasattr(target_storage, "store"):
            result = target_storage.store(transformed_data)
            if asyncio.iscoroutine(result):
                result = await result
            return {
                "records_loaded": records,
                "destination": "storage",
                "storage_result": result,
            }
        if isinstance(target_storage, dict):
            target_storage["data"] = transformed_data
            target_storage["records_loaded"] = records
            return {"records_loaded": records, "destination": "mapping"}
        raise TypeError("target_storage must be None, a mapping, or expose store()")

    async def _handle_transformation_error(
        self, error: Exception, transformation: Transformation, data: Any
    ) -> None:
        """Handle transformation-specific errors."""
        logger.error(
            f"Handling transformation error for {transformation.type}: {error}"
        )

        # Log error details
        if self.current_execution:
            log_entry = {
                "timestamp": datetime.now(UTC),
                "level": "error",
                "transformation": transformation.type,
                "message": str(error),
            }
            self.current_execution.logs.append(log_entry)

    def _recovery_delay(self, error: Exception | None, retry_index: int) -> float:
        """Return the backoff delay (seconds) before retry ``retry_index``.

        The delay grows exponentially with the retry index. For the
        INTELLIGENT_RETRY strategy the base delay is adapted to the error
        type: connection errors get a longer cooldown, validation errors are
        retried immediately because retrying the same data cannot repair it.
        """
        if self.error_recovery_strategy is ErrorRecoveryStrategy.INTELLIGENT_RETRY:
            error_type = type(error).__name__ if error is not None else ""
            if "ConnectionError" in error_type:
                base_delay = max(self.retry_delay, 2.0)
            elif "ValidationError" in error_type:
                base_delay = 0.0
            else:
                base_delay = self.retry_delay
        else:
            base_delay = self.retry_delay
        return base_delay * (2**retry_index)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get pipeline performance metrics."""
        if not self.performance_monitor:
            return {"monitoring_disabled": True}

        return self.performance_monitor.get_metrics()

    def _get_performance_metrics(self) -> dict[str, Any]:
        """Get current execution performance metrics."""
        if not self.current_execution or not self.performance_monitor:
            return {}

        execution_time = (
            (self.current_execution.completed_at - self.current_execution.started_at)
            if (
                self.current_execution.completed_at
                and self.current_execution.started_at
            )
            else timedelta()
        )

        return {
            "execution_time_seconds": execution_time.total_seconds(),
            "status": self.current_execution.status,
            "progress": self.current_execution.progress,
            "logs_count": len(self.current_execution.logs),
        }

    def identify_bottlenecks(self, metrics: dict[str, Any]) -> list[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if metrics.get("execution_time_seconds", 0) > 3600:  # 1 hour
            bottlenecks.append("Long execution time detected")

        if metrics.get("memory_usage", {}).get("peak", 0) > 0.9:  # 90% memory usage
            bottlenecks.append("High memory usage detected")

        # Add more bottleneck detection logic
        return bottlenecks
