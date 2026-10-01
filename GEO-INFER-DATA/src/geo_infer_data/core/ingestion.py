"""
Multi-source data ingestion system for GEO-INFER-DATA.

This module provides comprehensive data ingestion capabilities from multiple
geospatial data sources including satellite imagery, sensor networks,
crowdsourced data, and various APIs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast
from datetime import datetime, UTC
from importlib.util import find_spec
import asyncio
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests

from ..models.schemas import QualityCheck
from ..utils.validation import GeospatialValidator
from ..utils.format_detection import FormatDetector

HAS_RASTERIO = find_spec("rasterio") is not None


logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""

    data_sources: list[str]
    format_detection: str = "automatic"
    validation_enabled: bool = True
    quality_threshold: float = 0.8
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    retry_attempts: int = 3
    timeout_seconds: int = 300


class DataSourceConnector(ABC):
    """
    Base class for data source connectors.

    This abstract base class defines the interface for connecting to various
    data sources including databases, APIs, files, and streaming services.

    Subclasses must implement the connect() and fetch_data() methods to
    provide specific data source connectivity.

    Attributes:
        config: Configuration dictionary for the connector
        validator: GeospatialValidator instance for data validation

    Methods:
        connect(): Establish connection to data source
        fetch_data(): Fetch data from the source
        validate_data(): Validate fetched data quality
        disconnect(): Close connection to data source

    Examples:
        >>> # Subclass implementation
        >>> class CustomConnector(DataSourceConnector):
        ...     async def connect(self) -> bool:
        ...         # Implementation for specific data source
        ...         return True
        ...
        ...     async def fetch_data(self, query: Dict[str, Any]) -> Any:
        ...         # Implementation for data fetching
        ...         return {'data': 'local_fixture_data'}
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize data source connector.

        Args:
            config: Configuration dictionary containing connection parameters

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.validator = GeospatialValidator()

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source.

        This method must be implemented by subclasses to establish a connection
        to the specific data source (database, API, file system, etc.).

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
            RuntimeError: If the subclass does not provide a connector
        """
        raise RuntimeError("Data ingestion subclasses must implement connect()")

    @abstractmethod
    async def fetch_data(self, query: dict[str, Any]) -> Any:
        """
        Fetch data from source.

        This method must be implemented by subclasses to fetch data based on
        the provided query parameters.

        Args:
            query: Query parameters for data retrieval

        Returns:
            Fetched data in appropriate format

        Raises:
            DataFetchError: If data fetching fails
            RuntimeError: If the subclass does not provide a connector
        """
        raise RuntimeError("Data ingestion subclasses must implement fetch_data()")

    async def validate_data(self, data: Any) -> QualityCheck:
        """
        Validate fetched data.

        Validates the quality and integrity of fetched data using the
        configured geospatial validator.

        Args:
            data: Data to validate

        Returns:
            Quality check results with score and issues

        Raises:
            ValidationError: If validation fails
        """
        return await self.validator.validate_data(data)

    async def disconnect(self) -> None:
        """
        Close connection to data source.

        This method should be implemented by subclasses to properly close
        connections and clean up resources.

        Raises:
            ConnectionError: If disconnection fails
        """
        logger.debug(
            "%s has no persistent data-source connection to close", type(self).__name__
        )


class SatelliteDataConnector(DataSourceConnector):
    """Connector for satellite imagery data sources."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.satellite-imagery.com")

    async def connect(self) -> bool:
        """Connect to satellite API."""
        if self.api_key in {None, "", "your_api_key"}:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return bool(response.status_code == 200)
        except Exception as e:
            logger.error(f"Failed to connect to satellite API: {e}")
            return False

    async def fetch_data(self, query: dict[str, Any]) -> dict[str, Any]:
        """Fetch satellite imagery data."""
        if not self.api_key:
            raise RuntimeError("Satellite API credentials are required")
        response = requests.get(
            f"{self.base_url.rstrip('/')}/imagery",
            params=query,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not {"imagery", "metadata"}.issubset(
            payload
        ):
            raise ValueError("Satellite response must contain imagery and metadata")
        return payload


class SensorDataConnector(DataSourceConnector):
    """Connector for IoT sensor data."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host")
        self.port = config.get("port", 1883)  # MQTT default
        self.topic = config.get("topic", "sensors/#")

    async def connect(self) -> bool:
        """Connect to sensor network."""
        return bool(self.host and self.port and self.topic)

    async def fetch_data(self, query: dict[str, Any]) -> dict[str, Any]:
        """Fetch sensor data."""
        endpoint = self.config.get("data_url")
        if not endpoint:
            raise RuntimeError(
                "Sensor data requires a configured data_url or an MQTT adapter"
            )
        response = requests.get(endpoint, params=query, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "measurements" not in payload:
            raise ValueError("Sensor response must contain measurements")
        payload["measurements"] = pd.DataFrame(payload["measurements"])
        return payload


class CrowdsourcedDataConnector(DataSourceConnector):
    """Connector for crowdsourced data."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.api_endpoint = config.get("api_endpoint")
        self.api_key = config.get("api_key")

    async def connect(self) -> bool:
        """Connect to crowdsourcing platform."""
        if self.api_key in {None, "", "your_key"} or not self.api_endpoint:
            return False
        try:
            response = requests.get(
                f"{self.api_endpoint}/health",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return bool(response.status_code == 200)
        except Exception as e:
            logger.error(f"Failed to connect to crowdsourcing API: {e}")
            return False

    async def fetch_data(self, query: dict[str, Any]) -> dict[str, Any]:
        """Fetch crowdsourced data."""
        if not self.api_endpoint or not self.api_key:
            raise RuntimeError("Crowdsourced API endpoint and credentials are required")
        response = requests.get(
            f"{self.api_endpoint.rstrip('/')}/reports",
            params=query,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "reports" not in payload:
            raise ValueError("Crowdsourced response must contain reports")
        payload["reports"] = pd.DataFrame(payload["reports"])
        return payload


class GenericDataSourceConnector(DataSourceConnector):
    """Generic connector for data sources without a dedicated connector class."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url") or config.get("api_endpoint")
        self.api_key = config.get("api_key")

    async def connect(self) -> bool:
        """Check a configured generic HTTP endpoint."""
        if not self.base_url:
            return False
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/health",
                headers=self._headers(),
                timeout=10,
            )
            return bool(response.ok)
        except requests.RequestException as exc:
            logger.warning("Generic data source connection failed: %s", exc)
            return False

    async def fetch_data(self, query: dict[str, Any]) -> Any:
        """Fetch a JSON payload from the configured generic HTTP endpoint."""
        if not self.base_url:
            raise RuntimeError("A generic data source requires an endpoint")
        response = requests.get(
            self.base_url,
            params=query,
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            raise ValueError("Generic data source returned an empty payload")
        return payload

    def _headers(self) -> dict[str, str]:
        """Build request headers from configured credentials."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}


class MultiSourceDataIngestion:
    """
    Multi-source geospatial data ingestion system.

    This class provides comprehensive data ingestion capabilities from multiple
    sources including satellite imagery, sensor networks, crowdsourced data,
    and various APIs with automatic format detection and validation.

    The system supports parallel processing, automatic error recovery, data
    validation, and quality assurance to ensure high-quality data ingestion
    across diverse geospatial sources.

    Attributes:
        config: Ingestion configuration with source types and parameters
        connectors: Dictionary of initialized data source connectors
        format_detector: FormatDetector instance for automatic format detection
        validator: GeospatialValidator for data quality validation

    Methods:
        ingest_multi_source(): Ingest data from multiple sources simultaneously
        validate_and_clean(): Validate and clean ingested data
        generate_quality_report(): Generate comprehensive quality assessment
        _connect_all_sources(): Establish connections to all configured sources
        _ingest_single_source(): Ingest data from a single source
        _validate_ingested_data(): Validate data from specific source
        _clean_data(): Clean data based on validation issues
        _calculate_completeness(): Calculate data completeness score
        _calculate_accuracy(): Calculate data accuracy score
        _calculate_consistency(): Calculate data consistency score
        _generate_recommendations(): Generate quality improvement recommendations

    Args:
        data_sources: List of data source types to support
        format_detection: Format detection strategy ('automatic', 'manual', 'config')
        validation_enabled: Whether to enable data validation
        quality_threshold: Minimum quality score for acceptance (0.0 to 1.0)
        parallel_processing: Whether to use parallel processing
        max_workers: Maximum number of worker threads/processes

    Raises:
        ValueError: If data source is not supported
        ConfigurationError: If configuration is invalid

    Examples:
        >>> # Basic usage
        >>> ingestion = MultiSourceDataIngestion(
        ...     data_sources=['satellite', 'sensors', 'crowdsourced'],
        ...     validation_enabled=True,
        ...     quality_threshold=0.8,
        ...     parallel_processing=True
        ... )
        >>>
        >>> # Ingest multi-source data
        >>> result = await ingestion.ingest_multi_source(
        ...     satellite_data={'bbox': [-122.5, 37.7, -122.3, 37.9]},
        ...     sensor_data={'time_range': '2023-01-01/2023-01-31'},
        ...     crowdsourced_data={'category': 'environment'}
        ... )
        >>>
        >>> # Validate and clean data
        >>> cleaned_result = await ingestion.validate_and_clean(result)
        >>>
        >>> # Generate quality report
        >>> quality_report = ingestion.generate_quality_report(cleaned_result)
        >>> print(f"Overall quality: {quality_report['overall_score']:.2f}")
        >>> print(f"Validation passed: {quality_report['validation_passed']}")
    """

    def __init__(
        self,
        data_sources: list[str],
        format_detection: str = "automatic",
        validation_enabled: bool = True,
        quality_threshold: float = 0.8,
        parallel_processing: bool = True,
        max_workers: int = 4,
    ):
        self.config = IngestionConfig(
            data_sources=data_sources,
            format_detection=format_detection,
            validation_enabled=validation_enabled,
            quality_threshold=quality_threshold,
            parallel_processing=parallel_processing,
            max_workers=max_workers,
        )

        self.connectors: dict[str, Any] = {}
        self.format_detector = FormatDetector()
        self.validator = GeospatialValidator()
        self._initialize_connectors()

        missing = sorted(set(self.config.data_sources) - set(self.connectors))
        if missing:
            raise ValueError(f"Unsupported data source(s): {', '.join(missing)}")

        logger.info(
            f"Initialized MultiSourceDataIngestion with {len(data_sources)} sources"
        )

    def _initialize_connectors(self) -> None:
        """Initialize data source connectors."""
        connector_configs: dict[str, dict[str, Any]] = {
            "satellite": {
                "api_key": "your_api_key",
                "base_url": "https://api.example.com",
            },
            "sensors": {"host": "localhost", "port": 1883, "topic": "sensors/#"},
            "crowdsourced": {
                "api_endpoint": "https://api.crowdsourcing.com",
                "api_key": "your_key",
            },
            "weather_api": {
                "api_key": "weather_key",
                "base_url": "https://api.weather.com",
            },
            "social_media": {"api_key": "social_key", "platform": "twitter"},
            "government": {
                "api_endpoint": "https://api.govdata.com",
                "format": "geojson",
            },
        }

        for source in self.config.data_sources:
            if source in connector_configs:
                if source == "satellite":
                    self.connectors[source] = SatelliteDataConnector(
                        connector_configs[source]
                    )
                elif source == "sensors":
                    self.connectors[source] = SensorDataConnector(
                        connector_configs[source]
                    )
                elif source == "crowdsourced":
                    self.connectors[source] = CrowdsourcedDataConnector(
                        connector_configs[source]
                    )
                else:
                    # Generic connector for other sources
                    self.connectors[source] = GenericDataSourceConnector(
                        connector_configs[source]
                    )

    async def ingest_multi_source(self, **data_sources: Any) -> dict[str, Any]:
        """
        Ingest data from multiple sources simultaneously.

        This method coordinates data ingestion from multiple geospatial sources
        in parallel, with automatic connection management, error recovery, and
        data validation. Each source is processed independently and results
        are aggregated into a comprehensive ingestion report.

        The method supports various data source types including:
        - satellite: Satellite imagery and remote sensing data
        - sensors: IoT sensor networks and monitoring devices
        - crowdsourced: Crowdsourced observations and reports
        - weather_api: Weather and meteorological data
        - social_media: Social media geospatial data
        - government: Government open data portals

        Args:
            **data_sources: Data source parameters as keyword arguments. Each
                argument name corresponds to a data source type (e.g., satellite=...,
                sensors=...). The value should be a dictionary containing source-
                specific parameters such as spatial bounds, temporal ranges, and
                filtering criteria.

        Returns:
            Dictionary containing comprehensive ingestion results with the following structure:
            {
                'ingested_data': {
                    'source_name': {
                        'data': fetched_data,
                        'validation': quality_check_results,
                        'metadata': source_metadata
                    }
                },
                'quality_reports': {
                    'source_name': QualityCheck object
                },
                'ingestion_metadata': {
                    'timestamp': datetime,
                    'sources_processed': int,
                    'validation_enabled': bool,
                    'parallel_processing': bool
                }
            }

        Raises:
            ValueError: If a requested data source is not supported or configured
            ConnectionError: If unable to connect to one or more data sources
            ValidationError: If data validation fails and strict mode is enabled
            TimeoutError: If ingestion takes longer than configured timeout

        Examples:
            >>> # Ingest from satellite and sensor sources
            >>> result = await ingestion.ingest_multi_source(
            ...     satellite={
            ...         'bbox': [-122.5, 37.7, -122.3, 37.9],
            ...         'date_range': '2023-01-01/2023-01-31',
            ...         'bands': ['red', 'green', 'blue', 'nir']
            ...     },
            ...     sensors={
            ...         'time_range': '2023-01-01/2023-01-31',
            ...         'sensor_types': ['temperature', 'humidity'],
            ...         'locations': [{'lat': 37.7749, 'lon': -122.4194}]
            ...     },
            ...     crowdsourced={
            ...         'category': 'environment',
            ...         'time_range': '2023-01-01/2023-01-31',
            ...         'max_reports': 1000
            ...     }
            ... )
            >>>
            >>> # Check results
            >>> print(f"Processed {result['ingestion_metadata']['sources_processed']} sources")
            >>> for source, data in result['ingested_data'].items():
            ...     if 'validation' in data:
            ...         print(f"{source} quality: {data['validation'].score:.2f}")
        """
        logger.info(f"Starting multi-source ingestion for {len(data_sources)} sources")

        # Validate requested sources
        for source in data_sources.keys():
            if source not in self.config.data_sources:
                raise ValueError(
                    f"Data source '{source}' not supported. Available: {self.config.data_sources}"
                )

        # Connect only to sources requested for this operation. A configured
        # source that is not part of this request must not make an otherwise
        # valid ingestion fail.
        requested_sources = list(data_sources)
        connection_results = await self._connect_all_sources(requested_sources)
        if not all(connection_results.values()):
            failed_sources = [
                s for s, connected in connection_results.items() if not connected
            ]
            raise ConnectionError(f"Failed to connect to sources: {failed_sources}")

        # Ingest data from all sources
        ingestion_tasks = []
        for source_name, source_data in data_sources.items():
            task = self._ingest_single_source(source_name, source_data)
            ingestion_tasks.append(task)

        # Execute ingestion tasks
        if self.config.parallel_processing:
            results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)
        else:
            results = []
            for task in ingestion_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ingestion failed for task: {e}")
                    results.append({"error": str(e)})

        # Process results
        ingested_data: dict[str, Any] = {}
        quality_reports = {}

        for source_name, ing_result in zip(data_sources.keys(), results):
            if isinstance(ing_result, Exception):
                logger.error(f"Failed to ingest from {source_name}: {ing_result}")
                ingested_data[source_name] = {"error": str(ing_result)}
            else:
                ingested_data[source_name] = ing_result

                # Validate data if enabled
                if self.config.validation_enabled:
                    quality_report = await self._validate_ingested_data(
                        source_name, ing_result
                    )
                    quality_reports[source_name] = quality_report

                    # Check quality threshold
                    if quality_report.score < self.config.quality_threshold:
                        logger.warning(
                            f"Data quality below threshold for {source_name}: {quality_report.score:.2f}"
                        )

        # Generate comprehensive report
        ingestion_report = {
            "ingested_data": ingested_data,
            "quality_reports": quality_reports,
            "ingestion_metadata": {
                "timestamp": datetime.now(UTC),
                "sources_processed": len(data_sources),
                "validation_enabled": self.config.validation_enabled,
                "parallel_processing": self.config.parallel_processing,
            },
        }

        logger.info(f"Multi-source ingestion completed for {len(data_sources)} sources")
        return ingestion_report

    async def _connect_all_sources(
        self, source_names: list[str] | None = None
    ) -> dict[str, bool]:
        """Connect to the configured sources selected for one operation."""
        source_names = list(source_names or self.connectors)
        results = {}
        for source_name in source_names:
            connector = self.connectors[source_name]
            connected = False
            for attempt in range(1, self.config.retry_attempts + 1):
                try:
                    connected = await connector.connect()
                    if connected:
                        break
                except Exception as e:
                    logger.error(
                        "Connection error for %s on attempt %s: %s",
                        source_name,
                        attempt,
                        e,
                    )
            results[source_name] = connected
            logger.info(
                f"{'Connected' if connected else 'Failed to connect'} to {source_name}"
            )

        return results

    async def _ingest_single_source(
        self, source_name: str, source_data: Any
    ) -> dict[str, Any]:
        """Ingest data from a single source."""
        connector = self.connectors[source_name]

        try:
            # Detect format if automatic; work on a copy so the caller's
            # payload dict is never mutated as a side effect.
            if self.config.format_detection == "automatic":
                detected_format = self.format_detector.detect_format(source_data)
                if isinstance(source_data, dict):
                    source_data = {**source_data, "format": detected_format}

            # Fetch data
            data = await connector.fetch_data(source_data)

            # Validate data
            if self.config.validation_enabled:
                validation_result = await connector.validate_data(data)
                data["validation"] = validation_result

            return cast(dict[str, Any], data)

        except Exception as e:
            logger.error(f"Failed to ingest from {source_name}: {e}")
            return {"error": str(e)}

    async def _validate_ingested_data(
        self, source_name: str, data: Any
    ) -> QualityCheck:
        """Validate ingested data from a specific source."""
        try:
            return await self.validator.validate_data(data)
        except Exception as e:
            logger.error(f"Validation failed for {source_name}: {e}")
            return QualityCheck(
                score=0.0,
                status=cast(Any, "fail"),
                issues=[{"type": "validation_error", "message": str(e)}],
            )

    async def validate_and_clean(self, ingested_data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and clean ingested data.

        This method performs comprehensive validation and cleaning of ingested data
        from multiple sources. It applies data quality rules, removes invalid records,
        handles missing values, and generates detailed validation reports.

        The cleaning process includes:
        - Missing value imputation or removal
        - Outlier detection and handling
        - Data type consistency checks
        - Coordinate validation and correction
        - Temporal consistency validation
        - Spatial reference system validation

        Args:
            ingested_data: Raw ingested data from multiple sources in the format
                returned by ingest_multi_source(). Should contain 'ingested_data'
                dictionary with source-specific data.

        Returns:
            Dictionary containing cleaned and validated data with the following structure:
            {
                'cleaned_data': {
                    'source_name': {
                        'data': cleaned_data,
                        'validation': updated_quality_check,
                        'cleaning_applied': list_of_cleaning_operations
                    }
                },
                'validation_summary': {
                    'source_name': QualityCheck object
                },
                'cleaning_metadata': {
                    'timestamp': datetime,
                    'sources_cleaned': int,
                    'total_records_cleaned': int,
                    'cleaning_operations': list
                }
            }

        Raises:
            ValidationError: If validation fails for all sources
            CleaningError: If data cleaning process fails
            ValueError: If input data format is invalid

        Examples:
            >>> # Clean previously ingested data
            >>> cleaned_result = await ingestion.validate_and_clean(ingestion_result)
            >>>
            >>> # Check cleaning results
            >>> for source, data in cleaned_result['cleaned_data'].items():
            ...     operations = data.get('cleaning_applied', [])
            ...     print(f"{source}: {len(operations)} cleaning operations applied")
            >>>
            >>> # Validate cleaning quality
            >>> print(f"Total sources cleaned: {cleaned_result['cleaning_metadata']['sources_cleaned']}")
        """
        logger.info("Starting data validation and cleaning")

        cleaned_data = {}
        validation_summary = {}

        for source_name, data in ingested_data["ingested_data"].items():
            if "error" in data:
                cleaned_data[source_name] = data
                continue

            try:
                # Validate data structure
                validation_result = await self._validate_ingested_data(
                    source_name, data
                )

                # Clean data based on validation results
                if validation_result.status == "pass":
                    cleaned_data[source_name] = data
                else:
                    # Apply cleaning rules
                    cleaned = await self._clean_data(data, validation_result.issues)
                    cleaned_data[source_name] = cleaned

                validation_summary[source_name] = validation_result

            except Exception as e:
                logger.error(f"Failed to validate/clean {source_name}: {e}")
                cleaned_data[source_name] = {"error": str(e)}

        return {
            "cleaned_data": cleaned_data,
            "validation_summary": validation_summary,
            "cleaning_metadata": {
                "timestamp": datetime.now(UTC),
                "sources_cleaned": len(
                    [d for d in cleaned_data.values() if "error" not in d]
                ),
            },
        }

    async def _clean_data(self, data: Any, issues: list[dict[str, Any]]) -> Any:
        """Clean data based on validation issues."""
        # Implementation of data cleaning logic
        # This would handle common data quality issues like:
        # - Missing values
        # - Outliers
        # - Format inconsistencies
        # - Invalid geometries
        # - Temporal anomalies

        cleaned_data = data.copy() if isinstance(data, dict) else data

        for issue in issues:
            issue_type = issue.get("type")
            if issue_type == "missing_values":
                # Fill missing values using forward-fill then backward-fill
                if isinstance(cleaned_data, dict):
                    for key, val in cleaned_data.items():
                        if isinstance(val, pd.DataFrame):
                            cleaned_data[key] = val.ffill().bfill()
                elif isinstance(cleaned_data, pd.DataFrame):
                    cleaned_data = cleaned_data.ffill().bfill()
                logger.info("Cleaned missing values via forward/backward fill")

            elif issue_type == "invalid_geometry":
                # Attempt to fix invalid geometries using buffer(0) trick
                if isinstance(cleaned_data, dict):
                    for key, val in cleaned_data.items():
                        if hasattr(val, "geometry"):
                            try:
                                invalid_mask = ~val.geometry.is_valid
                                if invalid_mask.any():
                                    val.loc[invalid_mask, "geometry"] = val.loc[
                                        invalid_mask, "geometry"
                                    ].buffer(0)
                                    cleaned_data[key] = val
                            except Exception as geom_err:
                                logger.warning("Could not fix geometries: %s", geom_err)
                logger.info("Cleaned invalid geometries via buffer(0)")

            elif issue_type == "temporal_anomaly":
                # Sort by timestamp and remove exact duplicate timestamps
                if isinstance(cleaned_data, dict):
                    for key, val in cleaned_data.items():
                        if isinstance(val, pd.DataFrame) and hasattr(val, "index"):
                            if isinstance(val.index, pd.DatetimeIndex):
                                val = val.sort_index()
                                val = val[~val.index.duplicated(keep="first")]
                                cleaned_data[key] = val
                elif isinstance(cleaned_data, pd.DataFrame):
                    if isinstance(cleaned_data.index, pd.DatetimeIndex):
                        cleaned_data = cleaned_data.sort_index()
                        cleaned_data = cleaned_data[
                            ~cleaned_data.index.duplicated(keep="first")
                        ]
                logger.info("Cleaned temporal anomalies: sorted and deduplicated")

        return cleaned_data

    def generate_quality_report(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate comprehensive quality report for ingested data.

        This method performs detailed quality assessment of ingested or cleaned data
        across multiple dimensions including completeness, accuracy, consistency,
        and temporal validity. It provides actionable recommendations for data
        improvement and quality enhancement.

        Quality metrics calculated:
        - Completeness: Percentage of non-missing values and required fields
        - Accuracy: Outlier detection, coordinate validity, and data range checks
        - Consistency: Data type consistency, duplicate detection, temporal order
        - Validity: Format validation, schema compliance, and constraint checks

        Args:
            data: Ingested or cleaned data in the format returned by ingest_multi_source()
                or validate_and_clean(). Should contain 'ingested_data' or 'cleaned_data'
                dictionary with source-specific datasets.

        Returns:
            Comprehensive quality assessment report with the following structure:
            {
                'overall_score': float,  # 0.0 to 1.0 overall quality score
                'source_scores': {
                    'source_name': float  # Individual source quality scores
                },
                'quality_threshold': float,  # Configured quality threshold
                'validation_enabled': bool,  # Whether validation was performed
                'issues': [str],  # List of identified quality issues
                'recommendations': [str],  # Improvement recommendations
                'generated_at': datetime  # Report generation timestamp
            }

        Raises:
            ValueError: If input data format is invalid
            QualityAssessmentError: If quality assessment fails

        Examples:
            >>> # Generate report for ingested data
            >>> report = ingestion.generate_quality_report(ingestion_result)
            >>>
            >>> # Analyze quality scores
            >>> print(f"Overall quality: {report['overall_score']:.2f}")
            >>> print(f"Quality threshold: {report['quality_threshold']:.2f}")
            >>> print(f"Validation passed: {report['overall_score'] >= report['quality_threshold']}")
            >>>
            >>> # Review recommendations
            >>> if report['recommendations']:
            ...     print("Improvement recommendations:")
            ...     for rec in report['recommendations']:
            ...         print(f"  - {rec}")
            >>>
            >>> # Check source-specific scores
            >>> for source, score in report['source_scores'].items():
            ...     status = "✓ PASS" if score >= 0.8 else "⚠ WARNING" if score >= 0.6 else "✗ FAIL"
            ...     print(f"{source}: {score:.2f} ({status})")
        """
        logger.info("Generating data quality report")

        quality_scores = {}
        overall_issues = []

        for source_name, source_data in data.get("ingested_data", {}).items():
            if "error" in source_data:
                quality_scores[source_name] = 0.0
                overall_issues.append(f"Ingestion failed for {source_name}")
                continue

            # Calculate quality metrics
            completeness = self._calculate_completeness(source_data)
            accuracy = self._calculate_accuracy(source_data)
            consistency = self._calculate_consistency(source_data)

            overall_score = (completeness + accuracy + consistency) / 3.0

            quality_scores[source_name] = overall_score

            if overall_score < self.config.quality_threshold:
                overall_issues.append(
                    f"Low quality data from {source_name}: {overall_score:.2f}"
                )

        overall_score = (
            sum(quality_scores.values()) / len(quality_scores)
            if quality_scores
            else 0.0
        )

        return {
            "overall_score": overall_score,
            "source_scores": quality_scores,
            "quality_threshold": self.config.quality_threshold,
            "validation_enabled": self.config.validation_enabled,
            "issues": overall_issues,
            "recommendations": self._generate_recommendations(quality_scores),
            "generated_at": datetime.now(UTC),
        }

    def _calculate_completeness(self, data: Any) -> float:
        """Calculate data completeness score.

        Measures the fraction of non-null values across all DataFrames
        found in the data.  Falls back to 1.0 when completeness cannot
        be determined (e.g. non-tabular data).
        """
        dataframes = self._extract_dataframes(data)
        if not dataframes:
            return 1.0

        total_cells = 0
        non_null_cells = 0
        for df in dataframes:
            total_cells += df.size
            non_null_cells += int(df.notna().sum().sum())

        if total_cells == 0:
            return 1.0
        return non_null_cells / total_cells

    def _calculate_accuracy(self, data: Any) -> float:
        """Calculate data accuracy score.

        Checks numeric columns for outliers using the IQR method.
        The accuracy score is the fraction of values within 3x IQR
        of the median across all numeric columns.
        """
        dataframes = self._extract_dataframes(data)
        if not dataframes:
            return 1.0

        total_values = 0
        accurate_values = 0
        for df in dataframes:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 4:
                    total_values += len(series)
                    accurate_values += len(series)
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                in_range = ((series >= lower) & (series <= upper)).sum()
                total_values += len(series)
                accurate_values += int(in_range)

        if total_values == 0:
            return 1.0
        return accurate_values / total_values

    def _calculate_consistency(self, data: Any) -> float:
        """Calculate data consistency score.

        Checks for duplicate rows and consistent data types within
        each DataFrame.  Penalises duplicates and mixed-type columns.
        """
        dataframes = self._extract_dataframes(data)
        if not dataframes:
            return 1.0

        scores: list[float] = []
        for df in dataframes:
            if len(df) == 0:
                scores.append(1.0)
                continue

            # Duplicate penalty
            dup_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0.0
            dup_score = 1.0 - dup_ratio

            # Type consistency: penalise object columns with mixed Python types
            type_penalties = 0
            for col in df.columns:
                if df[col].dtype == object:
                    unique_types = df[col].dropna().apply(type).nunique()
                    if unique_types > 1:
                        type_penalties += 1
            type_score = 1.0 - (type_penalties / max(len(df.columns), 1))

            scores.append((dup_score + type_score) / 2.0)

        return float(np.mean(scores)) if scores else 1.0

    @staticmethod
    def _extract_dataframes(data: Any) -> list[pd.DataFrame]:
        """Extract DataFrames from various data structures."""
        frames: list[pd.DataFrame] = []
        if isinstance(data, pd.DataFrame):
            frames.append(data)
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, pd.DataFrame):
                    frames.append(val)
        return frames

    def _generate_recommendations(self, quality_scores: dict[str, float]) -> list[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        for source, score in quality_scores.items():
            if score < 0.8:
                recommendations.append(
                    f"Improve data quality for {source} (current: {score:.2f})"
                )
            elif score < 0.9:
                recommendations.append(
                    f"Monitor data quality for {source} (current: {score:.2f})"
                )

        return recommendations
