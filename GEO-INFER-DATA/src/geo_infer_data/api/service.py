"""
Core data service for GEO-INFER-DATA.

This module provides the core data service functionality including
dataset management, data access, and integration with other modules.
"""

import logging
import inspect
from typing import Any, cast
from datetime import datetime, UTC

import geopandas as gpd
import pandas as pd

from ..models.schemas import (
    Dataset,
    DatasetMetadata,
    DataQualityReport,
    DataType,
    DataFormat,
    StorageBackend,
)
from ..core.storage import AdaptiveDataStorage
from ..core.validation import DataQualityManager


logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Return sync values and await coroutine-like service results."""
    if inspect.isawaitable(value):
        return await value
    return value


class DataService:
    """
    Core data service for geospatial data management.

    This class provides comprehensive data service functionality including
    dataset management, data access patterns, and integration capabilities.

    Args:
        storage_service: Storage service instance
        quality_service: Quality service instance

    Examples:
        >>> service = DataService(storage, quality_manager)
        >>>
        >>> # List available datasets
        >>> datasets = service.list_datasets()
        >>>
        >>> # Get dataset data
        >>> data = service.get_dataset_data('dataset_123', bbox=[-122.5, 37.7, -122.3, 37.9])
        >>>
        >>> # Create new dataset
        >>> dataset_id = service.create_dataset(metadata, geodataframe)
    """

    def __init__(
        self,
        storage_service: AdaptiveDataStorage | None = None,
        quality_service: DataQualityManager | None = None,
    ):
        self.storage_service = storage_service or AdaptiveDataStorage(
            storage_backends=["local"], optimization_strategy="access_pattern_based"
        )

        self.quality_service = quality_service or DataQualityManager(
            validation_rules="comprehensive", quality_threshold=0.8
        )

        self.datasets: dict[str, Dataset] = {}
        self.access_log: list[dict[str, Any]] = []

        logger.info("Initialized DataService")

    async def list_datasets(
        self, filters: dict[str, Any] | None = None, limit: int = 50, offset: int = 0
    ) -> list[Dataset]:
        """
        List available datasets.

        Args:
            filters: Optional filters to apply
            limit: Maximum number of datasets to return
            offset: Offset for pagination

        Returns:
            List of datasets
        """
        logger.debug(f"Listing datasets with filters: {filters}")

        datasets = []
        for dataset in self.datasets.values():
            # Apply filters
            if filters:
                if "type" in filters and dataset.type != filters["type"]:
                    continue
                if "bbox" in filters:
                    # Check if dataset intersects with filter bbox
                    if dataset.metadata.spatial is None:
                        continue
                    dataset_bbox = dataset.metadata.spatial.bbox
                    filter_bbox = filters["bbox"]
                    if not self._bboxes_intersect(dataset_bbox, filter_bbox):
                        continue

            datasets.append(dataset)

        # Paginate over the filtered result set
        return datasets[offset : offset + limit]

    def _bboxes_intersect(self, bbox1: list[float], bbox2: list[float]) -> bool:
        """Check if two bounding boxes intersect."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False

        min_lon1, min_lat1, max_lon1, max_lat1 = bbox1[:4]
        min_lon2, min_lat2, max_lon2, max_lat2 = bbox2[:4]

        return not (
            max_lon1 < min_lon2
            or min_lon1 > max_lon2
            or max_lat1 < min_lat2
            or min_lat1 > max_lat2
        )

    async def get_dataset(self, dataset_id: str) -> Dataset | None:
        """
        Get dataset information.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset information or None if not found
        """
        logger.debug(f"Getting dataset: {dataset_id}")

        # Check cache first
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]

        return self.datasets.get(dataset_id)

    async def get_dataset_data(
        self,
        dataset_id: str,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple | None = None,
        format: str = "geojson",
    ) -> Any:
        """
        Get dataset data with optional filtering.

        Args:
            dataset_id: Dataset identifier
            spatial_bounds: Spatial bounds filter
            temporal_range: Temporal range filter
            format: Output format

        Returns:
            Filtered dataset data
        """
        logger.debug(f"Getting data for dataset {dataset_id} with format {format}")

        if dataset_id not in self.datasets:
            raise KeyError(f"Dataset {dataset_id!r} is not registered")

        # Log access
        self.access_log.append(
            {
                "dataset_id": dataset_id,
                "timestamp": datetime.now(UTC),
                "spatial_bounds": spatial_bounds,
                "temporal_range": temporal_range,
                "format": format,
            }
        )

        try:
            data = await self.storage_service.retrieve_geospatial_data(
                dataset_id,
                spatial_bounds=spatial_bounds,
                temporal_range=temporal_range,
            )

            # Convert format if needed
            if format == "geojson" and isinstance(
                data, (pd.DataFrame, gpd.GeoDataFrame)
            ):
                return data.to_json() if hasattr(data, "to_json") else data.to_dict()
            elif format == "csv" and isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                return data.to_csv()
            else:
                return data

        except Exception as e:
            logger.error(f"Failed to get dataset data: {e}")
            raise

    async def create_dataset(
        self, metadata: DatasetMetadata, data: Any, storage_backend: str = "auto"
    ) -> str:
        """
        Create a new dataset.

        Args:
            metadata: Dataset metadata
            data: Dataset data
            storage_backend: Storage backend to use

        Returns:
            Dataset ID
        """
        logger.info(f"Creating dataset: {metadata.title}")

        # Validate data quality
        quality_report = await self.quality_service.validator.validate_data(
            data, metadata
        )

        threshold = getattr(
            getattr(self.quality_service, "config", None), "quality_threshold", 0.8
        )
        if not isinstance(threshold, (int, float)):
            threshold = 0.8
        if quality_report.overall_score < threshold:
            logger.warning(
                f"Low quality data for dataset {metadata.title}: {quality_report.overall_score}"
            )

        # Store data
        access_patterns = self._analyze_access_patterns(metadata)
        dataset_id = await self.storage_service.store_geospatial_data(
            data, metadata, access_patterns
        )

        # Create dataset record
        dataset = Dataset(
            id=dataset_id,
            title=metadata.title,
            description=metadata.description,
            type=DataType.VECTOR,  # Would be determined from data
            format=DataFormat.GEOJSON,  # Would be determined from data
            metadata=metadata,
            storage_backend=(
                StorageBackend(storage_backend)
                if storage_backend != "auto"
                else StorageBackend.POSTGRESQL
            ),
        )

        self.datasets[dataset_id] = dataset
        self.quality_service.register_dataset(dataset_id, data, metadata)

        logger.info(f"Dataset created successfully: {dataset_id}")
        return dataset_id

    def _analyze_access_patterns(self, metadata: DatasetMetadata) -> dict[str, Any]:
        """Analyze expected access patterns for optimization."""
        patterns: dict[str, Any] = {
            "spatial_queries": [],
            "temporal_queries": [],
            "query_frequency": "medium",
        }

        # Analyze based on metadata
        if metadata.spatial:
            # Add spatial pattern based on bounds
            patterns["spatial_queries"].append(
                {"bbox": metadata.spatial.bbox, "frequency": "high"}
            )

        if metadata.temporal:
            # Add temporal pattern
            patterns["temporal_queries"].append(
                {
                    "start": metadata.temporal.start,
                    "end": metadata.temporal.end,
                    "frequency": "medium",
                }
            )

        return patterns

    async def update_dataset(self, dataset_id: str, updates: dict[str, Any]) -> bool:
        """
        Update dataset information.

        Args:
            dataset_id: Dataset identifier
            updates: Updates to apply

        Returns:
            True if successful
        """
        logger.info(f"Updating dataset {dataset_id}: {updates}")

        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            logger.warning("Cannot update unknown dataset %s", dataset_id)
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
            elif hasattr(dataset.metadata, key):
                setattr(dataset.metadata, key, value)

        dataset.updated_at = datetime.now(UTC)

        return True

    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if successful
        """
        logger.info(f"Deleting dataset: {dataset_id}")

        if dataset_id not in self.datasets:
            logger.warning("Cannot delete unknown dataset %s", dataset_id)
            return False

        del self.datasets[dataset_id]
        return True

    async def get_dataset_quality(self, dataset_id: str) -> DataQualityReport:
        """
        Get dataset quality report.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Quality report
        """
        return cast(
            DataQualityReport,
            await _maybe_await(self.quality_service.validate_dataset(dataset_id)),
        )

    def get_access_patterns(self, dataset_id: str | None = None) -> dict[str, Any]:
        """
        Get data access patterns.

        Args:
            dataset_id: Optional dataset to filter by

        Returns:
            Access pattern analysis
        """
        if dataset_id:
            # Get patterns for specific dataset
            dataset_access = [
                log for log in self.access_log if log["dataset_id"] == dataset_id
            ]
            return self._analyze_access_log(dataset_access)
        else:
            # Get overall patterns
            return self._analyze_access_log(self.access_log)

    def _analyze_access_log(self, access_log: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze access log for patterns."""
        if not access_log:
            return {"message": "No access data available"}

        # Analyze access patterns
        spatial_queries = [log for log in access_log if log.get("spatial_bounds")]
        temporal_queries = [log for log in access_log if log.get("temporal_range")]

        return {
            "total_accesses": len(access_log),
            "spatial_queries": len(spatial_queries),
            "temporal_queries": len(temporal_queries),
            "formats_requested": list(
                set(log.get("format", "unknown") for log in access_log)
            ),
            "peak_hours": self._find_peak_hours(access_log),
        }

    def _find_peak_hours(self, access_log: list[dict[str, Any]]) -> list[int]:
        """Find peak access hours."""
        hour_counts: dict[int, int] = {}

        for log in access_log:
            hour = log["timestamp"].hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # Find hours with highest access
        if hour_counts:
            max_count = max(hour_counts.values())
            return [hour for hour, count in hour_counts.items() if count == max_count]

        return []

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return self.storage_service.get_storage_stats()

    def optimize_performance(self) -> dict[str, Any]:
        """Optimize data service performance."""
        optimizations: dict[str, Any] = {
            "cache_optimization": {},
            "storage_optimization": {},
            "query_optimization": {},
        }

        # Analyze access patterns and optimize
        access_patterns = self.get_access_patterns()

        if access_patterns.get("total_accesses", 0) > 100:
            # Optimize frequently accessed datasets
            optimizations["cache_optimization"]["frequent_datasets"] = "optimized"

        # Storage optimization
        storage_stats = self.get_storage_stats()
        if storage_stats.get("total_size", 0) > 1000000000:  # 1GB
            optimizations["storage_optimization"]["compression"] = "enabled"

        return optimizations
