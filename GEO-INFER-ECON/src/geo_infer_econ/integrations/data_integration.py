"""
GEO-INFER-DATA Integration Adapter

Provides data loading wrapper for economic datasets.
"""

from typing import Any
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import GEO-INFER-DATA modules
try:
    from geo_infer_data.api.service import DataService
    from geo_infer_data.core.ingestion import MultiSourceDataIngestion

    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    logger.warning(
        "GEO-INFER-DATA not available. Data loading will be limited. "
        "Install geo-infer-data to enable full functionality."
    )


class DataIntegration:
    """
    Integration adapter for GEO-INFER-DATA.

    Provides data loading for economic analysis including:
    - Dataset access and querying
    - Spatial and temporal filtering
    - Data format conversion
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize data integration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        if not DATA_AVAILABLE:
            logger.warning(
                "DataIntegration initialized but GEO-INFER-DATA not available"
            )
            self.service = None
            self.ingestion = None
        else:
            try:
                self.service = DataService()
                self.ingestion = MultiSourceDataIngestion(
                    data_sources=["sensors"], validation_enabled=True
                )
                logger.info("DataIntegration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DataIntegration: {e}")
                self.service = None
                self.ingestion = None

    def load_dataset(
        self,
        dataset_id: str,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple[str, str] | None = None,
        format: str = "geopandas",
    ) -> pd.DataFrame | gpd.GeoDataFrame | None:
        """
        Load economic dataset with optional filtering.

        Args:
            dataset_id: Dataset identifier
            spatial_bounds: Optional spatial bounds [min_lon, min_lat, max_lon, max_lat]
            temporal_range: Optional temporal range (start_date, end_date)
            format: Output format ('geopandas', 'pandas', 'geojson', 'csv')

        Returns:
            Loaded dataset or None if unavailable
        """
        if not DATA_AVAILABLE or self.service is None:
            logger.warning("Data service not available for loading datasets")
            return None

        try:
            # Use async method if available, otherwise sync
            import asyncio

            try:
                data = asyncio.run(
                    self.service.get_dataset_data(
                        dataset_id=dataset_id,
                        spatial_bounds=spatial_bounds,
                        temporal_range=temporal_range,
                        format=format,
                    )
                )
            except AttributeError:
                # Fallback to sync method if async not available
                data = self.service.get_dataset_data(
                    dataset_id=dataset_id,
                    spatial_bounds=spatial_bounds,
                    temporal_range=temporal_range,
                    format=format,
                )

            # Convert format if needed
            if format == "geopandas" and isinstance(data, (dict, str)):
                if isinstance(data, str):
                    return gpd.read_file(data)
                else:
                    return gpd.GeoDataFrame.from_features(data.get("features", []))
            elif format == "pandas" and isinstance(data, (dict, str)):
                if isinstance(data, str):
                    return pd.read_csv(data)
                else:
                    return pd.DataFrame(data)

            return data
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return None

    def list_datasets(
        self, dataset_type: str | None = None, tags: list[str] | None = None
    ) -> list[dict[str, Any]] | None:
        """
        List available economic datasets.

        Args:
            dataset_type: Optional dataset type filter
            tags: Optional tags filter

        Returns:
            List of dataset metadata dictionaries or None if unavailable
        """
        if not DATA_AVAILABLE or self.service is None:
            logger.warning("Data service not available for listing datasets")
            return None

        try:
            import asyncio

            try:
                datasets = asyncio.run(self.service.list_datasets())
            except AttributeError:
                datasets = self.service.list_datasets()

            # Apply filters
            normalized = [
                d.model_dump() if hasattr(d, "model_dump") else d for d in datasets
            ]
            if dataset_type:
                normalized = [d for d in normalized if d.get("type") == dataset_type]
            if tags:
                normalized = [
                    d
                    for d in normalized
                    if any(tag in d.get("tags", []) for tag in tags)
                ]

            return normalized
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return None

    def load_economic_data(
        self, source: str | Path, source_type: str = "file", **kwargs: Any
    ) -> pd.DataFrame | gpd.GeoDataFrame | None:
        """
        Load economic data from various sources.

        Args:
            source: Data source (file path, URL, or dataset ID)
            source_type: Source type ('file', 'url', 'dataset')
            **kwargs: Additional parameters for data loading

        Returns:
            Loaded data or None if unavailable
        """
        if source_type == "dataset":
            return self.load_dataset(str(source), **kwargs)

        if not DATA_AVAILABLE or self.ingestion is None:
            # Fallback to direct file loading
            try:
                source_path = Path(source)
                if source_path.suffix == ".csv":
                    return pd.read_csv(source_path)
                elif source_path.suffix in [".geojson", ".json"]:
                    return gpd.read_file(source_path)
                elif source_path.suffix in [".shp", ".gpkg"]:
                    return gpd.read_file(source_path)
                else:
                    logger.warning(f"Unknown file format: {source_path.suffix}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load file {source}: {e}")
                return None

        try:
            return self.ingestion.ingest_data(source, source_type=source_type, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load economic data: {e}")
            return None

    def is_available(self) -> bool:
        """Check if GEO-INFER-DATA is available."""
        return DATA_AVAILABLE and self.service is not None
