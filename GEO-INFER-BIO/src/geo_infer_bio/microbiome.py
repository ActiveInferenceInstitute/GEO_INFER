"""
GEO-INFER-BIO Microbiome Data Processing Module

This module provides comprehensive microbiome data handling capabilities for spatial analysis,
specifically designed to work with real-world datasets like the Earth Microbiome Project (EMP),
American Gut Project, and other spatially-referenced microbiome studies.

Key Features:
- Earth Microbiome Project (EMP) data loading and processing
- 16S rRNA sequence data handling with spatial context
- Microbiome diversity metrics calculation
- Taxonomic classification and standardization
- Spatial coordinate validation and quality control
- Integration with GEO-INFER-SPACE H3 indexing
"""

import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MicrobiomeDataLoader:
    """
    Comprehensive microbiome data loading and processing for spatial analysis.

    Supports multiple data sources:
    - Earth Microbiome Project (EMP)
    - American Gut Project
    - Custom microbiome datasets with spatial coordinates
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the microbiome data loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".geo_infer_bio" / "cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Earth Microbiome Project configuration
        self.emp_config = {
            "base_url": "https://qiita.ucsd.edu/public_artifact_download/",
            "metadata_file": "emp_qiime_mapping_qc_filtered.tsv",
            "sample_data_url": "https://earthmicrobiome.org/protocols-and-standards/",
        }

        logger.info(
            f"MicrobiomeDataLoader initialized with cache directory: {self.cache_dir}"
        )

    def load_emp_data(
        self,
        region_bbox: tuple[float, float, float, float] | None = None,
        sample_types: list[str] | None = None,
        max_samples: int | None = None,
        quality_filters: bool = True,
        metadata_path: str | None = None,
    ) -> "MicrobiomeDataset":
        """
        Load Earth Microbiome Project data with spatial filtering.

        Args:
            region_bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            sample_types: Filter by sample types (e.g., ['soil', 'sediment', 'water'])
            max_samples: Maximum number of samples to load
            quality_filters: Apply data quality filters
            metadata_path: Local EMP metadata TSV/CSV file. When omitted, the
                configured cache file must already exist.

        Returns:
            MicrobiomeDataset object with loaded data
        """
        logger.info("Loading Earth Microbiome Project data...")

        source_path = (
            Path(metadata_path)
            if metadata_path
            else self.cache_dir / self.emp_config["metadata_file"]
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"EMP metadata file not found: {source_path}. "
                "Download the dataset and pass metadata_path or place it in the cache directory."
            )
        delimiter = "\t" if source_path.suffix.lower() in {".tsv", ".txt"} else ","
        metadata = pd.read_csv(source_path, sep=delimiter, index_col=0)
        required_columns = {"latitude", "longitude"}
        missing = required_columns - set(metadata.columns)
        if missing:
            raise ValueError(
                f"EMP metadata is missing required columns: {sorted(missing)}"
            )

        if region_bbox is not None:
            min_lon, min_lat, max_lon, max_lat = region_bbox
            metadata = metadata[
                metadata["latitude"].between(min_lat, max_lat)
                & metadata["longitude"].between(min_lon, max_lon)
            ]
        if sample_types:
            type_columns = [
                column for column in ("empo_1", "sample_type") if column in metadata
            ]
            if not type_columns:
                raise ValueError("EMP metadata does not contain a sample type column")
            metadata = metadata[metadata[type_columns[0]].isin(sample_types)]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            metadata = metadata.head(max_samples)

        # Apply quality filters
        if quality_filters:
            metadata = self._apply_quality_filters(metadata)

        logger.info(f"Successfully loaded EMP data: {len(metadata)} samples")

        return MicrobiomeDataset(metadata=metadata, data_source=str(source_path))

    def _apply_quality_filters(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters to metadata."""
        initial_count = len(metadata)

        # Remove samples with missing coordinates
        metadata = metadata.dropna(subset=["latitude", "longitude"])

        # Remove samples with invalid coordinates
        metadata = metadata[
            (metadata["latitude"].between(-90, 90))
            & (metadata["longitude"].between(-180, 180))
        ]

        # Remove samples with extreme pH values (likely errors), when pH data exists
        if "ph" in metadata.columns:
            metadata = metadata[metadata["ph"].between(0, 14)]
        else:
            logger.debug("Skipping pH quality filter: metadata has no 'ph' column")

        final_count = len(metadata)
        logger.info(
            f"Quality filtering: {initial_count} → {final_count} samples "
            f"({initial_count - final_count} removed)"
        )

        return metadata

    def load_custom_microbiome_data(
        self,
        metadata_path: str,
        coordinate_columns: tuple[str, str] = ("latitude", "longitude"),
    ) -> "MicrobiomeDataset":
        """
        Load custom microbiome dataset with spatial coordinates.

        Args:
            metadata_path: Path to sample metadata (CSV/TSV)
            coordinate_columns: Column names for latitude and longitude

        Returns:
            MicrobiomeDataset object
        """
        logger.info(f"Loading custom microbiome data from {metadata_path}")

        # Determine file format and load
        if metadata_path.endswith(".csv"):
            metadata = pd.read_csv(metadata_path, index_col=0)
        else:
            metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

        # Validate coordinate columns
        lat_col, lon_col = coordinate_columns
        if lat_col not in metadata.columns or lon_col not in metadata.columns:
            raise ValueError(
                f"Coordinate columns {coordinate_columns} not found in metadata"
            )

        # Rename coordinate columns to standard names
        metadata = metadata.rename(columns={lat_col: "latitude", lon_col: "longitude"})

        return MicrobiomeDataset(metadata=metadata, data_source="Custom dataset")


class MicrobiomeDataset:
    """
    Container for microbiome data with spatial analysis capabilities.

    Provides methods for:
    - Diversity metric calculation
    - Spatial coordinate access
    - Data quality assessment
    - Integration with GEO-INFER modules
    """

    def __init__(self, metadata: pd.DataFrame, data_source: str = "Unknown"):
        """
        Initialize microbiome dataset.

        Args:
            metadata: Sample metadata with spatial coordinates
            data_source: Description of data source
        """
        self.metadata = metadata
        self.data_source = data_source

        # Validate spatial coordinates
        self._validate_coordinates()

        logger.info(f"MicrobiomeDataset initialized: {len(self.metadata)} samples")

    def _validate_coordinates(self) -> None:
        """Validate that spatial coordinates are present and valid."""
        required_cols = ["latitude", "longitude"]
        missing_cols = [
            col for col in required_cols if col not in self.metadata.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required coordinate columns: {missing_cols}")

        # Check for valid coordinate ranges
        invalid_lat = ~self.metadata["latitude"].between(-90, 90)
        invalid_lon = ~self.metadata["longitude"].between(-180, 180)

        if invalid_lat.any() or invalid_lon.any():
            logger.warning(
                f"Found {invalid_lat.sum()} invalid latitudes and "
                f"{invalid_lon.sum()} invalid longitudes"
            )

    def get_coordinates(self) -> list[tuple[float, float]]:
        """
        Get sample coordinates as list of (latitude, longitude) tuples.

        Returns:
        """
        coords = []
        for _, row in self.metadata.iterrows():
            coords.append((row["latitude"], row["longitude"]))
        return coords

    def get_coordinates_gdf(self) -> gpd.GeoDataFrame:
        """
        Get sample coordinates as GeoDataFrame.

        Returns:
            GeoDataFrame with Point geometries
        """
        geometry = [Point(lon, lat) for lat, lon in self.get_coordinates()]
        gdf = gpd.GeoDataFrame(self.metadata, geometry=geometry, crs="EPSG:4326")
        return gdf

    def get_diversity_metrics(self) -> pd.DataFrame:
        """
        Get diversity metrics from metadata.

        Returns:
            DataFrame with diversity metrics
        """
        diversity_cols = [
            "shannon_diversity",
            "observed_species",
            "simpson_diversity",
            "chao1",
        ]
        available_cols = [col for col in diversity_cols if col in self.metadata.columns]

        if not available_cols:
            logger.warning("No diversity metrics found in metadata")
            return pd.DataFrame()

        return self.metadata[available_cols]

    def filter_by_coordinates(
        self, bbox: tuple[float, float, float, float]
    ) -> "MicrobiomeDataset":
        """
        Filter dataset by spatial bounding box.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            New MicrobiomeDataset with filtered samples
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        spatial_filter = (
            (self.metadata["latitude"] >= min_lat)
            & (self.metadata["latitude"] <= max_lat)
            & (self.metadata["longitude"] >= min_lon)
            & (self.metadata["longitude"] <= max_lon)
        )

        filtered_metadata = self.metadata[spatial_filter]

        return MicrobiomeDataset(
            metadata=filtered_metadata,
            data_source=f"{self.data_source} (spatially filtered)",
        )

    def export_for_h3_integration(self) -> dict[str, Any]:
        """
        Export data in format suitable for H3 spatial integration.

        Returns:
            Dictionary with coordinates and associated data
        """
        diversity_metrics = self.get_diversity_metrics()

        export_data = {
            "coordinates": self.get_coordinates(),
            "sample_ids": list(self.metadata.index),
            "diversity_metrics": (
                diversity_metrics.to_dict("index")
                if not diversity_metrics.empty
                else {}
            ),
            "environmental_metadata": self.metadata.select_dtypes(
                include=[np.number]
            ).to_dict("index"),
            "data_source": self.data_source,
        }

        logger.info(
            f"Exported {len(export_data['coordinates'])} samples for H3 integration"
        )
        return export_data

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.metadata)

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"MicrobiomeDataset(samples={len(self.metadata)}, "
            f"source='{self.data_source}')"
        )
