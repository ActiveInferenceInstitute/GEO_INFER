"""
File connectors for GEO-INFER-DATA.

This module provides comprehensive file connectivity for various geospatial
file formats including vector, raster, and tabular data formats.
"""

import logging
import importlib.util
from typing import Any, cast
from collections.abc import Iterator, AsyncGenerator
from pathlib import Path
import zipfile
import tarfile
import gzip

import geopandas as gpd
import pandas as pd
import numpy as np

from ..models.schemas import DatasetMetadata
from ..utils.format_detection import FormatDetector
from ..utils.dependencies import require_dependency


HAS_FIONA = importlib.util.find_spec("fiona") is not None


logger = logging.getLogger(__name__)


def _safe_extract_zip(archive: zipfile.ZipFile, extract_to: Path) -> None:
    """Extract a zip archive while rejecting path-traversal member names.

    Raises:
        ValueError: If any member path escapes the target directory.
    """
    target = extract_to.resolve()
    for member in archive.infolist():
        member_path = (extract_to / member.filename).resolve()
        if not member_path.is_relative_to(target):
            raise ValueError(f"Unsafe archive member path: {member.filename!r}")
    archive.extractall(extract_to)


def _safe_extract_tar(archive: tarfile.TarFile, extract_to: Path) -> None:
    """Extract a tar archive while rejecting path-traversal members.

    Uses Python 3.12+'s ``filter="data"`` (PEP 706) and additionally validates
    every member path against the target directory.

    Raises:
        ValueError: If any member path escapes the target directory.
    """
    target = extract_to.resolve()
    for member in archive.getmembers():
        member_path = (extract_to / member.name).resolve()
        if not member_path.is_relative_to(target):
            raise ValueError(f"Unsafe archive member path: {member.name!r}")
    archive.extractall(extract_to, filter="data")


class FileConnector:
    """
    Universal file connector for geospatial data formats.

    This class provides connectivity to various file formats including
    vector (GeoJSON, Shapefile, GeoPackage), raster (GeoTIFF, NetCDF),
    and tabular formats with automatic format detection and optimization.

    Args:
        base_path: Base directory for file operations
        format_detector: Format detection instance

    Examples:
        >>> # Read geospatial data
        >>> connector = FileConnector(base_path='/data')
        >>> gdf = await connector.read_geospatial('sensors.geojson')
        >>>
        >>> # Write processed data
        >>> await connector.write_geospatial(gdf, 'processed_data.gpkg', metadata)
        >>>
        >>> # List files by pattern
        >>> files = connector.list_files('*.geojson', recursive=True)
    """

    def __init__(
        self, base_path: str = ".", format_detector: FormatDetector | None = None
    ):
        self.base_path = Path(base_path)
        self.format_detector = format_detector or FormatDetector()

        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FileConnector for {base_path}")

    async def read_geospatial(
        self,
        file_path: str | Path,
        layer: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | gpd.GeoDataFrame | np.ndarray:
        """
        Read geospatial data from file.

        Args:
            file_path: Path to the file
            layer: Layer name for multi-layer files
            **kwargs: Additional reading parameters

        Returns:
            Data in appropriate format (DataFrame, GeoDataFrame, or array)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Reading geospatial data from {file_path}")

        try:
            # Detect format
            detected_format = self.format_detector.detect_from_path(file_path)

            # Read based on format
            if detected_format.name in ["GEOJSON", "SHAPEFILE", "GEOPACKAGE", "KML"]:
                return self._read_vector_file(
                    file_path, detected_format, layer, **kwargs
                )
            elif detected_format.name in ["GEOTIFF", "NETCDF", "HDF5"]:
                return self._read_raster_file(file_path, detected_format, **kwargs)
            elif detected_format.name in ["CSV", "PARQUET"]:
                return self._read_tabular_file(file_path, detected_format, **kwargs)
            else:
                # Fallback to pandas
                return pd.read_csv(file_path, **kwargs)

        except Exception as e:
            logger.error(f"Failed to read geospatial file {file_path}: {e}")
            raise

    def _read_vector_file(
        self,
        file_path: Path,
        format_type: Any,
        layer: str | None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Read vector geospatial file."""
        try:
            if format_type.name == "SHAPEFILE":
                # Handle shapefile (may have multiple files)
                gdf = gpd.read_file(str(file_path), **kwargs)
            elif format_type.name == "GEOPACKAGE":
                gdf = gpd.read_file(str(file_path), layer=layer, **kwargs)
            else:
                gdf = gpd.read_file(str(file_path), **kwargs)

            logger.info(f"Read {len(gdf)} features from {format_type.name} file")
            return gdf

        except Exception as e:
            logger.error(f"Failed to read vector file: {e}")
            raise

    def _read_raster_file(
        self, file_path: Path, format_type: Any, **kwargs: Any
    ) -> np.ndarray:
        """Read raster file."""
        rasterio = require_dependency("rasterio", "raster")
        try:
            if format_type.name == "GEOTIFF":
                with rasterio.open(file_path) as src:
                    # Read all bands
                    array = src.read()
                    logger.info(f"Read raster with shape {array.shape}")
                    return cast(np.ndarray, array)
            else:
                # For other raster formats, use rasterio if possible
                with rasterio.open(file_path) as src:
                    array = src.read()
                    return cast(np.ndarray, array)

        except Exception as e:
            logger.error(f"Failed to read raster file: {e}")
            raise

    def _read_tabular_file(
        self, file_path: Path, format_type: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """Read tabular file."""
        try:
            if format_type.name == "PARQUET":
                df = pd.read_parquet(file_path, **kwargs)
            elif format_type.name == "CSV":
                df = pd.read_csv(file_path, **kwargs)
            else:
                df = pd.read_csv(file_path, **kwargs)

            logger.info(f"Read {len(df)} rows from tabular file")
            return df

        except Exception as e:
            logger.error(f"Failed to read tabular file: {e}")
            raise

    async def write_geospatial(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame | np.ndarray,
        file_path: str | Path,
        metadata: DatasetMetadata | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Write geospatial data to file.

        Args:
            data: Data to write
            file_path: Output file path
            metadata: Dataset metadata
            **kwargs: Additional writing parameters

        Returns:
            Path to written file
        """
        file_path = Path(file_path)

        logger.info(f"Writing geospatial data to {file_path}")

        try:
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Detect or determine format from extension
            if file_path.suffix:
                format_hint = file_path.suffix.lower()
                if format_hint == ".geojson":
                    self._write_geojson(data, file_path, **kwargs)
                elif format_hint == ".gpkg":
                    self._write_geopackage(data, file_path, **kwargs)
                elif format_hint in [".tif", ".tiff"]:
                    self._write_geotiff(data, file_path, **kwargs)
                elif format_hint == ".parquet":
                    self._write_parquet(data, file_path, **kwargs)
                elif format_hint == ".csv":
                    self._write_csv(data, file_path, **kwargs)
                else:
                    self._write_generic(data, file_path, **kwargs)
            else:
                # Default to GeoJSON for geospatial data
                if isinstance(data, gpd.GeoDataFrame):
                    file_path = file_path.with_suffix(".geojson")
                    self._write_geojson(data, file_path, **kwargs)
                else:
                    file_path = file_path.with_suffix(".csv")
                    self._write_csv(data, file_path, **kwargs)

            # Write metadata if provided
            if metadata:
                await self._write_metadata(file_path, metadata)

            logger.info(f"Successfully wrote data to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to write geospatial file: {e}")
            raise

    def _write_geojson(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write data as GeoJSON."""
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(file_path, driver="GeoJSON", **kwargs)
        else:
            # Convert DataFrame to GeoDataFrame if possible
            if "latitude" in data.columns and "longitude" in data.columns:
                gdf = gpd.GeoDataFrame(
                    data,
                    geometry=gpd.points_from_xy(data.longitude, data.latitude),
                    crs="EPSG:4326",
                )
                gdf.to_file(file_path, driver="GeoJSON", **kwargs)
            else:
                # Save as regular JSON
                data.to_json(file_path, **kwargs)

    def _write_geopackage(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write data as GeoPackage."""
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(file_path, driver="GPKG", **kwargs)
        else:
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(data)
            gdf.to_file(file_path, driver="GPKG", **kwargs)

    def _write_geotiff(self, data: np.ndarray, file_path: Path, **kwargs: Any) -> None:
        """Write array as GeoTIFF."""
        # Implementation for GeoTIFF writing
        # Would need coordinate reference and geotransform information
        np.save(file_path.with_suffix(".npy"), data)

    def _write_parquet(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write data as Parquet."""
        if isinstance(data, gpd.GeoDataFrame):
            data.to_parquet(file_path, **kwargs)
        else:
            data.to_parquet(file_path, **kwargs)

    def _write_csv(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write data as CSV."""
        if isinstance(data, gpd.GeoDataFrame):
            # Drop geometry column for CSV
            csv_data = data.drop("geometry", axis=1)
            csv_data.to_csv(file_path, **kwargs)
        else:
            data.to_csv(file_path, **kwargs)

    def _write_generic(self, data: Any, file_path: Path, **kwargs: Any) -> None:
        """Write data in generic format."""
        if hasattr(data, "to_csv"):
            data.to_csv(file_path, **kwargs)
        elif hasattr(data, "to_json"):
            data.to_json(file_path, **kwargs)
        else:
            # Fallback to pickle
            import pickle

            with open(file_path, "wb") as f:
                pickle.dump(data, f)

    async def _write_metadata(self, file_path: Path, metadata: DatasetMetadata) -> None:
        """Write metadata file alongside data."""
        metadata_path = file_path.with_suffix(".json")

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

        logger.info(f"Metadata written to {metadata_path}")

    def list_files(
        self,
        pattern: str = "*",
        recursive: bool = False,
        file_types: list[str] | None = None,
    ) -> list[Path]:
        """
        List files matching pattern.

        Args:
            pattern: File pattern (e.g., '*.geojson')
            recursive: Whether to search recursively
            file_types: List of file extensions to include

        Returns:
            List of matching file paths
        """
        if recursive:
            files = list(self.base_path.rglob(pattern))
        else:
            files = list(self.base_path.glob(pattern))

        # Filter by file types if specified
        if file_types:
            file_types = [ft.lower().strip(".") for ft in file_types]
            files = [f for f in files if f.suffix.lower().strip(".") in file_types]

        logger.info(f"Found {len(files)} files matching pattern {pattern}")
        return files

    async def scan_directory(
        self, directory: str | Path | None = None, recursive: bool = True
    ) -> dict[str, Any]:
        """
        Scan directory for geospatial files.

        Args:
            directory: Directory to scan (default: base_path)
            recursive: Whether to scan recursively

        Returns:
            Scan results with file statistics
        """
        scan_path = Path(directory) if directory else self.base_path

        if not scan_path.exists():
            return {"error": f"Directory not found: {scan_path}"}

        logger.info(f"Scanning directory: {scan_path}")

        file_stats: dict[str, Any] = {
            "total_files": 0,
            "geospatial_files": 0,
            "by_format": {},
            "by_size": {"small": 0, "medium": 0, "large": 0},
            "total_size": 0,
        }

        # Supported geospatial formats
        geospatial_extensions = {
            ".geojson",
            ".json",
            ".shp",
            ".gpkg",
            ".kml",
            ".kmz",
            ".tif",
            ".tiff",
            ".nc",
            ".h5",
            ".hdf5",
            ".parquet",
            ".parq",
            ".csv",
            ".xlsx",
            ".xls",
            ".wkt",
            ".wkb",
        }

        for file_path in scan_path.rglob("*") if recursive else scan_path.glob("*"):
            if file_path.is_file():
                file_stats["total_files"] += 1
                file_size = file_path.stat().st_size
                file_stats["total_size"] += file_size

                # Categorize by size
                if file_size < 1024 * 1024:  # < 1MB
                    file_stats["by_size"]["small"] += 1
                elif file_size < 100 * 1024 * 1024:  # < 100MB
                    file_stats["by_size"]["medium"] += 1
                else:
                    file_stats["by_size"]["large"] += 1

                # Check if geospatial file
                if file_path.suffix.lower() in geospatial_extensions:
                    file_stats["geospatial_files"] += 1

                    # Count by format
                    format_name = file_path.suffix.lower().strip(".")
                    file_stats["by_format"][format_name] = (
                        file_stats["by_format"].get(format_name, 0) + 1
                    )

        # Convert size to MB
        file_stats["total_size_mb"] = file_stats["total_size"] / (1024 * 1024)

        logger.info(
            f"Directory scan completed: {file_stats['geospatial_files']} geospatial files found"
        )
        return file_stats

    async def compress_files(
        self,
        file_paths: list[str | Path],
        archive_path: str | Path,
        compression: str = "zip",
    ) -> str:
        """
        Compress multiple files into archive.

        Args:
            file_paths: List of files to compress
            archive_path: Output archive path
            compression: Compression type ('zip', 'tar', 'gzip', 'bz2')

        Returns:
            Path to created archive
        """
        archive_path = Path(archive_path)

        logger.info(f"Creating {compression} archive with {len(file_paths)} files")

        try:
            if compression == "zip":
                with zipfile.ZipFile(
                    archive_path, "w", zipfile.ZIP_DEFLATED
                ) as archive:
                    for file_path in file_paths:
                        archive.write(Path(file_path), Path(file_path).name)

            elif compression == "tar":
                with tarfile.open(archive_path, "w") as archive:
                    for file_path in file_paths:
                        archive.add(Path(file_path), arcname=Path(file_path).name)

            elif compression == "gzip":
                # For single file compression
                if len(file_paths) == 1:
                    with open(file_paths[0], "rb") as f_in:
                        with gzip.open(archive_path, "wb") as f_out:
                            f_out.writelines(f_in)

            logger.info(f"Archive created: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            raise

    async def extract_archive(
        self,
        archive_path: str | Path,
        extract_to: str | Path | None = None,
    ) -> list[str]:
        """
        Extract files from archive.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to (default: same as archive)

        Returns:
            List of extracted file paths
        """
        archive_path = Path(archive_path)
        extract_to = Path(extract_to) if extract_to else archive_path.parent

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        logger.info(f"Extracting archive {archive_path} to {extract_to}")

        extracted_files = []

        try:
            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path, "r") as archive:
                    _safe_extract_zip(archive, extract_to)
                    extracted_files = [
                        str(extract_to / name) for name in archive.namelist()
                    ]

            elif archive_path.suffix.lower() in [".tar", ".gz", ".bz2", ".xz"]:
                with tarfile.open(archive_path, "r") as archive:
                    _safe_extract_tar(archive, extract_to)
                    extracted_files = [
                        str(extract_to / member.name) for member in archive.getmembers()
                    ]

            elif archive_path.suffix.lower() == ".gz":
                # Single file compression
                output_path = extract_to / archive_path.stem
                with gzip.open(archive_path, "rb") as f_in:
                    with open(output_path, "wb") as f_out:
                        f_out.write(f_in.read())
                extracted_files = [str(output_path)]

            logger.info(f"Extracted {len(extracted_files)} files")
            return extracted_files

        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            raise


class StreamingFileConnector:
    """
    Streaming file connector for large file processing.

    This class provides streaming capabilities for processing large files
    without loading them entirely into memory.
    """

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        logger.info(f"Initialized StreamingFileConnector with chunk_size={chunk_size}")

    async def read_csv_streaming(
        self, file_path: str | Path, **kwargs: Any
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Read CSV file in streaming mode.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv parameters

        Yields:
            DataFrame chunks
        """
        logger.info(f"Streaming CSV file: {file_path}")

        try:
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs):
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream CSV file: {e}")
            raise

    async def read_parquet_streaming(
        self, file_path: str | Path, **kwargs: Any
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Read Parquet file in streaming mode.

        Args:
            file_path: Path to Parquet file
            **kwargs: Additional parameters

        Yields:
            DataFrame chunks
        """
        logger.info(f"Streaming Parquet file: {file_path}")

        try:
            # Read in chunks
            df = pd.read_parquet(file_path, **kwargs)

            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i : i + self.chunk_size]
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream Parquet file: {e}")
            raise

    async def write_csv_streaming(
        self,
        data_generator: Iterator[pd.DataFrame],
        file_path: str | Path,
        **kwargs: Any,
    ) -> str:
        """
        Write data to CSV in streaming mode.

        Args:
            data_generator: Generator yielding DataFrames
            file_path: Output file path
            **kwargs: Additional pandas to_csv parameters

        Returns:
            Path to written file
        """
        file_path = Path(file_path)

        logger.info(f"Streaming CSV write to: {file_path}")

        try:
            # Write first chunk to create file with headers
            first_chunk = True

            for chunk in data_generator:
                if first_chunk:
                    chunk.to_csv(file_path, index=False, **kwargs)
                    first_chunk = False
                else:
                    chunk.to_csv(
                        file_path, mode="a", header=False, index=False, **kwargs
                    )

            logger.info(f"Successfully wrote streaming CSV: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to write streaming CSV: {e}")
            raise
