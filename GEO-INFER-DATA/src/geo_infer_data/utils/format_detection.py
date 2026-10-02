"""
Format detection utilities for GEO-INFER-DATA.

This module provides automatic format detection capabilities for various
geospatial data formats including vector, raster, and tabular data.
"""

import logging
from typing import Any
from pathlib import Path
import json
import zipfile

import geopandas as gpd
import pandas as pd
import numpy as np

from ..models.schemas import DataFormat
from .dependencies import MissingOptionalDependency, require_dependency


logger = logging.getLogger(__name__)


class FormatDetector:
    """
    Automatic format detection for geospatial data.

    This class provides comprehensive format detection capabilities for
    various geospatial data formats including vector, raster, and tabular data.

    Examples:
        >>> detector = FormatDetector()
        >>>
        >>> # Detect format from file path
        >>> format_type = detector.detect_from_path('/path/to/data.geojson')
        >>> print(f"Detected format: {format_type}")
        >>>
        >>> # Detect format from data content
        >>> format_type = detector.detect_format(data)
        >>> print(f"Detected format: {format_type}")
    """

    def __init__(self) -> None:
        self.format_signatures = {
            DataFormat.GEOJSON: self._detect_geojson,
            DataFormat.SHAPEFILE: self._detect_shapefile,
            DataFormat.GEOPACKAGE: self._detect_geopackage,
            DataFormat.GEOTIFF: self._detect_geotiff,
            DataFormat.NETCDF: self._detect_netcdf,
            DataFormat.CSV: self._detect_csv,
            DataFormat.PARQUET: self._detect_parquet,
            DataFormat.KML: self._detect_kml,
            DataFormat.WKT: self._detect_wkt,
            DataFormat.HDF5: self._detect_hdf5,
        }

        logger.info("Initialized FormatDetector")

    def detect_from_path(self, file_path: str | Path) -> DataFormat:
        """
        Detect format from file path.

        Args:
            file_path: Path to the data file

        Returns:
            Detected data format

        Raises:
            ValueError: If format cannot be detected
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Check file extension
        extension = file_path.suffix.lower()

        extension_map = {
            ".geojson": DataFormat.GEOJSON,
            ".json": DataFormat.GEOJSON,
            ".shp": DataFormat.SHAPEFILE,
            ".gpkg": DataFormat.GEOPACKAGE,
            ".tif": DataFormat.GEOTIFF,
            ".tiff": DataFormat.GEOTIFF,
            ".nc": DataFormat.NETCDF,
            ".csv": DataFormat.CSV,
            ".parquet": DataFormat.PARQUET,
            ".parq": DataFormat.PARQUET,
            ".kml": DataFormat.KML,
            ".kmz": DataFormat.KML,
            ".wkt": DataFormat.WKT,
            ".h5": DataFormat.HDF5,
            ".hdf5": DataFormat.HDF5,
        }

        if extension in extension_map:
            detected_format = extension_map[extension]

            # Verify format by attempting to read
            try:
                if detected_format == DataFormat.SHAPEFILE:
                    # Check for associated files
                    base_name = file_path.stem
                    shp_dir = file_path.parent
                    required_files = [".shp", ".shx", ".dbf"]
                    missing_files = []

                    for ext in required_files:
                        if not (shp_dir / f"{base_name}{ext}").exists():
                            missing_files.append(ext)

                    if missing_files:
                        logger.warning(
                            f"Shapefile missing associated files: {missing_files}"
                        )
                elif detected_format == DataFormat.GEOJSON:
                    with open(file_path) as f:
                        data = json.load(f)
                        if not self._is_geojson_structure(data):
                            raise ValueError(
                                "File extension suggests GeoJSON but structure is invalid"
                            )
                elif detected_format == DataFormat.GEOTIFF:
                    rasterio = require_dependency("rasterio", "raster")
                    with rasterio.open(file_path) as src:
                        _ = src.count  # Opening and reading metadata verifies access.

                return detected_format

            except MissingOptionalDependency:
                raise
            except Exception as e:
                logger.warning(f"Format verification failed for {detected_format}: {e}")
                # Fall back to content-based detection

        # Fall back to content-based detection
        return self.detect_from_content(file_path)

    def detect_from_content(self, file_path: str | Path) -> DataFormat:
        """
        Detect format from file content.

        Args:
            file_path: Path to the data file

        Returns:
            Detected data format
        """
        file_path = Path(file_path)

        # Try to read as different formats
        for format_type, detector_func in self.format_signatures.items():
            try:
                if detector_func(file_path):
                    return format_type
            except MissingOptionalDependency:
                raise
            except Exception as e:
                logger.debug(f"Format {format_type} detection failed: {e}")
                continue

        # Default to unknown
        logger.warning(f"Could not detect format for {file_path}")
        return DataFormat.CSV  # Safe default

    def detect_format(self, data: Any) -> DataFormat:
        """
        Detect format from data object.

        Args:
            data: Data object to analyze

        Returns:
            Detected data format
        """
        # Check data type and structure
        if isinstance(data, gpd.GeoDataFrame):
            return DataFormat.GEOPACKAGE
        elif isinstance(data, pd.DataFrame):
            return DataFormat.CSV
        elif isinstance(data, np.ndarray):
            # Could be raster data
            return DataFormat.GEOTIFF
        elif isinstance(data, dict):
            if self._is_geojson_structure(data):
                return DataFormat.GEOJSON
            else:
                return DataFormat.CSV
        elif isinstance(data, (str, Path)):
            return self.detect_from_path(data)
        else:
            return DataFormat.CSV  # Safe default

    def _detect_geojson(self, file_path: Path) -> bool:
        """Detect GeoJSON format."""
        try:
            with open(file_path) as f:
                data = json.load(f)
                return self._is_geojson_structure(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

    def _detect_shapefile(self, file_path: Path) -> bool:
        """Detect Shapefile format."""
        # Check if it's a .shp file and has associated files
        if file_path.suffix.lower() != ".shp":
            return False

        base_name = file_path.stem
        shp_dir = file_path.parent

        # Check for required associated files
        required_files = [".shp", ".shx", ".dbf"]
        for ext in required_files:
            if not (shp_dir / f"{base_name}{ext}").exists():
                return False

        # Try to read as shapefile
        try:
            gpd.read_file(file_path)
            return True
        except Exception:
            return False

    def _detect_geopackage(self, file_path: Path) -> bool:
        """Detect GeoPackage format."""
        try:
            gpd.read_file(file_path)
            return True
        except Exception:
            return False

    def _detect_geotiff(self, file_path: Path) -> bool:
        """Detect GeoTIFF format."""
        with file_path.open("rb") as file:
            if file.read(4) not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                return False
        rasterio = require_dependency("rasterio", "raster")
        try:
            with rasterio.open(file_path):
                return True
        except Exception:
            return False

    def _detect_netcdf(self, file_path: Path) -> bool:
        """Detect NetCDF format."""
        try:
            import xarray as xr

            xr.open_dataset(file_path)
            return True
        except Exception:
            return False

    def _detect_csv(self, file_path: Path) -> bool:
        """Detect CSV format."""
        try:
            pd.read_csv(file_path, nrows=5)
            return True
        except Exception:
            return False

    def _detect_parquet(self, file_path: Path) -> bool:
        """Detect Parquet format."""
        try:
            pd.read_parquet(file_path)
            return True
        except Exception:
            return False

    def _detect_kml(self, file_path: Path) -> bool:
        """Detect KML format."""
        try:
            gpd.read_file(file_path)
            return True
        except Exception:
            # Try as KMZ (zipped KML)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    kml_files = [f for f in zip_ref.namelist() if f.endswith(".kml")]
                    return len(kml_files) > 0
            except Exception:
                return False

    def _detect_wkt(self, file_path: Path) -> bool:
        """Detect WKT format."""
        try:
            with open(file_path) as f:
                content = f.read()
                # Simple WKT detection
                return (
                    "POINT" in content
                    or "POLYGON" in content
                    or "LINESTRING" in content
                )
        except Exception:
            return False

    def _detect_hdf5(self, file_path: Path) -> bool:
        """Detect HDF5 format."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
                return header.startswith(b"\x89HDF\r\n\x1a\n")
        except Exception:
            return False

    def _is_geojson_structure(self, data: dict[str, Any]) -> bool:
        """Check if dictionary has GeoJSON structure."""
        # Check for GeoJSON required fields
        if "type" not in data:
            return False

        # Check for FeatureCollection, Feature, or geometry
        if data["type"] in ["FeatureCollection", "Feature"]:
            return True

        # Check for geometry object
        if data["type"] in [
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
        ]:
            return "coordinates" in data

        return False

    def get_supported_formats(self) -> list[DataFormat]:
        """Get list of supported formats."""
        return list(self.format_signatures.keys())

    def validate_format(
        self, file_path: str | Path, expected_format: DataFormat
    ) -> bool:
        """
        Validate that file matches expected format.

        Args:
            file_path: Path to file
            expected_format: Expected format

        Returns:
            True if format matches
        """
        detected_format = self.detect_from_path(file_path)
        return detected_format == expected_format
