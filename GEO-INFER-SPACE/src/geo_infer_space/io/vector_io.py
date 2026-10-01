"""
Vector data I/O operations for various geospatial formats.

This module provides reading and writing capabilities for vector data
including GeoJSON, Shapefile, GeoPackage, KML, and other formats
supported by GeoPandas and Fiona.
"""

import logging
from pathlib import Path
from typing import Any
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import json

logger = logging.getLogger(__name__)

# Supported vector formats
SUPPORTED_VECTOR_FORMATS = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".kml": "KML",
    ".gml": "GML",
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".parquet": "Parquet",
    ".feather": "Feather",
}


class VectorReader:
    """Reader class for vector geospatial data."""

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_VECTOR_FORMATS

    def read(self, file_path: str | Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """
        Read vector data from file.

        Args:
            file_path: Path to vector data file
            **kwargs: Additional parameters for specific formats

        Returns:
            GeoDataFrame with vector data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")

        try:
            if file_ext in [".geojson", ".json"]:
                return self._read_geojson(file_path, **kwargs)
            elif file_ext == ".csv":
                return self._read_csv(file_path, **kwargs)
            elif file_ext == ".xlsx":
                return self._read_excel(file_path, **kwargs)
            elif file_ext == ".parquet":
                return self._read_parquet(file_path, **kwargs)
            elif file_ext == ".feather":
                return self._read_feather(file_path, **kwargs)
            else:
                # Use GeoPandas for standard formats
                return gpd.read_file(file_path, **kwargs)

        except Exception as e:
            logger.error(f"Failed to read vector file {file_path}: {e}")
            raise

    def _read_geojson(self, file_path: Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read GeoJSON file with enhanced error handling."""
        try:
            with open(file_path, encoding="utf-8") as f:
                geojson_data = json.load(f)

            # Validate GeoJSON structure
            if not isinstance(geojson_data, dict):
                raise ValueError("Invalid GeoJSON: root must be an object")

            geojson_type = geojson_data.get("type")

            if geojson_type == "FeatureCollection":
                features = geojson_data.get("features", [])
                if not features:
                    logger.warning("GeoJSON FeatureCollection is empty")
                    return gpd.GeoDataFrame()

                return gpd.GeoDataFrame.from_features(features, **kwargs)

            elif geojson_type == "Feature":
                return gpd.GeoDataFrame.from_features([geojson_data], **kwargs)

            elif geojson_type in [
                "Point",
                "LineString",
                "Polygon",
                "MultiPoint",
                "MultiLineString",
                "MultiPolygon",
            ]:
                # Single geometry
                geom = shape(geojson_data)
                return gpd.GeoDataFrame([{"geometry": geom}], **kwargs)

            else:
                raise ValueError(f"Unsupported GeoJSON type: {geojson_type}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e

    def _read_csv(self, file_path: Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read CSV file and convert to GeoDataFrame."""
        # Extract geometry-related parameters
        x_col = kwargs.pop("x_col", "longitude")
        y_col = kwargs.pop("y_col", "latitude")
        z_col = kwargs.pop("z_col", None)
        crs = kwargs.pop("crs", "EPSG:4326")

        # Read CSV
        df = pd.read_csv(file_path, **kwargs)

        # Check for coordinate columns
        if x_col not in df.columns or y_col not in df.columns:
            # Try alternative column names
            possible_x = ["lon", "lng", "x", "easting"]
            possible_y = ["lat", "y", "northing"]

            x_col = next((col for col in possible_x if col in df.columns), None)
            y_col = next((col for col in possible_y if col in df.columns), None)

            if not x_col or not y_col:
                raise ValueError("Could not find coordinate columns in CSV")

        # Create geometries
        if z_col and z_col in df.columns:
            geometries = gpd.points_from_xy(df[x_col], df[y_col], df[z_col])
        else:
            geometries = gpd.points_from_xy(df[x_col], df[y_col])

        # Remove coordinate columns from dataframe
        df = df.drop(
            columns=[col for col in [x_col, y_col, z_col] if col in df.columns]
        )

        return gpd.GeoDataFrame(df, geometry=geometries, crs=crs)

    def _read_excel(self, file_path: Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read Excel file and convert to GeoDataFrame."""
        # Similar to CSV but using pandas.read_excel
        x_col = kwargs.pop("x_col", "longitude")
        y_col = kwargs.pop("y_col", "latitude")
        z_col = kwargs.pop("z_col", None)
        crs = kwargs.pop("crs", "EPSG:4326")

        df = pd.read_excel(file_path, **kwargs)

        if x_col not in df.columns or y_col not in df.columns:
            possible_x = ["lon", "lng", "x", "easting"]
            possible_y = ["lat", "y", "northing"]

            x_col = next((col for col in possible_x if col in df.columns), None)
            y_col = next((col for col in possible_y if col in df.columns), None)

            if not x_col or not y_col:
                raise ValueError("Could not find coordinate columns in Excel file")

        if z_col and z_col in df.columns:
            geometries = gpd.points_from_xy(df[x_col], df[y_col], df[z_col])
        else:
            geometries = gpd.points_from_xy(df[x_col], df[y_col])

        df = df.drop(
            columns=[col for col in [x_col, y_col, z_col] if col in df.columns]
        )

        return gpd.GeoDataFrame(df, geometry=geometries, crs=crs)

    def _read_parquet(self, file_path: Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read Parquet file with geospatial data."""
        try:
            # Try reading as GeoParquet first
            return gpd.read_parquet(file_path, **kwargs)
        except Exception as exc:
            # Fallback to regular parquet with geometry reconstruction
            df = pd.read_parquet(file_path, **kwargs)

            # Look for WKT or WKB geometry columns
            geom_col = None
            for col in df.columns:
                if (
                    "geom" in col.lower()
                    or "wkt" in col.lower()
                    or "wkb" in col.lower()
                ):
                    geom_col = col
                    break

            if geom_col:
                from shapely import wkt, wkb

                # Try WKT first, then WKB
                try:
                    geometries = df[geom_col].apply(wkt.loads)
                except Exception:
                    try:
                        geometries = df[geom_col].apply(wkb.loads)
                    except Exception as wkb_exc:
                        raise ValueError("Could not parse geometry column") from wkb_exc

                df = df.drop(columns=[geom_col])
                return gpd.GeoDataFrame(df, geometry=geometries)
            else:
                raise ValueError("No geometry column found in Parquet file") from exc

    def _read_feather(self, file_path: Path, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read Feather file with geospatial data."""
        df = pd.read_feather(file_path, **kwargs)

        # Similar geometry reconstruction as Parquet
        geom_col = None
        for col in df.columns:
            if "geom" in col.lower() or "wkt" in col.lower():
                geom_col = col
                break

        if geom_col:
            from shapely import wkt

            geometries = df[geom_col].apply(wkt.loads)
            df = df.drop(columns=[geom_col])
            return gpd.GeoDataFrame(df, geometry=geometries)
        else:
            raise ValueError("No geometry column found in Feather file")


class VectorWriter:
    """Writer class for vector geospatial data."""

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_VECTOR_FORMATS

    def write(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """
        Write GeoDataFrame to file.

        Args:
            gdf: GeoDataFrame to write
            file_path: Output file path
            **kwargs: Additional parameters for specific formats
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")

        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_ext in [".geojson", ".json"]:
                self._write_geojson(gdf, file_path, **kwargs)
            elif file_ext == ".csv":
                self._write_csv(gdf, file_path, **kwargs)
            elif file_ext == ".xlsx":
                self._write_excel(gdf, file_path, **kwargs)
            elif file_ext == ".parquet":
                self._write_parquet(gdf, file_path, **kwargs)
            elif file_ext == ".feather":
                self._write_feather(gdf, file_path, **kwargs)
            else:
                # Use GeoPandas for standard formats
                gdf.to_file(file_path, **kwargs)

            logger.info(f"Successfully wrote {len(gdf)} features to {file_path}")

        except Exception as e:
            logger.error(f"Failed to write vector file {file_path}: {e}")
            raise

    def _write_geojson(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write GeoDataFrame as GeoJSON with formatting options."""
        # Set default formatting options
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("ensure_ascii", False)

        gdf.to_file(file_path, driver="GeoJSON", **kwargs)

    def _write_csv(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write GeoDataFrame as CSV with coordinate columns."""
        df = gdf.copy()

        # Extract coordinates
        df["longitude"] = df.geometry.x
        df["latitude"] = df.geometry.y

        # Add Z coordinate if available
        if df.geometry.iloc[0].has_z:
            df["elevation"] = df.geometry.z

        # Remove geometry column
        df = df.drop(columns=["geometry"])

        df.to_csv(file_path, index=False, **kwargs)

    def _write_excel(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write GeoDataFrame as Excel with coordinate columns."""
        df = gdf.copy()

        df["longitude"] = df.geometry.x
        df["latitude"] = df.geometry.y

        if df.geometry.iloc[0].has_z:
            df["elevation"] = df.geometry.z

        df = df.drop(columns=["geometry"])

        df.to_excel(file_path, index=False, **kwargs)

    def _write_parquet(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write GeoDataFrame as GeoParquet."""
        try:
            # Try writing as GeoParquet
            gdf.to_parquet(file_path, **kwargs)
        except Exception:
            # Fallback to regular parquet with WKT geometry
            df = gdf.copy()
            df["geometry_wkt"] = df.geometry.to_wkt()
            df = df.drop(columns=["geometry"])
            df.to_parquet(file_path, **kwargs)

    def _write_feather(
        self,
        gdf: gpd.GeoDataFrame,
        file_path: Path,
        **kwargs: Any,
    ) -> None:
        """Write GeoDataFrame as Feather with WKT geometry."""
        df = gdf.copy()
        df["geometry_wkt"] = df.geometry.to_wkt()
        df = df.drop(columns=["geometry"])
        df.to_feather(file_path, **kwargs)


# Convenience functions
def read_vector_file(file_path: str | Path, **kwargs: Any) -> gpd.GeoDataFrame:
    """
    Read vector data from file using appropriate reader.

    Args:
        file_path: Path to vector data file
        **kwargs: Additional parameters for reading

    Returns:
        GeoDataFrame with vector data
    """
    reader = VectorReader()
    return reader.read(file_path, **kwargs)


def write_vector_file(
    gdf: gpd.GeoDataFrame,
    file_path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Write GeoDataFrame to file using appropriate writer.

    Args:
        gdf: GeoDataFrame to write
        file_path: Output file path
        **kwargs: Additional parameters for writing
    """
    writer = VectorWriter()
    writer.write(gdf, file_path, **kwargs)


def supported_vector_formats() -> dict[str, str]:
    """
    Get dictionary of supported vector formats.

    Returns:
        Dictionary mapping file extensions to format names
    """
    return SUPPORTED_VECTOR_FORMATS.copy()


def detect_vector_format(file_path: str | Path) -> str | None:
    """
    Detect vector format from file extension.

    Args:
        file_path: Path to vector file

    Returns:
        Format name or None if not supported
    """
    file_ext = Path(file_path).suffix.lower()
    return SUPPORTED_VECTOR_FORMATS.get(file_ext)


def validate_vector_file(file_path: str | Path) -> dict[str, Any]:
    """
    Validate vector file and return metadata.

    Args:
        file_path: Path to vector file

    Returns:
        Dictionary with validation results and metadata
    """
    file_path = Path(file_path)

    result: dict[str, Any] = {
        "valid": False,
        "format": None,
        "error": None,
        "metadata": {},
    }

    try:
        # Check if file exists
        if not file_path.exists():
            result["error"] = "File does not exist"
            return result

        # Detect format
        format_name = detect_vector_format(file_path)
        if not format_name:
            result["error"] = f"Unsupported format: {file_path.suffix}"
            return result

        result["format"] = format_name

        # Try to read file
        gdf = read_vector_file(file_path)

        # Extract metadata
        result["metadata"] = {
            "num_features": len(gdf),
            "columns": list(gdf.columns),
            "geometry_types": list(gdf.geometry.geom_type.unique()),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else None,
        }

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
