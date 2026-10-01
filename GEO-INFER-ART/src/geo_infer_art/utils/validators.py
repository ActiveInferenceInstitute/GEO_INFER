"""
Validation functions for file paths and geospatial data.
"""

import os
import geopandas as gpd
import numpy as np


def validate_file_path(file_path: str, extensions: list[str] | None = None) -> None:
    """
    Validate that a file path exists and has the correct extension.

    Args:
        file_path: Path to the file to validate
        extensions: List of valid file extensions (e.g., ['.geojson', '.json'])

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file has an invalid extension
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if extensions:
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in extensions:
            raise ValueError(
                f"Invalid file extension: {ext}. Expected one of: {extensions}"
            )


def validate_geospatial_data(data: gpd.GeoDataFrame | np.ndarray) -> None:
    """
    Validate that the data is a valid GeoDataFrame or numpy array.

    Args:
        data: Data to validate

    Raises:
        ValueError: If the data is not a valid GeoDataFrame or numpy array
    """
    if isinstance(data, gpd.GeoDataFrame):
        # Check if GeoDataFrame has a geometry column
        if not data.geometry.any():
            raise ValueError("GeoDataFrame has no valid geometries")

        # Check if GeoDataFrame has a CRS
        if data.crs is None:
            raise ValueError("GeoDataFrame has no CRS (Coordinate Reference System)")

    elif isinstance(data, np.ndarray):
        # Check if numpy array has valid dimensions
        if data.ndim not in [2, 3]:
            raise ValueError(
                f"Invalid array dimensions: {data.ndim}. Expected 2D or 3D array."
            )

        # For 3D arrays, check if it has valid number of channels
        if data.ndim == 3 and data.shape[2] not in [1, 3, 4]:
            raise ValueError(
                f"Invalid number of channels: {data.shape[2]}. Expected 1, 3, or 4."
            )

    else:
        raise ValueError(
            f"Invalid data type: {type(data)}. Expected GeoDataFrame or numpy array."
        )
