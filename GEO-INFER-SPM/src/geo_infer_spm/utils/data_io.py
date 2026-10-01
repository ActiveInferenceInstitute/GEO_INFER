"""
Data input/output utilities for GEO-INFER-SPM

This module provides functions for loading and saving geospatial data
in various formats commonly used in environmental and geospatial analysis.

Supported formats:
- GeoTIFF (raster data)
- NetCDF (multidimensional arrays)
- GeoJSON/GeoPackage (vector data)
- CSV with coordinates
- HDF5 (hierarchical data)
- Cloud-optimized GeoTIFF (COG)

All functions return standardized SPMData objects for consistent processing.
"""

import numpy as np
import pandas as pd
from typing import Any, cast
from collections.abc import Callable
import json

import geopandas as gpd
import h5py
import rasterio
import xarray as xr

from ..models.data_models import SPMData, SPMResult


def load_data(file_path: str, **kwargs: Any) -> SPMData:
    """
    Load geospatial data from file with automatic format detection.

    Args:
        file_path: Path to data file
        **kwargs: Format-specific loading parameters

    Returns:
        SPMData object with loaded data

    Raises:
        ValueError: If file format not supported or file not found
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Determine file format from extension
    file_extension = file_path.lower().split(".")[-1]

    format_loaders = {
        "tif": load_geotiff,
        "tiff": load_geotiff,
        "nc": load_netcdf,
        "netcdf": load_netcdf,
        "h5": load_hdf5,
        "hdf5": load_hdf5,
        "geojson": load_geojson,
        "gpkg": load_geopackage,
        "csv": load_csv_with_coords,
        "json": load_json_data,
    }

    if file_extension not in format_loaders:
        raise ValueError(f"Unsupported file format: {file_extension}")

    loader_func = cast(Callable[..., Any], format_loaders[file_extension])
    return cast(SPMData, loader_func(file_path, **kwargs))


def load_geotiff(
    file_path: str, band: int | None = None, nodata_value: float | None = None
) -> SPMData:
    """
    Load GeoTIFF raster data.

    Args:
        file_path: Path to GeoTIFF file
        band: Band number to load (default: first band)
        nodata_value: Value to treat as nodata

    Returns:
        SPMData object with raster data
    """
    with rasterio.open(file_path) as src:
        if band is None:
            band = 1

        # Read data
        data = src.read(band)

        # Handle nodata values
        if nodata_value is not None:
            data = np.ma.masked_equal(data, nodata_value)
        elif src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)

        # Get spatial coordinates
        height, width = data.shape
        transform = src.transform

        # Create coordinate grids
        x_coords = np.arange(width) * transform[0] + transform[2]
        y_coords = np.arange(height) * transform[3] + transform[5]

        # Create coordinate array (pixel centers)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        coordinates = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        # Flatten data to match coordinates
        data_flat = data.ravel()

        # Create metadata
        metadata = {
            "source_file": file_path,
            "crs": str(src.crs),
            "transform": transform,
            "band": band,
            "width": width,
            "height": height,
            "bounds": src.bounds,
            "data_type": "raster",
        }

        return SPMData(
            data=data_flat, coordinates=coordinates, metadata=metadata, crs=str(src.crs)
        )


def load_netcdf(
    file_path: str,
    variable: str | None = None,
    time_dim: str | None = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> SPMData:
    """
    Load NetCDF data.

    Args:
        file_path: Path to NetCDF file
        variable: Variable name to load (default: first variable)
        time_dim: Name of time dimension
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension

    Returns:
        SPMData object with NetCDF data
    """
    # Open dataset
    ds = xr.open_dataset(file_path)

    # Select variable
    if variable is None:
        # Find first data variable (not coordinate)
        data_vars = [v for v in ds.data_vars if v not in ds.coords]
        if not data_vars:
            raise ValueError("No data variables found in NetCDF file")
        variable = str(data_vars[0])

    data_array = ds[variable]

    # Handle temporal data
    time_coords = None
    if time_dim and time_dim in data_array.dims:
        time_coords = data_array[time_dim].values
        if hasattr(time_coords[0], "calendar"):  # datetime objects
            time_coords = np.arange(len(time_coords))  # Convert to indices

    # Get spatial coordinates
    if lat_dim in data_array.coords and lon_dim in data_array.coords:
        lat_vals = data_array[lat_dim].values
        lon_vals = data_array[lon_dim].values

        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        coordinates = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    else:
        raise ValueError(f"Spatial dimensions {lat_dim}, {lon_dim} not found")

    # Extract data
    data_values = data_array.values

    # Handle different dimensionalities
    if data_values.ndim == 2:  # Spatial only
        data_flat = data_values.ravel()
        final_time = None
    elif data_values.ndim == 3 and time_dim:  # Spatial + temporal
        # For now, take mean across time or return as 3D
        data_flat = data_values.reshape(data_values.shape[0], -1).T  # (time, space)
        final_time = time_coords
    else:
        # Reshape to 2D (assuming last two dims are spatial)
        spatial_size = data_values.shape[-2] * data_values.shape[-1]
        data_flat = data_values.reshape(-1, spatial_size).T
        final_time = None

    # Create metadata
    metadata = {
        "source_file": file_path,
        "variable": variable,
        "dimensions": list(data_array.dims),
        "data_type": "netcdf",
        "time_dimension": time_dim,
        "spatial_dimensions": [lat_dim, lon_dim],
    }

    return SPMData(
        data=data_flat,
        coordinates=coordinates,
        time=final_time,
        metadata=metadata,
        crs="EPSG:4326",  # Assume WGS84 for NetCDF
    )


def load_geojson(file_path: str, value_column: str | None = None) -> SPMData:
    """
    Load GeoJSON vector data.

    Args:
        file_path: Path to GeoJSON file
        value_column: Column containing values to analyze

    Returns:
        SPMData object with vector data
    """
    # Read GeoJSON
    gdf = gpd.read_file(file_path)

    # Extract coordinates from geometries
    centroids = gdf.geometry.centroid
    coordinates = np.column_stack([centroids.x.values, centroids.y.values])

    # Extract values
    if value_column is None:
        # Try to find a numeric column
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in GeoJSON")
        value_column = numeric_cols[0]

    data_values = gdf[value_column].values

    # Create metadata
    metadata = {
        "source_file": file_path,
        "value_column": value_column,
        "crs": str(gdf.crs),
        "n_features": len(gdf),
        "data_type": "vector",
    }

    return SPMData(
        data=data_values, coordinates=coordinates, metadata=metadata, crs=str(gdf.crs)
    )


def load_geopackage(
    file_path: str, layer: str | None = None, value_column: str | None = None
) -> SPMData:
    """
    Load GeoPackage vector data.

    Args:
        file_path: Path to GeoPackage file
        layer: Layer name to load
        value_column: Column containing values to analyze

    Returns:
        SPMData object with vector data
    """
    # Read GeoPackage
    gpd.read_file(file_path, layer=layer)

    # Use same logic as GeoJSON
    return load_geojson(file_path, value_column)


def load_csv_with_coords(
    file_path: str,
    x_column: str = "longitude",
    y_column: str = "latitude",
    value_column: str | None = None,
    **kwargs: Any,
) -> SPMData:
    """
    Load CSV data with coordinate columns.

    Args:
        file_path: Path to CSV file
        x_column: Name of x/longitude column
        y_column: Name of y/latitude column
        value_column: Name of value column (default: first numeric column)
        **kwargs: Additional pandas read_csv parameters

    Returns:
        SPMData object with CSV data
    """
    # Read CSV
    df = pd.read_csv(file_path, **kwargs)

    # Extract coordinates
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"Coordinate columns {x_column}, {y_column} not found")

    coordinates = df[[x_column, y_column]].values

    # Extract values
    if value_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = {x_column, y_column}
        value_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not value_cols:
            raise ValueError("No numeric value columns found")

        value_column = value_cols[0]

    data_values = df[value_column].values
    covariate_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {x_column, y_column, value_column}
    ]
    covariates = {col: df[col].values for col in covariate_columns}

    # Check for temporal data
    time_column = kwargs.get("time_column")
    time_values = None
    if time_column and time_column in df.columns:
        time_values = df[time_column].values

    # Create metadata
    metadata = {
        "source_file": file_path,
        "x_column": x_column,
        "y_column": y_column,
        "value_column": value_column,
        "n_points": len(df),
        "data_type": "csv",
    }

    return SPMData(
        data=data_values,
        coordinates=coordinates,
        time=time_values,
        covariates=covariates or None,
        metadata=metadata,
        crs="EPSG:4326",  # Assume WGS84
    )


def load_hdf5(
    file_path: str,
    dataset_path: str = "/",
    coordinate_datasets: dict[str, str] | None = None,
) -> SPMData:
    """
    Load HDF5 data.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within HDF5 file
        coordinate_datasets: Dictionary mapping coordinate names to dataset paths

    Returns:
        SPMData object with HDF5 data
    """
    with h5py.File(file_path, "r") as f:
        # Load main dataset
        if dataset_path not in f:
            raise ValueError(f"Dataset {dataset_path} not found in HDF5 file")

        data = f[dataset_path][()]

        # Load coordinates
        if coordinate_datasets is None:
            # Try to find coordinate datasets automatically
            coord_names = ["coordinates", "coords", "lon", "lat", "x", "y"]
            coordinate_datasets = {}
            for name in coord_names:
                if name in f:
                    if name in ["lon", "lat"]:
                        coordinate_datasets["lon"] = f["lon"][()]
                        coordinate_datasets["lat"] = f["lat"][()]
                        break
                    elif name in ["x", "y"]:
                        coordinate_datasets["x"] = f["x"][()]
                        coordinate_datasets["y"] = f["y"][()]
                        break
                    else:
                        coords = f[name][()]
                        if coords.shape[1] == 2:  # (n_points, 2)
                            coordinates = coords
                            break

        # Construct coordinates array
        if "coordinates" in locals():
            coordinates = coordinates
        elif "lon" in coordinate_datasets and "lat" in coordinate_datasets:
            lon_grid, lat_grid = np.meshgrid(
                coordinate_datasets["lon"], coordinate_datasets["lat"]
            )
            coordinates = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        elif "x" in coordinate_datasets and "y" in coordinate_datasets:
            x_grid, y_grid = np.meshgrid(
                coordinate_datasets["x"], coordinate_datasets["y"]
            )
            coordinates = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        else:
            raise ValueError("Could not determine coordinate system")

        # Handle data dimensionality
        if data.ndim > 2:
            # Flatten spatial dimensions
            original_shape = data.shape
            data_flat = data.reshape(original_shape[0], -1).T
        else:
            data_flat = data

        # Create metadata
        metadata = {
            "source_file": file_path,
            "dataset_path": dataset_path,
            "coordinate_datasets": coordinate_datasets,
            "original_shape": data.shape if hasattr(data, "shape") else None,
            "data_type": "hdf5",
        }

        return SPMData(data=data_flat, coordinates=coordinates, metadata=metadata)


def load_json_data(
    file_path: str, data_key: str = "data", coords_key: str = "coordinates"
) -> SPMData:
    """
    Load JSON data with custom structure.

    Args:
        file_path: Path to JSON file
        data_key: Key for data array in JSON
        coords_key: Key for coordinates array in JSON

    Returns:
        SPMData object with JSON data
    """
    with open(file_path) as f:
        json_data = json.load(f)

    # Extract data and coordinates
    if data_key not in json_data:
        raise ValueError(f"Data key '{data_key}' not found in JSON")

    if coords_key not in json_data:
        raise ValueError(f"Coordinates key '{coords_key}' not found in JSON")

    data_values = np.array(json_data[data_key])
    coordinates = np.array(json_data[coords_key])
    covariates = {
        key: np.asarray(value) for key, value in json_data.get("covariates", {}).items()
    }

    # Create metadata — start with the file's own metadata if present
    metadata = {}
    if "metadata" in json_data and isinstance(json_data["metadata"], dict):
        metadata.update(json_data["metadata"])
    metadata.update(
        {
            "source_file": file_path,
            "data_key": data_key,
            "coords_key": coords_key,
            "data_type": "json",
        }
    )

    crs = metadata.get("crs", "EPSG:4326")
    if str(crs).upper() == "EPSG:4326" and (
        np.any((coordinates[:, 0] < -180) | (coordinates[:, 0] > 180))
        or np.any((coordinates[:, 1] < -90) | (coordinates[:, 1] > 90))
    ):
        crs = "LOCAL"

    return SPMData(
        data=data_values,
        coordinates=coordinates,
        covariates=covariates or None,
        metadata=metadata,
        crs=crs,
    )


def save_spm(
    spm_result: SPMResult, file_path: str, format: str = "json", **kwargs: Any
) -> None:
    """
    Save SPM results to file.

    Args:
        spm_result: SPMResult object to save
        file_path: Output file path
        format: Output format ('json', 'hdf5', 'csv')
        **kwargs: Format-specific saving parameters
    """
    if format == "json":
        _save_spm_json(spm_result, file_path, **kwargs)
    elif format == "hdf5":
        _save_spm_hdf5(spm_result, file_path, **kwargs)
    elif format == "csv":
        _save_spm_csv(spm_result, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported save format: {format}")


def _save_spm_json(spm_result: SPMResult, file_path: str, **kwargs: Any) -> None:
    """Save SPM results as JSON."""
    # Convert numpy arrays to lists for JSON serialization
    result_dict = {
        "beta_coefficients": spm_result.beta_coefficients.tolist(),
        "residuals": spm_result.residuals.tolist(),
        "model_diagnostics": spm_result.model_diagnostics,
        "processing_metadata": spm_result.processing_metadata,
        "design_matrix_shape": spm_result.design_matrix.matrix.shape,
        "data_shape": (
            spm_result.spm_data.data.shape
            if hasattr(spm_result.spm_data.data, "shape")
            else None
        ),
        "coordinates_shape": spm_result.spm_data.coordinates.shape,
        "crs": spm_result.spm_data.crs,
        "has_time": spm_result.spm_data.has_temporal,
        "n_contrasts": len(spm_result.contrasts),
    }

    # Add contrast information
    if spm_result.contrasts:
        result_dict["contrasts"] = []
        for i, contrast in enumerate(spm_result.contrasts):
            contrast_dict = {
                "index": i,
                "contrast_vector": contrast.contrast_vector.tolist(),
                "t_statistic_shape": contrast.t_statistic.shape,
                "p_values_shape": contrast.p_values.shape,
                "n_significant": contrast.n_significant,
                "correction_method": contrast.correction_method,
            }
            result_dict["contrasts"].append(contrast_dict)

    with open(file_path, "w") as f:
        json.dump(result_dict, f, indent=2)


def _save_spm_hdf5(spm_result: SPMResult, file_path: str, **kwargs: Any) -> None:
    """Save SPM results as HDF5."""
    with h5py.File(file_path, "w") as f:
        # Save main results
        f.create_dataset("beta_coefficients", data=spm_result.beta_coefficients)
        f.create_dataset("residuals", data=spm_result.residuals)
        f.create_dataset("coordinates", data=spm_result.spm_data.coordinates)

        # Save design matrix
        f.create_dataset("design_matrix", data=spm_result.design_matrix.matrix)

        # Save contrasts
        if spm_result.contrasts:
            contrast_group = f.create_group("contrasts")
            for i, contrast in enumerate(spm_result.contrasts):
                c_group = contrast_group.create_group(f"contrast_{i}")
                c_group.create_dataset("contrast_vector", data=contrast.contrast_vector)
                c_group.create_dataset("t_statistic", data=contrast.t_statistic)
                c_group.create_dataset("p_values", data=contrast.p_values)
                c_group.attrs["correction_method"] = contrast.correction_method

        # Save metadata
        metadata_group = f.create_group("metadata")
        for key, value in spm_result.model_diagnostics.items():
            if isinstance(value, (int, float, str)):
                metadata_group.attrs[key] = value


def _save_spm_csv(spm_result: SPMResult, file_path: str, **kwargs: Any) -> None:
    """Save SPM results as CSV."""
    # Create DataFrame with coordinates and results
    df_data = {
        "longitude": spm_result.spm_data.coordinates[:, 0],
        "latitude": spm_result.spm_data.coordinates[:, 1],
        "residuals": spm_result.residuals,
    }

    # Add beta coefficients if 1D
    if spm_result.beta_coefficients.ndim == 1:
        for i, beta in enumerate(spm_result.beta_coefficients):
            df_data[f"beta_{i}"] = beta
    else:
        # For multi-dimensional beta, save first column
        df_data["beta"] = spm_result.beta_coefficients[:, 0]

    df = pd.DataFrame(df_data)
    df.to_csv(file_path, index=False)
