"""
Data processing utilities for Bayesian inference with geospatial data.

This module provides utilities for preparing, loading, and processing
geospatial data for use with Bayesian models.
"""

import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
import logging
import json
from .rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


def prepare_spatial_data(
    data: pd.DataFrame | np.ndarray,
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: str | None = None,
    time_col: str | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """
    Prepare spatial data for Bayesian inference.

    Args:
        data: Input data as DataFrame or array
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        value_col: Column name for values to model
        time_col: Column name for temporal data (optional)
        **kwargs: Additional processing parameters

    Returns:
        Tuple of (spatial_coords, values, temporal_coords, metadata)
    """
    logger.info("Preparing spatial data for Bayesian inference...")

    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        if data.shape[1] < 2:
            raise ValueError("Array must have at least 2 columns (lat, lon)")
        data = pd.DataFrame(
            data,
            columns=[lat_col, lon_col] + [f"col_{i}" for i in range(data.shape[1] - 2)],
        )

    # Extract spatial coordinates
    if lat_col not in data.columns or lon_col not in data.columns:
        raise ValueError(f"Columns {lat_col} and {lon_col} must be present in data")

    spatial_coords = data[[lat_col, lon_col]].values

    # Extract values
    if value_col is not None:
        if value_col not in data.columns:
            raise ValueError(f"Column {value_col} not found in data")
        values = data[value_col].values
    else:
        # Use first non-coordinate column as values
        value_cols = [
            col for col in data.columns if col not in [lat_col, lon_col, time_col]
        ]
        if not value_cols:
            raise ValueError("No value column specified and no suitable column found")
        values = data[value_cols[0]].values
        logger.info(f"Using column '{value_cols[0]}' as values")

    # Extract temporal coordinates if available
    temporal_coords = None
    if time_col is not None:
        if time_col in data.columns:
            temporal_coords = _process_temporal_data(data[time_col])
        else:
            logger.warning(f"Time column '{time_col}' not found in data")

    # Create metadata
    metadata = {
        "n_samples": len(data),
        "spatial_bounds": {
            "lat_min": spatial_coords[:, 0].min(),
            "lat_max": spatial_coords[:, 0].max(),
            "lon_min": spatial_coords[:, 1].min(),
            "lon_max": spatial_coords[:, 1].max(),
        },
        "value_stats": {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        },
    }

    if temporal_coords is not None:
        metadata["temporal_bounds"] = {
            "min": np.min(temporal_coords),
            "max": np.max(temporal_coords),
        }

    logger.info(f"Prepared {len(data)} samples with spatial coordinates")
    return spatial_coords, values, temporal_coords, metadata


def load_geospatial_data(
    file_path: str | Path, file_format: str | None = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Load geospatial data from various file formats.

    Args:
        file_path: Path to the data file
        file_format: Format of the file (auto-detected if None)
        **kwargs: Additional loading parameters

    Returns:
        Loaded data as DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Auto-detect format if not specified
    if file_format is None:
        file_format = _detect_file_format(file_path)

    logger.info(f"Loading geospatial data from {file_path} (format: {file_format})")

    try:
        if file_format.lower() in ["csv", "txt"]:
            data = pd.read_csv(file_path, **kwargs)
        elif file_format.lower() in ["json", "geojson"]:
            data = _load_json_data(file_path, **kwargs)
        elif file_format.lower() in ["parquet"]:
            data = pd.read_parquet(file_path, **kwargs)
        elif file_format.lower() in ["h5", "hdf5"]:
            data = pd.read_hdf(file_path, **kwargs)
        elif file_format.lower() in ["xlsx", "xls"]:
            data = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(
            f"Successfully loaded {len(data)} rows and {len(data.columns)} columns"
        )
        return data

    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


def _detect_file_format(file_path: Path) -> str:
    """Auto-detect file format based on extension."""
    suffix = file_path.suffix.lower()

    format_mapping = {
        ".csv": "csv",
        ".txt": "csv",
        ".json": "json",
        ".geojson": "geojson",
        ".parquet": "parquet",
        ".h5": "h5",
        ".hdf5": "h5",
        ".xlsx": "xlsx",
        ".xls": "xlsx",
    }

    return format_mapping.get(suffix, "csv")


def _load_json_data(file_path: Path, **kwargs: Any) -> pd.DataFrame:
    """Load data from JSON or GeoJSON files."""
    with open(file_path) as f:
        data = json.load(f)

    # Handle GeoJSON format
    if "features" in data:
        return _parse_geojson(data)
    else:
        # Regular JSON - try to convert to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON format")


def _parse_geojson(geojson_data: dict[str, Any]) -> pd.DataFrame:
    """Parse GeoJSON data into a DataFrame."""
    features = geojson_data.get("features", [])

    if not features:
        raise ValueError("No features found in GeoJSON data")

    # Extract properties and geometry
    rows = []
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # Extract coordinates if available
        if geometry.get("type") == "Point":
            coords = geometry.get("coordinates", [])
            if len(coords) >= 2:
                properties["lon"] = coords[0]
                properties["lat"] = coords[1]

        rows.append(properties)

    return pd.DataFrame(rows)


def _process_temporal_data(temporal_data: pd.Series) -> np.ndarray:
    """Process temporal data into numerical format."""
    # Try to convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(temporal_data):
        try:
            temporal_data = pd.to_datetime(temporal_data)
        except (TypeError, ValueError):
            try:
                temporal_data = pd.Series(
                    pd.to_numeric(temporal_data, errors="raise"),
                    index=temporal_data.index,
                )
            except (TypeError, ValueError) as numeric_error:
                raise ValueError(
                    "temporal_data must be datetime-like or numeric"
                ) from numeric_error

    # Convert to numerical representation
    if pd.api.types.is_datetime64_any_dtype(temporal_data):
        # Convert to Unix timestamp
        temporal_numeric: np.ndarray = (
            temporal_data.astype(np.int64).to_numpy() // 10**9
        )
    else:
        temporal_numeric = np.asarray(temporal_data.values)

    return temporal_numeric


def validate_spatial_data(
    spatial_coords: np.ndarray,
    values: np.ndarray,
    temporal_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Validate spatial data for Bayesian inference.

    Both latitude and longitude range violations are reported as errors.

    Args:
        spatial_coords: Spatial coordinates array
        values: Values array
        temporal_coords: Temporal coordinates array (optional)

    Returns:
        Validation results dictionary
    """
    errors: list[str] = []
    warnings: list[str] = []
    validation_results: dict[str, Any] = {
        "is_valid": True,
        "warnings": warnings,
        "errors": errors,
    }

    # Check spatial coordinates
    if spatial_coords.shape[1] != 2:
        errors.append("Spatial coordinates must have 2 columns (lat, lon)")
        validation_results["is_valid"] = False

    # Check for NaN values
    if np.any(np.isnan(spatial_coords)):
        warnings.append("Spatial coordinates contain NaN values")

    if np.any(np.isnan(values)):
        warnings.append("Values contain NaN values")

    # Check coordinate ranges
    lat_range = spatial_coords[:, 0]
    lon_range = spatial_coords[:, 1]

    if np.any(lat_range < -90) or np.any(lat_range > 90):
        errors.append("Latitude values out of range [-90, 90]")
        validation_results["is_valid"] = False

    if np.any(lon_range < -180) or np.any(lon_range > 180):
        errors.append("Longitude values out of range [-180, 180]")
        validation_results["is_valid"] = False

    # Check data consistency
    if len(spatial_coords) != len(values):
        errors.append("Spatial coordinates and values have different lengths")
        validation_results["is_valid"] = False

    if temporal_coords is not None:
        if len(temporal_coords) != len(values):
            errors.append("Temporal coordinates and values have different lengths")
            validation_results["is_valid"] = False

        if np.any(np.isnan(temporal_coords)):
            warnings.append("Temporal coordinates contain NaN values")

    # Check for duplicate coordinates
    unique_coords = np.unique(spatial_coords, axis=0)
    if len(unique_coords) < len(spatial_coords):
        warnings.append(
            f"Found {len(spatial_coords) - len(unique_coords)} duplicate spatial coordinates"
        )

    return validation_results


def create_spatial_grid(
    bounds: dict[str, float], resolution: float = 0.1, grid_type: str = "regular"
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create a spatial grid for prediction.

    Args:
        bounds: Dictionary with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
        resolution: Grid resolution in degrees
        grid_type: Type of grid ('regular' or 'adaptive')

    Returns:
        Tuple of (grid_coordinates, grid_metadata)
    """
    lat_min = bounds["lat_min"]
    lat_max = bounds["lat_max"]
    lon_min = bounds["lon_min"]
    lon_max = bounds["lon_max"]

    if grid_type == "regular":
        # Create regular grid
        lat_grid = np.arange(lat_min, lat_max + resolution, resolution)
        lon_grid = np.arange(lon_min, lon_max + resolution, resolution)

        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Flatten to coordinate pairs
        grid_coords = np.column_stack([lat_mesh.flatten(), lon_mesh.flatten()])

        grid_metadata = {
            "grid_type": "regular",
            "resolution": resolution,
            "n_points": len(grid_coords),
            "bounds": bounds,
            "shape": (len(lat_grid), len(lon_grid)),
        }

    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")

    return grid_coords, grid_metadata


def sample_spatial_data(
    spatial_coords: np.ndarray,
    values: np.ndarray,
    n_samples: int,
    method: str = "random",
    random_seed: SeedLike = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample spatial data for training/validation.

    Args:
        spatial_coords: Spatial coordinates
        values: Values
        n_samples: Number of samples to take
        method: Sampling method ('random', 'stratified', 'systematic')
        random_seed: Seed or generator for the sampling draws. ``None``
            (default) means a generator seeded from OS entropy, so the sample
            is not replayable; pass an int to replay. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        **kwargs: Additional sampling parameters

    Returns:
        Tuple of (sampled_coords, sampled_values)
    """
    rng = resolve_rng(random_seed)

    n_total = len(spatial_coords)

    if n_samples >= n_total:
        logger.warning(f"Requested {n_samples} samples but only {n_total} available")
        return spatial_coords, values

    if method == "random":
        # Random sampling
        indices = rng.choice(n_total, n_samples, replace=False)

    elif method == "stratified":
        # Stratified sampling based on value quantiles
        n_strata = kwargs.get("n_strata", 5)
        quantiles = np.percentile(values, np.linspace(0, 100, n_strata + 1))

        stratified_indices: list[int] = []
        samples_per_stratum = n_samples // n_strata

        for i in range(n_strata):
            mask = (values >= quantiles[i]) & (values < quantiles[i + 1])
            stratum_indices = np.where(mask)[0]

            if len(stratum_indices) > 0:
                stratum_sample = rng.choice(
                    stratum_indices,
                    min(samples_per_stratum, len(stratum_indices)),
                    replace=False,
                )
                stratified_indices.extend(list(stratum_sample))

        # Add remaining samples randomly
        remaining = n_samples - len(stratified_indices)
        if remaining > 0:
            used_indices = set(stratified_indices)
            available_indices = [i for i in range(n_total) if i not in used_indices]
            additional_indices = rng.choice(available_indices, remaining, replace=False)
            stratified_indices.extend(list(additional_indices))

        indices = np.array(stratified_indices)

    elif method == "systematic":
        # Systematic sampling
        step = n_total // n_samples
        indices = np.arange(0, n_total, step)[:n_samples]

    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    return spatial_coords[indices], values[indices]


def save_processed_data(
    data: pd.DataFrame,
    output_path: str | Path,
    format: str = "csv",
    **kwargs: Any,
) -> None:
    """
    Save processed data to file.

    Args:
        data: Data to save
        output_path: Output file path
        format: Output format
        **kwargs: Additional saving parameters
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_path}")

    try:
        if format.lower() == "csv":
            data.to_csv(output_path, index=False, **kwargs)
        elif format.lower() == "parquet":
            data.to_parquet(output_path, index=False, **kwargs)
        elif format.lower() == "json":
            data.to_json(output_path, orient="records", **kwargs)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Successfully saved data to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save data to {output_path}: {e}")
        raise
