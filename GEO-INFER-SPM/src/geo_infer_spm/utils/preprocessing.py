"""
Data preprocessing utilities for GEO-INFER-SPM

This module provides functions for preprocessing geospatial data before SPM analysis,
including normalization, missing data handling, outlier detection, and spatial
data preparation.

Preprocessing steps ensure data quality and compatibility with SPM statistical methods.
"""

import numpy as np
from typing import Any, cast
from scipy import stats
import warnings

from ..models.data_models import SPMData
from .validation import validate_spm_data


def preprocess_data(
    data: SPMData, steps: list[str] | None = None, **kwargs: Any
) -> SPMData:
    """
    Apply preprocessing pipeline to SPM data.

    Args:
        data: Input SPMData
        steps: List of preprocessing steps to apply
        **kwargs: Parameters for individual preprocessing steps

    Returns:
        Preprocessed SPMData

    Example:
        >>> processed = preprocess_data(data, steps=['normalize', 'remove_outliers'])
    """
    if steps is None:
        steps = ["validate", "handle_missing", "normalize"]

    processed_data = data

    for step in steps:
        if step == "validate":
            processed_data = validate_spm_data(processed_data)
        elif step == "handle_missing":
            processed_data = handle_missing_data(
                processed_data, **kwargs.get("missing_params", {})
            )
        elif step == "normalize":
            processed_data = normalize_data(
                processed_data, **kwargs.get("normalize_params", {})
            )
        elif step == "remove_outliers":
            processed_data = remove_outliers(
                processed_data, **kwargs.get("outlier_params", {})
            )
        elif step == "spatial_filter":
            processed_data = spatial_filter(
                processed_data, **kwargs.get("spatial_params", {})
            )
        elif step == "temporal_filter":
            processed_data = temporal_filter(
                processed_data, **kwargs.get("temporal_params", {})
            )
        else:
            warnings.warn(f"Unknown preprocessing step: {step}")

    # Update metadata
    processed_data.metadata["preprocessing_steps"] = steps
    processed_data.metadata["preprocessing_params"] = kwargs

    return processed_data


def handle_missing_data(
    data: SPMData, method: str = "interpolate", max_missing_fraction: float = 0.1
) -> SPMData:
    """
    Handle missing data in SPM dataset.

    Args:
        data: Input SPMData
        method: Method for handling missing data ('drop', 'interpolate', 'mean')
        max_missing_fraction: Maximum fraction of missing data allowed

    Returns:
        SPMData with missing data handled
    """
    if isinstance(data.data, np.ma.MaskedArray):
        missing_mask = data.data.mask
    else:
        missing_mask = np.isnan(data.data)

    if data.data.ndim > 1:
        missing_fraction = np.mean(missing_mask, axis=tuple(range(1, data.data.ndim)))
    else:
        missing_fraction = np.mean(missing_mask)

    # Check if too much data is missing
    if np.any(missing_fraction > max_missing_fraction):
        raise ValueError(
            f"Too much missing data: {missing_fraction.max():.2%} > {max_missing_fraction:.2%}"
        )

    if method == "drop":
        # Remove points with missing data
        if data.data.ndim == 1:
            valid_mask = ~missing_mask
        else:
            valid_mask = ~np.any(missing_mask, axis=tuple(range(1, data.data.ndim)))

        new_data = (
            data.data[valid_mask] if data.data.ndim == 1 else data.data[valid_mask]
        )
        new_coordinates = data.coordinates[valid_mask]

        if data.time is not None:
            new_time = data.time[valid_mask]
        else:
            new_time = None

    elif method == "interpolate":
        # Spatial interpolation of missing values
        new_data = _spatial_interpolate_missing(
            data.data, data.coordinates, missing_mask
        )

        new_coordinates = data.coordinates
        new_time = data.time

    elif method == "mean":
        # Replace with mean value
        if data.data.ndim == 1:
            mean_val = np.nanmean(data.data)
            new_data = np.where(missing_mask, mean_val, data.data)
        else:
            # Mean along spatial dimensions for each variable
            mean_vals = np.nanmean(
                data.data, axis=tuple(range(1, data.data.ndim)), keepdims=True
            )
            new_data = np.where(missing_mask, mean_vals, data.data)

        new_coordinates = data.coordinates
        new_time = data.time

    else:
        raise ValueError(f"Unknown missing data method: {method}")

    # Create new SPMData object
    new_metadata = data.metadata.copy()
    new_metadata["missing_data_handled"] = {
        "method": method,
        "original_missing_fraction": float(missing_fraction),
        "max_missing_fraction": max_missing_fraction,
    }

    return SPMData(
        data=new_data,
        coordinates=new_coordinates,
        time=new_time,
        covariates=data.covariates,
        metadata=new_metadata,
        crs=data.crs,
    )


def _spatial_interpolate_missing(
    data: np.ndarray, coordinates: np.ndarray, missing_mask: np.ndarray
) -> np.ndarray:
    """Interpolate missing values using spatial neighbors."""
    if data.ndim == 1:
        # 1D interpolation using inverse distance weighting
        filled_data = data.copy()

        missing_indices = np.where(missing_mask)[0]
        for idx in missing_indices:
            point = coordinates[idx]
            distances = np.linalg.norm(coordinates - point, axis=1)

            # Use nearest 5 points for interpolation
            valid_indices = np.where(~missing_mask)[0]
            nearest_indices = valid_indices[np.argsort(distances[valid_indices])[:5]]
            weights = 1 / (distances[nearest_indices] + 1e-10)  # Avoid division by zero
            weights /= np.sum(weights)

            interpolated_value = np.sum(data[nearest_indices] * weights)
            filled_data[idx] = interpolated_value

        return filled_data
    else:
        # For multi-dimensional data, interpolate each column separately
        filled_data = data.copy()
        for col in range(data.shape[1]):
            col_data = data[:, col]
            col_missing = missing_mask[:, col]
            if np.any(col_missing):
                filled_col = _spatial_interpolate_missing(
                    col_data, coordinates, col_missing
                )
                filled_data[:, col] = filled_col
        return filled_data


def normalize_data(
    data: SPMData, method: str = "zscore", axis: int | None = None
) -> SPMData:
    """
    Normalize data values for SPM analysis.

    Args:
        data: Input SPMData
        method: Normalization method ('zscore', 'minmax', 'robust')
        axis: Axis along which to normalize (None for global)

    Returns:
        Normalized SPMData
    """
    normalized_data = data.data.copy()

    if method == "zscore":
        # Z-score normalization: (x - mean) / std
        mean_val = np.mean(normalized_data, axis=axis, keepdims=True)
        std_val = np.std(normalized_data, axis=axis, keepdims=True)
        std_val = np.where(std_val == 0, 1, std_val)  # Avoid division by zero
        normalized_data = (normalized_data - mean_val) / std_val

    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(normalized_data, axis=axis, keepdims=True)
        max_val = np.max(normalized_data, axis=axis, keepdims=True)
        denominator = max_val - min_val
        denominator = np.where(denominator == 0, 1, denominator)
        normalized_data = (normalized_data - min_val) / denominator

    elif method == "robust":
        # Robust normalization using median and MAD
        if axis is None:
            median_val = np.median(normalized_data)
            mad_val = stats.median_abs_deviation(normalized_data, axis=None)
        else:
            median_val = np.median(normalized_data, axis=axis, keepdims=True)
            mad_val = np.expand_dims(
                stats.median_abs_deviation(normalized_data, axis=axis), axis=axis
            )
        mad_val = np.where(mad_val == 0, 1, mad_val)
        normalized_data = (normalized_data - median_val) / mad_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Update metadata
    new_metadata = data.metadata.copy()
    new_metadata["normalization"] = {
        "method": method,
        "axis": axis,
        "original_mean": float(np.mean(data.data)),
        "original_std": float(np.std(data.data)),
    }

    return SPMData(
        data=normalized_data,
        coordinates=data.coordinates,
        time=data.time,
        covariates=data.covariates,
        metadata=new_metadata,
        crs=data.crs,
    )


def remove_outliers(
    data: SPMData, method: str = "iqr", threshold: float = 1.5
) -> SPMData:
    """
    Remove outlier data points.

    Args:
        data: Input SPMData
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection

    Returns:
        SPMData with outliers removed
    """
    if data.data.ndim > 1:
        # For multi-dimensional data, detect outliers for each column
        outlier_masks = []

        for col in range(data.data.shape[1]):
            col_data = data.data[:, col]
            col_outliers = _detect_outliers_1d(col_data, method, threshold)
            outlier_masks.append(col_outliers)

        # Point is outlier if it's outlier in any dimension
        outlier_mask = np.any(outlier_masks, axis=0)
    else:
        outlier_mask = _detect_outliers_1d(data.data, method, threshold)

    # Remove outliers
    keep_mask = ~outlier_mask

    new_data = data.data[keep_mask] if data.data.ndim == 1 else data.data[keep_mask]
    new_coordinates = data.coordinates[keep_mask]

    if data.time is not None:
        new_time = data.time[keep_mask]
    else:
        new_time = None

    # Update covariates if present
    new_covariates = None
    if data.covariates is not None:
        new_covariates = {}
        for name, values in data.covariates.items():
            new_covariates[name] = values[keep_mask]

    # Update metadata
    new_metadata = data.metadata.copy()
    new_metadata["outlier_removal"] = {
        "method": method,
        "threshold": threshold,
        "n_outliers_removed": int(np.sum(outlier_mask)),
        "n_points_remaining": int(np.sum(keep_mask)),
    }

    return SPMData(
        data=new_data,
        coordinates=new_coordinates,
        time=new_time,
        covariates=new_covariates,
        metadata=new_metadata,
        crs=data.crs,
    )


def _detect_outliers_1d(data: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """Detect outliers in 1D data."""
    if method == "iqr":
        # Interquartile range method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold

    elif method == "isolation_forest":
        # Isolation Forest method
        try:
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data.reshape(-1, 1)) == -1
        except ImportError:
            warnings.warn(
                "scikit-learn required for isolation forest. Using IQR method."
            )
            outliers = _detect_outliers_1d(data, "iqr", threshold)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return cast(np.ndarray, outliers)


def spatial_filter(
    data: SPMData, method: str = "gaussian", sigma: float = 1.0
) -> SPMData:
    """
    Apply spatial filtering to smooth data.

    Args:
        data: Input SPMData
        method: Filtering method ('gaussian', 'median', 'mean')
        sigma: Standard deviation for Gaussian filter

    Returns:
        Spatially filtered SPMData
    """

    # Convert point data to grid for filtering
    if data.data.ndim == 1:
        # Point data: neighbor-weighted local filtering.
        filtered_data = data.data.copy()

        # Apply local filtering using spatial neighbors
        for i in range(len(data.coordinates)):
            point = data.coordinates[i]
            distances = np.linalg.norm(data.coordinates - point, axis=1)

            # Find nearby points within 3*sigma
            nearby_mask = distances <= 3 * sigma
            nearby_values = data.data[nearby_mask]

            if len(nearby_values) > 1:
                if method == "gaussian":
                    # Gaussian weighted average
                    weights = np.exp(-(distances[nearby_mask] ** 2) / (2 * sigma**2))
                    weights /= np.sum(weights)
                    filtered_data[i] = np.sum(nearby_values * weights)

                elif method == "median":
                    filtered_data[i] = np.median(nearby_values)

                elif method == "mean":
                    filtered_data[i] = np.mean(nearby_values)
    else:
        # For gridded data, apply 2D filtering
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Gridded spatial filtering").
        filtered_data = data.data.copy()
        for var_idx in range(data.data.shape[1]):
            var_data = data.data[:, var_idx]
            filtered_data[:, var_idx] = var_data.flatten()

    # Update metadata
    new_metadata = data.metadata.copy()
    new_metadata["spatial_filter"] = {"method": method, "sigma": sigma}

    return SPMData(
        data=filtered_data,
        coordinates=data.coordinates,
        time=data.time,
        covariates=data.covariates,
        metadata=new_metadata,
        crs=data.crs,
    )


def temporal_filter(
    data: SPMData, method: str = "moving_average", window_size: int = 5
) -> SPMData:
    """
    Apply temporal filtering to time series data.

    Args:
        data: Input SPMData with temporal dimension
        method: Filtering method ('moving_average', 'exponential', 'savitzky_golay')
        window_size: Window size for filtering

    Returns:
        Temporally filtered SPMData
    """
    if data.time is None:
        raise ValueError("Temporal filtering requires time dimension")

    filtered_data = data.data.copy()

    if data.data.ndim == 1:
        # 1D time series filtering
        if method == "moving_average":
            filtered_data = _moving_average_filter(data.data, window_size)

        elif method == "exponential":
            alpha = 2.0 / (window_size + 1)  # Exponential smoothing parameter
            filtered_data = _exponential_filter(data.data, alpha)

        elif method == "savitzky_golay":
            try:
                from scipy.signal import savgol_filter

                filtered_data = savgol_filter(data.data, window_size, 2)
            except ImportError:
                warnings.warn(
                    "SciPy required for Savitzky-Golay filter. Using moving average."
                )
                filtered_data = _moving_average_filter(data.data, window_size)

    else:
        # Multi-dimensional filtering (apply to each spatial point)
        for spatial_idx in range(data.data.shape[1]):
            time_series = data.data[:, spatial_idx]

            if method == "moving_average":
                filtered_series = _moving_average_filter(time_series, window_size)
            elif method == "exponential":
                alpha = 2.0 / (window_size + 1)
                filtered_series = _exponential_filter(time_series, alpha)
            else:
                filtered_series = time_series  # No filtering

            filtered_data[:, spatial_idx] = filtered_series

    # Update metadata
    new_metadata = data.metadata.copy()
    new_metadata["temporal_filter"] = {"method": method, "window_size": window_size}

    return SPMData(
        data=filtered_data,
        coordinates=data.coordinates,
        time=data.time,
        covariates=data.covariates,
        metadata=new_metadata,
        crs=data.crs,
    )


def _moving_average_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average filter."""
    from scipy.ndimage import uniform_filter1d

    return cast(np.ndarray, uniform_filter1d(data.astype(float), window_size))


def _exponential_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential smoothing filter."""
    filtered = data.copy().astype(float)
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]
    return filtered
