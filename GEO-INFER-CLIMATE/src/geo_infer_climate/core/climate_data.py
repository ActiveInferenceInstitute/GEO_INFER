"""
Climate data processing module.

Handles loading, processing, and validation of climate datasets including
CMIP models, reanalysis data, and observational data.
"""

import importlib.util
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _detrend_series(ts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove the least-squares linear trend of ``ts`` against ``x``."""
    if np.isnan(ts).all():
        return ts
    trend = np.polyfit(x, ts, 1)
    return ts - np.polyval(trend, x)


class ClimateDataProcessor:
    """
    Process and validate climate datasets.

    Supports CMIP models, reanalysis datasets (ERA5, NCEP), and
    observational climate data.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize climate data processor.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.supported_formats = ["netcdf", "grib"]
        self.supported_datasets = ["cmip6", "era5", "ncep", "observations"]

    def load_dataset(
        self, file_path: str, dataset_type: str, variables: list[str] | None = None
    ) -> xr.Dataset:
        """
        Load climate dataset from file.

        Args:
            file_path: Path to climate data file
            dataset_type: Type of dataset (cmip6, era5, ncep, observations)
            variables: Optional list of variables to load

        Returns:
            xarray Dataset with climate data

        Raises:
            ValueError: If dataset type is not supported
            FileNotFoundError: If file does not exist
        """
        if dataset_type.lower() not in self.supported_datasets:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type}. "
                f"Supported types: {self.supported_datasets}"
            )

        try:
            if file_path.endswith(".nc") or file_path.endswith(".netcdf"):
                ds = xr.open_dataset(file_path)
            elif file_path.endswith(".grib") or file_path.endswith(".grib2"):
                if importlib.util.find_spec("cfgrib") is None:
                    raise ImportError(
                        "Reading GRIB files requires the cfgrib engine "
                        "(install the 'grib' extra of geo-infer-climate)"
                    )
                ds = xr.open_dataset(file_path, engine="cfgrib")
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            if variables:
                ds = ds[variables]

            logger.info(
                f"Loaded {dataset_type} dataset with {len(ds.data_vars)} variables"
            )
            return ds

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def validate_dataset(self, dataset: xr.Dataset) -> dict[str, bool]:
        """
        Validate climate dataset structure and data quality.

        Args:
            dataset: xarray Dataset to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "has_coordinates": False,
            "has_time_dimension": False,
            "has_spatial_dimensions": False,
            "data_complete": False,
            "no_missing_values": False,
        }

        # Check coordinates
        if len(dataset.coords) > 0:
            validation_results["has_coordinates"] = True

        # Check time dimension
        if "time" in dataset.dims or "time" in dataset.coords:
            validation_results["has_time_dimension"] = True

        # Check spatial dimensions
        spatial_dims = ["lat", "latitude", "lon", "longitude", "x", "y"]
        if any(dim in dataset.dims or dim in dataset.coords for dim in spatial_dims):
            validation_results["has_spatial_dimensions"] = True

        # Check data completeness
        if len(dataset.data_vars) > 0:
            validation_results["data_complete"] = True

        # Check for missing values
        has_missing = False
        for var in dataset.data_vars:
            if dataset[var].isnull().any():
                has_missing = True
                break
        validation_results["no_missing_values"] = not has_missing

        return validation_results

    def preprocess_dataset(
        self, dataset: xr.Dataset, operations: list[str] | None = None
    ) -> xr.Dataset:
        """
        Preprocess climate dataset with common operations.

        Supported operations:
            - 'standardize_coords': rename latitude/longitude to lat/lon
            - 'sort_time': sort the dataset along the time dimension
            - 'detrend': remove linear trends along the time dimension
            - 'remove_outliers': drop z-score outliers from data variables

        Note: 'resample' and 'regrid' are advertised by earlier documentation
        but are not implemented; requesting an unsupported operation raises
        ValueError rather than being silently ignored.

        Args:
            dataset: Input dataset
            operations: List of operations to apply (defaults to
                ['standardize_coords', 'sort_time'])

        Returns:
            Preprocessed dataset

        Raises:
            ValueError: If an unsupported operation name is requested
        """
        operations = operations or ["standardize_coords", "sort_time"]
        processed = dataset.copy()

        for op in operations:
            if op == "standardize_coords":
                processed = self._standardize_coordinates(processed)
            elif op == "sort_time":
                if "time" in processed.dims:
                    processed = processed.sortby("time")
            elif op == "detrend":
                processed = self._detrend_data(processed)
            elif op == "remove_outliers":
                processed = self._remove_outliers(processed)
            else:
                raise ValueError(
                    f"Unsupported preprocessing operation: {op!r}. "
                    f"Supported operations: standardize_coords, sort_time, "
                    f"detrend, remove_outliers"
                )

        return processed

    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names."""
        rename_dict = {}

        if "latitude" in dataset.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in dataset.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            dataset = dataset.rename(rename_dict)

        return dataset

    def _detrend_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Remove linear trend from time series data."""
        detrended = dataset.copy()

        for var in dataset.data_vars:
            if "time" in dataset[var].dims:
                # Simple linear detrending
                data = dataset[var].values
                if data.ndim == 1:
                    if not np.isnan(data).all():
                        trend = np.polyfit(range(len(data)), data, 1)
                        detrended[var].values = data - np.polyval(
                            trend, range(len(data))
                        )
                elif data.ndim >= 2:
                    # Detrend along time dimension
                    time_axis = dataset[var].dims.index("time")
                    x = np.arange(data.shape[time_axis])
                    # Fit and remove linear trend along time_axis
                    detrended[var].values = np.apply_along_axis(
                        _detrend_series, time_axis, data, x
                    )

        return detrended

    def _remove_outliers(
        self, dataset: xr.Dataset, threshold: float = 3.0
    ) -> xr.Dataset:
        """Remove statistical outliers using z-score method."""
        cleaned = dataset.copy()

        for var in dataset.data_vars:
            data = dataset[var].values
            if data.size > 0:
                mean = np.nanmean(data)
                std = np.nanstd(data)
                if std > 0:
                    z_scores = np.abs((data - mean) / std)
                    outliers = z_scores > threshold
                    cleaned[var].values[outliers] = np.nan

        return cleaned

    def extract_temporal_subset(
        self, dataset: xr.Dataset, start_date: str, end_date: str
    ) -> xr.Dataset:
        """
        Extract temporal subset of dataset.

        Args:
            dataset: Input dataset
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            Subsetted dataset
        """
        if "time" not in dataset.coords and "time" not in dataset.dims:
            raise ValueError("Dataset does not have time dimension")

        time_slice = slice(start_date, end_date)
        return dataset.sel(time=time_slice)

    def extract_spatial_subset(
        self,
        dataset: xr.Dataset,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
    ) -> xr.Dataset:
        """
        Extract spatial subset of dataset.

        Args:
            dataset: Input dataset
            lat_range: (min_lat, max_lat) tuple
            lon_range: (min_lon, max_lon) tuple

        Returns:
            Subsetted dataset
        """
        lat_name = "lat" if "lat" in dataset.coords else "latitude"
        lon_name = "lon" if "lon" in dataset.coords else "longitude"

        lat_slice = slice(lat_range[0], lat_range[1])
        lon_slice = slice(lon_range[0], lon_range[1])

        return dataset.sel({lat_name: lat_slice, lon_name: lon_slice})
