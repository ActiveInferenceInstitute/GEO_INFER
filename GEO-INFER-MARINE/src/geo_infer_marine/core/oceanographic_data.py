"""
Oceanographic data processing module.

Handles processing of oceanographic data including temperature, salinity, currents.
"""

import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class OceanographicDataProcessor:
    """
    Process oceanographic datasets.

    Supports 3D oceanographic data (temperature, salinity, currents, depth).
    """

    def __init__(self, config: dict | None = None):
        """Initialize oceanographic data processor."""
        self.config = config or {}

    def load_oceanographic_data(
        self, file_path: str, variables: list[str] | None = None
    ) -> xr.Dataset:
        """
        Load oceanographic dataset.

        Args:
            file_path: Path to oceanographic data file
            variables: Optional list of variables to load

        Returns:
            xarray Dataset with oceanographic data
        """
        try:
            if file_path.endswith(".nc") or file_path.endswith(".netcdf"):
                ds = xr.open_dataset(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            if variables:
                ds = ds[variables]

            logger.info(
                f"Loaded oceanographic dataset with {len(ds.data_vars)} variables"
            )
            return ds

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def process_3d_ocean_data(
        self, dataset: xr.Dataset, depth_levels: list[float] | None = None
    ) -> xr.Dataset:
        """
        Process 3D oceanographic data.

        Args:
            dataset: Oceanographic dataset with depth dimension
            depth_levels: Optional specific depth levels to extract

        Returns:
            Processed 3D dataset
        """
        processed = dataset.copy()

        if depth_levels and "depth" in processed.dims:
            processed = processed.sel(depth=depth_levels)

        return processed

    def calculate_ocean_currents(
        self, u_velocity: xr.DataArray, v_velocity: xr.DataArray
    ) -> xr.Dataset:
        """
        Calculate ocean current magnitude and direction.

        Args:
            u_velocity: U-component of velocity
            v_velocity: V-component of velocity

        Returns:
            Dataset with current magnitude and direction
        """
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
        direction = np.arctan2(v_velocity, u_velocity) * 180 / np.pi

        return xr.Dataset(
            {"current_magnitude": magnitude, "current_direction": direction}
        )
