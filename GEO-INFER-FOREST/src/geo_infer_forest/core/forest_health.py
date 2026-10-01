"""Forest health monitoring."""

import logging
from typing import Any
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ForestHealthMonitor:
    """Monitor forest health."""

    def __init__(self, config: dict | None = None):
        """Initialize forest health monitor."""
        self.config = config or {}

    def assess_forest_health(
        self,
        ndvi: xr.DataArray,
        temperature: xr.DataArray | None = None,
        precipitation: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess forest health using NDVI and climate data.

        Args:
            ndvi: Normalized Difference Vegetation Index
            temperature: Optional temperature data
            precipitation: Optional precipitation data

        Returns:
            Forest health assessment
        """
        # NDVI-based health (0-1 scale, higher is better)
        health_index = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-10)

        results: dict[str, Any] = {"health_index": health_index, "ndvi": ndvi}

        if temperature is not None:
            # Temperature stress
            optimal_temp = float(self.config.get("optimal_temperature_c", 20.0))
            temp_stress = np.clip(
                np.abs(temperature - optimal_temp) / max(abs(optimal_temp), 1e-6),
                0.0,
                1.0,
            )
            results["temperature_stress"] = temp_stress

        if precipitation is not None:
            # Water stress
            optimal_precip = float(self.config.get("optimal_precipitation_mm", 1000.0))
            water_stress = np.clip(
                np.abs(precipitation - optimal_precip) / max(optimal_precip, 1e-6),
                0.0,
                1.0,
            )
            results["water_stress"] = water_stress

        return xr.Dataset(results)

    def detect_deforestation(
        self, forest_cover_time_series: xr.DataArray, threshold: float = 0.1
    ) -> xr.Dataset:
        """
        Detect deforestation from time series.

        Args:
            forest_cover_time_series: Forest cover over time
            threshold: Minimum change to detect

        Returns:
            Deforestation detection results
        """
        # Calculate change
        initial_cover = forest_cover_time_series.isel(time=0)
        current_cover = forest_cover_time_series.isel(time=-1)
        change = initial_cover - current_cover

        # Detect significant loss
        deforestation = change > threshold

        return xr.Dataset(
            {
                "deforestation": deforestation,
                "cover_change": change,
                "deforestation_area": deforestation.sum(),
            }
        )
