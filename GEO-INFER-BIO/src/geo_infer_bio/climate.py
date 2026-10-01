"""
GEO-INFER-BIO Climate Data Processing Module

This module provides climate data processing capabilities for biological spatial analysis,
designed to work with real-world climate datasets like WorldClim bioclimatic rasters
and other meteorological data sources with biological relevance.

Key Features:
- WorldClim bioclimatic variable sampling from local GeoTIFF rasters
- Custom climate raster loading
- Climate data sampling at biological sampling coordinates
- Multi-variable climate data alignment
- Support for user-supplied climate change scenario rasters
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Raster processing (optional; required only for WorldClim/custom raster loading)
try:
    import rasterio

    HAS_RASTER_DEPS = True
except ImportError:
    HAS_RASTER_DEPS = False
    logger.warning(
        "Raster processing dependencies not available. Install with: uv pip install rasterio"
    )


class ClimateDataProcessor:
    """
    Climate data processing for biological spatial analysis.

    Supports multiple climate data sources:
    - WorldClim bioclimatic variables (sampled from local GeoTIFF rasters)
    - Custom climate rasters
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize climate data processor.

        Args:
            cache_dir: Directory for caching downloaded climate data
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".geo_infer_bio" / "climate_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # WorldClim configuration
        self.worldclim_config: dict[str, Any] = {
            "base_url": "https://biogeo.ucdavis.edu/data/worldclim/v2.1/",
            "variables": {
                "bio1": "Annual Mean Temperature",
                "bio2": "Mean Diurnal Range",
                "bio3": "Isothermality",
                "bio4": "Temperature Seasonality",
                "bio5": "Max Temperature of Warmest Month",
                "bio6": "Min Temperature of Coldest Month",
                "bio7": "Temperature Annual Range",
                "bio8": "Mean Temperature of Wettest Quarter",
                "bio9": "Mean Temperature of Driest Quarter",
                "bio10": "Mean Temperature of Warmest Quarter",
                "bio11": "Mean Temperature of Coldest Quarter",
                "bio12": "Annual Precipitation",
                "bio13": "Precipitation of Wettest Month",
                "bio14": "Precipitation of Driest Month",
                "bio15": "Precipitation Seasonality",
                "bio16": "Precipitation of Wettest Quarter",
                "bio17": "Precipitation of Driest Quarter",
                "bio18": "Precipitation of Warmest Quarter",
                "bio19": "Precipitation of Coldest Quarter",
            },
            "resolutions": ["30s", "2.5m", "5m", "10m"],
        }

        logger.info(
            f"ClimateDataProcessor initialized with cache directory: {self.cache_dir}"
        )

    def load_worldclim_data(
        self,
        variables: list[str],
        coordinates: list[tuple[float, float]],
        buffer_km: float = 5.0,
        resolution: str = "30s",
        data_path: str | None = None,
    ) -> "ClimateDataset":
        """
        Load WorldClim bioclimatic variables for specified coordinates.

        Args:
            variables: List of bioclimatic variable codes (e.g., ['bio1', 'bio12'])
            coordinates: List of (latitude, longitude) tuples
            buffer_km: Buffer distance around coordinates in kilometers
            resolution: WorldClim resolution ('30s', '2.5m', '5m', or '10m')
            data_path: Directory containing WorldClim GeoTIFF files, or a
                single GeoTIFF when loading one variable.

        Returns:
            ClimateDataset object with climate data
        """
        logger.info(
            f"Loading WorldClim data for {len(variables)} variables at {resolution} resolution"
        )

        if not coordinates:
            raise ValueError("coordinates must not be empty")
        if not variables:
            raise ValueError("variables must not be empty")
        if not HAS_RASTER_DEPS:
            raise ImportError("rasterio and geopandas are required for WorldClim data")
        if data_path is None:
            raise ValueError("data_path is required for WorldClim raster data")

        # Calculate bounding box with buffer
        bbox = self._calculate_bbox_with_buffer(coordinates, buffer_km)

        climate_data = {}
        source = Path(data_path)
        for var in variables:
            if var not in self.worldclim_config["variables"]:
                logger.warning(f"Unknown WorldClim variable: {var}")
                continue

            if source.is_file():
                raster_path: Path | None = source
            else:
                number = var.removeprefix("bio")
                candidates = (
                    source / f"wc2.1_{resolution}_bio_{number}.tif",
                    source / f"wc2.1_{resolution}_{var}.tif",
                    source / f"{var}.tif",
                )
                raster_path = next(
                    (candidate for candidate in candidates if candidate.is_file()), None
                )
            if raster_path is None or not Path(raster_path).is_file():
                raise FileNotFoundError(
                    f"WorldClim raster not found for {var}: {source}"
                )

            with rasterio.open(raster_path) as raster:
                values = list(raster.sample([(lon, lat) for lat, lon in coordinates]))
                sampled = []
                for (lat, lon), value in zip(coordinates, values):
                    numeric_value = float(value[0])
                    if raster.nodata is not None and numeric_value == raster.nodata:
                        numeric_value = float("nan")
                    sampled.append(
                        {"latitude": lat, "longitude": lon, "value": numeric_value}
                    )
                climate_data[var] = {
                    "variable": var,
                    "description": self.worldclim_config["variables"][var],
                    "coordinates": sampled,
                    "bbox": bbox,
                    "units": self._get_variable_units(var),
                }

        logger.info(
            f"Successfully loaded climate data for {len(climate_data)} variables"
        )

        return ClimateDataset(
            data=climate_data, coordinates=coordinates, data_source=str(source)
        )

    def _calculate_bbox_with_buffer(
        self, coordinates: list[tuple[float, float]], buffer_km: float
    ) -> tuple[float, float, float, float]:
        """Calculate bounding box with buffer around coordinates."""
        if not coordinates:
            raise ValueError("No coordinates provided")

        lats, lons = zip(*coordinates)
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Convert buffer from km to degrees (approximate)
        buffer_deg = buffer_km / 111.0  # 1 degree ≈ 111 km

        bbox = (
            min_lon - buffer_deg,  # min_lon
            min_lat - buffer_deg,  # min_lat
            max_lon + buffer_deg,  # max_lon
            max_lat + buffer_deg,  # max_lat
        )

        return bbox

    def _get_variable_units(self, variable: str) -> str:
        """Get units for WorldClim variables."""
        if variable in [
            "bio1",
            "bio2",
            "bio5",
            "bio6",
            "bio7",
            "bio8",
            "bio9",
            "bio10",
            "bio11",
        ]:
            return "°C * 10"
        elif variable in [
            "bio12",
            "bio13",
            "bio14",
            "bio16",
            "bio17",
            "bio18",
            "bio19",
        ]:
            return "mm"
        elif variable == "bio3":
            return "%"
        elif variable == "bio4":
            return "°C * 100"
        elif variable == "bio15":
            return "Coefficient of Variation"
        else:
            return "Unknown"

    def load_custom_climate_data(
        self,
        raster_path: str,
        coordinates: list[tuple[float, float]],
        variable_name: str = "custom_climate",
    ) -> "ClimateDataset":
        """
        Load custom climate raster data.

        Args:
            raster_path: Path to climate raster file
            coordinates: List of (latitude, longitude) tuples
            variable_name: Name for the climate variable

        Returns:
            ClimateDataset object
        """
        logger.info(f"Loading custom climate data from {raster_path}")

        if not HAS_RASTER_DEPS:
            raise ImportError("rasterio and geopandas are required for raster data")
        if not coordinates:
            raise ValueError("coordinates must not be empty")
        if not Path(raster_path).is_file():
            raise FileNotFoundError(f"Climate raster not found: {raster_path}")

        try:
            with rasterio.open(raster_path) as raster:
                values = list(raster.sample([(lon, lat) for lat, lon in coordinates]))
                sampled = []
                for (lat, lon), value in zip(coordinates, values):
                    numeric_value = float(value[0])
                    if raster.nodata is not None and numeric_value == raster.nodata:
                        numeric_value = float("nan")
                    sampled.append(
                        {"latitude": lat, "longitude": lon, "value": numeric_value}
                    )

            return ClimateDataset(
                data={
                    variable_name: {
                        "variable": variable_name,
                        "coordinates": sampled,
                        "bbox": self._calculate_bbox_with_buffer(coordinates, 0.0),
                        "units": self._get_variable_units(variable_name),
                    }
                },
                coordinates=coordinates,
                data_source=f"Custom raster: {raster_path}",
            )

        except Exception as e:
            logger.error(f"Failed to load custom climate data: {e}")
            raise


class ClimateDataset:
    """
    Container for climate data with spatial analysis capabilities.

    Provides methods for:
    - Climate variable access
    - Spatial interpolation
    - Integration with biological data
    - Export for spatial analysis
    """

    def __init__(
        self,
        data: dict[str, Any],
        coordinates: list[tuple[float, float]],
        data_source: str = "Unknown",
    ):
        """
        Initialize climate dataset.

        Args:
            data: Dictionary of climate variable data
            coordinates: List of coordinate tuples
            data_source: Description of data source
        """
        self.data = data
        self.coordinates = coordinates
        self.data_source = data_source

        logger.info(
            f"ClimateDataset initialized: {len(self.data)} variables, "
            f"{len(self.coordinates)} locations"
        )

    def get_variables(self) -> list[str]:
        """Get list of available climate variables."""
        return list(self.data.keys())

    def get_variable_data(self, variable: str) -> pd.DataFrame:
        """
        Get data for a specific climate variable.

        Args:
            variable: Variable name

        Returns:
            DataFrame with coordinate and value data
        """
        if variable not in self.data:
            raise ValueError(f"Variable {variable} not found in dataset")

        var_data = self.data[variable]
        df = pd.DataFrame(var_data["coordinates"])
        df["variable"] = variable
        df["units"] = var_data.get("units", "Unknown")

        return df

    def get_all_variables_dataframe(self) -> pd.DataFrame:
        """
        Get all climate variables as a single DataFrame.

        Returns:
            DataFrame with all variables and coordinates
        """
        dfs: list[pd.DataFrame] = [
            self.get_variable_data(variable).rename(columns={"value": variable})[
                ["latitude", "longitude", variable]
            ]
            for variable in self.get_variables()
        ]

        if not dfs:
            return pd.DataFrame()

        # Merge all dataframes on coordinates
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=["latitude", "longitude"], how="outer")

        return result

    def export_for_h3_integration(self) -> dict[str, Any]:
        """
        Export climate data for H3 spatial integration.

        Returns:
            Dictionary with coordinates and climate data
        """
        all_data = self.get_all_variables_dataframe()

        export_data = {
            "coordinates": self.coordinates,
            "climate_variables": self.get_variables(),
            "climate_data": all_data.to_dict("records") if not all_data.empty else [],
            "data_source": self.data_source,
        }

        logger.info(
            f"Exported climate data: {len(export_data['climate_variables'])} variables, "
            f"{len(export_data['coordinates'])} locations"
        )

        return export_data

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"ClimateDataset(variables={len(self.data)}, "
            f"locations={len(self.coordinates)}, "
            f"source='{self.data_source}')"
        )
