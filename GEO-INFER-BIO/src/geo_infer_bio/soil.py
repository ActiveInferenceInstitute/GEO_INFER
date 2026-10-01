"""
GEO-INFER-BIO Soil Data Processing Module

This module provides soil data processing capabilities for biological spatial analysis,
designed to work with real-world soil datasets like ISRIC SoilGrids and other
pedological data sources relevant to biological research.

Key Features:
- ISRIC SoilGrids global soil property data (REST query endpoint)
- Custom soil dataset loading from CSV/TSV files
- Multi-depth soil profile processing
- Integration with biological sampling coordinates
- Soil health and quality indicators
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class SoilDataIntegrator:
    """
    Soil data integration for biological spatial analysis.

    Supports multiple soil data sources:
    - ISRIC SoilGrids global soil properties
    - Custom soil datasets (CSV/TSV)
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize soil data integrator.

        Args:
            cache_dir: Directory for caching downloaded soil data
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".geo_infer_bio" / "soil_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ISRIC SoilGrids configuration
        self.soilgrids_config: dict[str, Any] = {
            "base_url": "https://rest.isric.org/soilgrids/v2.0/",
            "properties": {
                "bdod": "Bulk density of the fine earth fraction",
                "cec": "Cation Exchange Capacity of the soil",
                "cfvo": "Volumetric fraction of coarse fragments",
                "clay": "Proportion of clay particles",
                "nitrogen": "Total nitrogen",
                "ocd": "Organic carbon density",
                "ocs": "Organic carbon stock",
                "phh2o": "Soil pH in H2O",
                "sand": "Proportion of sand particles",
                "silt": "Proportion of silt particles",
                "soc": "Soil organic carbon content",
            },
            "depths": {
                "0-5cm": "sl1",
                "5-15cm": "sl2",
                "15-30cm": "sl3",
                "30-60cm": "sl4",
                "60-100cm": "sl5",
                "100-200cm": "sl6",
            },
        }

        logger.info(
            f"SoilDataIntegrator initialized with cache directory: {self.cache_dir}"
        )

    def load_soilgrids_data(
        self,
        coordinates: list[tuple[float, float]],
        properties: list[str],
        depths: list[str] | None = None,
    ) -> "SoilDataset":
        """
        Load ISRIC SoilGrids data for specified coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples
            properties: List of soil property codes (e.g., ['phh2o', 'soc', 'clay'])
            depths: List of depth intervals (e.g., ['0-5cm', '5-15cm'])

        Returns:
            SoilDataset object with soil data
        """
        depths = list(depths) if depths is not None else ["0-5cm", "5-15cm"]

        logger.info(
            f"Loading SoilGrids data for {len(properties)} properties at {len(depths)} depths"
        )

        if not coordinates:
            raise ValueError("coordinates must not be empty")
        if not properties:
            raise ValueError("properties must not be empty")
        unknown_properties = sorted(
            set(properties) - set(self.soilgrids_config["properties"])
        )
        if unknown_properties:
            raise ValueError(f"Unknown SoilGrids properties: {unknown_properties}")
        unknown_depths = sorted(set(depths) - set(self.soilgrids_config["depths"]))
        if unknown_depths:
            raise ValueError(f"Unknown SoilGrids depths: {unknown_depths}")

        soil_data = {
            f"{prop}_{depth}": {
                "property": prop,
                "depth": depth,
                "description": self.soilgrids_config["properties"][prop],
                "coordinates": [],
                "units": "SoilGrids mapped units",
            }
            for prop in properties
            for depth in depths
        }

        endpoint = urljoin(self.soilgrids_config["base_url"], "properties/query")
        for latitude, longitude in coordinates:
            response = requests.get(
                endpoint,
                params={"lat": latitude, "lon": longitude},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            layers = {
                layer.get("name"): layer
                for layer in payload.get("properties", {}).get("layers", [])
            }
            for prop in properties:
                layer = layers.get(prop)
                if layer is None:
                    raise ValueError(f"SoilGrids response has no layer for {prop}")
                for depth in depths:
                    depth_record = next(
                        (
                            item
                            for item in layer.get("depths", [])
                            if item.get("label") == depth
                        ),
                        None,
                    )
                    if depth_record is None or "mean" not in depth_record.get(
                        "values", {}
                    ):
                        raise ValueError(
                            f"SoilGrids response has no value for {prop} at {depth}"
                        )
                    soil_data[f"{prop}_{depth}"]["coordinates"].append(
                        {
                            "latitude": latitude,
                            "longitude": longitude,
                            "value": depth_record["values"]["mean"],
                            "depth": depth,
                        }
                    )

        logger.info(
            f"Successfully loaded soil data for {len(soil_data)} property-depth combinations"
        )

        return SoilDataset(
            data=soil_data, coordinates=coordinates, data_source=endpoint
        )

    def load_custom_soil_data(
        self,
        soil_data_path: str,
        coordinates: list[tuple[float, float]],
        property_columns: dict[str, str],
    ) -> "SoilDataset":
        """
        Load custom soil dataset.

        Args:
            soil_data_path: Path to soil data file (CSV/TSV)
            coordinates: List of (latitude, longitude) tuples
            property_columns: Mapping of column names to soil properties

        Returns:
            SoilDataset object
        """
        logger.info(f"Loading custom soil data from {soil_data_path}")

        try:
            # Load soil data
            if soil_data_path.endswith(".csv"):
                soil_df = pd.read_csv(soil_data_path)
            else:
                soil_df = pd.read_csv(soil_data_path, sep="\t")

            if soil_df.empty:
                raise ValueError("soil data file is empty")
            latitude_column = next(
                (name for name in ("latitude", "lat") if name in soil_df), None
            )
            longitude_column = next(
                (name for name in ("longitude", "lon", "lng") if name in soil_df), None
            )
            if latitude_column is None or longitude_column is None:
                if len(soil_df) != len(coordinates):
                    raise ValueError(
                        "soil data must contain coordinates or match the supplied coordinate count"
                    )
                soil_df = soil_df.copy()
                soil_df["latitude"] = [lat for lat, _ in coordinates]
                soil_df["longitude"] = [lon for _, lon in coordinates]
                latitude_column = "latitude"
                longitude_column = "longitude"

            soil_data = {}
            for col_name, prop_name in property_columns.items():
                if col_name not in soil_df.columns:
                    raise ValueError(f"Column {col_name!r} is absent from soil data")
                soil_data[f"{prop_name}_0-30cm"] = {
                    "property": prop_name,
                    "depth": "0-30cm",
                    "coordinates": [
                        {
                            "latitude": float(row[latitude_column]),
                            "longitude": float(row[longitude_column]),
                            "value": row[col_name],
                            "depth": "0-30cm",
                        }
                        for _, row in soil_df.iterrows()
                    ],
                    "units": "source-defined",
                }

            return SoilDataset(
                data=soil_data,
                coordinates=coordinates,
                data_source=f"Custom soil data: {soil_data_path}",
            )

        except Exception as e:
            logger.error(f"Failed to load custom soil data: {e}")
            raise


class SoilDataset:
    """
    Container for soil data with spatial analysis capabilities.

    Provides methods for:
    - Soil property access
    - Multi-depth profile analysis
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
        Initialize soil dataset.

        Args:
            data: Dictionary of soil property data
            coordinates: List of coordinate tuples
            data_source: Description of data source
        """
        self.data = data
        self.coordinates = coordinates
        self.data_source = data_source

        # Parse properties and depths
        self._parse_properties_and_depths()

        logger.info(
            f"SoilDataset initialized: {len(self.properties)} properties, "
            f"{len(self.depths)} depths, {len(self.coordinates)} locations"
        )

    def _parse_properties_and_depths(self) -> None:
        """Parse available properties and depths from data keys."""
        properties = set()
        depths = set()

        for key in self.data.keys():
            if "_" in key:
                prop, depth = key.rsplit("_", 1)
                properties.add(prop)
                depths.add(depth)
            else:
                properties.add(key)
                depths.add("unknown")

        self.properties = sorted(list(properties))
        self.depths = sorted(list(depths))

    def get_properties(self) -> list[str]:
        """Get list of available soil properties."""
        return self.properties

    def get_depths(self) -> list[str]:
        """Get list of available depth intervals."""
        return self.depths

    def get_property_data(
        self, property_name: str, depth: str | None = None
    ) -> pd.DataFrame:
        """
        Get data for a specific soil property.

        Args:
            property_name: Property name
            depth: Depth interval (if None, returns all depths)

        Returns:
            DataFrame with coordinate and value data
        """
        if depth:
            key = f"{property_name}_{depth}"
            if key not in self.data:
                raise ValueError(f"Property {property_name} at depth {depth} not found")

            prop_data = self.data[key]
            df = pd.DataFrame(prop_data["coordinates"])
            df["property"] = property_name
            df["units"] = prop_data.get("units", "Unknown")

            return df
        else:
            # Return all depths for this property
            dfs = []
            for d in self.depths:
                key = f"{property_name}_{d}"
                if key in self.data:
                    prop_data = self.data[key]
                    df = pd.DataFrame(prop_data["coordinates"])
                    df["property"] = property_name
                    df["units"] = prop_data.get("units", "Unknown")
                    dfs.append(df)

            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

    def get_soil_profile(
        self, latitude: float, longitude: float, tolerance: float = 0.01
    ) -> pd.DataFrame:
        """
        Get soil profile data for a specific location.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            tolerance: Coordinate tolerance for matching

        Returns:
            DataFrame with soil profile by depth
        """
        profile_data = []

        for prop in self.properties:
            for depth in self.depths:
                key = f"{prop}_{depth}"
                if key in self.data:
                    prop_data = self.data[key]

                    # Find closest coordinate match
                    for coord_data in prop_data["coordinates"]:
                        if (
                            abs(coord_data["latitude"] - latitude) <= tolerance
                            and abs(coord_data["longitude"] - longitude) <= tolerance
                        ):
                            profile_data.append(
                                {
                                    "property": prop,
                                    "depth": depth,
                                    "value": coord_data["value"],
                                    "units": prop_data.get("units", "Unknown"),
                                    "description": prop_data.get("description", ""),
                                }
                            )
                            break

        return pd.DataFrame(profile_data)

    def get_all_properties_dataframe(self) -> pd.DataFrame:
        """
        Get all soil properties as a single DataFrame.

        Returns:
            DataFrame with all properties, depths, and coordinates
        """
        all_data = []

        for key, prop_data in self.data.items():
            df = pd.DataFrame(prop_data["coordinates"])
            df["property_depth"] = key
            df["property"] = prop_data.get("property", key.split("_")[0])
            df["depth"] = prop_data.get("depth", "unknown")
            df["units"] = prop_data.get("units", "Unknown")
            all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_soil_health_indicators(self) -> pd.DataFrame:
        """
        Calculate soil health indicators from available properties.

        Returns:
            DataFrame with soil health metrics
        """
        health_data = []

        # Group data by coordinates
        coord_groups: dict[tuple[float, float], dict[str, Any]] = {}
        for key, prop_data in self.data.items():
            for coord_data in prop_data["coordinates"]:
                coord_key = (coord_data["latitude"], coord_data["longitude"])
                if coord_key not in coord_groups:
                    coord_groups[coord_key] = {}
                coord_groups[coord_key][key] = coord_data["value"]

        # Calculate health indicators for each location
        for (lat, lon), values in coord_groups.items():
            health_indicators = {"latitude": lat, "longitude": lon}

            # Soil pH health score (optimal range 6.0-7.0)
            ph_keys = [k for k in values.keys() if "phh2o" in k]
            if ph_keys:
                avg_ph = np.mean([values[k] for k in ph_keys])
                if 6.0 <= avg_ph <= 7.0:
                    health_indicators["ph_health_score"] = 1.0
                else:
                    health_indicators["ph_health_score"] = max(
                        0, 1 - abs(avg_ph - 6.5) / 2.0
                    )

            # Organic carbon health score (higher is better, up to 3%)
            soc_keys = [k for k in values.keys() if "soc" in k]
            if soc_keys:
                avg_soc = np.mean([values[k] for k in soc_keys])
                health_indicators["organic_carbon_score"] = min(
                    1.0, avg_soc / 30.0
                )  # Scale to 3%

            # Soil texture balance score
            clay_keys = [k for k in values.keys() if "clay" in k]
            sand_keys = [k for k in values.keys() if "sand" in k]
            if clay_keys and sand_keys:
                avg_clay = np.mean([values[k] for k in clay_keys])
                avg_sand = np.mean([values[k] for k in sand_keys])
                # Optimal balance: 20-40% clay, 30-60% sand
                clay_score = (
                    1.0 if 20 <= avg_clay <= 40 else max(0, 1 - abs(avg_clay - 30) / 30)
                )
                sand_score = (
                    1.0 if 30 <= avg_sand <= 60 else max(0, 1 - abs(avg_sand - 45) / 45)
                )
                health_indicators["texture_balance_score"] = (
                    clay_score + sand_score
                ) / 2

            # Overall soil health score
            scores = [v for k, v in health_indicators.items() if k.endswith("_score")]
            if scores:
                health_indicators["overall_soil_health"] = float(np.mean(scores))

            health_data.append(health_indicators)

        return pd.DataFrame(health_data)

    def export_for_h3_integration(self) -> dict[str, Any]:
        """
        Export soil data for H3 spatial integration.

        Returns:
            Dictionary with coordinates and soil data
        """
        properties_out = self.coordinates
        soil_props = self.properties
        export_data: dict[str, Any] = {
            "coordinates": properties_out,
            "soil_properties": soil_props,
            "depths": self.depths,
            "soil_data": self.data,
            "data_source": self.data_source,
        }

        logger.info(
            f"Exported soil data: {len(soil_props)} properties, "
            f"{len(properties_out)} locations"
        )

        return export_data

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"SoilDataset(properties={len(self.properties)}, "
            f"depths={len(self.depths)}, "
            f"locations={len(self.coordinates)}, "
            f"source='{self.data_source}')"
        )
