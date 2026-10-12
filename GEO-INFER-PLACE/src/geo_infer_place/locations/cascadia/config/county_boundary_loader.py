#!/usr/bin/env python3
"""
County Boundary Loader for Cascadia Analysis

This utility loads county boundary data from YAML configuration and GeoJSON files,
providing proper geometry objects for H3 geo_to_cells operations and spatial analysis.
Updated to use H3 v4 API.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Any
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping
import requests
import tempfile
import zipfile

logger = logging.getLogger(__name__)


class CountyBoundaryLoader:
    """Loads and manages county boundary data for the Cascadia analysis"""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.boundaries_config = self._load_boundaries_config()
        self.county_geometries = {}

    def _load_boundaries_config(self) -> dict[str, Any]:
        """Load the county boundaries configuration file"""
        config_path = self.config_dir / "county_boundaries.yaml"

        if not config_path.exists():
            logger.warning(f"County boundaries config not found: {config_path}")
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded county boundaries configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load county boundaries config: {e}")
            return {}

    def get_county_info(self, county_key: str) -> dict[str, Any] | None:
        """Get county information from configuration"""
        return self.boundaries_config.get(county_key)

    def load_county_geometry(self, county_key: str) -> Polygon | dict[str, Any] | None:
        """
        Load county geometry from GeoJSON file

        Returns either a Shapely geometry object or a properly formatted GeoJSON dict
        suitable for H3 v4 geo_to_cells API
        """
        if county_key in self.county_geometries:
            return self.county_geometries[county_key]

        county_info = self.get_county_info(county_key)
        if not county_info:
            logger.error(f"County info not found for: {county_key}")
            return None

        geometry_file = county_info.get("geometry_file")
        if not geometry_file:
            logger.error(f"No geometry file specified for county: {county_key}")
            return None

        geometry_path = self.config_dir / geometry_file

        try:
            if geometry_path.exists():
                # First try loading as GeoJSON directly
                try:
                    with open(geometry_path) as f:
                        geojson_data = json.load(f)

                    # If it's a FeatureCollection, extract the first feature's geometry
                    if geojson_data.get(
                        "type"
                    ) == "FeatureCollection" and geojson_data.get("features"):
                        feature = geojson_data["features"][0]
                        if feature.get("geometry"):
                            geometry = feature["geometry"]
                            # Ensure proper GeoJSON structure for H3 v4
                            if geometry.get("type") == "Polygon" and geometry.get(
                                "coordinates"
                            ):
                                if not isinstance(
                                    geometry["coordinates"][0][0], (list, tuple)
                                ):
                                    geometry["coordinates"] = [geometry["coordinates"]]

                            self.county_geometries[county_key] = geometry
                            logger.info(
                                f"Loaded GeoJSON geometry for {county_key} from {geometry_path}"
                            )
                            return geometry

                    # If it's a Feature, extract its geometry
                    elif geojson_data.get("type") == "Feature" and geojson_data.get(
                        "geometry"
                    ):
                        geometry = geojson_data["geometry"]
                        # Ensure proper GeoJSON structure for H3 v4
                        if geometry.get("type") == "Polygon" and geometry.get(
                            "coordinates"
                        ):
                            if not isinstance(
                                geometry["coordinates"][0][0], (list, tuple)
                            ):
                                geometry["coordinates"] = [geometry["coordinates"]]

                        self.county_geometries[county_key] = geometry
                        logger.info(
                            f"Loaded GeoJSON geometry for {county_key} from {geometry_path}"
                        )
                        return geometry

                    # If it's a Geometry object directly
                    elif geojson_data.get("type") in ("Polygon", "MultiPolygon"):
                        # Ensure proper GeoJSON structure for H3 v4
                        if geojson_data.get("type") == "Polygon" and geojson_data.get(
                            "coordinates"
                        ):
                            if not isinstance(
                                geojson_data["coordinates"][0][0], (list, tuple)
                            ):
                                geojson_data["coordinates"] = [
                                    geojson_data["coordinates"]
                                ]

                        self.county_geometries[county_key] = geojson_data
                        logger.info(
                            f"Loaded GeoJSON geometry for {county_key} from {geometry_path}"
                        )
                        return geojson_data

                except Exception as json_error:
                    logger.warning(
                        f"Failed to load as direct GeoJSON: {json_error}, trying geopandas"
                    )

                # Fall back to geopandas if direct JSON loading fails
                gdf = gpd.read_file(geometry_path)
                if not gdf.empty:
                    geometry = gdf.iloc[0].geometry
                    if isinstance(geometry, (Polygon, MultiPolygon)):
                        # Convert to GeoJSON for H3 v4 compatibility
                        geojson_geometry = mapping(geometry)
                        # Ensure proper GeoJSON structure for H3 v4
                        if geojson_geometry.get(
                            "type"
                        ) == "Polygon" and geojson_geometry.get("coordinates"):
                            if not isinstance(
                                geojson_geometry["coordinates"][0][0], (list, tuple)
                            ):
                                geojson_geometry["coordinates"] = [
                                    geojson_geometry["coordinates"]
                                ]

                        self.county_geometries[county_key] = geojson_geometry
                        logger.info(
                            f"Loaded geometry for {county_key} from {geometry_path} using geopandas"
                        )
                        return geojson_geometry
                    else:
                        logger.error(
                            f"Invalid geometry type for {county_key}: {type(geometry)}"
                        )
                        return None
                else:
                    logger.error(f"Empty GeoJSON file for {county_key}")
                    return None
            else:
                logger.warning(f"Geometry file not found: {geometry_path}")
                # Try to create from bounds
                return self._create_geometry_from_bounds(county_info)

        except Exception as e:
            logger.error(f"Failed to load geometry for {county_key}: {e}")
            return None

    def _create_geometry_from_bounds(
        self, county_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create a simple polygon from county bounds as GeoJSON for H3 v4"""
        bounds = county_info.get("bounds", {})
        if not bounds:
            logger.error("No bounds information available")
            return None

        try:
            # Create a simple bounding box polygon
            west = bounds.get("west", -121.5)
            east = bounds.get("east", -119.5)
            south = bounds.get("south", 40.0)
            north = bounds.get("north", 41.2)

            # Create GeoJSON directly for H3 v4
            geojson_polygon = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]
                ],
            }

            county_key = county_info.get("name", "unknown").lower().replace(" ", "_")
            self.county_geometries[county_key] = geojson_polygon
            logger.info(f"Created GeoJSON geometry from bounds for {county_key}")
            return geojson_polygon

        except Exception as e:
            logger.error(f"Failed to create geometry from bounds: {e}")
            return None

    def download_county_boundary(self, county_key: str) -> bool:
        """Download county boundary from official sources"""
        county_info = self.get_county_info(county_key)
        if not county_info:
            logger.error(f"County info not found for: {county_key}")
            return False

        data_sources = county_info.get("data_sources", [])
        if not data_sources:
            logger.error(f"No data sources for {county_key}")
            return False

        # Try the first data source
        primary_source = data_sources[0]

        url = primary_source.get("url")
        format_type = primary_source.get("format")

        if not url or not format_type:
            logger.error(f"Invalid data source configuration for {county_key}")
            return False

        try:
            logger.info(f"Downloading boundary data for {county_key} from {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if format_type == "shapefile":
                return self._process_shapefile(response.content, county_key)
            elif format_type == "geojson":
                return self._process_geojson(response.content, county_key)
            else:
                logger.error(f"Unsupported format: {format_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to download boundary for {county_key}: {e}")
            return False

    def _process_shapefile(self, content: bytes, county_key: str) -> bool:
        """Process downloaded shapefile content"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "county_boundary.zip"

                # Save zip content
                with open(zip_path, "wb") as f:
                    f.write(content)

                # Extract shapefile
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)

                # Find the .shp file
                shp_files = list(temp_path.glob("*.shp"))
                if not shp_files:
                    logger.error("No shapefile found in downloaded content")
                    return False

                # Load with geopandas
                gdf = gpd.read_file(shp_files[0])

                # Filter for the specific county if needed
                county_info = self.get_county_info(county_key)
                fips_code = county_info.get("fips_code")

                if fips_code and "GEOID" in gdf.columns:
                    gdf = gdf[gdf["GEOID"] == fips_code]

                if not gdf.empty:
                    # Save to local file as GeoJSON
                    geometry_file = county_info.get(
                        "geometry_file", f"{county_key}_boundary.geojson"
                    )
                    geometry_path = self.config_dir / geometry_file
                    gdf.to_file(geometry_path, driver="GeoJSON")

                    # Create GeoJSON for H3 v4
                    geometry = gdf.iloc[0].geometry
                    geojson_geometry = mapping(geometry)

                    # Ensure proper GeoJSON structure for H3 v4
                    if geojson_geometry.get(
                        "type"
                    ) == "Polygon" and geojson_geometry.get("coordinates"):
                        if not isinstance(
                            geojson_geometry["coordinates"][0][0], (list, tuple)
                        ):
                            geojson_geometry["coordinates"] = [
                                geojson_geometry["coordinates"]
                            ]

                    self.county_geometries[county_key] = geojson_geometry

                    logger.info(
                        f"Successfully processed and saved boundary for {county_key}"
                    )
                    return True
                else:
                    logger.error(f"No geometry found for {county_key}")
                    return False

        except Exception as e:
            logger.error(f"Failed to process shapefile for {county_key}: {e}")
            return False

    def _process_geojson(self, content: bytes, county_key: str) -> bool:
        """Process downloaded GeoJSON content"""
        try:
            # Parse GeoJSON
            geojson_data = json.loads(content.decode("utf-8"))

            # Save to local file
            county_info = self.get_county_info(county_key)
            geometry_file = county_info.get(
                "geometry_file", f"{county_key}_boundary.geojson"
            )
            geometry_path = self.config_dir / geometry_file

            with open(geometry_path, "w") as f:
                json.dump(geojson_data, f)

            # Extract geometry for H3 v4
            if geojson_data.get("type") == "FeatureCollection" and geojson_data.get(
                "features"
            ):
                feature = geojson_data["features"][0]
                if feature.get("geometry"):
                    geometry = feature["geometry"]
                    # Ensure proper GeoJSON structure for H3 v4
                    if geometry.get("type") == "Polygon" and geometry.get(
                        "coordinates"
                    ):
                        if not isinstance(geometry["coordinates"][0][0], (list, tuple)):
                            geometry["coordinates"] = [geometry["coordinates"]]

                    self.county_geometries[county_key] = geometry
                    logger.info(
                        f"Successfully processed and saved boundary for {county_key}"
                    )
                    return True

            logger.error(f"Invalid or missing geometry in GeoJSON for {county_key}")
            return False

        except Exception as e:
            logger.error(f"Failed to process GeoJSON for {county_key}: {e}")
            return False

    def get_all_county_geometries(
        self, target_counties: dict[str, list[str]]
    ) -> dict[str, dict[str, Any]]:
        """Get geometries for all target counties"""
        geometries = {}

        for state, counties in target_counties.items():
            if state not in geometries:
                geometries[state] = {}

            if counties == ["all"] or "all" in counties:
                # Get all counties for this state
                state_counties = [
                    key
                    for key in self.boundaries_config.keys()
                    if key.startswith(f"{state.lower()}_")
                ]
            else:
                # Get specific counties
                state_counties = [
                    f"{state.lower()}_{county.lower().replace(' ', '_')}"
                    for county in counties
                ]

            for county_key in state_counties:
                geometry = self.load_county_geometry(county_key)
                if geometry:
                    # Extract county name from key (e.g., 'ca_lassen' -> 'Lassen')
                    parts = county_key.split("_", 1)
                    if len(parts) == 2:
                        county_name = parts[1].title()
                        geometries[state][county_name] = geometry
                    else:
                        # Use full key if we can't parse it
                        geometries[state][county_key] = geometry
                else:
                    logger.warning(f"Could not load geometry for {county_key}")

        return geometries

    def validate_geometry(self, geometry: dict[str, Any] | Polygon) -> bool:
        """Validate that a geometry is suitable for H3 geo_to_cells"""
        # For GeoJSON dict
        if isinstance(geometry, dict):
            if geometry.get("type") not in ("Polygon", "MultiPolygon"):
                logger.warning(f"Invalid GeoJSON geometry type: {geometry.get('type')}")
                return False

            if not geometry.get("coordinates"):
                logger.warning("Missing coordinates in GeoJSON geometry")
                return False

            return True

        # For Shapely geometry
        elif isinstance(geometry, (Polygon, MultiPolygon)):
            if not geometry.is_valid:
                logger.warning("Geometry is not valid")
                return False

            if geometry.is_empty:
                logger.warning("Geometry is empty")
                return False

            if geometry.area < 0.001:  # Very small area
                logger.warning("Geometry area is too small")
                return False

            return True

        else:
            logger.warning(f"Unsupported geometry type: {type(geometry)}")
            return False


def create_county_boundary_loader() -> CountyBoundaryLoader:
    """Factory function to create a county boundary loader"""
    config_dir = Path(__file__).parent
    return CountyBoundaryLoader(config_dir)


if __name__ == "__main__":
    # Test the county boundary loader
    logging.basicConfig(level=logging.INFO)

    loader = create_county_boundary_loader()

    # Test loading Lassen County
    lassen_geometry = loader.load_county_geometry("ca_lassen")
    if lassen_geometry:
        print(f"Successfully loaded Lassen County geometry: {lassen_geometry}")
        print(f"Geometry type: {type(lassen_geometry)}")

        # For GeoJSON dict
        if isinstance(lassen_geometry, dict):
            print(f"GeoJSON type: {lassen_geometry.get('type')}")
        # For Shapely geometry
        elif isinstance(lassen_geometry, (Polygon, MultiPolygon)):
            print(f"Geometry area: {lassen_geometry.area}")
    else:
        print("Failed to load Lassen County geometry")
