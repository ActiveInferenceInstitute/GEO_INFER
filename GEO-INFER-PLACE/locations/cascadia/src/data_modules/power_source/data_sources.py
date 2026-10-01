"""
Cascadian Power Source Data Sources

This module is responsible for fetching and loading power infrastructure
data (e.g., transmission lines) from the HIFLD open data portal.
"""

import logging
import os
import geopandas as gpd
import requests
import io
import json
from shapely.geometry import Polygon

import pandas as pd

try:
    from geo_infer_space.utils.h3_utils import (
        cell_to_latlngjson,
        latlng_to_cell,
        cell_to_latlng,
        cell_to_latlng_boundary,
        polygon_to_cells,
    )
except ImportError:
    import h3

    latlng_to_cell = h3.latlng_to_cell
    cell_to_latlng = h3.cell_to_latlng
    cell_to_latlng_boundary = h3.cell_to_boundary
    polygon_to_cells = h3.polygon_to_cells
    cell_to_latlngjson = None  # Not used in this module

logger = logging.getLogger(__name__)


class CascadianPowerSourceDataSources:
    """Handles fetching and loading of power infrastructure data."""

    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "data_urls.json")
        try:
            with open(config_path) as f:
                self.config = json.load(f).get("power_source", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse power_source config: {e}")
            self.config = {}

        self.hifld_transmission_url = self.config.get("hifld_transmission_url")
        self.hifld_power_plants_url = self.config.get("hifld_power_plants_url")

    def _calculate_bbox_from_hexagons(
        self, hexagons: list[str]
    ) -> tuple[float, float, float, float]:
        """Calculates a bounding box from a list of H3 hexagons."""
        boundaries = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)]) for h in hexagons
        ]
        min_lon = min(b.bounds[0] for b in boundaries)
        min_lat = min(b.bounds[1] for b in boundaries)
        max_lon = max(b.bounds[2] for b in boundaries)
        max_lat = max(b.bounds[3] for b in boundaries)
        return (min_lon, min_lat, max_lon, max_lat)

    def _query_hifld_service(
        self, service_url: str, layer_name: str, hexagons: list[str]
    ) -> gpd.GeoDataFrame:
        """Generic function to query a HIFLD service layer for a given list of hexagons."""
        if not service_url:
            logger.error(f"HIFLD service URL for '{layer_name}' not configured.")
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

        if not hexagons:
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

        bbox = self._calculate_bbox_from_hexagons(hexagons)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        params = {
            "where": "1=1",
            "outFields": "*",
            "geometry": bbox_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "f": "geojson",
        }

        logger.info(f"Querying HIFLD {layer_name} service with bbox: {bbox_str}")
        try:
            # The query endpoint is typically at '/query' relative to the layer URL
            query_url = f"{service_url}/query"
            response = requests.get(query_url, params=params, timeout=180)
            response.raise_for_status()

            # Check for empty response
            if not response.text or not response.text.strip():
                logger.info(f"Empty response from HIFLD {layer_name} service.")
                return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

            gdf = gpd.read_file(io.StringIO(response.text))
            logger.info(f"Successfully loaded {len(gdf)} {layer_name} records.")
            return gdf
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {layer_name} data from HIFLD service: {e}")
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
        except Exception as e:
            logger.error(f"Failed to read GeoJSON response from HIFLD {layer_name} service: {e}")
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

    def _query_osm_overpass_power(self, bbox: tuple[float, float, float, float]) -> dict:
        """Query OSM Overpass for power infrastructure."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Query for power lines and plants
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way["power"="line"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["power"="plant"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["power"="plant"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["power"="plant"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.info("PowerSource: Querying OSM for power lines/plants (fallback/augment)...")

        try:
            response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}

            lines = []
            plants = []

            from shapely.geometry import LineString, Polygon, Point

            for element in elements:
                geom = None

                if element["type"] == "node" and element.get("tags", {}).get("power") == "plant":
                    geom = Point(element["lon"], element["lat"])
                    plants.append(
                        {
                            "geometry": geom,
                            "source": "OSM",
                            "name": element["tags"].get("name", "Unknown Plant"),
                        }
                    )

                elif element["type"] == "way" and "nodes" in element:
                    c_list = [nodes.get(n) for n in element["nodes"]]
                    c_list = [c for c in c_list if c]

                    if len(c_list) >= 2:
                        # Check if it's a line or plant
                        if element.get("tags", {}).get("power") == "line":
                            lines.append(
                                {
                                    "geometry": LineString(c_list),
                                    "source": "OSM",
                                    "voltage": element["tags"].get("voltage", "Unknown"),
                                }
                            )
                        elif element.get("tags", {}).get("power") == "plant" and len(c_list) >= 4:
                            try:
                                poly = Polygon(c_list)
                                if poly.is_valid:
                                    plants.append(
                                        {
                                            "geometry": poly,
                                            "source": "OSM",
                                            "name": element["tags"].get("name", "Unknown Plant"),
                                        }
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "OSM plant feature construction failed; skipping feature: %s",
                                    exc,
                                )

            lines_gdf = (
                gpd.GeoDataFrame(lines, crs="EPSG:4326")
                if lines
                else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            )
            plants_gdf = (
                gpd.GeoDataFrame(plants, crs="EPSG:4326")
                if plants
                else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            )

            logger.info(
                f"PowerSource: Fetched {len(lines_gdf)} lines and {len(plants_gdf)} plants from OSM."
            )
            return {"transmission_lines": lines_gdf, "power_plants": plants_gdf}

        except Exception as e:
            logger.error(f"OSM Overpass query failed for power: {e}")
            return {
                "transmission_lines": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
                "power_plants": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
            }

    def fetch_power_infrastructure_features(self, hexagons: list[str]) -> dict:
        """
        Fetches all power infrastructure (transmission lines, power plants)
        for the area covered by the given hexagons.
        """
        logger.info("Fetching all power infrastructure features from HIFLD...")

        transmission_lines = self._query_hifld_service(
            self.hifld_transmission_url, "transmission lines", hexagons
        )

        power_plants = self._query_hifld_service(
            self.hifld_power_plants_url, "power plants", hexagons
        )

        # Augment/Fallback with OSM
        if transmission_lines.empty or power_plants.empty:
            logger.info("Incomplete HIFLD data, querying OSM...")
            bbox = self._calculate_bbox_from_hexagons(hexagons)
            osm_data = self._query_osm_overpass_power(bbox)

            if transmission_lines.empty:
                transmission_lines = osm_data["transmission_lines"]
            elif not osm_data["transmission_lines"].empty:
                transmission_lines = pd.concat(
                    [transmission_lines, osm_data["transmission_lines"]],
                    ignore_index=True,
                )

            if power_plants.empty:
                power_plants = osm_data["power_plants"]
            elif not osm_data["power_plants"].empty:
                power_plants = pd.concat(
                    [power_plants, osm_data["power_plants"]], ignore_index=True
                )

        return {"transmission_lines": transmission_lines, "power_plants": power_plants}
