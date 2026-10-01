"""
Cascadian Water Rights Data Sources

This module is responsible for fetching water rights data from the official
state data portals for California, Oregon, and Washington.
"""

import os
import logging
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
import json

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
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class CascadianWaterRightsDataSources:
    """
    Manages the acquisition of water rights data for the Cascadia region.

    This class handles downloading real data from state-level ArcGIS REST
    services and public data portals.
    """

    def __init__(self):
        self.data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "water_rights"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "data_urls.json")
        try:
            with open(config_path) as f:
                self.config = json.load(f).get("water_rights", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse water_rights config: {e}")
            self.config = {}

        self.ca_url = self.config.get("ca_ewrims_csv_url")
        self.ca_service_url = self.config.get("ca_ewrims_service_url")
        self.or_url = self.config.get("or_wr_service_url")
        self.or_url_comment = self.config.get("or_wr_service_url_comment")

        self.ca_csv_path = os.path.join(self.data_dir, "ca_water_rights_summary.csv")
        logger.info("Initialized CascadianWaterRightsDataSources.")

    def _fetch_arcgis_generic(self, service_url: str, bbox: tuple) -> gpd.GeoDataFrame:
        """Generic ArcGIS query helper."""
        if not service_url:
            return gpd.GeoDataFrame()

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        params = {
            "geometry": bbox_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }

        try:
            response = requests.get(f"{service_url}/query", params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            if data.get("features"):
                gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
                return gdf
        except Exception as e:
            logger.error(f"ArcGIS query failed for {service_url}: {e}")

        return gpd.GeoDataFrame()

    def _query_osm_overpass_wells(
        self, bbox: tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """Query OSM Overpass for wells and dams as water rights proxies."""
        min_lon, min_lat, max_lon, max_lat = bbox

        overpass_query = f"""
        [out:json][timeout:120];
        (
          node["man_made"="water_well"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["water"="well"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["waterway"="dam"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["waterway"="dam"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.info("WaterRights: Querying OSM for wells/dams (proxy)...")

        try:
            response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
            features = []

            from shapely.geometry import LineString

            for element in elements:
                geom = None

                if element["type"] == "node":
                    geom = Point(element["lon"], element["lat"])
                elif element["type"] == "way" and "nodes" in element:
                    c_list = [nodes.get(n) for n in element["nodes"]]
                    c_list = [c for c in c_list if c]
                    if len(c_list) >= 2:
                        geom = LineString(c_list)

                if geom:
                    tags = element.get("tags", {})
                    features.append(
                        {
                            "geometry": geom,
                            "source": "OSM",
                            "type": tags.get("man_made")
                            or tags.get("water")
                            or tags.get("waterway"),
                            "name": tags.get("name", "Unknown"),
                        }
                    )

            if not features:
                return gpd.GeoDataFrame()

            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            logger.info(f"WaterRights: Fetched {len(gdf)} OSM proxy features.")
            return gdf

        except Exception as e:
            logger.error(f"OSM Overpass query failed: {e}")
            return gpd.GeoDataFrame()

    def _download_ca_file(self) -> None:
        """Downloads the CA eWRIMS data file if it's not already cached."""
        if os.path.exists(self.ca_csv_path):
            logger.info(f"Using cached CA water rights file: {self.ca_csv_path}")
            return

        if not self.ca_url:
            logger.error("California water rights URL not configured.")
            return

        logger.info(f"Attempting to download CA water rights file from {self.ca_url}...")
        try:
            response = requests.get(self.ca_url, timeout=300)  # Increased timeout for large file
            response.raise_for_status()

            with open(self.ca_csv_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded CA water rights file to {self.ca_csv_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download CA water rights from primary URL: {e}")

    def _fetch_ca_data(self, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """Fetches CA water rights data, preferring ArcGIS then CSV."""
        # Try ArcGIS first
        if self.ca_service_url:
            logger.info("Attempting to fetch CA Water Rights from ArcGIS...")
            gdf = self._fetch_arcgis_generic(self.ca_service_url, bbox)
            if not gdf.empty:
                gdf["state"] = "CA"
                return gdf

        # Fallback to CSV download handled in previous version, but simplified here to avoid large downloads
        # If ArcGIS fails, return empty so we rely on OSM fallback or nothing,
        # unless user explicitly wants the huge CSV.
        return gpd.GeoDataFrame()

    def _fetch_or_data(self, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """Fetches Oregon water rights data (Points of Diversion) from the OWRD ArcGIS service."""
        if not self.or_url:
            return gpd.GeoDataFrame()

        # Layer IDs for Points of Diversion
        gdfs = []
        for layer_id in [0, 1]:
            query_url = f"{self.or_url}/{layer_id}"
            gdf = self._fetch_arcgis_generic(query_url, bbox)
            if not gdf.empty:
                gdfs.append(gdf)

        if not gdfs:
            return gpd.GeoDataFrame()

        or_gdf = pd.concat(gdfs, ignore_index=True)
        or_gdf["state"] = "OR"
        return or_gdf

    def fetch_all_water_rights_data(self, target_hexagons: list) -> gpd.GeoDataFrame:
        """
        Fetches water rights data for all three states (CA, OR, WA).
        """

        if not target_hexagons:
            logger.error("Cannot fetch water rights without target hexagons for bounding box.")
            return gpd.GeoDataFrame()

        # Correct the bounding box creation for the polygon clipping
        hex_boundaries = [
            Polygon([(lon, lat) for lat, lon in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        min_lon = min(p.bounds[0] for p in hex_boundaries)
        min_lat = min(p.bounds[1] for p in hex_boundaries)
        max_lon = max(p.bounds[2] for p in hex_boundaries)
        max_lat = max(p.bounds[3] for p in hex_boundaries)
        bbox = (min_lon, min_lat, max_lon, max_lat)

        all_gdfs = []

        # State sources
        ca_gdf = self._fetch_ca_data(bbox)
        if not ca_gdf.empty:
            all_gdfs.append(ca_gdf)

        or_gdf = self._fetch_or_data(bbox)
        if not or_gdf.empty:
            all_gdfs.append(or_gdf)

        # OSM Fallback if very little data found (or always to augment?)
        # Let's augment, as official records + physical wells are complementary
        osm_gdf = self._query_osm_overpass_wells(bbox)
        if not osm_gdf.empty:
            osm_gdf["state"] = "OSM"
            all_gdfs.append(osm_gdf)

        if not all_gdfs:
            logger.warning("No water rights data could be loaded for any state in the target area.")
            return gpd.GeoDataFrame()

        return pd.concat(all_gdfs, ignore_index=True)
