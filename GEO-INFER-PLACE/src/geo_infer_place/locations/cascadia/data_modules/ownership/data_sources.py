"""
Cascadian Ownership Data Sources

This module is responsible for fetching parcel-level ownership data by
querying public ArcGIS REST services.
"""

from importlib.resources import files

import json
import logging
import os
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Polygon

from geo_infer_space.utils.h3_utils import (
    cell_to_latlng_boundary,
)

logger = logging.getLogger(__name__)


class CascadianOwnershipDataSources:
    """
    Handles fetching of ownership parcel data from public ArcGIS services.
    """

    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "data", "parcels")
        os.makedirs(self.data_dir, exist_ok=True)

        config_path = (
            Path(str(files("geo_infer_place.locations.cascadia.config")))
            / "data_urls.json"
        )
        try:
            with open(config_path) as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse config file at {config_path}: {e}")
            self.config = {}

        self.arcgis_service_urls = {
            "CA": self.config.get("data_sources", {}).get("ca_parcels", {}).get("url"),
            "DEL_NORTE": self.config.get("data_sources", {})
            .get("del_norte_parcels", {})
            .get("url"),
        }
        self.arcgis_service_urls = {
            k: v for k, v in self.arcgis_service_urls.items() if v
        }

    def _fetch_arcgis_parcels(self, service_url: str, bbox: tuple) -> gpd.GeoDataFrame:
        """
        Fetches parcel data from an ArcGIS Feature/MapServer for a given bounding box.
        """
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        params = {
            "geometry": bbox_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",  # Get all available fields
            "outSR": "4326",
            "f": "geojson",
        }

        query_url = f"{service_url}/query"
        logger.info(f"Querying ArcGIS service: {query_url} with bbox: {bbox_str}")

        try:
            response = requests.get(query_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Check for error response
            if "error" in data:
                logger.error(f"ArcGIS error: {data['error']}")
                return gpd.GeoDataFrame()

            if not data.get("features"):
                logger.info(
                    "No features returned from the ArcGIS service for the given extent."
                )
                return gpd.GeoDataFrame()

            # The geojson response from ArcGIS is directly readable by GeoPandas
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            gdf.set_crs("EPSG:4326", inplace=True)

            logger.info(f"Successfully fetched {len(gdf)} parcels from {service_url}")
            return gdf

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying ArcGIS service at {service_url}: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing GeoJSON response from {service_url}: {e}")

        return gpd.GeoDataFrame()

    def _query_osm_overpass(self, bbox: tuple) -> gpd.GeoDataFrame:
        """
        Query OpenStreetMap Overpass API for building footprints as parcel proxies.
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Overpass QL query for buildings (proxy for parcels)
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.info(
            "Ownership: Querying OpenStreetMap Overpass API for buildings (parcel proxy)..."
        )

        try:
            response = requests.post(
                overpass_url, data={"data": overpass_query}, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            nodes = {
                e["id"]: (e["lon"], e["lat"])
                for e in data.get("elements", [])
                if e["type"] == "node"
            }
            features = []

            for element in data.get("elements", []):
                if element["type"] == "way" and "tags" in element:
                    coords = [nodes.get(n) for n in element.get("nodes", [])]
                    coords = [c for c in coords if c is not None]

                    if len(coords) >= 4:
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid:
                                tags = element["tags"]
                                feature = {
                                    "geometry": poly,
                                    "source": "OSM",
                                    "parcel_id": str(element["id"]),
                                    "OWNER_NAME": tags.get("operator")
                                    or tags.get("name")
                                    or "Unknown",
                                    "LAND_USE": tags.get("building"),
                                    "ACRES": poly.area
                                    * 111000
                                    * 111000
                                    * 0.000247105,  # Very rough estimate, mainly for relative size
                                }
                                features.append(feature)
                        except Exception as exc:
                            logger.warning(
                                "OSM parcel feature construction failed; skipping feature: %s",
                                exc,
                            )

            if not features:
                logger.info("Ownership: No OSM buildings found.")
                return gpd.GeoDataFrame()

            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            logger.info(
                f"Ownership: Fetched {len(gdf)} OSM buildings as parcel proxies."
            )
            return gdf

        except Exception as e:
            logger.error(f"Ownership: Overpass API error: {e}")
            return gpd.GeoDataFrame()

    def fetch_all_parcel_data(self, target_hexagons: list[str]) -> gpd.GeoDataFrame:
        """
        Fetches all available parcel data from configured ArcGIS services or OSM.
        """
        if not target_hexagons:
            logger.warning("Cannot fetch parcel data without target hexagons.")
            return gpd.GeoDataFrame()

        # 1. Calculate the total bounding box for the H3 hexagons
        hex_polygons = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        min_lon = min(p.bounds[0] for p in hex_polygons)
        min_lat = min(p.bounds[1] for p in hex_polygons)
        max_lon = max(p.bounds[2] for p in hex_polygons)
        max_lat = max(p.bounds[3] for p in hex_polygons)
        bbox = (min_lon, min_lat, max_lon, max_lat)

        all_gdfs = []

        # 2. Query each configured ArcGIS service
        for state, url in self.arcgis_service_urls.items():
            logger.info(f"Fetching parcel data for {state}...")
            state_gdf = self._fetch_arcgis_parcels(url, bbox)
            if not state_gdf.empty:
                state_gdf["source_state"] = state
                state_gdf["source"] = "ARCGIS"
                all_gdfs.append(state_gdf)

        # 3. If no ArcGIS data, fall back to OSM
        if not all_gdfs:
            logger.info("No ArcGIS parcel data found. Falling back to OSM buildings.")
            osm_gdf = self._query_osm_overpass(bbox)
            if not osm_gdf.empty:
                all_gdfs.append(osm_gdf)

        if not all_gdfs:
            logger.warning("No parcel data was loaded from any source.")
            return gpd.GeoDataFrame()

        logger.info(f"Successfully loaded data from {len(all_gdfs)} sources.")

        # 4. Combine and standardize the data
        combined_gdf = gpd.pd.concat(all_gdfs, ignore_index=True)

        # Ensure parcel_id exists
        if "parcel_id" not in combined_gdf.columns:
            id_cols = ["PARNO", "APN", "parcel_id", "OBJECTID"]
            parcel_id_col = next(
                (col for col in id_cols if col in combined_gdf.columns), "generated_id"
            )
            if parcel_id_col == "generated_id":
                combined_gdf["parcel_id"] = [
                    f"gen_{i}" for i in range(len(combined_gdf))
                ]
            else:
                combined_gdf.rename(columns={parcel_id_col: "parcel_id"}, inplace=True)

        return combined_gdf
