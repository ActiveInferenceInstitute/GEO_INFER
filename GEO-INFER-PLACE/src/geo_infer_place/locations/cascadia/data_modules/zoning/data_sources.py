"""
Data source handling for the Zoning module.

Fetches real zoning/farmland classification data from:
- California FMMP (Farmland Mapping and Monitoring Program)
- Del Norte County Zoning (when available)
"""

import json
import logging
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

logger = logging.getLogger(__name__)

# Default bounding box for Del Norte County, CA
DEL_NORTE_BBOX = (-124.5, 41.4, -123.5, 42.0)


class CascadianZoningDataSources:
    """
    Manages the acquisition of zoning data from various state and county sources.
    """

    def __init__(self, data_dir: Path, config_path: Path | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_path = self.data_dir / "raw_zoning_data.geojson"

        # Load config for URLs
        if config_path is None:
            config_path = (
                self.data_dir.parent.parent.parent / "config" / "data_urls.json"
            )

        self.config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
                logger.info(f"[zoning] Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"[zoning] Could not load config: {e}")

        # Get data source URLs
        self.fmmp_url = (
            self.config.get("data_sources", {}).get("ca_fmmp", {}).get("url")
        )
        self.zoning_url = (
            self.config.get("data_sources", {}).get("del_norte_zoning", {}).get("url")
        )

    def _query_arcgis_service(
        self, service_url: str, bbox: tuple, max_features: int = 5000
    ) -> gpd.GeoDataFrame | None:
        """
        Query an ArcGIS REST service for features within a bounding box.

        Args:
            service_url: Base URL of the ArcGIS service
            bbox: (min_lon, min_lat, max_lon, max_lat)
            max_features: Maximum features to fetch

        Returns:
            GeoDataFrame with features, or None on error
        """
        if not service_url:
            return None

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        query_url = f"{service_url}/query"

        params = {
            "geometry": bbox_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "returnGeometry": "true",
            "resultRecordCount": max_features,
            "f": "geojson",
        }

        logger.info(f"[zoning] Querying ArcGIS: {service_url}")
        logger.debug(f"[zoning] Bbox: {bbox_str}")

        try:
            response = requests.get(query_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"[zoning] ArcGIS error: {data['error']}")
                return None

            features = data.get("features", [])
            if not features:
                logger.info(f"[zoning] No features returned from {service_url}")
                return None

            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            logger.info(f"[zoning] Fetched {len(gdf)} features from ArcGIS")
            return gdf

        except requests.exceptions.Timeout:
            logger.error(f"[zoning] Timeout querying {service_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[zoning] Request error: {e}")
        except Exception as e:
            logger.error(f"[zoning] Error parsing response: {e}")

        return None

    def _query_osm_overpass(self, bbox: tuple) -> gpd.GeoDataFrame | None:
        """
        Query OpenStreetMap Overpass API for land use data.

        This provides global coverage when local/regional sources are unavailable.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)

        Returns:
            GeoDataFrame with land use features, or None on error
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Overpass QL query for land use polygons
        # Note: Overpass uses (south, west, north, east) format
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way["landuse"~"farmland|meadow|orchard|vineyard|forest|residential|commercial|industrial|retail"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["landuse"~"farmland|meadow|orchard|vineyard|forest|residential|commercial|industrial|retail"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        overpass_url = "https://overpass-api.de/api/interpreter"

        logger.info("[zoning] Querying OpenStreetMap Overpass API for land use data...")

        try:
            response = requests.post(
                overpass_url, data={"data": overpass_query}, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            # Parse OSM elements into geometries
            nodes = {
                e["id"]: (e["lon"], e["lat"])
                for e in data.get("elements", [])
                if e["type"] == "node"
            }
            features = []

            for element in data.get("elements", []):
                if element["type"] == "way" and "tags" in element:
                    # Build polygon from node refs
                    coords = [nodes.get(n) for n in element.get("nodes", [])]
                    coords = [c for c in coords if c is not None]

                    if len(coords) >= 4:  # Need at least 4 points for a valid polygon
                        try:
                            from shapely.geometry import Polygon as ShapelyPolygon

                            poly = ShapelyPolygon(coords)
                            if poly.is_valid:
                                landuse = element["tags"].get("landuse", "unknown")
                                features.append(
                                    {
                                        "geometry": poly,
                                        "CI_CLASSNM": self._osm_to_fmmp_class(landuse),
                                        "osm_landuse": landuse,
                                        "source": "OSM",
                                    }
                                )
                        except Exception as exc:
                            logger.warning(
                                "OSM zoning feature construction failed; skipping feature: %s",
                                exc,
                            )

            if not features:
                logger.info("[zoning] No valid land use features from OSM")
                return None

            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            logger.info(
                f"[zoning] Fetched {len(gdf)} land use features from OpenStreetMap"
            )
            return gdf

        except requests.exceptions.Timeout:
            logger.error("[zoning] Timeout querying Overpass API")
        except requests.exceptions.RequestException as e:
            logger.error(f"[zoning] Overpass API request error: {e}")
        except Exception as e:
            logger.error(f"[zoning] Error parsing Overpass response: {e}")

        return None

    def _osm_to_fmmp_class(self, osm_landuse: str) -> str:
        """Map OSM landuse tags to FMMP-style classifications."""
        mapping = {
            "farmland": "Prime Farmland",
            "meadow": "Grazing Land",
            "orchard": "Prime Farmland",
            "vineyard": "Unique Farmland",
            "forest": "Other Land",
            "residential": "Urban and Built-up Land",
            "commercial": "Urban and Built-up Land",
            "industrial": "Urban and Built-up Land",
            "retail": "Urban and Built-up Land",
        }
        return mapping.get(osm_landuse, "Other Land")

    def fetch_all_zoning_data(
        self, bbox: tuple = DEL_NORTE_BBOX, force_refresh: bool = False
    ) -> Path:
        """
        Fetches zoning data from all relevant sources and saves it.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            force_refresh: If True, re-fetch even if cached

        Returns:
            Path to the consolidated raw data file
        """
        if self.raw_data_path.exists() and not force_refresh:
            logger.info(f"[zoning] Using cached data at {self.raw_data_path}")
            return self.raw_data_path

        logger.info("[zoning] Fetching zoning data from sources...")
        all_gdfs = []

        # 1. Try California FMMP (farmland classification)
        if self.fmmp_url:
            fmmp_gdf = self._query_arcgis_service(self.fmmp_url, bbox)
            if fmmp_gdf is not None and not fmmp_gdf.empty:
                fmmp_gdf["source"] = "CA_FMMP"
                all_gdfs.append(fmmp_gdf)
                logger.info(f"[zoning] Added {len(fmmp_gdf)} FMMP features")

        # 2. Try Del Norte County Zoning
        if self.zoning_url:
            zoning_gdf = self._query_arcgis_service(self.zoning_url, bbox)
            if zoning_gdf is not None and not zoning_gdf.empty:
                zoning_gdf["source"] = "DEL_NORTE_ZONING"
                all_gdfs.append(zoning_gdf)
                logger.info(
                    f"[zoning] Added {len(zoning_gdf)} Del Norte zoning features"
                )

        # 3. If no ArcGIS data, try OpenStreetMap Overpass API (global coverage)
        if not all_gdfs:
            logger.info("[zoning] No ArcGIS data available, trying OpenStreetMap...")
            osm_gdf = self._query_osm_overpass(bbox)
            if osm_gdf is not None and not osm_gdf.empty:
                all_gdfs.append(osm_gdf)
                logger.info(f"[zoning] Added {len(osm_gdf)} OSM land use features")

        # 4. Require an authoritative source.
        if all_gdfs:
            combined_gdf = pd.concat(all_gdfs, ignore_index=True)
            logger.info(
                f"[zoning] Combined {len(combined_gdf)} total features from {len(all_gdfs)} sources"
            )
        else:
            raise RuntimeError(
                "No authoritative zoning data was returned by configured sources"
            )

        # Standardize column names
        if "CI_CLASSNM" not in combined_gdf.columns:
            # Try other common zoning fields
            for col in ["ZONE_CODE", "ZONE_DESC", "ZONING", "LAND_USE"]:
                if col in combined_gdf.columns:
                    combined_gdf["CI_CLASSNM"] = combined_gdf[col]
                    break
            else:
                combined_gdf["CI_CLASSNM"] = "Unknown"

        # Save
        logger.info(
            f"[zoning] Saving {len(combined_gdf)} features to {self.raw_data_path}"
        )
        combined_gdf.to_file(self.raw_data_path, driver="GeoJSON")

        return self.raw_data_path
