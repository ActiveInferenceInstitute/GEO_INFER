"""
Cascadian Improvements Data Sources

This module is responsible for fetching and consolidating agricultural
improvements data, including building footprints and estimated values.
"""

import logging
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import zipfile

from geo_infer_space.utils.h3_utils import (
    cell_to_latlng_boundary,
)
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class CascadianImprovementsDataSources:
    """
    Manages acquisition and processing of building footprints and estimated property values.

    This class downloads building footprint data from the Microsoft USBuildingFootprints
    dataset and property value data from Zillow's ZHVI. It then estimates
    improvement and land values based on the combination of these sources.
    """

    def __init__(self, *, random_seed: int = 0):
        self.rng = np.random.default_rng(random_seed)
        self.data_dir = os.path.join(os.getcwd(), "data", "improvements")
        os.makedirs(self.data_dir, exist_ok=True)

        self.state_urls = {
            "CA": "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/California.geojson.zip",
            "OR": "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Oregon.geojson.zip",
        }
        self.zillow_zhvi_url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"

        logger.info("Initialized CascadianImprovementsDataSources.")

    def _fetch_zillow_zhvi_data(self) -> gpd.GeoDataFrame:
        """Downloads Zillow Home Value Index data and pre-processes it."""
        zhvi_path = os.path.join(
            self.data_dir, "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        )

        if not os.path.exists(zhvi_path):
            logger.info("Downloading Zillow ZHVI data...")
            try:
                response = requests.get(self.zillow_zhvi_url, timeout=(30, 300))
                response.raise_for_status()
                with open(zhvi_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download Zillow data: {e}")
                return gpd.GeoDataFrame()

        logger.info("Processing Zillow ZHVI data...")
        df = pd.read_csv(zhvi_path)
        # Get the most recent month's value
        latest_month = df.columns[-1]
        df = df[["RegionName", "State", latest_month]].copy()
        df.rename(
            columns={"RegionName": "zip_code", latest_month: "median_home_value"},
            inplace=True,
        )
        df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

        # To make this spatial, we need zip code boundaries.
        zip_boundaries_dir = os.path.join(self.data_dir, "tl_2023_us_zcta520")
        zip_boundaries_shapefile = os.path.join(
            zip_boundaries_dir, "tl_2023_us_zcta520.shp"
        )

        if not os.path.exists(zip_boundaries_shapefile):
            logger.info("Downloading and unzipping US Zip Code boundaries...")
            url = "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
            zip_path = os.path.join(self.data_dir, "tl_2023_us_zcta520.zip")
            try:
                response = requests.get(url, timeout=(30, 300))
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(response.content)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    os.makedirs(zip_boundaries_dir, exist_ok=True)
                    zip_ref.extractall(zip_boundaries_dir)
                os.remove(zip_path)  # Clean up the zip file

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download zip code boundaries: {e}")
                return gpd.GeoDataFrame()
            except zipfile.BadZipFile:
                logger.error("Downloaded zip code boundaries file is not a valid zip.")
                return gpd.GeoDataFrame()

        logger.info(f"Reading zip code boundaries from {zip_boundaries_shapefile}")
        gdf_zips = gpd.read_file(zip_boundaries_shapefile)
        gdf_zips = gdf_zips[["ZCTA5CE20", "geometry"]].rename(
            columns={"ZCTA5CE20": "zip_code"}
        )
        gdf_zips["zip_code"] = gdf_zips["zip_code"].astype(str)

        # Merge Zillow data with zip boundaries
        zhvi_gdf = gdf_zips.merge(df, on="zip_code", how="inner")
        return zhvi_gdf

    def _download_and_unzip(self, state: str, url: str):
        zip_path = os.path.join(self.data_dir, f"{state}_buildings.zip")
        geojson_path = os.path.join(self.data_dir, f"{state}.geojson")

        if os.path.exists(geojson_path):
            logger.info(f"GeoJSON file already exists for {state}, skipping download.")
            return geojson_path

        if not os.path.exists(zip_path):
            logger.info(f"Downloading {state} building footprints from {url}...")
            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Download complete for {state}.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading file for {state}: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return None

        logger.info(f"Unzipping {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                geojson_files = [
                    name for name in zip_ref.namelist() if name.endswith(".geojson")
                ]
                if not geojson_files:
                    logger.error(f"No GeoJSON found in {zip_path}")
                    os.remove(zip_path)
                    return None
                geojson_filename = geojson_files[0]
                zip_ref.extract(geojson_filename, path=self.data_dir)
                os.rename(os.path.join(self.data_dir, geojson_filename), geojson_path)
            logger.info(f"Unzipped and renamed to {geojson_path}.")
            return geojson_path
        except zipfile.BadZipFile:
            logger.error(
                f"File {zip_path} is not a valid zip file. Removing it to retry next time."
            )
            try:
                os.remove(zip_path)
            except OSError:
                pass
            return None
        except Exception as e:
            logger.error(f"Error unzipping {state}: {e}")
            return None

    def _estimate_improvement_values(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Estimates 'improvement_value' and 'land_value' based on geometry and Zillow data.
        Assumes 'median_home_value' is already joined to the GeoDataFrame.
        """
        if gdf.empty or "median_home_value" not in gdf.columns:
            logger.warning("Cannot estimate values without median_home_value.")
            return gdf.assign(improvement_value=0, land_value=0)

        gdf["median_home_value"].fillna(gdf["median_home_value"].median(), inplace=True)

        gdf_proj = gdf.to_crs(epsg=3310)
        gdf["area_sqm"] = gdf_proj.geometry.area

        # Heuristic: Assume median home value is for a 150 sqm (1600 sqft) home.
        # This gives us a local $/sqm estimate.
        estimated_cost_per_sqm = gdf["median_home_value"] / 150

        # Improvement value is area * local $/sqm
        gdf["improvement_value"] = (gdf["area_sqm"] * estimated_cost_per_sqm).round(0)

        # Land value is a fraction of the improvement value.
        land_value_multiplier = self.rng.normal(loc=0.5, scale=0.15, size=len(gdf))
        gdf["land_value"] = (
            (gdf["improvement_value"] * land_value_multiplier).clip(lower=0).round(0)
        )

        gdf["parcel_id"] = [f"est_parcel_{i}" for i in range(len(gdf))]

        return gdf.drop(columns=["area_sqm"])

    def _query_osm_overpass_buildings(
        self, bbox: tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """Query OSM Overpass for building footprints."""
        min_lon, min_lat, max_lon, max_lat = bbox

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
        logger.info("Improvements: Querying OSM for building footprints (fallback)...")

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

            from shapely.geometry import Polygon

            for element in data.get("elements", []):
                if element["type"] == "way" and "nodes" in element:
                    coords = [nodes.get(n) for n in element["nodes"]]
                    coords = [c for c in coords if c is not None]

                    if len(coords) >= 4:
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid:
                                features.append({"geometry": poly, "source": "OSM"})
                        except Exception as exc:
                            logger.warning(
                                "OSM feature construction failed; skipping feature: %s",
                                exc,
                            )

            if not features:
                return gpd.GeoDataFrame()

            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            logger.info(
                f"Improvements: Fetched {len(gdf)} building footprints from OSM."
            )
            return gdf

        except Exception as e:
            logger.error(f"OSM Overpass query failed for buildings: {e}")
            return gpd.GeoDataFrame()

    def fetch_all_improvements_data(self, target_hexagons: list) -> gpd.GeoDataFrame:
        """
        Fetches building footprints and Zillow data, merges them, and returns
        a consolidated GeoDataFrame with estimated financial data.
        """
        if not target_hexagons:
            logger.warning("Cannot fetch improvements data without target hexagons.")
            return gpd.GeoDataFrame()

        hex_boundaries = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        minx = min(p.bounds[0] for p in hex_boundaries)
        miny = min(p.bounds[1] for p in hex_boundaries)
        maxx = max(p.bounds[2] for p in hex_boundaries)
        maxy = max(p.bounds[3] for p in hex_boundaries)
        bbox = (minx, miny, maxx, maxy)

        # Fetch Zillow data first
        zhvi_gdf = self._fetch_zillow_zhvi_data()
        if zhvi_gdf.empty:
            raise RuntimeError("Zillow valuation data acquisition returned no records")

        all_gdfs = []

        # Try Official Sources first
        for state, url in self.state_urls.items():
            geojson_path = self._download_and_unzip(state, url)
            if geojson_path:
                try:
                    logger.info(
                        f"Loading building footprints for {state} within bounding box: {bbox}..."
                    )
                    gdf = gpd.read_file(geojson_path, bbox=bbox)
                    if not gdf.empty:
                        all_gdfs.append(gdf)
                except Exception as e:
                    logger.error(f"Failed to load or process GeoJSON for {state}: {e}")

        # Fallback to OSM if no official data
        if not all_gdfs:
            logger.info("No official building footprints found. Falling back to OSM.")
            osm_gdf = self._query_osm_overpass_buildings(bbox)
            if not osm_gdf.empty:
                all_gdfs.append(osm_gdf)

        if not all_gdfs:
            raise RuntimeError(
                "No authoritative improvements data was returned by configured sources"
            )

        combined_gdf = pd.concat(all_gdfs, ignore_index=True)

        # Ensure CRS match for spatial join
        if combined_gdf.crs != zhvi_gdf.crs:
            combined_gdf = combined_gdf.to_crs(zhvi_gdf.crs)

        logger.info("Spatially joining buildings with Zillow data...")
        # Spatial join to get median home value for each building
        gdf_merged = gpd.sjoin(combined_gdf, zhvi_gdf, how="left", predicate="within")

        logger.info("Estimating improvement and land values...")
        gdf_valued = self._estimate_improvement_values(gdf_merged)

        logger.info(
            f"Successfully loaded and processed {len(gdf_valued)} improvement records."
        )
        return gdf_valued
