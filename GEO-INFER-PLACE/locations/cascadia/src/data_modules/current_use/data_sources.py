"""
Cascadian Current Use Data Sources

This module provides real-time data fetching capabilities for current agricultural
land use analysis across the Cascadian bioregion (Northern California + Oregon).
"""

import logging
import os
from datetime import datetime
from typing import NamedTuple, Any
from pathlib import Path
import requests
import zipfile

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask

# Import H3 utilities from the unified backend's path
try:
    from geo_infer_space.utils.h3_utils import (
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

logger = logging.getLogger(__name__)


class CropClassification(NamedTuple):
    id: int
    name: str
    crop_category: str
    water_intensity: str
    value_per_acre: float
    growth_cycle: str


class CascadianCurrentUseDataSources:
    """Multi-source agricultural land use classification for Cascadian bioregion"""

    def __init__(self):
        self.nass_cdl_url_base = (
            "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
        )
        self.data_dir = Path(__file__).parent.joinpath("data", "cdl")
        os.makedirs(self.data_dir, exist_ok=True)
        self.crop_classifications = self._init_crop_classifications()
        self.data_sources = self._init_data_sources()
        self.target_ca_counties = [
            "Butte",
            "Colusa",
            "Del Norte",
            "Glenn",
            "Humboldt",
            "Lake",
            "Lassen",
            "Mendocino",
            "Modoc",
            "Nevada",
            "Plumas",
            "Shasta",
            "Sierra",
            "Siskiyou",
            "Tehama",
            "Trinity",
        ]
        logger.info("Initialized CascadianCurrentUseDataSources")

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

    def _init_data_sources(self) -> dict[str, dict]:
        """Initialize data sources configuration"""
        return {
            "nass_cdl": {
                "url": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php",
                "api_endpoint": None,  # Direct download now, no API
                "spatial_resolution": "30 meters",
                "temporal_coverage": "2008-present",
                "classification_categories": "50+ crop-specific categories",
                "api_access": "USDA APIs with rate limiting",
                "format": "GeoTIFF raster",
            },
            "land_iq": {
                "partnership": "California Department of Water Resources",
                "spatial_resolution": "0.5-2.0 acre minimum mapping units",
                "coverage": "15.4 million acres",
                "accuracy": "98%+",
                "update_frequency": "Annual (water year basis)",
                "api_endpoint": "https://www.landiq.com/api/v1/",
                "format": "Vector polygons",
            },
            "oregon_farm_reports": {
                "source": "Oregon Department of Land Conservation and Development",
                "coverage": "EFU and forest zone land use decisions",
                "frequency": "Biennial",
                "data_elements": "Agricultural land conversion tracking",
                "api_endpoint": "https://www.oregon.gov/lcd/api/",
                "format": "Tabular and spatial data",
            },
            "usda_nass_stats": {
                "api_endpoint": "https://quickstats.nass.usda.gov/api",
                "coverage": "County-level agricultural statistics",
                "update_frequency": "Annual",
                "data_elements": "Crop acreage, production, value",
                "format": "JSON API",
            },
        }

    def _init_crop_classifications(self) -> dict[int, CropClassification]:
        """Initialize standardized crop classification system"""
        return {
            1: CropClassification(1, "Corn", "Field Crops", "High", 800.0, "Annual"),
            5: CropClassification(5, "Soybeans", "Field Crops", "Medium", 600.0, "Annual"),
            24: CropClassification(24, "Winter Wheat", "Field Crops", "Low", 400.0, "Annual"),
            36: CropClassification(36, "Alfalfa", "Forage", "High", 600.0, "Perennial"),
            61: CropClassification(61, "Fallow/Idle Cropland", "Fallow", "None", 0.0, "Seasonal"),
            111: CropClassification(111, "Open Water", "Water", "None", 0.0, "Permanent"),
            121: CropClassification(
                121, "Developed/Open Space", "Developed", "None", 0.0, "Permanent"
            ),
            122: CropClassification(
                122, "Developed/Low Intensity", "Developed", "None", 0.0, "Permanent"
            ),
            123: CropClassification(
                123, "Developed/Med Intensity", "Developed", "None", 0.0, "Permanent"
            ),
            124: CropClassification(
                124, "Developed/High Intensity", "Developed", "None", 0.0, "Permanent"
            ),
            141: CropClassification(141, "Deciduous Forest", "Forest", "Low", 100.0, "Perennial"),
            176: CropClassification(
                176, "Grassland/Pasture", "Grassland", "Low", 200.0, "Perennial"
            ),
        }

    def fetch_nass_cdl_data_for_hexagons(
        self, year: int, hexagons: list[str]
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Fetches and processes NASS CDL data for a list of hexagons, chunking requests.
        For each hexagon, it returns the list of crop codes and their percentage coverage.
        """
        if not hexagons:
            return {}

        hex_results = {}
        # Group hexagons by state to fetch appropriate raster
        hex_states = self._group_hexagons_by_state(hexagons)

        for state, state_hexagons in hex_states.items():
            if not state_hexagons:
                continue

            bbox = self._calculate_bbox_from_hexagons(state_hexagons)
            raster_data = self._fetch_cdl_raster_for_bbox(year, bbox, state)

            if not raster_data:
                logger.warning(f"Could not fetch a CDL raster for {state}.")
                continue

            src = raster_data["src"]

            # Process each hexagon in the current batch with the fetched raster
            for h3_index in state_hexagons:
                try:
                    hex_poly = Polygon(
                        [(lng, lat) for lat, lng in cell_to_latlng_boundary(h3_index)]
                    )
                    out_image, out_transform = mask(src, [hex_poly], crop=True)

                    unique, counts = np.unique(
                        out_image[out_image != src.nodata], return_counts=True
                    )
                    total_pixels = np.sum(counts)

                    if total_pixels > 0:
                        crop_percentages = [
                            (int(code), (count / total_pixels) * 100)
                            for code, count in zip(unique, counts)
                        ]
                        hex_results[h3_index] = crop_percentages
                except Exception as e:
                    logger.error(f"Error processing hexagon {h3_index} with raster data: {e}")

            src.close()  # Important to close the raster file

        return hex_results

    def _get_state_for_bbox(self, bbox: tuple[float, float, float, float]) -> str:
        """Determines the state based on the bounding box's center."""
        center_lon = (bbox[0] + bbox[2]) / 2
        # Simple longitude check for CA, OR, WA
        if center_lon < -114 and center_lon > -125:  # California, Oregon, Washington
            center_lat = (bbox[1] + bbox[3]) / 2
            if center_lat < 42:
                return "CA"
            elif center_lat < 46:
                return "OR"
            else:
                return "WA"
        return "CONUS"  # Default to conterminous US if not clearly in one state

    def _group_hexagons_by_state(self, hexagons: list[str]) -> dict[str, list[str]]:
        """Groups hexagons by their approximate state location."""
        states = {"CA": [], "OR": [], "WA": [], "Other": []}
        for h in hexagons:
            lat, lon = cell_to_latlng(h)
            if lon < -114 and lon > -125:
                if lat < 42:
                    states["CA"].append(h)
                elif lat < 46:
                    states["OR"].append(h)
                else:
                    states["WA"].append(h)
            else:
                states["Other"].append(h)
        return states

    def _download_and_clip_national_cdl(self, year: int, state: str) -> Path | None:
        """
        Downloads the national CDL file, unzips it, and clips it to the state boundary.
        Manages local cache to avoid re-downloads and re-processing.
        """
        national_zip_filename = f"{year}_30m_cdls.zip"
        national_zip_filepath = self.data_dir / national_zip_filename
        unzip_dir = self.data_dir / f"{year}_30m_cdls"

        # Download the national file if it doesn't exist
        if not national_zip_filepath.exists():
            url = f"{self.nass_cdl_url_base}{national_zip_filename}"
            logger.info(f"Downloading national CDL data from {url}...")
            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                with open(national_zip_filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
                logger.info(f"Downloaded {national_zip_filepath}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download national CDL data for {year}: {e}")
                return None

        # Unzip the file if the directory doesn't exist
        if not unzip_dir.exists():
            logger.info(f"Unzipping {national_zip_filepath}...")
            with zipfile.ZipFile(national_zip_filepath, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)
            logger.info(f"Unzipped to {unzip_dir}")

        national_tif_files = list(unzip_dir.glob("*.tif"))
        if not national_tif_files:
            logger.error(f"No TIF file found in {unzip_dir}")
            return None
        national_tif_path = national_tif_files[0]

        # Now clip to the state
        state_clipped_path = self.data_dir / f"{year}_{state}_clipped_cdl.tif"
        if state_clipped_path.exists():
            logger.info(f"Using cached state-clipped CDL raster: {state_clipped_path}")
            return state_clipped_path

        try:
            # Load state boundary
            state_boundaries = gpd.read_file("path/to/state_boundaries.shp")  # Add actual path
            state_geom = state_boundaries[state_boundaries["STUSPS"] == state].geometry.iloc[0]

            # Clip national raster to state
            with rasterio.open(national_tif_path) as src:
                clipped_data, clipped_transform = mask(src, [state_geom], crop=True)

            # Save clipped raster
            clipped_path = self.data_dir / f"clipped_{state}_{year}.tif"
            with rasterio.open(
                clipped_path,
                "w",
                driver="GTiff",
                height=clipped_data.shape[1],
                width=clipped_data.shape[2],
                count=1,
                dtype=clipped_data.dtype,
                crs=src.crs,
                transform=clipped_transform,
            ) as dst:
                dst.write(clipped_data, 1)

            return clipped_path

        except Exception as e:
            logger.error(f"Failed during clipping process for {state} {year}: {e}")
            return None

    def _fetch_cdl_raster_for_bbox(
        self, year: int, bbox: tuple[float, float, float, float], state: str
    ) -> dict[str, Any] | None:
        """
        Fetches a NASS Cropland Data Layer raster for a given bounding box by
        downloading, caching, and clipping the national dataset.
        """
        for year_to_try in range(year, year - 4, -1):
            clipped_raster_path = self._download_and_clip_national_cdl(year_to_try, state)

            if clipped_raster_path and clipped_raster_path.exists():
                try:
                    logger.info(f"Opening raster {clipped_raster_path} for state {state}")
                    src = rasterio.open(clipped_raster_path)
                    # Further mask to the specific bbox of the hexagon batch
                    masked_src, masked_transform = mask(
                        src, [Polygon.from_bounds(*bbox)], crop=True
                    )

                    # Save this smaller chunk to a temporary in-memory raster
                    memfile = rasterio.io.MemoryFile()
                    with memfile.open(
                        driver="GTiff",
                        height=masked_src.shape[1],
                        width=masked_src.shape[2],
                        count=1,
                        dtype=masked_src.dtype,
                        crs=src.crs,
                        transform=masked_transform,
                    ) as dataset:
                        dataset.write(masked_src)

                    src.close()  # Close original file

                    # Reopen the in-memory file to pass to the processing function
                    final_src = memfile.open()

                    return {"src": final_src}
                except rasterio.errors.RasterioIOError as e:
                    logger.error(
                        f"Could not open or process raster file {clipped_raster_path}: {e}"
                    )
                    continue

            logger.warning(
                f"Could not obtain clipped raster for {state} for year {year_to_try}. Trying previous year."
            )

        logger.error(
            f"Could not obtain CDL data for years {year} down to {year - 3}. Cannot proceed."
        )
        return None

    def fetch_land_iq_data(self, county: str) -> gpd.GeoDataFrame:
        """
        Fetches detailed Land IQ land use data for a specific county in California.
        """
        raise RuntimeError(f"Land IQ data acquisition is not configured for {county}")

    def get_usda_county_stats(self, county_fips: str, year: int) -> dict | None:
        """
        Fetches USDA NASS county-level statistics for validation.
        """
        raise RuntimeError(
            f"USDA county statistics acquisition is not configured for {county_fips} ({year})"
        )

    def fetch_oregon_efu_reports(self, year: int = None) -> pd.DataFrame:
        """
        Fetch Oregon EFU land use reports from local file
        """
        try:
            year = year or datetime.now().year
            local_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "oregon_efu",
                f"efu_report_{year}.csv",
            )
            if not os.path.exists(local_path):
                logger.error(f"Local Oregon EFU file missing: {local_path}")
                return pd.DataFrame()

            return pd.read_csv(local_path)

        except Exception as e:
            logger.error(f"Error fetching Oregon EFU reports: {str(e)}")
            return pd.DataFrame()

    def get_crop_classification(self, crop_code: int) -> CropClassification | None:
        """
        Get crop classification information
        """
        return self.crop_classifications.get(crop_code)

    def classify_crop_category(self, crop_code: int) -> str:
        """
        Classify crop into a general category
        """
        classification = self.get_crop_classification(crop_code)
        return classification.crop_category if classification else "Unknown"

    def estimate_water_requirements(self, crop_code: int) -> str:
        """
        Estimate water requirements for a crop
        """
        classification = self.get_crop_classification(crop_code)
        return classification.water_intensity if classification else "Unknown"

    def estimate_economic_value(self, crop_code: int) -> float:
        """
        Estimate economic value for a crop
        """
        classification = self.get_crop_classification(crop_code)
        return classification.value_per_acre if classification else 0.0

    def get_seasonal_pattern(self, crop_code: int) -> str:
        """
        Get seasonal pattern for a crop
        """
        classification = self.get_crop_classification(crop_code)
        return classification.growth_cycle if classification else "Unknown"

    def validate_data_availability(self, year: int, source: str = "nass_cdl") -> bool:
        """
        Validate data availability for a given year and source.
        """
        if source == "nass_cdl":
            # NASS CDL data is generally available from 2008 onwards
            return 2008 <= year <= datetime.now().year
        elif source == "land_iq":
            # Land IQ data has specific year coverage
            return year in [2014, 2016, 2018, 2020]  # Example years
        elif source == "oregon_efu":
            # Check for local file existence
            local_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "oregon_efu",
                f"efu_report_{year}.csv",
            )
            return os.path.exists(local_path)
        return False

    def get_available_years(self, source: str = "nass_cdl") -> list[int]:
        """
        Get available years for a data source.
        """
        current_year = datetime.now().year
        if source == "nass_cdl":
            return list(range(2008, current_year + 1))
        elif source == "land_iq":
            return [2014, 2016, 2018, 2020]  # Example years
        elif source == "oregon_efu":
            # Scan local directory for available report years
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "oregon_efu")
            if not os.path.exists(data_dir):
                return []
            years = []
            for f in os.listdir(data_dir):
                if f.startswith("efu_report_") and f.endswith(".csv"):
                    try:
                        year_str = f.replace("efu_report_", "").replace(".csv", "")
                        years.append(int(year_str))
                    except ValueError:
                        continue
            return sorted(years)
        return []

    def get_target_counties(self, state: str = "CA") -> list[str]:
        """
        Get target counties for analysis.
        """
        if state == "CA":
            return self.target_ca_counties
        return []
