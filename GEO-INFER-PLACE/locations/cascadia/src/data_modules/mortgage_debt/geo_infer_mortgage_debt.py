"""
GeoInfer Mortgage & Debt Module

This module analyzes agricultural mortgage and debt data within an H3 grid by
leveraging public Home Mortgage Disclosure Act (HMDA) data.
"""

import logging
from typing import Any
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import requests
import zipfile
import os

from .data_sources import CascadianMortgageDataSources
from geo_infer_space.utils.h3_utils import cell_to_latlng_boundary
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferMortgageDebt:
    """Processes and analyzes mortgage data aggregated at the census tract level."""

    module_name: str = "mortgage_debt"

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        """Initialize with shared backend to access target hexagons and output dirs.

        Args:
            backend: Cascadian shared backend providing target hexagons and context
        """
        self.backend = backend
        # Prefer backend's resolution if provided; fall back to 8
        self.resolution = getattr(backend, "h3_resolution", 8)
        self.target_hexagons = list(getattr(backend, "target_hexagons", []))
        self.mortgage_data_source = CascadianMortgageDataSources()
        self.census_data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "census"
        )
        os.makedirs(self.census_data_dir, exist_ok=True)
        # Will be injected by orchestrator
        self.data_manager = None  # type: ignore[attr-defined]
        self.h3_fusion = None  # type: ignore[attr-defined]
        logger.info("Initialized GeoInferMortgageDebt with backend integration")

    def _fetch_census_tract_geometries(self, year: int, state_fips: str) -> gpd.GeoDataFrame:
        """
        Downloads and caches US Census TIGER/Line shapefiles for a given state.
        """
        zip_path = os.path.join(self.census_data_dir, f"tl_{year}_{state_fips}_tract.zip")
        shapefile_dir = os.path.join(self.census_data_dir, f"tl_{year}_{state_fips}_tract")

        if not os.path.exists(shapefile_dir):
            if not os.path.exists(zip_path):
                url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
                logger.info(
                    f"Downloading census tract geometries for state FIPS {state_fips} from {url}"
                )
                try:
                    response = requests.get(url, timeout=(30, 300))
                    response.raise_for_status()
                    with open(zip_path, "wb") as f:
                        f.write(response.content)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download census tract data for FIPS {state_fips}: {e}")
                    return gpd.GeoDataFrame()

            logger.info(f"Unzipping {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(shapefile_dir)

        try:
            gdf = gpd.read_file(shapefile_dir)
            # GEOID is the full census tract FIPS code
            gdf.rename(columns={"GEOID": "census_tract"}, inplace=True)
            return gdf[["census_tract", "geometry"]]
        except Exception as e:
            logger.error(f"Failed to read shapefile from {shapefile_dir}: {e}")
            return gpd.GeoDataFrame()

    def acquire_raw_data(self, year: int = 2022) -> Path:
        """Acquire and cache raw mortgage data merged to census tract geometries.

        Saves a GeoJSON with tract geometries and mortgage metrics into the
        module's raw data path and returns that path. Respects cache if present.
        """
        if not hasattr(self, "data_manager") or self.data_manager is None:
            # Fallback: store under output/data if data_manager not injected yet
            raw_out = Path("output/data/raw_mortgage_debt_data.geojson")
        else:
            paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[union-attr]
            raw_out = paths["raw_data"]

        if raw_out.exists():
            logger.info(f"[{self.module_name}] Using cached raw data at {raw_out}")
            return raw_out

        logger.info(f"[{self.module_name}] Fetching HMDA mortgage aggregates and census tracts...")
        # 1. Fetch HMDA aggregates
        mortgage_df = self.mortgage_data_source.fetch_all_mortgage_data(year)
        if mortgage_df.empty:
            # Save an empty file to avoid repeated downloads
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_out, driver="GeoJSON")
            return raw_out

        # 2. Load tract geometries for CA (06) and OR (41)
        ca_tracts = self._fetch_census_tract_geometries(year, "06")
        or_tracts = self._fetch_census_tract_geometries(year, "41")
        census_tract_gdf = pd.concat([ca_tracts, or_tracts], ignore_index=True)
        if census_tract_gdf.empty:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_out, driver="GeoJSON")
            return raw_out

        # 3. Merge tabular aggregates to spatial tracts
        mortgage_gdf = census_tract_gdf.merge(mortgage_df, on="census_tract", how="inner")
        if mortgage_gdf.empty:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_out, driver="GeoJSON")
            return raw_out

        # 4. Save raw merged GeoJSON
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        mortgage_gdf.to_file(raw_out, driver="GeoJSON")
        logger.info(f"[{self.module_name}] Saved raw mortgage tract data to {raw_out}")
        return raw_out

    def run_analysis(
        self, target_hexagons: list[str], year: int = 2022
    ) -> dict[str, dict[str, Any]]:
        """
        Spatially joins HMDA mortgage data with H3 hexagons and calculates debt metrics.
        """
        logger.info(f"Starting mortgage & debt analysis for {len(target_hexagons)} hexagons.")

        # Prefer cached raw data if available
        try:
            raw_path = self.acquire_raw_data(year)
            mortgage_gdf = gpd.read_file(raw_path)
        except Exception:
            # Fallback to live fetch if needed
            mortgage_df = self.mortgage_data_source.fetch_all_mortgage_data(year)
            if mortgage_df.empty:
                logger.warning("No mortgage data found. Aborting analysis.")
                return {}
            ca_tracts = self._fetch_census_tract_geometries(year, "06")
            or_tracts = self._fetch_census_tract_geometries(year, "41")
            census_tract_gdf = pd.concat([ca_tracts, or_tracts], ignore_index=True)
            if census_tract_gdf.empty:
                logger.error("Could not load any census tract geometries. Aborting.")
                return {}
            mortgage_gdf = census_tract_gdf.merge(mortgage_df, on="census_tract", how="inner")
            if mortgage_gdf.empty:
                logger.warning("No matching census tracts found for the loaded mortgage data.")
                return {}

        # 4. Create a GeoDataFrame for the target hexagons
        hex_geometries = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        # Ensure a valid CRS is provided (WGS84) for hexagon polygons, then align to census tract CRS
        hex_gdf = gpd.GeoDataFrame(
            {"hex_id": target_hexagons}, geometry=hex_geometries, crs="EPSG:4326"
        )

        # 5. Ensure CRS alignment before join (project hexagons to census CRS)
        try:
            if mortgage_gdf.crs is None:
                mortgage_gdf.set_crs("EPSG:4326", inplace=True)
            hex_gdf = hex_gdf.to_crs(mortgage_gdf.crs)
        except Exception:
            # Fallback: project mortgage data to WGS84 to match hex_gdf
            try:
                mortgage_gdf = mortgage_gdf.to_crs("EPSG:4326")
            except Exception as exc:
                logger.warning("Mortgage CRS fallback projection failed: %s", exc)

        # 6. Perform the spatial join
        logger.info("Performing spatial join between hexagons and mortgage census tracts...")
        # Use 'intersects' as a hexagon can overlap with multiple tracts
        joined_gdf = gpd.sjoin(hex_gdf, mortgage_gdf, how="inner", predicate="intersects")

        if joined_gdf.empty:
            logger.warning(
                "Spatial join resulted in no matches between hexagons and mortgage data."
            )
            return {}

        # 7. Aggregate results and calculate metrics per hexagon
        logger.info("Aggregating mortgage results per hexagon...")
        h3_mortgage = {}
        for hex_id, group in joined_gdf.groupby("hex_id"):
            # Weighted average of metrics based on intersection area is too complex here.
            # We will do a simpler aggregation: average of the tract values.
            h3_mortgage[hex_id] = {
                "total_loan_volume": group["total_loan_volume"].sum(),
                "avg_loan_to_value_ratio": group["loan_to_value_ratio"].mean(),
                "avg_income": group["average_income"].mean(),
                "number_of_loans": group["number_of_loans"].sum(),
                "number_of_source_tracts": group["census_tract"].nunique(),
            }

        logger.info(f"Completed mortgage analysis. Processed {len(h3_mortgage)} hexagons.")
        return h3_mortgage

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize H3 mortgage features into per-hex metrics if needed.

        If upstream already fused metrics, this acts as a pass-through that
        rescales or validates values.
        """
        results: dict[str, Any] = {}
        for hex_id, items in h3_data.items():
            # items may be a list of dicts with mortgage fields
            try:
                if isinstance(items, dict) and "mortgage" in items:
                    # Already summarized upstream
                    results[hex_id] = items["mortgage"]
                    continue
                if isinstance(items, list) and items:
                    df = pd.DataFrame(items)
                    results[hex_id] = {
                        "total_loan_volume": float(
                            df.get("total_loan_volume", pd.Series(dtype=float)).sum()
                        ),
                        "avg_loan_to_value_ratio": float(
                            df.get("loan_to_value_ratio", pd.Series(dtype=float)).mean()
                        ),
                        "avg_income": float(
                            df.get("average_income", pd.Series(dtype=float)).mean()
                        ),
                        "number_of_loans": int(
                            df.get("number_of_loans", pd.Series(dtype=float)).sum()
                        ),
                        "number_of_source_tracts": int(
                            df.get("census_tract", pd.Series(dtype=str)).nunique()
                        ),
                    }
            except Exception:
                results[hex_id] = {}
        return results
