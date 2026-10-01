"""
GeoInfer Water Rights Module

This module analyzes agricultural water rights using H3 indexing by fetching
and integrating real data from multiple state-level sources.
"""

import logging
from typing import Any
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

from .data_sources import CascadianWaterRightsDataSources
from geo_infer_space.utils.h3_utils import cell_to_latlng_boundary
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferWaterRights:
    """Processes and analyzes real water rights data within an H3 grid."""

    module_name: str = "water_rights"

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        self.backend = backend
        self.resolution = getattr(backend, "h3_resolution", 8)
        self.target_hexagons = list(getattr(backend, "target_hexagons", []))
        self.data_source = CascadianWaterRightsDataSources()
        # Will be injected
        self.data_manager = None  # type: ignore[attr-defined]
        self.h3_fusion = None  # type: ignore[attr-defined]
        logger.info(f"Initialized GeoInferWaterRights with resolution {self.resolution}")

    def acquire_raw_data(self) -> Path:
        """Acquire and cache raw water rights points for the target area."""
        if not hasattr(self, "data_manager") or self.data_manager is None:
            raw_out = Path("output/data/raw_water_rights_data.geojson")
        else:
            paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[union-attr]
            raw_out = paths["raw_data"]

        if raw_out.exists():
            return raw_out

        gdf = self.data_source.fetch_all_water_rights_data(self.target_hexagons)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        if gdf is None or gdf.empty:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_out, driver="GeoJSON")
        else:
            gdf.to_file(raw_out, driver="GeoJSON")
        return raw_out

    def _find_col(self, df, potential_names):
        """Finds the first matching column name from a list of potential names."""
        for name in potential_names:
            if name in df.columns:
                return name
        return None

    def run_analysis(self, target_hexagons: list[str]) -> dict[str, dict[str, Any]]:
        """
        Spatially joins real water rights data with H3 hexagons and aggregates metrics.
        This is the main entry point for the module.

        Args:
            target_hexagons: A list of H3 hexagon IDs to analyze.

        Returns:
            A dictionary mapping H3 hexagons to their water rights analysis.
        """
        logger.info(f"Starting water rights analysis for {len(target_hexagons)} hexagons.")

        water_rights_gdf = self.data_source.fetch_all_water_rights_data(target_hexagons)
        if water_rights_gdf.empty:
            logger.warning("No water rights data found. Aborting analysis.")
            return {
                hex_id: {"error": "No water rights data available."} for hex_id in target_hexagons
            }

        hex_gdf = gpd.GeoDataFrame(
            {"hex_id": target_hexagons},
            geometry=[
                Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
                for h in target_hexagons
            ],
            crs="EPSG:4326",
        )

        water_rights_gdf = water_rights_gdf.to_crs(hex_gdf.crs)
        logger.info("Performing spatial join between hexagons and water rights points...")
        joined_gdf = gpd.sjoin(hex_gdf, water_rights_gdf, how="inner", predicate="contains")

        if joined_gdf.empty:
            logger.info("Spatial join resulted in no matches. All hexagons have 0 rights.")
            results = {}
            for hex_id in target_hexagons:
                results[hex_id] = {
                    "number_of_rights": 0,
                    "number_of_active_rights": 0,
                    "total_flow_rate_cfs": 0.0,
                    "active_flow_rate_cfs": 0.0,
                    "data_sources": [],
                }
            return results

        logger.info("Aggregating water rights results per hexagon...")
        h3_water_rights = {}

        status_col = self._find_col(joined_gdf, ["status", "STATUS", "pod_status"])
        flow_rate_col = self._find_col(
            joined_gdf, ["flow_rate_cfs", "POD_FLOW_RATE", "face_value_flow_rate", "wr_flow"]
        )

        if not flow_rate_col:
            logger.warning(
                "No recognizable flow rate column found in the data. Flow rates will be 0."
            )
        else:
            # Ensure the flow rate column is numeric, coercing errors
            joined_gdf[flow_rate_col] = pd.to_numeric(
                joined_gdf[flow_rate_col], errors="coerce"
            ).fillna(0)

        for hex_id, group in joined_gdf.groupby("hex_id"):
            # Determine active rights
            active_rights_mask = pd.Series(False, index=group.index)
            if status_col:
                active_rights_mask = (
                    group[status_col].astype(str).str.lower().isin(["active", "licensed", "actv"])
                )
            active_rights = group[active_rights_mask]

            # Calculate flow rates
            total_flow = group[flow_rate_col].sum() if flow_rate_col else 0.0
            active_flow = (
                active_rights[flow_rate_col].sum()
                if flow_rate_col and not active_rights.empty
                else 0.0
            )

            h3_water_rights[hex_id] = {
                "number_of_rights": len(group),
                "number_of_active_rights": len(active_rights),
                "total_flow_rate_cfs": total_flow,
                "active_flow_rate_cfs": active_flow,
                "data_sources": group["state"].unique().tolist(),
            }

        # Add empty results for hexagons that had no matching water rights
        for hex_id in target_hexagons:
            if hex_id not in h3_water_rights:
                h3_water_rights[hex_id] = {
                    "number_of_rights": 0,
                    "number_of_active_rights": 0,
                    "total_flow_rate_cfs": 0.0,
                    "active_flow_rate_cfs": 0.0,
                    "data_sources": [],
                }

        logger.info(f"Completed water rights analysis. Processed {len(h3_water_rights)} hexagons.")
        return h3_water_rights
