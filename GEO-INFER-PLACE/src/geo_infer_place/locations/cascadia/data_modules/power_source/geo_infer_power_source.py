"""
GeoInfer Power Source Module

This module analyzes proximity and density of high-voltage transmission
lines within an H3 grid, using data from the HIFLD open data portal.
"""

import logging
from typing import Any
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from .data_sources import CascadianPowerSourceDataSources
from geo_infer_space.utils.h3_utils import cell_to_latlng_boundary

# For type checking only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferPowerSource:
    """Analyzes proximity to high-voltage power infrastructure."""

    module_name: str = "power_source"

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        self.backend = backend
        self.resolution = getattr(backend, "h3_resolution", 8)
        self.target_hexagons = list(getattr(backend, "target_hexagons", []))
        self.data_source = CascadianPowerSourceDataSources()
        # Will be injected
        self.data_manager = None  # type: ignore[attr-defined]
        self.h3_fusion = None  # type: ignore[attr-defined]
        logger.info(
            f"Initialized GeoInferPowerSource with resolution {self.resolution}"
        )

    def acquire_raw_data(self) -> Path:
        """Acquire and cache raw HIFLD infrastructure features for target hexagons."""
        if not hasattr(self, "data_manager") or self.data_manager is None:
            raw_out = Path("output/data/raw_power_source_data.geojson")
        else:
            paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[union-attr]
            raw_out = paths["raw_data"]

        if raw_out.exists():
            return raw_out

        infra = self.data_source.fetch_power_infrastructure_features(
            self.target_hexagons
        )
        trans = infra.get("transmission_lines") or gpd.GeoDataFrame(
            geometry=[], crs="EPSG:4326"
        )
        plants = infra.get("power_plants") or gpd.GeoDataFrame(
            geometry=[], crs="EPSG:4326"
        )

        frames = []
        if not trans.empty:
            # Buffer lines to small polygons for H3 polygon processing
            try:
                tproj = trans.to_crs("EPSG:3310")
                tproj["geometry"] = tproj.buffer(30)  # ~30m
                trans = tproj.to_crs("EPSG:4326")
            except Exception as exc:
                logger.warning(
                    "Transmission line buffering skipped; CRS projection failed: %s",
                    exc,
                )
            trans["layer"] = "transmission_lines"
            frames.append(trans)
        if not plants.empty:
            plants["layer"] = "power_plants"
            frames.append(plants)

        gdf = (
            gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
            if frames
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(raw_out, driver="GeoJSON")
        return raw_out

    def run_analysis(self, target_hexagons: list[str]) -> dict[str, dict[str, Any]]:
        """
        Calculates the density of transmission lines and average voltage
        for each H3 hexagon.

        Args:
            target_hexagons: A list of H3 hexagon IDs to analyze.

        Returns:
            A dictionary mapping H3 hexagons to their power source analysis.
        """
        logger.info(
            f"Starting power source analysis for {len(target_hexagons)} hexagons."
        )
        if not target_hexagons:
            return {}

        # 1. Fetch all power infrastructure data for the target area
        infra_data = self.data_source.fetch_power_infrastructure_features(
            target_hexagons
        )

        trans_lines_gdf = infra_data.get("transmission_lines")

        if trans_lines_gdf is None or trans_lines_gdf.empty:
            logger.warning(
                "No transmission line data found for the target area. Aborting analysis."
            )
            return {
                hex_id: {"transmission_line_km": 0, "avg_voltage_kv": 0}
                for hex_id in target_hexagons
            }

        # 2. Create a GeoDataFrame for the target hexagons
        hex_geometries = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        hex_gdf = gpd.GeoDataFrame(
            {"hex_id": target_hexagons}, geometry=hex_geometries, crs="EPSG:4326"
        )

        # 3. Ensure CRS alignment
        trans_lines_gdf = trans_lines_gdf.to_crs(hex_gdf.crs)

        # 4. Perform a spatial join (intersection)
        logger.info(
            "Performing spatial join between hexagons and transmission lines..."
        )
        # An intersection is needed to clip lines to the hexagon boundaries
        joined_gdf = gpd.overlay(
            hex_gdf, trans_lines_gdf, how="intersection", keep_geom_type=False
        )

        if joined_gdf.empty:
            logger.warning(
                "Spatial join resulted in no matches. No hexagons contain transmission lines."
            )
            # Return a default "no grid" score for all hexagons
            return {
                hex_id: {"transmission_line_km": 0, "avg_voltage_kv": 0}
                for hex_id in target_hexagons
            }

        # 5. Calculate metrics for each hexagon
        # Project to an equal-area projection to calculate length accurately
        joined_gdf_proj = joined_gdf.to_crs(
            "EPSG:3310"
        )  # Albers Equal Area for North America
        joined_gdf_proj["line_length_km"] = joined_gdf_proj.geometry.length / 1000

        # The 'VOLTAGE' column might be named differently, we'll need to be robust.
        # Based on HIFLD data, it's typically 'VOLTAGE'.
        voltage_col = "VOLTAGE"
        if voltage_col not in joined_gdf_proj.columns:
            logger.warning(
                f"'{voltage_col}' column not found in data. Voltage analysis will be skipped."
            )
            voltage_col = None

        h3_power = {}
        for hex_id, group in joined_gdf_proj.groupby("hex_id"):
            total_length_km = group["line_length_km"].sum()

            avg_voltage = 0
            if voltage_col:
                # Weighted average of voltage by line length
                avg_voltage = (
                    (group[voltage_col] * group["line_length_km"]).sum()
                    / total_length_km
                    if total_length_km > 0
                    else 0
                )

            h3_power[hex_id] = {
                "transmission_line_km": round(total_length_km, 2),
                "avg_voltage_kv": round(avg_voltage, 2),
            }

        # Add hexagons that had no intersecting lines with default zero values
        for hex_id in target_hexagons:
            if hex_id not in h3_power:
                h3_power[hex_id] = {"transmission_line_km": 0, "avg_voltage_kv": 0}

        logger.info(
            f"Completed power source analysis for {len(target_hexagons)} hexagons."
        )
        return h3_power

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize H3-indexed power infrastructure features into per-hex metrics.

        Expects h3_data[hex] to be a list of feature dicts with optional 'layer' and
        voltage fields. Returns counts and presence booleans.
        """
        results: dict[str, Any] = {}
        for hex_id, items in h3_data.items():
            try:
                df = pd.DataFrame(items) if isinstance(items, list) else pd.DataFrame()
                trans_count = (
                    int((df.get("layer") == "transmission_lines").sum())
                    if not df.empty
                    else 0
                )
                plant_count = (
                    int((df.get("layer") == "power_plants").sum())
                    if not df.empty
                    else 0
                )
                results[hex_id] = {
                    "has_transmission": trans_count > 0,
                    "transmission_feature_count": trans_count,
                    "power_plant_count": plant_count,
                }
            except Exception:
                results[hex_id] = {
                    "has_transmission": False,
                    "transmission_feature_count": 0,
                    "power_plant_count": 0,
                }
        return results
