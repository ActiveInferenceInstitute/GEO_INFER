"""
GeoInfer Surface Water Module

This module analyzes surface water resources (rivers, tributaries, lakes) within
an H3 grid by using data from the USGS National Hydrography Dataset (NHD / NHDPlus HR)
and provides river network topology traversal, validation, and spatial analysis.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from geo_infer_space.utils.h3_utils import cell_to_latlng_boundary
from .data_sources import CascadianSurfaceWaterDataSources
from .flowline_network import CascadiaFlowlineNetwork

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferSurfaceWater:
    """Analyzes surface water features by quantifying water body area, flowline length,

    river network connectivity, Strahler stream order hierarchy, and topology traversal
    within H3 hexagons across the Cascadia Bioregion.
    """

    module_name: str = "surface_water"

    def __init__(
        self,
        backend: "CascadianAgriculturalH3Backend",
        *,
        data_source: CascadianSurfaceWaterDataSources | None = None,
        allow_projection_fallback: bool = False,
    ) -> None:
        self.backend = backend
        self.resolution = getattr(backend, "h3_resolution", 8)
        self.target_hexagons = list(getattr(backend, "target_hexagons", []))
        self.data_source = (
            data_source
            if data_source is not None
            else CascadianSurfaceWaterDataSources()
        )
        self._network: CascadiaFlowlineNetwork | None = None
        self.allow_projection_fallback = allow_projection_fallback
        self.projection_degraded = False
        # Will be injected
        self.data_manager = None  # type: ignore[attr-defined]
        self.h3_fusion = None  # type: ignore[attr-defined]
        logger.info(
            f"Initialized GeoInferSurfaceWater with resolution {self.resolution}"
        )

    @property
    def network(self) -> CascadiaFlowlineNetwork:
        """Get or lazily build the Cascadia flowline network."""
        if self._network is None:
            self._network = self.data_source.get_flowline_network(min_stream_order=4)
        return self._network

    def get_flowline_network(
        self, min_stream_order: int = 4
    ) -> CascadiaFlowlineNetwork:
        """Return full connectivity; the threshold selects a view without dropping tributaries."""
        return self.data_source.get_flowline_network(min_stream_order=min_stream_order)

    def validate_network_topology(self, min_stream_order: int = 4) -> dict[str, Any]:
        """Validate network topology and Strahler stream order monotonicity."""
        net = self.get_flowline_network(min_stream_order=min_stream_order)
        return net.validate()

    def trace_upstream_tributaries(self, comid: int) -> list[dict[str, Any]]:
        """Trace all upstream tributaries feeding into a flowline reach."""
        return self.network.trace_upstream(comid)

    def trace_downstream_flowpath(self, comid: int) -> list[dict[str, Any]]:
        """Trace downstream flowpath from a reach to its terminal basin outlet."""
        return self.network.trace_downstream(comid)

    def acquire_raw_data(self) -> Path:
        """Acquire and cache raw NHD flowlines/waterbodies for target area."""
        if not hasattr(self, "data_manager") or self.data_manager is None:
            raw_out = Path("output/data/raw_surface_water_data.geojson")
        else:
            paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[union-attr]
            raw_out = paths["raw_data"]

        bbox = self._get_analysis_bbox(self.target_hexagons)
        nhd = self.data_source.fetch_surface_water_features(bbox)
        waterbodies = nhd["waterbodies"].copy()
        flowlines = nhd["flowlines"].copy()

        self.projection_degraded = False
        frames = []
        if not waterbodies.empty:
            waterbodies["layer"] = "waterbodies"
            frames.append(waterbodies)
        if not flowlines.empty:
            # Buffer lines to narrow polygons for H3 polygon processing
            try:
                fproj = flowlines.to_crs("EPSG:5070")
                fproj["geometry"] = fproj.buffer(10)  # ~10m buffer
                flowlines = fproj.to_crs("EPSG:4326")
            except Exception as exc:
                logger.warning("Flowline buffering projection failed: %s", exc)
                if not self.allow_projection_fallback:
                    raise
                self.projection_degraded = True
                logger.warning(
                    "Explicit projection fallback enabled; writing unbuffered flowlines"
                )
            flowlines["projection_degraded"] = self.projection_degraded
            flowlines["layer"] = "flowlines"
            frames.append(flowlines)
        gdf = (
            gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
            if frames
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(raw_out, driver="GeoJSON")
        return raw_out

    def _get_analysis_bbox(
        self, target_hexagons: list[str]
    ) -> tuple[float, float, float, float]:
        """Calculates the total bounding box for a list of H3 hexagons."""
        if not target_hexagons:
            raise ValueError(
                "Explicit target hexagons are required; no regional download default"
            )
        polygons = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        min_lons, min_lats, max_lons, max_lats = [], [], [], []
        for p in polygons:
            min_lon, min_lat, max_lon, max_lat = p.bounds
            min_lons.append(min_lon)
            min_lats.append(min_lat)
            max_lons.append(max_lon)
            max_lats.append(max_lat)
        return (min(min_lons), min(min_lats), max(max_lons), max(max_lats))

    def run_analysis(self, target_hexagons: list[str]) -> dict[str, dict[str, Any]]:
        """Calculates water body area, flowline length, stream order metrics, and

        network connectivity for each target hexagon.
        """
        logger.info(
            f"Starting surface water analysis for {len(target_hexagons)} hexagons."
        )

        if not target_hexagons:
            return {}

        # 1. Get the bounding box for all hexagons
        analysis_bbox = self._get_analysis_bbox(target_hexagons)

        # 2. Fetch all surface water features from NHD
        nhd_data = self.data_source.fetch_surface_water_features(analysis_bbox)
        flowlines_gdf = nhd_data.get("flowlines")
        waterbodies_gdf = nhd_data.get("waterbodies")

        if (flowlines_gdf is None or flowlines_gdf.empty) and (
            waterbodies_gdf is None or waterbodies_gdf.empty
        ):
            logger.warning("No surface water features found in the target area.")
            return {
                hex_id: {
                    "water_body_area_sqkm": None
                    if waterbodies_gdf is None
                    or waterbodies_gdf.attrs.get("status") == "not_queried"
                    else 0.0,
                    "flowline_length_km": 0.0,
                    "max_stream_order": 0,
                    "has_high_order_river": False,
                }
                for hex_id in target_hexagons
            }

        # 3. Create a GeoDataFrame for the target hexagons
        hex_geometries = [
            Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h)])
            for h in target_hexagons
        ]
        hex_gdf = gpd.GeoDataFrame(
            {"hex_id": target_hexagons}, geometry=hex_geometries, crs="EPSG:4326"
        )

        # 4. Initialize results
        results: dict[str, dict[str, Any]] = {
            hex_id: {
                "water_body_area_sqkm": None
                if waterbodies_gdf is None
                or waterbodies_gdf.attrs.get("status") == "not_queried"
                else 0.0,
                "flowline_length_km": 0.0,
                "max_stream_order": 0,
                "has_high_order_river": False,
            }
            for hex_id in target_hexagons
        }

        # 5. Analyze water bodies (polygons)
        if waterbodies_gdf is not None and not waterbodies_gdf.empty:
            waterbodies_gdf = waterbodies_gdf.to_crs(hex_gdf.crs)
            logger.info("Intersecting hexagons with water bodies...")
            intersected_wb = gpd.overlay(hex_gdf, waterbodies_gdf, how="intersection")

            if not intersected_wb.empty:
                # Project to an equal-area projection for accurate area calculation
                intersected_wb_proj = intersected_wb.to_crs("EPSG:5070")
                intersected_wb_proj["area_sqkm"] = (
                    intersected_wb_proj.geometry.area / 1_000_000
                )

                area_by_hex = intersected_wb_proj.groupby("hex_id")["area_sqkm"].sum()
                for hex_id, area in area_by_hex.items():
                    results[hex_id]["water_body_area_sqkm"] = round(float(area), 4)

        # 6. Analyze flowlines (linestrings)
        if flowlines_gdf is not None and not flowlines_gdf.empty:
            flowlines_gdf = flowlines_gdf.to_crs(hex_gdf.crs)
            logger.info("Intersecting hexagons with flowlines...")
            intersected_fl = gpd.overlay(
                hex_gdf, flowlines_gdf, how="intersection", keep_geom_type=False
            )

            if not intersected_fl.empty:
                # Project to an equal-area projection for accurate length calculation
                intersected_fl_proj = intersected_fl.to_crs("EPSG:5070")
                intersected_fl_proj["length_km"] = (
                    intersected_fl_proj.geometry.length / 1000.0
                )

                length_by_hex = intersected_fl_proj.groupby("hex_id")["length_km"].sum()
                for hex_id, length in length_by_hex.items():
                    results[hex_id]["flowline_length_km"] = round(float(length), 4)

                order_col = (
                    "stream_order"
                    if "stream_order" in intersected_fl_proj.columns
                    else "StreamOrder"
                    if "StreamOrder" in intersected_fl_proj.columns
                    else None
                )
                if order_col:
                    max_order_by_hex = intersected_fl_proj.groupby("hex_id")[
                        order_col
                    ].max()
                    for hex_id, max_order in max_order_by_hex.items():
                        if pd.isna(max_order):
                            continue
                        results[hex_id]["max_stream_order"] = int(max_order)
                        results[hex_id]["has_high_order_river"] = int(max_order) >= 5

        logger.info(
            f"Completed surface water analysis for {len(target_hexagons)} hexagons."
        )
        return results

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize H3-indexed water features into per-hex metrics if provided as features."""
        results: dict[str, Any] = {}
        for hex_id, items in h3_data.items():
            if not isinstance(items, list) or any(
                not isinstance(item, dict) for item in items
            ):
                raise ValueError(
                    "Each H3 cell must contain a list of feature dictionaries"
                )
            waterbodies = sum(item.get("layer") == "waterbodies" for item in items)
            flowlines = [item for item in items if item.get("layer") == "flowlines"]
            orders = [
                int(item["stream_order"])
                for item in flowlines
                if item.get("stream_order") is not None
                and not pd.isna(item["stream_order"])
                and int(item["stream_order"]) >= 0
            ]
            max_order = max(orders) if orders else None if flowlines else 0
            results[hex_id] = {
                "has_water": bool(waterbodies or flowlines),
                "waterbody_feature_count": waterbodies,
                "flowline_feature_count": len(flowlines),
                "max_stream_order": max_order,
                "has_high_order_river": max_order is not None and max_order >= 5,
            }
        return results
