import logging
from typing import Any
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
from .data_sources import CascadianGroundWaterDataSources
from geo_infer_space.utils.h3_utils import cell_to_latlng_boundary
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferGroundWater:
    """
    Analyzes groundwater availability by fetching real well data from the USGS
    for a given set of H3 hexagons.
    """

    module_name: str = "ground_water"

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        self.backend = backend
        self.resolution = getattr(backend, "h3_resolution", 8)
        self.target_hexagons = list(getattr(backend, "target_hexagons", []))
        self.data_source = CascadianGroundWaterDataSources()
        # Will be injected
        self.data_manager = None  # type: ignore[attr-defined]
        self.h3_fusion = None  # type: ignore[attr-defined]
        logger.info(f"Initialized GeoInferGroundWater with resolution {self.resolution}")

    def acquire_raw_data(self) -> Path:
        """Acquire and cache raw groundwater wells in target area."""
        if not hasattr(self, "data_manager") or self.data_manager is None:
            raw_out = Path("output/data/raw_ground_water_data.geojson")
        else:
            paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[union-attr]
            raw_out = paths["raw_data"]

        if raw_out.exists():
            return raw_out

        wells = self.data_source.fetch_groundwater_data(self.target_hexagons)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        if wells is None or wells.empty:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_out, driver="GeoJSON")
        else:
            wells.to_file(raw_out, driver="GeoJSON")
        return raw_out

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Pass-through for groundwater presence when features already summarized upstream."""
        return {hex_id: v for hex_id, v in h3_data.items() if isinstance(v, dict)}

    def run_analysis(self, target_hexagons: list[str]) -> dict[str, dict[str, Any]]:
        """
        Performs groundwater analysis by querying the USGS NWIS for wells within
        the target hexagons.

        Args:
            target_hexagons: A list of H3 hexagon IDs to analyze.

        Returns:
            A dictionary mapping H3 hexagon IDs to their groundwater analysis.
        """
        logger.info(f"Starting groundwater analysis for {len(target_hexagons)} hexagons.")
        if not target_hexagons:
            return {}

        # The data source now handles batching and bounding box calculation
        wells_gdf = self.data_source.fetch_groundwater_data(target_hexagons)

        if wells_gdf.empty:
            logger.warning(
                "No groundwater wells found in the entire analysis area. Returning empty results."
            )
            return {}

        # Spatially index the wells for efficient lookups
        wells_sindex = wells_gdf.sindex

        results = {}
        for h3_index in target_hexagons:
            try:
                hex_poly = Polygon([(lng, lat) for lat, lng in cell_to_latlng_boundary(h3_index)])

                # Find wells that intersect with the hexagon's bounding box first
                possible_matches_index = list(wells_sindex.intersection(hex_poly.bounds))
                possible_matches = wells_gdf.iloc[possible_matches_index]

                # Perform precise intersection check
                precise_matches = possible_matches[possible_matches.intersects(hex_poly)]

                well_count = len(precise_matches)
                if well_count > 0:
                    # Score is a simple heuristic: 1.0 if any wells exist.
                    # A more complex score could be based on well depth, yield, etc.
                    score = 1.0
                    results[h3_index] = {
                        "has_ground_water": True,
                        "groundwater_well_count": well_count,
                        "groundwater_availability_score": score,
                    }
                else:
                    results[h3_index] = {
                        "has_ground_water": False,
                        "groundwater_well_count": 0,
                        "groundwater_availability_score": 0.0,
                    }
            except Exception as e:
                logger.error(
                    f"Failed to process hexagon {h3_index} for groundwater analysis: {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Completed groundwater analysis. Found data for {len(results)} hexagons.")
        return results
