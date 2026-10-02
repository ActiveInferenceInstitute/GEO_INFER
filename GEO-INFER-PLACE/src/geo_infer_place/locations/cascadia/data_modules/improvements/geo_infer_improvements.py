"""
GeoInfer Improvements Module

This module analyzes agricultural improvement data within an H3 grid.
"""

import logging
from typing import Any
from pathlib import Path
import geopandas as gpd

from .data_sources import CascadianImprovementsDataSources
from geo_infer_space.core.base_module import BaseAnalysisModule

# A forward declaration for type hinting the backend without circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferImprovements(BaseAnalysisModule):
    """Processes and analyzes improvements data within an H3 grid."""

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        super().__init__("improvements", h3_resolution=8)
        self.backend = backend
        self.data_dir = Path("output/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.target_hexagons = backend.target_hexagons
        self.improvements_data_source = CascadianImprovementsDataSources()
        logger.info("Initialized GeoInferImprovements module.")

    def acquire_raw_data(self) -> Path:
        """
        Acquire raw improvements data for Del Norte county.
        Returns path to the raw data file.
        """
        logger.info(f"[{self.module_name}] 🔍 Acquiring raw improvements data...")

        # Resolve standardized paths if data_manager is available
        try:
            if hasattr(self, "data_manager") and self.data_manager is not None:  # type: ignore[attr-defined]
                paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[attr-defined]
                empirical_data_path = paths["empirical_data"]
                raw_data_path = paths["raw_data"]
            else:
                empirical_data_path = Path(
                    "output/data/empirical_improvements_data.geojson"
                )
                raw_data_path = Path("output/data/raw_improvements_data.geojson")
        except Exception:
            empirical_data_path = Path(
                "output/data/empirical_improvements_data.geojson"
            )
            raw_data_path = Path("output/data/raw_improvements_data.geojson")

        # Cache-first: empirical, then raw.
        if empirical_data_path.exists():
            logger.info(
                f"[{self.module_name}] ✅ Found empirical improvements data: {empirical_data_path}"
            )
            return empirical_data_path
        if raw_data_path.exists():
            logger.info(
                f"[{self.module_name}] ✅ Found cached raw improvements data: {raw_data_path}"
            )
            return raw_data_path
        # Attempt real acquisition via data source using target hexagon bbox
        try:
            if not self.target_hexagons:
                raise ValueError(
                    "Target hexagons are required for improvements acquisition"
                )

            gdf = self.improvements_data_source.fetch_all_improvements_data(
                list(self.target_hexagons)
            )
            if gdf is None or gdf.empty:
                raise ValueError("Improvements data source returned no records")

            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(raw_data_path, driver="GeoJSON")
            logger.info(
                f"[{self.module_name}] 💾 Saved raw improvements data: {raw_data_path}"
            )
            return raw_data_path
        except Exception as e:
            raise RuntimeError(
                f"[{self.module_name}] Improvements acquisition failed"
            ) from e

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """
        Perform improvements analysis on H3-indexed data.

        Args:
            h3_data: Dictionary of H3 cell ID -> improvements data

        Returns:
            Dictionary of H3 cell ID -> improvements analysis results
        """
        logger.info(f"Running improvements analysis on {len(h3_data)} H3 cells...")

        if not h3_data:
            logger.warning("No H3 data available for improvements analysis")
            return {}

        analysis_results = {}

        for h3_id, cell_data in h3_data.items():
            if not cell_data or "improvements" not in cell_data:
                continue

            improvements = cell_data["improvements"]
            if not improvements:
                continue

            # Convert improvements list back to GeoDataFrame for analysis
            improvements_gdf = gpd.GeoDataFrame(improvements)

            # Perform improvements analysis
            improvement_metrics = self._analyze_improvements(improvements_gdf)

            analysis_results[h3_id] = {
                "total_improvement_value": improvement_metrics.get(
                    "total_improvement_value", 0.0
                ),
                "total_land_value": improvement_metrics.get("total_land_value", 0.0),
                "improvement_to_land_value_ratio": improvement_metrics.get(
                    "improvement_ratio", 0.0
                ),
                "number_of_improvements": improvement_metrics.get(
                    "improvement_count", 0
                ),
                "modernization_score": improvement_metrics.get(
                    "modernization_score", 0.0
                ),
                "building_density": improvement_metrics.get("building_density", 0.0),
                "average_building_value": improvement_metrics.get(
                    "avg_building_value", 0.0
                ),
                "score": improvement_metrics.get("redevelopment_score", 0.0),
            }

        logger.info(
            f"Completed improvements analysis for {len(analysis_results)} cells"
        )
        return analysis_results

    def _analyze_improvements(
        self, improvements_gdf: gpd.GeoDataFrame
    ) -> dict[str, Any]:
        """
        Analyze improvements and building patterns.

        Args:
            improvements_gdf: GeoDataFrame of improvements/buildings

        Returns:
            Dictionary of improvement metrics
        """
        if improvements_gdf.empty:
            return {
                "total_improvement_value": 0.0,
                "total_land_value": 0.0,
                "improvement_ratio": 0.0,
                "improvement_count": 0,
                "modernization_score": 0.0,
                "building_density": 0.0,
                "avg_building_value": 0.0,
                "redevelopment_score": 0.0,
            }

        # Try to find value columns (different data sources use different names)
        imp_val_col = self._find_column(
            improvements_gdf,
            [
                "improvement_value",
                "bldg_value",
                "building_value",
                "structure_value",
                "IMPROVEMENT_VALUE",
            ],
        )
        land_val_col = self._find_column(
            improvements_gdf,
            ["land_value", "parcel_value", "assessed_value", "LAND_VALUE"],
        )

        # Calculate basic metrics
        improvement_count = len(improvements_gdf)
        total_improvement_value = (
            improvements_gdf[imp_val_col].sum() if imp_val_col else 0.0
        )
        total_land_value = improvements_gdf[land_val_col].sum() if land_val_col else 0.0
        avg_building_value = (
            total_improvement_value / improvement_count
            if improvement_count > 0
            else 0.0
        )

        # Calculate improvement-to-land ratio
        improvement_ratio = (
            (total_improvement_value / total_land_value)
            if total_land_value > 0
            else 0.0
        )

        # Calculate building density (buildings per km²)
        # Approximate H3 cell area at resolution 8 is ~0.46 km²
        h3_area_km2 = 0.46
        building_density = improvement_count / h3_area_km2

        # Calculate modernization score based on improvement patterns
        modernization_score = self._calculate_modernization_score(
            improvement_ratio, building_density, avg_building_value, improvement_count
        )

        # Calculate redevelopment potential score
        redevelopment_score = self._calculate_redevelopment_score(
            improvement_ratio, building_density, modernization_score, improvement_count
        )

        return {
            "total_improvement_value": total_improvement_value,
            "total_land_value": total_land_value,
            "improvement_ratio": improvement_ratio,
            "improvement_count": improvement_count,
            "modernization_score": modernization_score,
            "building_density": building_density,
            "avg_building_value": avg_building_value,
            "redevelopment_score": redevelopment_score,
        }

    def _find_column(self, gdf: gpd.GeoDataFrame, potential_names: list[str]) -> str:
        """Find the first matching column name in the GeoDataFrame."""
        for name in potential_names:
            if name in gdf.columns:
                return name
        return None

    def _calculate_modernization_score(
        self,
        improvement_ratio: float,
        building_density: float,
        avg_building_value: float,
        building_count: int,
    ) -> float:
        """
        Calculate modernization score based on improvement characteristics.

        Higher scores indicate more modern/valuable improvements.

        Args:
            improvement_ratio: Ratio of improvement value to land value
            building_density: Buildings per km²
            avg_building_value: Average value per building
            building_count: Total number of buildings

        Returns:
            Modernization score (0.0 to 1.0)
        """
        score = 0.0

        # Improvement ratio factor (moderate ratios indicate good balance)
        if improvement_ratio > 0:
            # Optimal range: 0.3-1.5 (reasonable improvement-to-land ratios)
            if 0.3 <= improvement_ratio <= 1.5:
                ratio_score = 1.0
            elif improvement_ratio < 0.3:
                ratio_score = improvement_ratio / 0.3  # Under-improved
            else:
                ratio_score = max(0.2, 1.5 / improvement_ratio)  # Over-improved
        else:
            ratio_score = 0.0

        # Building density factor (moderate density preferred)
        if building_density > 0:
            # Optimal range: 2-10 buildings per km² for agricultural areas
            if 2 <= building_density <= 10:
                density_score = 1.0
            elif building_density < 2:
                density_score = building_density / 2
            else:
                density_score = max(0.3, 10 / building_density)
        else:
            density_score = 0.0

        # Building value factor (higher values indicate more modern structures)
        if avg_building_value > 0:
            # Normalize against typical agricultural building values ($50k-$500k)
            if avg_building_value >= 500000:
                value_score = 1.0
            elif avg_building_value >= 50000:
                value_score = (avg_building_value - 50000) / 450000
            else:
                value_score = avg_building_value / 50000
        else:
            value_score = 0.0

        # Building count factor (some infrastructure needed but not too much)
        if building_count > 0:
            # Optimal range: 3-15 buildings per H3 cell for agricultural
            if 3 <= building_count <= 15:
                count_score = 1.0
            elif building_count < 3:
                count_score = building_count / 3
            else:
                count_score = max(0.2, 15 / building_count)
        else:
            count_score = 0.0

        # Weighted combination
        score = (
            ratio_score * 0.35
            + density_score * 0.25
            + value_score * 0.25
            + count_score * 0.15
        )

        return min(1.0, max(0.0, score))

    def _calculate_redevelopment_score(
        self,
        improvement_ratio: float,
        building_density: float,
        modernization_score: float,
        building_count: int,
    ) -> float:
        """
        Calculate redevelopment potential score based on improvement patterns.

        Higher scores indicate better redevelopment potential:
        - Existing infrastructure but not over-developed
        - Reasonable improvement values
        - Good modernization potential

        Args:
            improvement_ratio: Ratio of improvement value to land value
            building_density: Buildings per km²
            modernization_score: Score indicating modernization level
            building_count: Total number of buildings

        Returns:
            Redevelopment potential score (0.0 to 1.0)
        """
        score = 0.0

        # Infrastructure readiness factor
        if building_count > 0:
            # Some infrastructure exists (good for redevelopment)
            infrastructure_score = min(
                1.0, building_count / 5
            )  # Up to 5 buildings = full score
        else:
            infrastructure_score = 0.0  # No infrastructure

        # Under-development factor (opportunity for improvement)
        if improvement_ratio > 0:
            # Lower ratios indicate under-development (more potential)
            if improvement_ratio <= 0.5:
                underdevelopment_score = 1.0 - improvement_ratio
            elif improvement_ratio <= 1.0:
                underdevelopment_score = 0.5
            else:
                underdevelopment_score = max(0.1, 1.0 / improvement_ratio)
        else:
            underdevelopment_score = 1.0  # No improvements = high potential

        # Modernization potential (room for improvement)
        modernization_potential = 1.0 - modernization_score  # Inverse of modernization

        # Density factor (not too dense, not too sparse)
        if 1 <= building_density <= 8:
            density_factor = 1.0
        elif building_density < 1:
            density_factor = building_density
        else:
            density_factor = max(0.2, 8 / building_density)

        # Weighted combination
        score = (
            infrastructure_score * 0.3
            + underdevelopment_score * 0.3
            + modernization_potential * 0.25
            + density_factor * 0.15
        )

        return min(1.0, max(0.0, score))
