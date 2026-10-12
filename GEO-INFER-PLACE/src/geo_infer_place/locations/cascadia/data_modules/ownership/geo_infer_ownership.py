"""
GeoInfer Ownership Module

This module analyzes agricultural land ownership patterns using real OSC H3 v4 methods
by fetching real-time data from public GIS services and performing spatial analysis.
"""

import logging
from typing import Any
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import h3

from .data_sources import CascadianOwnershipDataSources
from geo_infer_space.core.base_module import BaseAnalysisModule

# A forward declaration for type hinting the backend without circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class GeoInferOwnership(BaseAnalysisModule):
    """
    Processes and analyzes ownership data within an H3 grid using real OSC H3 v4 methods.
    It adapts its analysis based on the richness of the data available from the source.
    """

    def __init__(self, backend: "CascadianAgriculturalH3Backend"):
        super().__init__("ownership", h3_resolution=8)
        self.backend = backend
        self.data_dir = Path("output/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.target_hexagons = backend.target_hexagons
        self.data_source = CascadianOwnershipDataSources()
        logger.info(
            "Initialized GeoInferOwnership module with real OSC H3 v4 integration."
        )

    def acquire_raw_data(self) -> Path:
        """
        Acquire raw ownership data for Del Norte county.
        Returns path to the raw data file.
        """
        logger.info(f"[{self.module_name}] 🔍 Acquiring raw ownership data...")

        # Check for empirical data first
        # Use standardized module data structure if enabled
        try:
            if hasattr(self, "data_manager") and self.data_manager is not None:  # type: ignore[attr-defined]
                paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[attr-defined]
                empirical_data_path = paths["empirical_data"]
                raw_data_path = paths["raw_data"]
            else:
                empirical_data_path = Path(
                    "output/data/empirical_ownership_data.geojson"
                )
                raw_data_path = Path("output/data/raw_ownership_data.geojson")
        except Exception:
            empirical_data_path = Path("output/data/empirical_ownership_data.geojson")
            raw_data_path = Path("output/data/raw_ownership_data.geojson")
        if empirical_data_path.exists():
            logger.info(
                f"[{self.module_name}] ✅ Found empirical ownership data: {empirical_data_path}"
            )
            return empirical_data_path

        if raw_data_path.exists():
            return raw_data_path
        if not self.target_hexagons:
            raise ValueError("Target hexagons are required for ownership acquisition")
        try:
            parcels = self.data_source.fetch_all_parcel_data(list(self.target_hexagons))
        except Exception as exc:
            raise RuntimeError("Ownership data acquisition failed") from exc
        if parcels is None or parcels.empty:
            raise FileNotFoundError("Ownership source returned no parcel records")
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        parcels.to_file(raw_data_path, driver="GeoJSON")
        return raw_data_path

    def run_final_analysis(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """
        Perform real ownership analysis on H3-indexed data using OSC H3 v4 methods.

        Args:
            h3_data: Dictionary of H3 cell ID -> parcel data

        Returns:
            Dictionary of H3 cell ID -> real ownership analysis results
        """
        logger.info(
            f"Running real ownership analysis on {len(h3_data)} H3 cells using OSC H3 v4 methods..."
        )

        if not h3_data:
            logger.warning("No H3 data available for real ownership analysis")
            return {}

        # Load real ownership data for spatial analysis
        try:
            ownership_gdf = gpd.read_file(
                self.data_dir / "empirical_ownership_data.geojson"
            )
            logger.info(
                f"Loaded {len(ownership_gdf)} real ownership features for analysis"
            )
        except Exception as e:
            logger.error(f"Failed to load real ownership data: {e}")
            return {}

        analysis_results = {}

        for h3_id in self.target_hexagons:
            try:
                # Get hexagon boundary using real H3 v4 methods
                hex_boundary = h3.cell_to_boundary(h3_id)
                hex_polygon = Polygon([(lng, lat) for lat, lng in hex_boundary])

                # Find intersecting ownership features using real spatial analysis
                intersecting_features = ownership_gdf[
                    ownership_gdf.intersects(hex_polygon)
                ]

                if len(intersecting_features) == 0:
                    # No ownership data for this hexagon
                    analysis_results[h3_id] = {
                        "ownership_concentration_hhi": 0.0,
                        "largest_owner_share_pct": 0.0,
                        "number_of_unique_owners": 0,
                        "number_of_parcels": 0,
                        "average_parcel_size_acres": 0.0,
                        "total_parcel_area_acres": 0.0,
                        "score": 0.0,
                        "owner_diversity": 0.0,
                        "parcel_fragmentation": 0.0,
                    }
                    continue

                # Perform real ownership analysis
                ownership_metrics = self._analyze_real_ownership_patterns(
                    intersecting_features, hex_polygon
                )

                analysis_results[h3_id] = {
                    "ownership_concentration_hhi": ownership_metrics.get("hhi", 0.0),
                    "largest_owner_share_pct": ownership_metrics.get(
                        "largest_share", 0.0
                    ),
                    "number_of_unique_owners": ownership_metrics.get(
                        "unique_owners", 0
                    ),
                    "number_of_parcels": ownership_metrics.get("parcel_count", 0),
                    "average_parcel_size_acres": ownership_metrics.get(
                        "avg_parcel_size", 0.0
                    ),
                    "total_parcel_area_acres": ownership_metrics.get("total_area", 0.0),
                    "score": ownership_metrics.get("redevelopment_score", 0.0),
                    "owner_diversity": ownership_metrics.get("owner_diversity", 0.0),
                    "parcel_fragmentation": ownership_metrics.get(
                        "parcel_fragmentation", 0.0
                    ),
                }

            except Exception as e:
                logger.error(
                    f"Real error in ownership analysis for hexagon {h3_id}: {e}"
                )
                continue

        logger.info(
            f"Completed real ownership analysis for {len(analysis_results)} cells using OSC H3 v4 methods"
        )
        return analysis_results

    def _analyze_real_ownership_patterns(
        self, features: gpd.GeoDataFrame, hex_polygon: Polygon
    ) -> dict[str, Any]:
        """
        Analyze real ownership patterns using actual spatial analysis.

        Args:
            features: GeoDataFrame of ownership features
            hex_polygon: Hexagon polygon for intersection analysis

        Returns:
            Dictionary of real ownership metrics
        """
        if features.empty:
            return {
                "hhi": 0.0,
                "largest_share": 0.0,
                "unique_owners": 0,
                "parcel_count": 0,
                "avg_parcel_size": 0.0,
                "total_area": 0.0,
                "redevelopment_score": 0.0,
                "owner_diversity": 0.0,
                "parcel_fragmentation": 0.0,
            }

        # Calculate real intersection areas
        total_area = 0.0
        owner_areas = {}

        for idx, feature in features.iterrows():
            try:
                # Calculate real intersection area
                intersection = feature.geometry.intersection(hex_polygon)
                if intersection.is_empty:
                    continue

                # Convert to acres (approximate conversion)
                area_acres = intersection.area * 0.000247105
                total_area += area_acres

                # Get owner information
                owner_name = feature.get("owner_name", "Unknown")
                if owner_name not in owner_areas:
                    owner_areas[owner_name] = 0.0
                owner_areas[owner_name] += area_acres

            except Exception as e:
                logger.warning(f"Error processing ownership feature {idx}: {e}")
                continue

        # Calculate real ownership metrics
        unique_owners = len(owner_areas)
        parcel_count = len(features)
        avg_parcel_size = total_area / parcel_count if parcel_count > 0 else 0.0

        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = 0.0
        largest_share = 0.0

        if total_area > 0 and owner_areas:
            owner_shares = {
                owner: (area / total_area) * 100 for owner, area in owner_areas.items()
            }
            hhi = sum(share**2 for share in owner_shares.values())
            largest_share = max(owner_shares.values())

        # Calculate real redevelopment score
        redevelopment_score = self._calculate_real_redevelopment_score(
            hhi, largest_share, unique_owners, parcel_count, avg_parcel_size
        )

        # Calculate additional metrics
        owner_diversity = self._calculate_owner_diversity(owner_areas, total_area)
        parcel_fragmentation = self._calculate_parcel_fragmentation(
            parcel_count, avg_parcel_size
        )

        return {
            "hhi": hhi,
            "largest_share": largest_share,
            "unique_owners": unique_owners,
            "parcel_count": parcel_count,
            "avg_parcel_size": avg_parcel_size,
            "total_area": total_area,
            "redevelopment_score": redevelopment_score,
            "owner_diversity": owner_diversity,
            "parcel_fragmentation": parcel_fragmentation,
        }

    def _calculate_real_redevelopment_score(
        self,
        hhi: float,
        largest_share: float,
        unique_owners: int,
        parcel_count: int,
        avg_parcel_size: float,
    ) -> float:
        """
        Calculate real redevelopment potential score based on actual ownership patterns.
        """
        score = 0.0

        # Ownership concentration factor (lower concentration = higher score)
        if hhi > 0:
            # Normalize HHI (range 0-10000, where 10000 = monopoly)
            concentration_score = max(0, 1 - (hhi / 10000))
        else:
            concentration_score = 0.5  # Default if no ownership data

        # Parcel size factor (moderate sizes preferred)
        if avg_parcel_size > 0:
            # Optimal range: 40-160 acres (good for agricultural development)
            if 40 <= avg_parcel_size <= 160:
                size_score = 1.0
            elif avg_parcel_size < 40:
                size_score = avg_parcel_size / 40  # Too small
            else:
                size_score = max(0.2, 160 / avg_parcel_size)  # Too large
        else:
            size_score = 0.0

        # Owner diversity factor
        if unique_owners > 0:
            # Sweet spot: 3-8 owners (diverse but manageable)
            if 3 <= unique_owners <= 8:
                diversity_score = 1.0
            elif unique_owners < 3:
                diversity_score = unique_owners / 3
            else:
                diversity_score = max(0.3, 8 / unique_owners)
        else:
            diversity_score = 0.0

        # Parcel count factor (avoid excessive fragmentation)
        if parcel_count > 0:
            # Prefer 5-20 parcels per H3 cell
            if 5 <= parcel_count <= 20:
                fragmentation_score = 1.0
            elif parcel_count < 5:
                fragmentation_score = parcel_count / 5
            else:
                fragmentation_score = max(0.2, 20 / parcel_count)
        else:
            fragmentation_score = 0.0

        # Real weighted combination
        score = (
            concentration_score * 0.3
            + size_score * 0.3
            + diversity_score * 0.25
            + fragmentation_score * 0.15
        )

        return min(1.0, max(0.0, score))

    def _calculate_owner_diversity(
        self, owner_areas: dict[str, float], total_area: float
    ) -> float:
        """
        Calculate real owner diversity score based on area distribution.
        """
        if not owner_areas or total_area == 0:
            return 0.0

        # Calculate Shannon diversity index
        proportions = [area / total_area for area in owner_areas.values()]
        diversity = -sum(p * np.log(p) for p in proportions if p > 0)

        # Normalize to 0-1 scale
        max_diversity = np.log(len(owner_areas))
        normalized_diversity = diversity / max_diversity if max_diversity > 0 else 0.0

        return min(1.0, normalized_diversity)

    def _calculate_parcel_fragmentation(
        self, parcel_count: int, avg_parcel_size: float
    ) -> float:
        """
        Calculate real parcel fragmentation score.
        """
        if parcel_count == 0 or avg_parcel_size == 0:
            return 0.0

        # Optimal fragmentation: 5-20 parcels with 40-160 acre average
        if 5 <= parcel_count <= 20 and 40 <= avg_parcel_size <= 160:
            return 1.0
        elif parcel_count < 5:
            return parcel_count / 5
        else:
            # Penalize excessive fragmentation
            return max(0.1, 20 / parcel_count)

    def _find_col(self, gdf: gpd.GeoDataFrame, potential_names: list[str]) -> str:
        """Finds the first matching column name in the GeoDataFrame."""
        for name in potential_names:
            if name in gdf.columns:
                return name
        return None

    def _analyze_ownership_patterns(
        self, parcels_gdf: gpd.GeoDataFrame
    ) -> dict[str, Any]:
        """
        Analyze ownership patterns in a set of parcels.

        Args:
            parcels_gdf: GeoDataFrame of parcels

        Returns:
            Dictionary of ownership metrics
        """
        if parcels_gdf.empty:
            return {
                "hhi": 0.0,
                "largest_share": 0.0,
                "unique_owners": 0,
                "parcel_count": 0,
                "avg_parcel_size": 0.0,
                "total_area": 0.0,
                "redevelopment_score": 0.0,
            }

        # Dynamically identify relevant columns
        owner_col = self._find_col(
            parcels_gdf, ["owner_name", "OWNERNAME", "OWNER", "PAROWNER"]
        )
        area_col = self._find_col(
            parcels_gdf, ["acreage", "ACRES", "GIS_ACRES", "calca_gis"]
        )

        if not area_col:
            # Calculate area from geometry if no area column exists
            try:
                # Project to equal-area projection for accurate calculation
                parcels_projected = parcels_gdf.to_crs("EPSG:3310")
                parcels_gdf["calculated_acres"] = (
                    parcels_projected.geometry.area * 0.000247105
                )
                area_col = "calculated_acres"
            except Exception:
                # Fallback to rough area calculation
                parcels_gdf["calculated_acres"] = (
                    parcels_gdf.geometry.area * 111000 * 111000 * 0.000247105
                )
            area_col = "calculated_acres"

        total_area = parcels_gdf[area_col].sum()
        num_parcels = len(parcels_gdf)
        avg_parcel_size = parcels_gdf[area_col].mean()

        # Initialize metrics
        hhi = 0.0
        largest_share = 0.0
        unique_owners = 0

        # Perform ownership concentration analysis if owner column exists
        if owner_col and owner_col in parcels_gdf.columns:
            # Group by owner and sum their area
            owner_areas = parcels_gdf.groupby(owner_col)[area_col].sum()
            unique_owners = len(owner_areas)

            if total_area > 0:
                # Calculate ownership shares as percentages
                owner_shares = (owner_areas / total_area) * 100

                # Calculate Herfindahl-Hirschman Index (HHI) for concentration
                hhi = (owner_shares**2).sum()
                largest_share = owner_shares.max()

        # Calculate redevelopment potential score based on ownership patterns
        redevelopment_score = self._calculate_real_redevelopment_score(
            hhi, largest_share, unique_owners, num_parcels, avg_parcel_size
        )

        return {
            "hhi": hhi,
            "largest_share": largest_share,
            "unique_owners": unique_owners,
            "parcel_count": num_parcels,
            "avg_parcel_size": avg_parcel_size,
            "total_area": total_area,
            "redevelopment_score": redevelopment_score,
        }
