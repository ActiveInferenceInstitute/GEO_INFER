"""
Place Analyzer - Spatial place analysis for GEO-INFER-SPACE.

This module provides comprehensive place-based spatial analysis capabilities
including demographic analysis, environmental assessment, and spatial indexing.
"""

import json
import logging
from typing import Any, cast
from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import h3

logger = logging.getLogger(__name__)


class PlaceAnalyzer:
    """
    Advanced place-based spatial analysis for GEO-INFER framework.

    Provides comprehensive spatial analysis capabilities including:
    - Demographic analysis
    - Environmental assessment
    - Spatial indexing and querying
    - Place-based data integration
    """

    def __init__(self, base_dir: str | None = None) -> None:
        """
        Initialize PlaceAnalyzer with base directory.

        Args:
            base_dir: Base directory for data storage and configuration
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.config_dir = self.base_dir / "config"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analysis components
        self.spatial_index: dict[str, Any] = {}
        self.place_data: dict[str, Any] = {}
        self.analysis_results: dict[str, Any] = {}

        logger.info(f"PlaceAnalyzer initialized with base_dir: {self.base_dir}")

    def analyze_place(
        self,
        place_name: str,
        coordinates: tuple[float, float],
        radius_km: float = 10.0,
        synthetic: bool = True,
    ) -> dict[str, Any]:
        """
        Perform comprehensive place analysis.

        Args:
            place_name: Name of the place to analyze
            coordinates: (latitude, longitude) coordinates
            radius_km: Analysis radius in kilometers
            synthetic: If True (default), the environmental factors and
                accessibility metrics are SYNTHETIC DEMO DATA — deterministic
                demo values generated from the coordinates, not real
                observations. Synthetic results are clearly labelled and are
                excluded from environmental/accessibility scores in
                get_analysis_summary(). Real data integrations (e.g. OSM
                street networks, POI databases) are not implemented yet;
                passing synthetic=False skips these sections entirely.

        Returns:
            Dictionary containing analysis results. When synthetic=True the
            result carries 'synthetic': True and both metric sections carry
            'data_provenance': 'synthetic_demo'.
        """
        lat, lon = coordinates

        if synthetic:
            logger.warning(
                f"Place analysis for '{place_name}' is running in SYNTHETIC "
                "demo mode: environmental factors and accessibility metrics "
                "are generated demo data, not real observations, and "
                "will be excluded from scores."
            )

        # Create analysis area
        analysis_area = self._create_analysis_area(lat, lon, radius_km)

        # Perform spatial analysis
        results = {
            "place_name": place_name,
            "coordinates": coordinates,
            "radius_km": radius_km,
            "synthetic": synthetic,
            "analysis_area": analysis_area,
            "h3_cells": self._get_h3_cells(lat, lon, radius_km),
            "spatial_metrics": self._calculate_spatial_metrics(analysis_area),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        if synthetic:
            environmental = self._analyze_environmental_factors(lat, lon, radius_km)
            accessibility = self._calculate_accessibility(lat, lon, radius_km)
            environmental["data_provenance"] = "synthetic_demo"
            accessibility["data_provenance"] = "synthetic_demo"
            results["environmental_factors"] = environmental
            results["accessibility_metrics"] = accessibility

        # Store results
        self.analysis_results[place_name] = results

        logger.info(f"Completed analysis for {place_name}")
        return results

    def _create_analysis_area(
        self, lat: float, lon: float, radius_km: float
    ) -> Polygon:
        """
        Create analysis area polygon.

        Note:
            Approximation: converts the radius to degrees with a flat 111 km
            per degree factor, ignoring latitude-dependent convergence of
            meridians. Accuracy degrades away from the equator.
        """
        center = Point(lon, lat)
        return center.buffer(radius_km / 111.0)

    def _get_h3_cells(self, lat: float, lon: float, radius_km: float) -> list[str]:
        """Get H3 cells covering the analysis area."""
        # Determine appropriate H3 resolution based on radius
        if radius_km <= 1:
            resolution = 9
        elif radius_km <= 5:
            resolution = 8
        elif radius_km <= 20:
            resolution = 7
        else:
            resolution = 6
        # Get center cell
        center_cell = h3.latlng_to_cell(lat, lon, resolution)
        # Get cells within radius
        cells = h3.grid_disk(center_cell, int(radius_km / 2))
        return list(cells)

    def _calculate_spatial_metrics(self, area: Polygon) -> dict[str, float]:
        """
        Calculate spatial metrics for the analysis area.

        Note:
            area_km2/perimeter_km use a crude degree-to-kilometre conversion
            (1 degree ~= 111 km) valid only near the equator; values are
            rough estimates, not geodesic measurements.
        """
        return {
            "area_km2": area.area * 111.0 * 111.0,
            "perimeter_km": area.length * 111.0,
            "compact_cellsness": 4 * np.pi * area.area / (area.length**2)
            if area.length > 0
            else 0,
        }

    def _analyze_environmental_factors(
        self, lat: float, lon: float, radius_km: float
    ) -> dict[str, Any]:
        """
        Generate SYNTHETIC demo environmental factors.

        These values are deterministic placeholders seeded from the
        coordinates — they are NOT real environmental observations and must
        not be used for real-world decisions. Callers must treat this output
        as demo data (flagged via 'data_provenance': 'synthetic_demo').
        """
        rng = np.random.RandomState(int(abs(lat * lon * 100)) % 10000)
        return {
            "elevation_range": {
                "min": float(rng.uniform(0, 100)),
                "max": float(rng.uniform(100, 2000)),
                "mean": float(rng.uniform(50, 500)),
            },
            "climate_zone": rng.choice(
                ["temperate", "tropical", "arid", "continental", "polar"]
            ),
            "vegetation_cover": float(rng.uniform(0.1, 0.95)),
            "water_bodies": int(rng.randint(0, 5)),
            "protected_areas": int(rng.randint(0, 3)),
        }

    def _calculate_accessibility(
        self, lat: float, lon: float, radius_km: float
    ) -> dict[str, float]:
        """
        Generate SYNTHETIC demo accessibility metrics.

        Hardcoded demo values — NOT real measurements of the place.
        Flagged via 'data_provenance': 'synthetic_demo' by analyze_place.
        """
        return {
            "road_density": 2.5,  # km/km2 (demo value)
            "transit_stops": 15,
            "healthcare_facilities": 3,
            "educational_institutions": 5,
            "commercial_centers": 8,
        }

    def get_analysis_summary(self, place_name: str) -> dict[str, Any]:
        """Get summary of analysis results for a place."""
        if place_name not in self.analysis_results:
            raise ValueError(f"No analysis results found for {place_name}")

        results = self.analysis_results[place_name]

        summary = {
            "place_name": results["place_name"],
            "analysis_date": results["timestamp"],
            "spatial_coverage": len(results["h3_cells"]),
            "area_km2": results["spatial_metrics"]["area_km2"],
        }
        if results.get("synthetic", False):
            # Synthetic demo data must not be presented as a real score.
            summary["synthetic"] = True
            summary["environmental_score"] = None
            summary["accessibility_score"] = None
            summary["note"] = (
                "environmental/accessibility data are synthetic demo values; "
                "scores omitted"
            )
        elif "environmental_factors" in results and "accessibility_metrics" in results:
            summary["environmental_score"] = self._calculate_environmental_score(
                results["environmental_factors"]
            )
            summary["accessibility_score"] = self._calculate_accessibility_score(
                results["accessibility_metrics"]
            )
        else:
            # synthetic=False: sections were skipped, so no scores exist.
            summary["environmental_score"] = None
            summary["accessibility_score"] = None
            summary["note"] = "environmental/accessibility metrics were not analyzed"
        return summary

    def _calculate_environmental_score(self, factors: dict[str, Any]) -> float:
        """Calculate environmental quality score."""
        # Simple scoring algorithm
        score = 0.0
        score += factors.get("vegetation_cover", 0) * 0.3
        score += (1 - factors.get("elevation_range", {}).get("mean", 0) / 1000) * 0.2
        score += min(factors.get("water_bodies", 0) / 5, 1) * 0.3
        score += min(factors.get("protected_areas", 0) / 3, 1) * 0.2
        return cast(float, min(score, 1.0))

    def _calculate_accessibility_score(self, metrics: dict[str, float]) -> float:
        """Calculate accessibility score."""
        # Simple scoring algorithm
        score = 0.0
        score += min(metrics.get("road_density", 0) / 5, 1) * 0.2
        score += min(metrics.get("transit_stops", 0) / 20, 1) * 0.2
        score += min(metrics.get("healthcare_facilities", 0) / 5, 1) * 0.2
        score += min(metrics.get("educational_institutions", 0) / 10, 1) * 0.2
        score += min(metrics.get("commercial_centers", 0) / 15, 1) * 0.2
        return min(score, 1.0)

    def export_results(self, place_name: str, format: str = "json") -> str:
        """Export analysis results to file."""
        if place_name not in self.analysis_results:
            raise ValueError(f"No analysis results found for {place_name}")

        results = self.analysis_results[place_name]
        output_file = self.data_dir / f"{place_name}_analysis.{format}"

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

        logger.info(f"Exported results to {output_file}")
        return str(output_file)
