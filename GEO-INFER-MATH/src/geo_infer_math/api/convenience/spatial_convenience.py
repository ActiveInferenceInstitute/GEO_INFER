"""
Enhanced Spatial Analysis Convenience Methods

This module provides enhanced convenience methods for spatial analysis.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def _load_spatial_analysis_api() -> Any:
    """Load the Flask-backed spatial API only when spatial helpers are used."""
    try:
        from geo_infer_math.api.spatial_analysis import SpatialAnalysisAPI
    except ImportError as exc:
        raise ImportError(
            "Spatial convenience helpers require the GEO-INFER-MATH web extra. "
            "Install with `uv sync --package geo-infer-math --extra web` or "
            "`uv run --package geo-infer-math --extra web ...`."
        ) from exc
    return SpatialAnalysisAPI


def enhanced_spatial_analysis(
    coordinates: np.ndarray,
    values: np.ndarray,
    analysis_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Enhanced spatial analysis combining multiple methods.

    Args:
        coordinates: Spatial coordinates
        values: Values at locations
        analysis_types: Types of analysis to perform

    Returns:
        Dictionary of analysis results
    """
    SpatialAnalysisAPI = _load_spatial_analysis_api()
    analysis_types = analysis_types or [
        "autocorrelation",
        "descriptive",
        "interpolation",
    ]

    api = SpatialAnalysisAPI()
    results = {}

    if "autocorrelation" in analysis_types:
        results["autocorrelation"] = api.autocorrelation_analysis(
            values, coordinates, method="moran"
        )

    if "descriptive" in analysis_types:
        results["descriptive"] = api.descriptive_statistics(values, coordinates)

    if "interpolation" in analysis_types:
        # Create query grid
        x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        x_grid = np.linspace(x_min, x_max, 20)
        y_grid = np.linspace(y_min, y_max, 20)
        xx, yy = np.meshgrid(x_grid, y_grid)
        query_points = np.column_stack([xx.flatten(), yy.flatten()])

        results["interpolation"] = api.spatial_interpolation(
            coordinates, values, query_points, method="idw"
        )

    return results


class SpatialConvenience:
    """
    Convenience class for enhanced spatial analysis.

    Provides high-level methods for comprehensive spatial analysis.
    """

    def __init__(self) -> None:
        """Initialize spatial convenience class."""
        SpatialAnalysisAPI = _load_spatial_analysis_api()
        self.api = SpatialAnalysisAPI()

    def comprehensive_analysis(
        self, coordinates: np.ndarray, values: np.ndarray, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Perform comprehensive spatial analysis.

        Args:
            coordinates: Spatial coordinates
            values: Values at locations
            **kwargs: Additional parameters

        Returns:
            Dictionary of analysis results
        """
        return enhanced_spatial_analysis(coordinates, values, **kwargs)
