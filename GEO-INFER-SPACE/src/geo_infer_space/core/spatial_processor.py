"""Spatial processing module for geospatial operations."""

import logging
import geopandas as gpd
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


class SpatialProcessor:
    """Spatial processing engine for geospatial operations."""

    def __init__(self) -> None:
        """Initialize the SpatialProcessor."""
        self.logger = logger
        self._operation_count: int = 0

    def buffer_analysis(
        self, gdf: gpd.GeoDataFrame, buffer_distance: float
    ) -> gpd.GeoDataFrame:
        """Create buffers around geometries.

        Args:
            gdf: Input GeoDataFrame
            buffer_distance: Distance in units of the CRS

        Returns:
            GeoDataFrame with buffered geometries
        """
        if gdf.empty or "geometry" not in gdf.columns:
            raise ValueError("Input GeoDataFrame is empty or missing geometry column")
        try:
            buffered = gdf.copy()
            original_crs = buffered.crs
            if original_crs is not None and original_crs.is_geographic:
                buffered = buffered.to_crs("EPSG:3857")
                buffered["geometry"] = buffered.geometry.buffer(buffer_distance)
                buffered = buffered.to_crs(original_crs)
            else:
                buffered["geometry"] = buffered.geometry.buffer(buffer_distance)
            return buffered
        except Exception as e:
            logger.error(f"Buffer analysis failed: {e}")
            raise

    def proximity_analysis(
        self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
    ) -> dict[str, Any]:
        """Calculate proximity between two sets of features.

        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame

        Returns:
            Dictionary with min, max, mean distance
        """
        if gdf1.empty or gdf2.empty:
            raise ValueError("Input GeoDataFrames cannot be empty")

        # Ensure CRS match
        if gdf1.crs != gdf2.crs and gdf1.crs is not None and gdf2.crs is not None:
            gdf2 = gdf2.to_crs(gdf1.crs)

        try:
            if gdf1.crs is not None and gdf1.crs.is_geographic:
                gdf1 = gdf1.to_crs("EPSG:3857")
                gdf2 = gdf2.to_crs("EPSG:3857")
            # Vectorized nearest distance calculation
            # Use sjoin_nearest to get distances natively (O(N log M))
            # Requires geopandas >= 0.10.0
            nearest = gpd.sjoin_nearest(gdf1, gdf2, how="left", distance_col="distance")

            # Since multiple matches can exist for identical distances, group by index
            distances = nearest["distance"].groupby(nearest.index).min()

            return {
                "min_distance": float(distances.min()),
                "max_distance": float(distances.max()),
                "mean_distance": float(distances.mean()),
            }
        except Exception as e:
            logger.error(f"Proximity analysis failed: {e}")
            raise

    def perform_multi_overlay(
        self, spatial_datasets: dict[str, gpd.GeoDataFrame]
    ) -> gpd.GeoDataFrame:
        """
        Perform multi-layer spatial overlay on multiple GeoDataFrames.

        Args:
            spatial_datasets: Dictionary of dataset names to GeoDataFrames

        Returns:
            Single GeoDataFrame with overlaid geometries and attributes
        """
        if not spatial_datasets:
            raise ValueError("No spatial datasets provided")

        # Get the first dataset as base
        base_name = list(spatial_datasets.keys())[0]
        base_gdf = spatial_datasets[base_name].copy()

        # Add domain identifier
        base_gdf["domain"] = base_name

        # Overlay with other datasets
        for name, gdf in spatial_datasets.items():
            if name == base_name:
                continue

            # Ensure same CRS
            if gdf.crs != base_gdf.crs:
                gdf = gdf.to_crs(base_gdf.crs)

            # Add domain identifier
            gdf = gdf.copy()
            gdf["domain"] = name

            # Concatenate
            base_gdf = gpd.GeoDataFrame(
                pd.concat([base_gdf, gdf], ignore_index=True), crs=base_gdf.crs
            )

        return base_gdf

    def calculate_spatial_correlation(self, gdf: gpd.GeoDataFrame) -> dict[str, float]:
        """
        Calculate spatial correlation metrics for a GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame

        Returns:
            Dictionary with correlation metrics
        """
        import numpy as np

        try:
            if len(gdf) < 2:
                return {"spatial_correlation": 0.0}

            # Use numpy for rapid vectorized pairwise distance correlation proxy
            # Extract coordinates as numpy array using centroids for all geometry types
            centroids = gdf.geometry.centroid
            coords = np.column_stack((centroids.x, centroids.y))

            # Remove invalid coordinates (e.g. empty geometries)
            coords = coords[~np.isnan(coords).any(axis=1)]

            if len(coords) < 2:
                return {"spatial_correlation": 0.0}

            # Vectorized pairwise Euclidean distance squared via broadcasting
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            sq_dists = np.sum(diffs**2, axis=-1)

            # Extract upper triangle (excluding diagonal)
            i_upper = np.triu_indices(len(coords), k=1)
            distances = np.sqrt(sq_dists[i_upper])

            if len(distances) > 0:
                mean_dist = float(np.mean(distances))
                return {"spatial_correlation": 1.0 / (1.0 + mean_dist)}
            else:
                return {"spatial_correlation": 0.0}
        except Exception as e:
            logger.error(f"Spatial correlation calculation failed: {e}")
            return {"spatial_correlation": 0.0}
