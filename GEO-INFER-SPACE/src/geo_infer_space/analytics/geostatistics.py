"""
Geostatistics module for advanced spatial analysis.

This module provides comprehensive geostatistical operations including
spatial interpolation, clustering analysis, hotspot detection, and
spatial autocorrelation using scikit-learn, scipy, and specialized libraries.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Any, cast
from shapely.geometry import Point
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.cluster import DBSCAN, KMeans

logger = logging.getLogger(__name__)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False
    logger.warning("Scikit-learn Gaussian Process not available for kriging")


def spatial_interpolation(
    points_gdf: gpd.GeoDataFrame,
    value_column: str,
    grid_bounds: tuple[float, float, float, float],
    grid_resolution: float,
    method: str = "idw",
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Perform spatial interpolation on point data.

    Args:
        points_gdf: GeoDataFrame with point observations
        value_column: Column name containing values to interpolate
        grid_bounds: (minx, miny, maxx, maxy) bounds for interpolation grid
        grid_resolution: Grid cell size for interpolation
        method: Interpolation method ('idw', 'kriging', 'rbf', 'nearest')
        **kwargs: Additional parameters for specific methods

    Returns:
        GeoDataFrame with interpolated grid points
    """
    if value_column not in points_gdf.columns:
        raise ValueError(f"Column '{value_column}' not found in GeoDataFrame")

    # Extract coordinates and values
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    values = points_gdf[value_column].values

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    coords = coords[valid_mask]
    values = values[valid_mask]

    if len(coords) == 0:
        raise ValueError("No valid data points for interpolation")

    # Create interpolation grid
    minx, miny, maxx, maxy = grid_bounds
    x_grid = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_grid = np.arange(miny, maxy + grid_resolution, grid_resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Perform interpolation based on method
    if method == "idw":
        power = kwargs.get("power", 2)
        interpolated_values = _idw_interpolation(coords, values, grid_points, power)

    elif method == "kriging" and SKLEARN_GP_AVAILABLE:
        interpolated_values = _kriging_interpolation(
            coords, values, grid_points, **kwargs
        )

    elif method == "rbf":
        from scipy.interpolate import Rbf

        rbf_function = kwargs.get("function", "multiquadric")
        rbf = Rbf(coords[:, 0], coords[:, 1], values, function=rbf_function)
        interpolated_values = rbf(grid_points[:, 0], grid_points[:, 1])

    elif method == "nearest":
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        _, indices = tree.query(grid_points)
        interpolated_values = values[indices]

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Create result GeoDataFrame
    grid_geoms = [Point(x, y) for x, y in grid_points]
    result_gdf = gpd.GeoDataFrame(
        {"geometry": grid_geoms, f"{value_column}_interpolated": interpolated_values},
        crs=points_gdf.crs,
    )

    logger.info(
        f"Spatial interpolation ({method}) completed: {len(result_gdf)} grid points"
    )
    return result_gdf


def clustering_analysis(
    points_gdf: gpd.GeoDataFrame, method: str = "dbscan", **kwargs: Any
) -> gpd.GeoDataFrame:
    """
    Perform spatial clustering analysis on point data.

    Args:
        points_gdf: GeoDataFrame with point data
        method: Clustering method ('dbscan', 'kmeans', 'hierarchical')
        **kwargs: Parameters for clustering algorithms

    Returns:
        GeoDataFrame with cluster labels
    """
    # Extract coordinates
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

    result_gdf = points_gdf.copy()

    if method == "dbscan":
        eps = kwargs.get("eps", 0.01)  # Distance threshold
        min_samples = kwargs.get("min_samples", 5)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords)

    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 5)

        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(coords)

    elif method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering

        n_clusters = kwargs.get("n_clusters", 5)
        linkage = kwargs.get("linkage", "ward")

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = clustering.fit_predict(coords)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    result_gdf["cluster"] = cluster_labels

    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Noise points in DBSCAN
            continue

        cluster_points = coords[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)

        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "num_points": len(cluster_points),
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
            }
        )

    logger.info(
        f"Clustering analysis ({method}) completed: {len(cluster_stats)} clusters"
    )
    return result_gdf


def hotspot_detection(
    points_gdf: gpd.GeoDataFrame,
    value_column: str | None = None,
    method: str = "getis_ord",
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Detect spatial hotspots and coldspots.

    Args:
        points_gdf: GeoDataFrame with point data
        value_column: Column with values for analysis (if None, uses point density)
        method: Detection method ('getis_ord', 'local_moran', 'kernel_density')
        **kwargs: Additional parameters

    Returns:
        GeoDataFrame with hotspot statistics
    """
    result_gdf = points_gdf.copy()

    if method == "getis_ord":
        result_gdf = _getis_ord_gi_star(points_gdf, value_column, **kwargs)

    elif method == "local_moran":
        result_gdf = _local_morans_i(points_gdf, value_column, **kwargs)

    elif method == "kernel_density":
        result_gdf = _kernel_density_hotspots(points_gdf, **kwargs)

    else:
        raise ValueError(f"Unknown hotspot detection method: {method}")

    logger.info(f"Hotspot detection ({method}) completed")
    return result_gdf


def spatial_autocorrelation(
    points_gdf: gpd.GeoDataFrame, value_column: str, method: str = "morans_i"
) -> dict[str, float | str]:
    """
    Calculate global spatial autocorrelation statistics.

    Args:
        points_gdf: GeoDataFrame with point data
        value_column: Column name with values for analysis
        method: Method for autocorrelation ('morans_i', 'geary_c')

    Returns:
        Dictionary with autocorrelation statistics
    """
    if value_column not in points_gdf.columns:
        raise ValueError(f"Column '{value_column}' not found")

    # Extract coordinates and values
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    values = points_gdf[value_column].values

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    coords = coords[valid_mask]
    values = values[valid_mask]

    if len(values) < 3:
        return {"error": "Insufficient data for autocorrelation analysis"}

    # Create spatial weights matrix (inverse distance)
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)  # Avoid division by zero
    weights = 1.0 / distances
    weights[distances == np.inf] = 0

    # Normalize weights
    row_sums = np.sum(weights, axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]

    if method == "morans_i":
        # Calculate Moran's I
        n = len(values)
        mean_val = np.mean(values)

        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += (
                        weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
                    )
            denominator += (values[i] - mean_val) ** 2

        W = np.sum(weights)  # Sum of all weights
        morans_i = (n / W) * (numerator / denominator) if denominator != 0 else 0

        # Expected value and variance (simplified)
        expected_i = -1.0 / (n - 1)
        variance_i = (n * n - 3 * n + 3) / ((n - 1) * (n - 2) * (n - 3))
        z_score = (morans_i - expected_i) / np.sqrt(variance_i) if variance_i > 0 else 0

        return {
            "morans_i": morans_i,
            "expected_i": expected_i,
            "variance_i": variance_i,
            "z_score": z_score,
        }

    elif method == "geary_c":
        # Calculate Geary's C
        n = len(values)
        mean_val = np.mean(values)

        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * (values[i] - values[j]) ** 2
            denominator += (values[i] - mean_val) ** 2

        W = np.sum(weights)
        geary_c = (
            ((n - 1) / (2 * W)) * (numerator / denominator) if denominator != 0 else 1
        )

        return {"geary_c": geary_c, "expected_c": 1.0}

    else:
        raise ValueError(f"Unknown autocorrelation method: {method}")


def variogram_analysis(
    points_gdf: gpd.GeoDataFrame,
    value_column: str,
    max_distance: float | None = None,
    n_lags: int = 15,
) -> pd.DataFrame:
    """
    Calculate experimental variogram for spatial data.

    Args:
        points_gdf: GeoDataFrame with point data
        value_column: Column name with values for analysis
        max_distance: Maximum distance for variogram calculation
        n_lags: Number of distance lags

    Returns:
        DataFrame with variogram results
    """
    if value_column not in points_gdf.columns:
        raise ValueError(f"Column '{value_column}' not found")

    # Extract coordinates and values
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    values = points_gdf[value_column].values

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    coords = coords[valid_mask]
    values = values[valid_mask]

    if len(values) < 3:
        return pd.DataFrame()

    # Calculate pairwise distances and value differences
    distances = pdist(coords)
    value_diffs = []

    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            value_diffs.append((values[i] - values[j]) ** 2)

    value_diffs_array = np.array(value_diffs)

    # Set maximum distance if not provided
    if max_distance is None:
        max_distance = np.max(distances) * 0.5

    # Create distance lags
    lag_size = max_distance / n_lags
    lags = np.arange(0, max_distance + lag_size, lag_size)

    # Calculate variogram for each lag
    variogram_results = []

    for i in range(len(lags) - 1):
        lag_min = lags[i]
        lag_max = lags[i + 1]

        # Find pairs within this distance lag
        mask = (distances >= lag_min) & (distances < lag_max)

        if np.sum(mask) > 0:
            lag_distances = distances[mask]
            lag_value_diffs = value_diffs_array[mask]

            # Calculate semivariance (half the mean squared difference)
            semivariance = np.mean(lag_value_diffs) / 2.0
            mean_distance = np.mean(lag_distances)
            n_pairs = len(lag_distances)

            variogram_results.append(
                {
                    "distance": mean_distance,
                    "semivariance": semivariance,
                    "n_pairs": n_pairs,
                    "lag_min": lag_min,
                    "lag_max": lag_max,
                }
            )

    result_df = pd.DataFrame(variogram_results)
    logger.info(f"Variogram analysis completed: {len(result_df)} lags")
    return result_df


# Helper functions for hotspot detection


def _getis_ord_gi_star(
    points_gdf: gpd.GeoDataFrame,
    value_column: str | None,
    distance_threshold: float = 1000,
) -> gpd.GeoDataFrame:
    """Calculate Getis-Ord Gi* statistic for hotspot detection."""
    result_gdf = points_gdf.copy()

    if value_column is None:
        # Use point density
        values = np.ones(len(points_gdf))
    else:
        values = points_gdf[value_column].values

    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

    gi_star_values = []
    z_scores = []

    for coord in coords:
        # Find neighbors within distance threshold
        distances = np.sqrt(np.sum((coords - coord) ** 2, axis=1))
        neighbors = distances <= distance_threshold

        if np.sum(neighbors) > 1:
            neighbor_values = values[neighbors]

            # Calculate Gi* statistic
            sum_neighbors = np.sum(neighbor_values)
            n_neighbors = len(neighbor_values)

            # Global statistics
            global_mean = np.mean(values)
            global_std = np.std(values)
            n_total = len(values)

            # Expected value and variance
            expected = global_mean * n_neighbors
            variance = (global_std**2 * n_neighbors * (n_total - n_neighbors)) / (
                n_total - 1
            )

            if variance > 0:
                gi_star = (sum_neighbors - expected) / np.sqrt(variance)
                z_score = gi_star
            else:
                gi_star = 0
                z_score = 0
        else:
            gi_star = 0
            z_score = 0

        gi_star_values.append(gi_star)
        z_scores.append(z_score)

    result_gdf["gi_star"] = gi_star_values
    result_gdf["z_score"] = z_scores
    result_gdf["hotspot_type"] = np.where(
        np.array(z_scores) > 1.96,
        "Hot Spot",
        np.where(np.array(z_scores) < -1.96, "Cold Spot", "Not Significant"),
    )

    return result_gdf


def _local_morans_i(
    points_gdf: gpd.GeoDataFrame,
    value_column: str | None,
    distance_threshold: float = 1000,
) -> gpd.GeoDataFrame:
    """Calculate Local Moran's I for hotspot detection."""
    result_gdf = points_gdf.copy()

    if value_column is None:
        values = np.ones(len(points_gdf))
    else:
        values = points_gdf[value_column].values

    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    global_mean = np.mean(values)

    local_morans_i = []

    for i, coord in enumerate(coords):
        distances = np.sqrt(np.sum((coords - coord) ** 2, axis=1))
        neighbors = distances <= distance_threshold
        neighbors[i] = False  # Exclude self

        if np.sum(neighbors) > 0:
            neighbor_values = values[neighbors]

            # Calculate local Moran's I
            zi = values[i] - global_mean
            sum_wij_zj = np.sum(neighbor_values - global_mean)

            local_i = zi * sum_wij_zj
        else:
            local_i = 0

        local_morans_i.append(local_i)

    result_gdf["local_morans_i"] = local_morans_i
    return result_gdf


def _kernel_density_hotspots(
    points_gdf: gpd.GeoDataFrame, bandwidth: float = 1000
) -> gpd.GeoDataFrame:
    """Calculate kernel density for hotspot detection."""
    from sklearn.neighbors import KernelDensity

    result_gdf = points_gdf.copy()
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

    # Fit kernel density estimator
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords)

    # Calculate density at each point
    log_density = kde.score_samples(coords)
    density = np.exp(log_density)

    # Standardize density values
    z_scores = zscore(density)

    result_gdf["density"] = density
    result_gdf["density_zscore"] = z_scores
    result_gdf["hotspot_type"] = np.where(
        z_scores > 1.96,
        "Hot Spot",
        np.where(z_scores < -1.96, "Cold Spot", "Not Significant"),
    )

    return result_gdf


# Helper functions for interpolation


def _idw_interpolation(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    grid_coords: np.ndarray,
    power: float = 2,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation."""
    interpolated = np.zeros(len(grid_coords))

    for i, grid_point in enumerate(grid_coords):
        distances = np.sqrt(np.sum((known_coords - grid_point) ** 2, axis=1))

        # Handle exact matches
        if np.any(distances == 0):
            exact_idx = np.where(distances == 0)[0][0]
            interpolated[i] = known_values[exact_idx]
        else:
            weights = 1.0 / (distances**power)
            interpolated[i] = np.sum(weights * known_values) / np.sum(weights)

    return interpolated


def _kriging_interpolation(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    grid_coords: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Simple kriging using Gaussian Process."""
    if not SKLEARN_GP_AVAILABLE:
        raise ImportError("Scikit-learn required for kriging interpolation")

    # Set up Gaussian Process with RBF kernel
    length_scale = kwargs.get("length_scale", 1.0)
    kernel = ConstantKernel() * RBF(length_scale=length_scale)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    gp.fit(known_coords, known_values)

    # Predict at grid points
    interpolated, _ = gp.predict(grid_coords, return_std=True)

    return cast(np.ndarray, interpolated)
