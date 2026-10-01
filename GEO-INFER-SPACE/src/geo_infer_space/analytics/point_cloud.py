"""
Point cloud processing module for advanced spatial analysis.

This module provides comprehensive point cloud processing operations including
filtering, feature extraction, classification, and surface generation
using numpy, scipy, and specialized libraries when available.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Any, cast
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

try:
    import laspy  # type: ignore[import-untyped]

    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.warning("laspy not available. LAS/LAZ file support will be limited.")


class PointCloud:
    """
    Point cloud data structure for spatial analysis.

    Attributes:
        points: Array of (x, y, z) coordinates
        colors: Optional RGB color values
        intensities: Optional intensity values
        classifications: Optional classification labels
        metadata: Additional metadata dictionary
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        classifications: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize point cloud.

        Args:
            points: Array of shape (n, 3) with x, y, z coordinates
            colors: Optional array of shape (n, 3) with RGB values
            intensities: Optional array of shape (n,) with intensity values
            classifications: Optional array of shape (n,) with class labels
            metadata: Optional metadata dictionary
        """
        if points.shape[1] != 3:
            raise ValueError("Points array must have shape (n, 3)")

        self.points = points
        self.colors = colors
        self.intensities = intensities
        self.classifications = classifications
        self.metadata = metadata or {}

    @property
    def num_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.points)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Bounding box as (minx, miny, minz, maxx, maxy, maxz)."""
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        return tuple(mins) + tuple(maxs)

    def to_geoDataFrame(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame with Point geometries."""
        geometries = [Point(x, y) for x, y in self.points[:, :2]]

        data = {"geometry": geometries, "z": self.points[:, 2]}

        if self.intensities is not None:
            data["intensity"] = self.intensities

        if self.classifications is not None:
            data["classification"] = self.classifications

        if self.colors is not None:
            data["red"] = self.colors[:, 0]
            data["green"] = self.colors[:, 1]
            data["blue"] = self.colors[:, 2]

        return gpd.GeoDataFrame(data, crs=crs)


def load_point_cloud(file_path: str) -> PointCloud:
    """
    Load point cloud from various file formats.

    Args:
        file_path: Path to point cloud file

    Returns:
        PointCloud object
    """
    file_ext = file_path.lower().split(".")[-1]

    if file_ext in ["las", "laz"] and LASPY_AVAILABLE:
        return _load_las_file(file_path)
    elif file_ext in ["txt", "csv", "xyz"]:
        return _load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def point_cloud_filtering(
    point_cloud: PointCloud, filter_type: str, **kwargs: Any
) -> PointCloud:
    """
    Apply filtering operations to point cloud data.

    Args:
        point_cloud: Input PointCloud object
        filter_type: Type of filter ('statistical', 'radius', 'voxel', 'ground')
        **kwargs: Filter-specific parameters

    Returns:
        Filtered PointCloud object
    """
    if filter_type == "statistical":
        return _statistical_outlier_filter(point_cloud, **kwargs)
    elif filter_type == "radius":
        return _radius_outlier_filter(point_cloud, **kwargs)
    elif filter_type == "voxel":
        return _voxel_grid_filter(point_cloud, **kwargs)
    elif filter_type == "ground":
        return _ground_filter(point_cloud, **kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def feature_extraction(
    point_cloud: PointCloud, neighborhood_size: int = 50, search_radius: float = 1.0
) -> pd.DataFrame:
    """
    Extract geometric features from point cloud neighborhoods.

    Args:
        point_cloud: Input PointCloud object
        neighborhood_size: Number of neighbors for feature calculation
        search_radius: Search radius for neighborhood definition

    Returns:
        DataFrame with extracted features
    """
    points = point_cloud.points
    features = []

    # Build nearest neighbors index
    nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="auto")
    nbrs.fit(points)

    for i, point in enumerate(points):
        try:
            # Find neighbors
            distances, indices = nbrs.kneighbors([point])
            neighbor_points = points[indices[0]]

            # Calculate geometric features
            feature_dict = _calculate_point_features(neighbor_points, distances[0])
            feature_dict["point_id"] = i
            features.append(feature_dict)

        except Exception as e:
            logger.warning(f"Feature extraction failed for point {i}: {e}")
            features.append({"point_id": i})

    return pd.DataFrame(features)


def classification(
    point_cloud: PointCloud,
    features_df: pd.DataFrame,
    method: str = "ground_vegetation",
    **kwargs: Any,
) -> PointCloud:
    """
    Classify point cloud points into different categories.

    Args:
        point_cloud: Input PointCloud object
        features_df: DataFrame with extracted features
        method: Classification method ('ground_vegetation', 'building_detection', 'clustering')
        **kwargs: Method-specific parameters

    Returns:
        PointCloud with classification labels
    """
    if method == "ground_vegetation":
        classifications = _ground_vegetation_classification(
            point_cloud, features_df, **kwargs
        )
    elif method == "building_detection":
        classifications = _building_detection(point_cloud, features_df, **kwargs)
    elif method == "clustering":
        classifications = _clustering_classification(point_cloud, **kwargs)
    else:
        raise ValueError(f"Unknown classification method: {method}")

    # Create new point cloud with classifications
    return PointCloud(
        points=point_cloud.points,
        colors=point_cloud.colors,
        intensities=point_cloud.intensities,
        classifications=classifications,
        metadata=point_cloud.metadata,
    )


def surface_generation(
    point_cloud: PointCloud,
    method: str = "triangulation",
    grid_resolution: float = 1.0,
    **kwargs: Any,
) -> gpd.GeoDataFrame | np.ndarray:
    """
    Generate surfaces from point cloud data.

    Args:
        point_cloud: Input PointCloud object
        method: Surface generation method ('triangulation', 'grid', 'contours')
        grid_resolution: Resolution for grid-based methods
        **kwargs: Method-specific parameters

    Returns:
        Surface representation (GeoDataFrame or array)
    """
    if method == "triangulation":
        return _delaunay_triangulation(point_cloud, **kwargs)
    elif method == "grid":
        return _grid_interpolation(point_cloud, grid_resolution, **kwargs)
    elif method == "contours":
        return _contour_generation(point_cloud, **kwargs)
    else:
        raise ValueError(f"Unknown surface generation method: {method}")


# Helper functions for file loading


def _load_las_file(file_path: str) -> PointCloud:
    """Load LAS/LAZ file using laspy."""
    if not LASPY_AVAILABLE:
        raise ImportError("laspy required for LAS/LAZ file support")

    with laspy.open(file_path) as las_file:
        las_data = las_file.read()

        # Extract coordinates
        points = np.column_stack([las_data.x, las_data.y, las_data.z])

        # Extract optional attributes
        intensities = las_data.intensity if hasattr(las_data, "intensity") else None
        classifications = (
            las_data.classification if hasattr(las_data, "classification") else None
        )

        # Extract colors if available
        colors = None
        if (
            hasattr(las_data, "red")
            and hasattr(las_data, "green")
            and hasattr(las_data, "blue")
        ):
            colors = np.column_stack([las_data.red, las_data.green, las_data.blue])

        # Extract metadata
        metadata = {
            "file_path": file_path,
            "point_format": las_data.point_format.id,
            "scale": las_data.header.scale,
            "offset": las_data.header.offset,
        }

        return PointCloud(points, colors, intensities, classifications, metadata)


def _load_text_file(file_path: str) -> PointCloud:
    """Load point cloud from text file (CSV, XYZ, etc.)."""
    try:
        # Try to load as CSV with headers
        data = pd.read_csv(file_path)

        # Identify coordinate columns
        coord_cols = []
        for col_set in [["x", "y", "z"], ["X", "Y", "Z"], ["lon", "lat", "elevation"]]:
            if all(col in data.columns for col in col_set):
                coord_cols = col_set
                break

        if not coord_cols:
            # Assume first three columns are coordinates
            coord_cols = data.columns[:3].tolist()

        points = data[coord_cols].values

        # Extract optional attributes
        intensities = data["intensity"].values if "intensity" in data.columns else None
        classifications = (
            data["classification"].values if "classification" in data.columns else None
        )

        # Extract colors
        colors = None
        color_cols = ["red", "green", "blue"]
        if all(col in data.columns for col in color_cols):
            colors = data[color_cols].values

        metadata = {"file_path": file_path, "format": "text"}

        return PointCloud(points, colors, intensities, classifications, metadata)

    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        raise


# Helper functions for filtering


def _statistical_outlier_filter(
    point_cloud: PointCloud, k_neighbors: int = 50, std_ratio: float = 2.0
) -> PointCloud:
    """Remove statistical outliers based on neighbor distances."""
    points = point_cloud.points

    # Build KNN index
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto")
    nbrs.fit(points)

    # Calculate mean distances to neighbors
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

    # Calculate statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # Filter outliers
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances <= threshold

    return _filter_point_cloud(point_cloud, inlier_mask)


def _radius_outlier_filter(
    point_cloud: PointCloud, radius: float = 1.0, min_neighbors: int = 5
) -> PointCloud:
    """Remove points with too few neighbors within radius."""
    points = point_cloud.points

    # Build radius neighbors index
    nbrs = NearestNeighbors(radius=radius, algorithm="auto")
    nbrs.fit(points)

    # Count neighbors within radius
    neighbor_counts = []
    for point in points:
        indices = nbrs.radius_neighbors([point], return_distance=False)[0]
        neighbor_counts.append(len(indices) - 1)  # Exclude self

    # Filter points with sufficient neighbors
    inlier_mask = np.array(neighbor_counts) >= min_neighbors

    return _filter_point_cloud(point_cloud, inlier_mask)


def _voxel_grid_filter(point_cloud: PointCloud, voxel_size: float = 0.1) -> PointCloud:
    """Downsample point cloud using voxel grid."""
    points = point_cloud.points

    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Find unique voxels and their first occurrence
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    return _filter_point_cloud(point_cloud, unique_indices)


def _ground_filter(
    point_cloud: PointCloud, grid_size: float = 1.0, height_threshold: float = 0.5
) -> PointCloud:
    """Simple ground filtering based on local height minima."""
    points = point_cloud.points

    # Create 2D grid
    xy_points = points[:, :2]
    grid_indices = np.floor(xy_points / grid_size).astype(int)

    # Find minimum height in each grid cell
    ground_mask = np.zeros(len(points), dtype=bool)

    unique_cells = np.unique(grid_indices, axis=0)

    for cell in unique_cells:
        # Find points in this cell
        cell_mask = np.all(grid_indices == cell, axis=1)
        cell_points = points[cell_mask]

        if len(cell_points) > 0:
            # Find minimum height
            min_height = np.min(cell_points[:, 2])

            # Mark points near ground as ground
            height_diff = cell_points[:, 2] - min_height
            ground_points = height_diff <= height_threshold

            # Update global mask
            cell_indices = np.where(cell_mask)[0]
            ground_mask[cell_indices[ground_points]] = True

    return _filter_point_cloud(point_cloud, ground_mask)


def _filter_point_cloud(point_cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    """Apply boolean mask to filter point cloud."""
    filtered_points = point_cloud.points[mask]

    filtered_colors = (
        point_cloud.colors[mask] if point_cloud.colors is not None else None
    )
    filtered_intensities = (
        point_cloud.intensities[mask] if point_cloud.intensities is not None else None
    )
    filtered_classifications = (
        point_cloud.classifications[mask]
        if point_cloud.classifications is not None
        else None
    )

    return PointCloud(
        points=filtered_points,
        colors=filtered_colors,
        intensities=filtered_intensities,
        classifications=filtered_classifications,
        metadata=point_cloud.metadata,
    )


# Helper functions for feature extraction


def _calculate_point_features(
    neighbor_points: np.ndarray, distances: np.ndarray
) -> dict[str, float]:
    """Calculate geometric features for a point neighborhood."""
    if len(neighbor_points) < 3:
        return {}

    # Center the points
    centroid = np.mean(neighbor_points, axis=0)
    centered_points = neighbor_points - centroid

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_points.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Normalize eigenvalues
    eigenvalue_sum = np.sum(eigenvalues)
    if eigenvalue_sum > 0:
        normalized_eigenvalues = eigenvalues / eigenvalue_sum
    else:
        normalized_eigenvalues = np.zeros(3)

    # Calculate features
    features = {
        "linearity": (normalized_eigenvalues[0] - normalized_eigenvalues[1])
        / normalized_eigenvalues[0]
        if normalized_eigenvalues[0] > 0
        else 0,
        "planarity": (normalized_eigenvalues[1] - normalized_eigenvalues[2])
        / normalized_eigenvalues[0]
        if normalized_eigenvalues[0] > 0
        else 0,
        "sphericity": normalized_eigenvalues[2] / normalized_eigenvalues[0]
        if normalized_eigenvalues[0] > 0
        else 0,
        "omnivariance": np.cbrt(np.prod(normalized_eigenvalues))
        if np.all(normalized_eigenvalues > 0)
        else 0,
        "anisotropy": (normalized_eigenvalues[0] - normalized_eigenvalues[2])
        / normalized_eigenvalues[0]
        if normalized_eigenvalues[0] > 0
        else 0,
        "eigenentropy": -np.sum(
            normalized_eigenvalues * np.log(normalized_eigenvalues + 1e-10)
        ),
        "height_std": np.std(neighbor_points[:, 2]),
        "height_range": np.ptp(neighbor_points[:, 2]),
        "density": len(neighbor_points) / (4 / 3 * np.pi * np.max(distances) ** 3)
        if len(distances) > 0 and np.max(distances) > 0
        else 0,
    }

    return features


# Helper functions for classification


def _ground_vegetation_classification(
    point_cloud: PointCloud,
    features_df: pd.DataFrame,
    height_threshold: float = 2.0,
    planarity_threshold: float = 0.7,
) -> np.ndarray:
    """Simple ground/vegetation classification."""
    points = point_cloud.points
    classifications = np.zeros(len(points), dtype=int)

    # Class labels: 0=unclassified, 1=ground, 2=vegetation

    for i, (_, features) in enumerate(features_df.iterrows()):
        height = points[i, 2]
        planarity = features.get("planarity", 0)

        if height < height_threshold and planarity > planarity_threshold:
            classifications[i] = 1  # Ground
        elif height >= height_threshold:
            classifications[i] = 2  # Vegetation

    return classifications


def _building_detection(
    point_cloud: PointCloud,
    features_df: pd.DataFrame,
    height_threshold: float = 3.0,
    planarity_threshold: float = 0.8,
) -> np.ndarray:
    """Simple building detection based on height and planarity."""
    points = point_cloud.points
    classifications = np.zeros(len(points), dtype=int)

    # Class labels: 0=unclassified, 1=ground, 2=vegetation, 3=building

    for i, (_, features) in enumerate(features_df.iterrows()):
        height = points[i, 2]
        planarity = features.get("planarity", 0)
        linearity = features.get("linearity", 0)

        if (
            height > height_threshold
            and planarity > planarity_threshold
            and linearity < 0.5
        ):
            classifications[i] = 3  # Building
        elif height < 2.0 and planarity > 0.7:
            classifications[i] = 1  # Ground
        elif height >= 2.0:
            classifications[i] = 2  # Vegetation

    return classifications


def _clustering_classification(
    point_cloud: PointCloud, eps: float = 0.5, min_samples: int = 10
) -> np.ndarray:
    """Classification using DBSCAN clustering."""
    points = point_cloud.points

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(points)

    # Convert cluster labels to classification (add 1 to avoid negative labels)
    classifications = cluster_labels + 1
    classifications[cluster_labels == -1] = 0  # Noise points as unclassified

    return cast(np.ndarray, classifications)


# Helper functions for surface generation


def _delaunay_triangulation(point_cloud: PointCloud, **kwargs: Any) -> gpd.GeoDataFrame:
    """Generate triangulated surface using Delaunay triangulation."""
    points_2d = point_cloud.points[:, :2]

    try:
        tri = Delaunay(points_2d)

        triangles = []
        for simplex in tri.simplices:
            # Get triangle vertices
            triangle_points = points_2d[simplex]
            triangle_geom = Polygon(triangle_points)

            # Calculate triangle properties
            heights = point_cloud.points[simplex, 2]
            mean_height = np.mean(heights)
            height_std = np.std(heights)

            triangles.append(
                {
                    "geometry": triangle_geom,
                    "mean_height": mean_height,
                    "height_std": height_std,
                    "area": triangle_geom.area,
                }
            )

        return gpd.GeoDataFrame(triangles)

    except Exception as e:
        logger.error(f"Delaunay triangulation failed: {e}")
        return gpd.GeoDataFrame()


def _grid_interpolation(
    point_cloud: PointCloud, grid_resolution: float, **kwargs: Any
) -> np.ndarray:
    """Generate regular grid surface using interpolation."""
    points = point_cloud.points

    # Define grid bounds
    minx, miny = np.min(points[:, :2], axis=0)
    maxx, maxy = np.max(points[:, :2], axis=0)

    # Create grid
    x_coords = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coords = np.arange(miny, maxy + grid_resolution, grid_resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Simple nearest neighbor interpolation
    from scipy.spatial import cKDTree

    tree = cKDTree(points[:, :2])

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    distances, indices = tree.query(grid_points)

    # Interpolate heights
    interpolated_heights = points[indices, 2]
    height_grid = interpolated_heights.reshape(xx.shape)

    return cast(np.ndarray, height_grid)


def _contour_generation(point_cloud: PointCloud, **kwargs: Any) -> gpd.GeoDataFrame:
    """Generate iso-elevation contour polygons from a point cloud.

    Each contour is the convex hull of the points falling within half a
    contour interval of the target elevation level.
    """

    points = point_cloud.points

    # Create simple height-based contours
    min_height = np.min(points[:, 2])
    max_height = np.max(points[:, 2])

    contour_interval = kwargs.get("interval", (max_height - min_height) / 10)
    contour_levels = np.arange(
        min_height, max_height + contour_interval, contour_interval
    )

    contours = []

    for level in contour_levels:
        # Find points near this height level
        height_diff = np.abs(points[:, 2] - level)
        near_level = height_diff <= contour_interval / 2

        if np.sum(near_level) > 2:
            level_points = points[near_level, :2]

            # Create convex hull as simplified contour
            try:
                hull = ConvexHull(level_points)
                hull_points = level_points[hull.vertices]
                contour_geom = Polygon(hull_points)

                contours.append(
                    {
                        "geometry": contour_geom,
                        "elevation": level,
                        "num_points": len(level_points),
                    }
                )
            except Exception:
                continue

    return gpd.GeoDataFrame(contours)
