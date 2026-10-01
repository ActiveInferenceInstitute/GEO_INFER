"""
Clustering Models Module

This module provides various clustering algorithms specialized for geospatial data,
including spatially constrained clustering, density-based clustering, and
hierarchical clustering methods.
"""

import numpy as np
from typing import Any, cast
from dataclasses import dataclass
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import logging

from ..utils.rng import resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResults:
    """Container for clustering analysis results."""

    labels: np.ndarray
    n_clusters: int
    centroids: np.ndarray | None = None
    silhouette_score: float | None = None
    cluster_sizes: np.ndarray | None = None
    within_cluster_distances: np.ndarray | None = None


class SpatialKMeans:
    """Spatially constrained K-means clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        """
        Initialize spatial K-means.

        Args:
            n_clusters: Number of clusters
            init: Initialization method
            n_init: Number of random initializations
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> "SpatialKMeans":
        """
        Fit K-means clustering.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates (optional spatial constraint)

        Returns:
            Self for method chaining
        """
        rng = resolve_rng(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids
        if self.init == "k-means++":
            self.cluster_centers_ = self._kmeans_plus_plus_init(X, rng)
        elif self.init == "random":
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError(f"Unknown init method: {self.init}")

        self.labels_ = np.zeros(n_samples, dtype=int)
        self.inertia_ = 0

        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            old_centers = self.cluster_centers_.copy()

            for i in range(n_samples):
                distances = np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1)
                self.labels_[i] = np.argmin(distances)

            # Update centroids
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    self.cluster_centers_[k] = np.mean(X[mask], axis=0)
                else:
                    # Reinitialize empty cluster
                    self.cluster_centers_[k] = X[rng.choice(n_samples)]

            # Check convergence
            center_shift = np.sum((self.cluster_centers_ - old_centers) ** 2)
            if center_shift < self.tol:
                self.n_iter_ = iteration + 1
                break

        # Calculate inertia
        self.inertia_ = 0
        for i in range(n_samples):
            self.inertia_ += np.sum(
                (X[i] - self.cluster_centers_[self.labels_[i]]) ** 2
            )

        self.is_fitted = True
        return self

    def _kmeans_plus_plus_init(
        self, X: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """K-means++ initialization."""
        if rng is None:
            rng = resolve_rng(self.random_state)
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))

        # First center: random selection
        centers[0] = X[rng.integers(n_samples)]

        for k in range(1, self.n_clusters):
            # Calculate distances to nearest existing center
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float("inf")
                for j in range(k):
                    dist = np.sum((X[i] - centers[j]) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist

            # Select next center with probability proportional to distance squared
            probabilities = distances**2 / np.sum(distances**2)
            cumulative_prob = np.cumsum(probabilities)
            r = rng.random()

            selected_idx = np.searchsorted(cumulative_prob, r)
            centers[k] = X[selected_idx]

        return centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.cluster_centers_ is not None

        labels = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            distances = np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1)
            labels[i] = np.argmin(distances)

        return labels

    def fit_predict(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Cluster labels
        """
        return cast(np.ndarray, self.fit(X, coordinates).labels_)


class SpatiallyConstrainedKMeans:
    """Spatially constrained K-means clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        spatial_weight: float = 0.5,
        max_iter: int = 100,
        random_state: int | None = None,
    ):
        """
        Initialize spatially constrained K-means.

        Args:
            n_clusters: Number of clusters
            spatial_weight: Weight for spatial constraint (0-1)
            max_iter: Maximum iterations
            random_state: Random state
        """
        self.n_clusters = n_clusters
        self.spatial_weight = spatial_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, coordinates: np.ndarray
    ) -> "SpatiallyConstrainedKMeans":
        """
        Fit spatially constrained K-means.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Self for method chaining
        """
        rng = resolve_rng(self.random_state)
        n_samples = len(X)

        # Initialize centroids randomly
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()

        self.labels_ = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iter):
            old_labels = self.labels_.copy()

            # Assign points to clusters
            for i in range(n_samples):
                # Calculate feature distance
                feature_distances = np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1)

                # Calculate spatial distances to cluster centroids
                spatial_centers = np.zeros((self.n_clusters, 2))
                for k in range(self.n_clusters):
                    cluster_mask = self.labels_ == k
                    if np.any(cluster_mask):
                        spatial_centers[k] = np.mean(coordinates[cluster_mask], axis=0)
                    else:
                        spatial_centers[k] = coordinates[indices[k]]

                spatial_distances = np.sum(
                    (spatial_centers - coordinates[i]) ** 2, axis=1
                )

                # Combine distances
                combined_distances = (
                    1 - self.spatial_weight
                ) * feature_distances + self.spatial_weight * spatial_distances

                self.labels_[i] = np.argmin(combined_distances)

            # Update centroids
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    self.cluster_centers_[k] = np.mean(X[mask], axis=0)

            # Check convergence
            if np.array_equal(old_labels, self.labels_):
                break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        assert self.labels_ is not None
        assert self.cluster_centers_ is not None

        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)

        # Calculate current spatial centroids
        spatial_centers = np.zeros((self.n_clusters, 2))
        for k in range(self.n_clusters):
            cluster_mask = self.labels_ == k
            if np.any(cluster_mask):
                spatial_centers[k] = np.mean(coordinates[cluster_mask], axis=0)

        for i in range(n_samples):
            # Calculate distances
            feature_distances = np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1)
            spatial_distances = np.sum((spatial_centers - coordinates[i]) ** 2, axis=1)

            combined_distances = (
                1 - self.spatial_weight
            ) * feature_distances + self.spatial_weight * spatial_distances

            labels[i] = np.argmin(combined_distances)

        return labels


class SpatialDBSCAN:
    """Density-based spatial clustering of applications with noise (DBSCAN)."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
    ):
        """
        Initialize spatial DBSCAN.

        Args:
            eps: Maximum distance between two samples for neighborhood
            min_samples: Minimum number of samples in neighborhood for core point
            metric: Distance metric
            algorithm: Algorithm to use
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm

        self.core_sample_indices_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> "SpatialDBSCAN":
        """
        Fit DBSCAN clustering.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates (if None, uses X as coordinates)

        Returns:
            Self for method chaining
        """
        if coordinates is None:
            coordinates = X

        # Use sklearn's DBSCAN for efficiency
        if self.metric == "euclidean":
            dbscan = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric="euclidean",
                algorithm=self.algorithm,
            )
            dbscan.fit(coordinates)
        else:
            # For other metrics, use precomputed distance matrix
            from scipy.spatial.distance import pdist, squareform

            distance_matrix = squareform(pdist(coordinates, metric=self.metric))
            dbscan = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric="precomputed",
                algorithm=self.algorithm,
            )
            dbscan.fit(distance_matrix)

        self.core_sample_indices_ = dbscan.core_sample_indices_
        self.components_ = dbscan.components_
        self.labels_ = dbscan.labels_
        self.n_features_in_ = coordinates.shape[1]

        self.is_fitted = True
        return self

    def fit_predict(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Cluster labels (-1 for noise)
        """
        return cast(np.ndarray, self.fit(X, coordinates).labels_)


class SKATERClustering:
    """Spatial 'K'luster Analysis by Tree Edge Removal (SKATER) clustering."""

    def __init__(self, n_clusters: int = 8, min_cluster_size: int = 2):
        """
        Initialize SKATER clustering.

        Args:
            n_clusters: Number of clusters
            min_cluster_size: Minimum cluster size
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

        self.labels_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, coordinates: np.ndarray) -> "SKATERClustering":
        """
        Fit SKATER clustering.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Self for method chaining
        """
        n_samples = len(X)

        # Create minimum spanning tree based on spatial distances
        from scipy.spatial.distance import pdist, squareform

        # Spatial distance matrix
        spatial_distances = squareform(pdist(coordinates, metric="euclidean"))

        # Feature distance matrix (normalized)
        feature_distances = squareform(pdist(X, metric="euclidean"))
        if np.max(feature_distances) > 0:
            feature_distances /= np.max(feature_distances)

        # Combined distance matrix
        combined_distances = spatial_distances + feature_distances

        # Create MST using Kruskal's algorithm
        edges = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                edges.append((i, j, combined_distances[i, j]))

        # Sort edges by weight
        edges.sort(key=lambda x: x[2])

        # Union-Find structure
        parent = list(range(n_samples))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> bool:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Build MST
        mst_edges = []
        for edge in edges:
            if union(edge[0], edge[1]):
                mst_edges.append(edge)
                if len(mst_edges) == n_samples - 1:
                    break

        # Remove edges to create clusters
        n_edges_to_remove = n_samples - self.n_clusters

        # Sort MST edges by weight (descending) for removal
        mst_edges.sort(key=lambda x: x[2], reverse=True)

        # Remove edges while ensuring minimum cluster size
        edges_to_remove = []
        cluster_sizes = [1] * n_samples
        cluster_ids = list(range(n_samples))

        for edge in mst_edges[:n_edges_to_remove]:
            i, j, weight = edge

            # Check if removing this edge would violate minimum cluster size
            cluster_i = cluster_ids[i]
            cluster_j = cluster_ids[j]

            if (
                cluster_sizes[cluster_i] > self.min_cluster_size
                and cluster_sizes[cluster_j] > self.min_cluster_size
            ):
                edges_to_remove.append(edge)

                # Merge clusters
                for k in range(n_samples):
                    if cluster_ids[k] == cluster_j:
                        cluster_ids[k] = cluster_i
                        cluster_sizes[cluster_i] += 1

        # Create final cluster labels
        unique_clusters = {}
        current_label = 0

        self.labels_ = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            cluster_id = cluster_ids[i]
            if cluster_id not in unique_clusters:
                unique_clusters[cluster_id] = current_label
                current_label += 1
            self.labels_[i] = unique_clusters[cluster_id]

        self.is_fitted = True
        return self


class HierarchicalClustering:
    """Hierarchical clustering with spatial constraints."""

    def __init__(
        self,
        n_clusters: int = 8,
        method: str = "ward",
        metric: str = "euclidean",
        spatial_weight: float = 0.0,
    ):
        """
        Initialize hierarchical clustering.

        Args:
            n_clusters: Number of clusters
            method: Linkage method ('single', 'complete', 'average', 'ward')
            metric: Distance metric
            spatial_weight: Weight for spatial constraint (0-1)
        """
        self.n_clusters = n_clusters
        self.method = method
        self.metric = metric
        self.spatial_weight = spatial_weight

        self.labels_: np.ndarray | None = None
        self.linkage_matrix_: np.ndarray | None = None
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> "HierarchicalClustering":
        """
        Fit hierarchical clustering.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Self for method chaining
        """
        # Calculate distance matrices
        feature_distances = pdist(X, metric=self.metric)

        if coordinates is not None and self.spatial_weight > 0:
            spatial_distances = pdist(coordinates, metric="euclidean")

            # Normalize distances
            if np.max(feature_distances) > 0:
                feature_distances = feature_distances / np.max(feature_distances)
            if np.max(spatial_distances) > 0:
                spatial_distances = spatial_distances / np.max(spatial_distances)

            # Combine distances
            combined_distances = (
                1 - self.spatial_weight
            ) * feature_distances + self.spatial_weight * spatial_distances
        else:
            combined_distances = feature_distances

        # Perform hierarchical clustering
        self.linkage_matrix_ = linkage(combined_distances, method=self.method)

        # Cut tree to get clusters
        self.labels_ = fcluster(
            self.linkage_matrix_, self.n_clusters, criterion="maxclust"
        )

        # Convert to zero-based indexing
        self.labels_ = self.labels_ - 1

        self.is_fitted = True
        return self

    def fit_predict(
        self, X: np.ndarray, coordinates: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            X: Feature matrix
            coordinates: Spatial coordinates

        Returns:
            Cluster labels
        """
        return cast(np.ndarray, self.fit(X, coordinates).labels_)


def spatial_clustering_analysis(
    X: np.ndarray, coordinates: np.ndarray, method: str = "kmeans", **kwargs: Any
) -> ClusteringResults:
    """
    Perform spatial clustering analysis.

    Args:
        X: Feature matrix
        coordinates: Spatial coordinates
        method: Clustering method ('kmeans', 'constrained_kmeans', 'dbscan', 'skater', 'hierarchical')
        **kwargs: Method-specific parameters

    Returns:
        Clustering results
    """
    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 8)
        model_kmeans = SpatialKMeans(n_clusters=n_clusters)
        model_kmeans.fit(X, coordinates)
        labels = cast(np.ndarray, model_kmeans.labels_)

        # Calculate cluster sizes
        cluster_sizes = np.bincount(labels)

        # Calculate within-cluster distances
        within_distances = np.zeros(n_clusters)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                cluster_points = X[mask]
                centroid = cast(np.ndarray, model_kmeans.cluster_centers_)[k]
                within_distances[k] = np.mean(
                    np.sum((cluster_points - centroid) ** 2, axis=1)
                )

        results = ClusteringResults(
            labels=labels,
            n_clusters=n_clusters,
            centroids=model_kmeans.cluster_centers_,
            cluster_sizes=cluster_sizes,
            within_cluster_distances=within_distances,
        )

    elif method == "constrained_kmeans":
        n_clusters = kwargs.get("n_clusters", 8)
        spatial_weight = kwargs.get("spatial_weight", 0.5)
        model_constrained = SpatiallyConstrainedKMeans(
            n_clusters=n_clusters, spatial_weight=spatial_weight
        )
        model_constrained.fit(X, coordinates)

        results = ClusteringResults(
            labels=cast(np.ndarray, model_constrained.labels_), n_clusters=n_clusters
        )

    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        model_dbscan = SpatialDBSCAN(eps=eps, min_samples=min_samples)
        model_dbscan.fit(X, coordinates)
        labels = cast(np.ndarray, model_dbscan.labels_)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        results = ClusteringResults(labels=labels, n_clusters=n_clusters)

    elif method == "skater":
        n_clusters = kwargs.get("n_clusters", 8)
        model_skater = SKATERClustering(n_clusters=n_clusters)
        model_skater.fit(X, coordinates)

        results = ClusteringResults(
            labels=cast(np.ndarray, model_skater.labels_), n_clusters=n_clusters
        )

    elif method == "hierarchical":
        n_clusters = kwargs.get("n_clusters", 8)
        linkage_method = kwargs.get("linkage_method", "ward")
        spatial_weight = kwargs.get("spatial_weight", 0.0)
        model_hier = HierarchicalClustering(
            n_clusters=n_clusters, method=linkage_method, spatial_weight=spatial_weight
        )
        model_hier.fit(X, coordinates)

        results = ClusteringResults(
            labels=cast(np.ndarray, model_hier.labels_), n_clusters=n_clusters
        )

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return results
