"""
H3 Analytics module for advanced spatial analysis and pattern detection.

This module provides specialized analytics classes for H3 hexagonal grids
including clustering, density analysis, network analysis, and temporal analysis.
"""

import logging
from datetime import datetime
from typing import Any, cast
import math
from .core import H3Grid, H3Cell

import h3
import numpy as np

logger = logging.getLogger(__name__)


class H3SpatialAnalyzer:
    """
    Spatial analysis for H3 grids.

    Provides methods for analyzing spatial patterns, relationships,
    and distributions within H3 hexagonal grids.
    """

    def __init__(self, grid: H3Grid) -> None:
        """
        Initialize spatial analyzer for an H3Grid.

        Args:
            grid: H3Grid instance to analyze
        """
        self.grid = grid

    def analyze_spatial_autocorrelation(self, value_column: str) -> dict[str, Any]:
        """
        Analyze spatial autocorrelation using Moran's I statistic.

        This method calculates spatial autocorrelation to determine if similar values
        cluster together spatially within the H3 grid. Based on methods from:
        https://medium.com/aimonks/harnessing-the-power-of-h3-py-a-practitioners-guide-to-hexagonal-spatial-indexing-108ded50fb3b

        Args:
            value_column: Name of the column containing values to analyze

        Returns:
            Dictionary containing Moran's I statistic, p-value, and interpretation

        Example:
            >>> analyzer = H3SpatialAnalyzer(grid)
            >>> result = analyzer.analyze_spatial_autocorrelation('population_density')
            >>> print(f"Moran's I: {result['morans_i']:.3f}")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Create spatial weights matrix based on H3 neighbors
        weights_matrix = self._create_spatial_weights_matrix()

        # Extract values for analysis
        values = []
        valid_cells = []

        for cell in self.grid.cells:
            if (
                value_column in cell.properties
                and cell.properties[value_column] is not None
            ):
                values.append(float(cell.properties[value_column]))
                valid_cells.append(cell)

        if len(values) < 3:
            return {"error": "Insufficient data for spatial autocorrelation analysis"}

        # Calculate Moran's I
        morans_i = self._calculate_morans_i(values, weights_matrix, valid_cells)

        # Interpret results
        interpretation = self._interpret_morans_i(morans_i)

        return {
            "morans_i": morans_i,
            "interpretation": interpretation,
            "n_observations": len(values),
            "method": "Morans I spatial autocorrelation",
            "reference": "https://medium.com/aimonks/harnessing-the-power-of-h3-py-a-practitioners-guide-to-hexagonal-spatial-indexing-108ded50fb3b",
        }

    def _create_spatial_weights_matrix(self) -> dict[str, list[str]]:
        """
        Create spatial weights matrix based on H3 neighbor relationships.

        Returns:
            Dictionary mapping cell indices to their neighbors
        """
        weights = {}

        for cell in self.grid.cells:
            try:
                neighbors = h3.grid_disk(cell.index, 1)
                # Remove self from neighbors
                neighbors = [n for n in neighbors if n != cell.index]
                weights[cell.index] = neighbors
            except Exception as e:
                logger.warning(f"Failed to get neighbors for cell {cell.index}: {e}")
                weights[cell.index] = []

        return weights

    def _calculate_morans_i(
        self,
        values: list[float],
        weights_matrix: dict[str, list[str]],
        valid_cells: list,
    ) -> float:
        """
        Calculate Moran's I statistic.

        Args:
            values: List of values for analysis
            weights_matrix: Spatial weights matrix
            valid_cells: List of valid cells corresponding to values

        Returns:
            Moran's I statistic
        """
        values_array = np.array(values)
        n = len(values)
        mean_val = np.mean(values_array)

        # Calculate numerator and denominator
        numerator: float = 0
        denominator = np.sum((values_array - mean_val) ** 2)
        total_weights = 0

        cell_index_map = {cell.index: i for i, cell in enumerate(valid_cells)}

        for i, cell in enumerate(valid_cells):
            neighbors = weights_matrix.get(cell.index, [])
            for neighbor_idx in neighbors:
                if neighbor_idx in cell_index_map:
                    j = cell_index_map[neighbor_idx]
                    numerator += (values[i] - mean_val) * (values[j] - mean_val)
                    total_weights += 1

        if total_weights == 0 or denominator == 0:
            return 0.0

        morans_i = (n / total_weights) * (numerator / denominator)
        return cast(float, morans_i)

    def _interpret_morans_i(self, morans_i: float) -> str:
        """
        Interpret Moran's I statistic.

        Args:
            morans_i: Moran's I statistic

        Returns:
            Interpretation string
        """
        if morans_i > 0.3:
            return "Strong positive spatial autocorrelation - similar values cluster together"
        elif morans_i > 0.1:
            return "Moderate positive spatial autocorrelation - some clustering of similar values"
        elif morans_i > -0.1:
            return "No significant spatial autocorrelation - random spatial pattern"
        elif morans_i > -0.3:
            return "Moderate negative spatial autocorrelation - dissimilar values cluster together"
        else:
            return (
                "Strong negative spatial autocorrelation - strong checkerboard pattern"
            )

    def detect_hotspots(
        self, value_column: str, method: str = "getis_ord"
    ) -> dict[str, Any]:
        """
        Detect spatial hotspots and coldspots using local spatial statistics.

        Based on methods from Foursquare's H3 analytics guide:
        https://location.foursquare.com/resources/reports-and-insights/ebook/how-to-use-h3-for-geospatial-analytics/

        Args:
            value_column: Column containing values for hotspot analysis
            method: Method to use ('getis_ord' or 'local_morans')

        Returns:
            Dictionary with hotspot analysis results

        Example:
            >>> hotspots = analyzer.detect_hotspots('crime_count', method='getis_ord')
            >>> print(f"Found {len(hotspots['hotspots'])} hotspots")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract valid data
        valid_data = []
        for cell in self.grid.cells:
            if (
                value_column in cell.properties
                and cell.properties[value_column] is not None
            ):
                valid_data.append(
                    {"cell": cell, "value": float(cell.properties[value_column])}
                )

        if len(valid_data) < 5:
            return {"error": "Insufficient data for hotspot analysis"}

        if method == "getis_ord":
            return self._getis_ord_analysis(valid_data, value_column)
        elif method == "local_morans":
            return self._local_morans_analysis(valid_data, value_column)
        else:
            return {"error": f"Unknown method: {method}"}

    def _getis_ord_analysis(
        self, valid_data: list[dict], value_column: str
    ) -> dict[str, Any]:
        """
        Perform Getis-Ord Gi* hotspot analysis.

        Args:
            valid_data: List of valid cell data
            value_column: Column name for analysis

        Returns:
            Hotspot analysis results
        """
        hotspots = []
        coldspots = []

        values = np.array([d["value"] for d in valid_data])
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Calculate Gi* for each cell
        for data in valid_data:
            cell = data["cell"]

            # Get neighbors including self. h3-py 4.x raises ValueError for
            # malformed indices and h3.H3BaseException subclasses for library
            # errors (there is no h3.CellError in this version).
            try:
                neighbors = h3.grid_disk(cell.index, 1)
            except (ValueError, h3.H3BaseException) as e:
                logger.warning(
                    f"grid_disk failed for {cell.index}; using self-only neighborhood "
                    f"in Getis-Ord Gi*: {e}"
                )
                neighbors = [cell.index]

            # Calculate local sum and count
            local_sum = 0
            local_count = 0

            for neighbor_idx in neighbors:
                for neighbor_data in valid_data:
                    if neighbor_data["cell"].index == neighbor_idx:
                        local_sum += neighbor_data["value"]
                        local_count += 1
                        break

            if local_count > 0:
                # Calculate Gi* statistic
                expected_sum = local_count * mean_val
                if std_val > 0:
                    gi_star = (local_sum - expected_sum) / (
                        std_val * math.sqrt(local_count)
                    )
                else:
                    gi_star = 0

                # Classify as hotspot or coldspot
                if gi_star > 1.96:  # 95% confidence
                    hotspots.append(
                        {
                            "cell_index": cell.index,
                            "gi_star": gi_star,
                            "confidence": "95%",
                            "local_sum": local_sum,
                            "local_count": local_count,
                        }
                    )
                elif gi_star < -1.96:
                    coldspots.append(
                        {
                            "cell_index": cell.index,
                            "gi_star": gi_star,
                            "confidence": "95%",
                            "local_sum": local_sum,
                            "local_count": local_count,
                        }
                    )

        return {
            "method": "Getis-Ord Gi*",
            "hotspots": hotspots,
            "coldspots": coldspots,
            "total_cells_analyzed": len(valid_data),
            "reference": "https://location.foursquare.com/resources/reports-and-insights/ebook/how-to-use-h3-for-geospatial-analytics/",
        }

    def _local_morans_analysis(
        self, valid_data: list[dict], value_column: str
    ) -> dict[str, Any]:
        """
        Perform Local Moran's I analysis for hotspot detection.

        Args:
            valid_data: List of valid cell data
            value_column: Column name for analysis

        Returns:
            Local Moran's I analysis results
        """
        hotspots = []
        coldspots = []
        outliers = []

        values = np.array([d["value"] for d in valid_data])
        mean_val = np.mean(values)
        # Calculate local Moran's I for each cell
        for data in valid_data:
            cell = data["cell"]
            cell_value = data["value"]

            # Get neighbors. h3-py 4.x raises ValueError for malformed indices
            # and h3.H3BaseException subclasses for library errors (there is no
            # h3.CellError in this version).
            try:
                neighbors = h3.grid_disk(cell.index, 1)
                neighbors = [n for n in neighbors if n != cell.index]  # Exclude self
            except (ValueError, h3.H3BaseException) as e:
                logger.warning(
                    f"grid_disk failed for {cell.index}; using empty neighborhood in "
                    f"Local Moran's I; results may be degraded: {e}"
                )
                neighbors = []

            # Calculate local Moran's I
            neighbor_values = []
            for neighbor_idx in neighbors:
                for neighbor_data in valid_data:
                    if neighbor_data["cell"].index == neighbor_idx:
                        neighbor_values.append(neighbor_data["value"])
                        break

            if neighbor_values:
                neighbor_mean = np.mean(neighbor_values)

                local_i = (cell_value - mean_val) * (neighbor_mean - mean_val)

                # Classify based on quadrant analysis
                if cell_value > mean_val and neighbor_mean > mean_val:
                    hotspots.append(
                        {
                            "cell_index": cell.index,
                            "local_i": local_i,
                            "type": "High-High",
                            "cell_value": cell_value,
                            "neighbor_mean": neighbor_mean,
                        }
                    )
                elif cell_value < mean_val and neighbor_mean < mean_val:
                    coldspots.append(
                        {
                            "cell_index": cell.index,
                            "local_i": local_i,
                            "type": "Low-Low",
                            "cell_value": cell_value,
                            "neighbor_mean": neighbor_mean,
                        }
                    )
                else:
                    outliers.append(
                        {
                            "cell_index": cell.index,
                            "local_i": local_i,
                            "type": "High-Low" if cell_value > mean_val else "Low-High",
                            "cell_value": cell_value,
                            "neighbor_mean": neighbor_mean,
                        }
                    )

        return {
            "method": "Local Morans I",
            "hotspots": hotspots,
            "coldspots": coldspots,
            "outliers": outliers,
            "total_cells_analyzed": len(valid_data),
        }

    def find_hotspots(
        self, value_column: str, threshold_percentile: float = 90
    ) -> list[H3Cell]:
        """
        Find hotspot cells based on value threshold.

        Args:
            value_column: Column name to analyze
            threshold_percentile: Percentile threshold for hotspots

        Returns:
            List of hotspot cells
        """
        if not self.grid.cells:
            return []

        # Extract values
        values = []
        for cell in self.grid.cells:
            value = cell.properties.get(value_column, 0)
            values.append((cell, value))

        if not values:
            return []

        # Calculate threshold
        sorted_values = sorted([v[1] for v in values])
        threshold_index = int(len(sorted_values) * threshold_percentile / 100)
        threshold = (
            sorted_values[threshold_index]
            if threshold_index < len(sorted_values)
            else sorted_values[-1]
        )

        # Find hotspots
        hotspots = [cell for cell, value in values if value >= threshold]

        return hotspots


class H3ClusterAnalyzer:
    """
    Clustering analysis for H3 grids.

    Provides methods for spatial clustering, density-based clustering,
    and hierarchical clustering within H3 hexagonal grids.
    Based on methods from Helsinki bike sharing analysis:
    https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2
    """

    def density_based_clustering(
        self,
        value_column: str,
        min_density: float | None = None,
        eps_rings: int = 1,
    ) -> dict[str, Any]:
        """
        Perform density-based clustering using H3 spatial relationships.

        Similar to DBSCAN but adapted for hexagonal grids. Based on methods from:
        https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2

        Args:
            value_column: Column containing density values
            min_density: Minimum density threshold (auto-calculated if None)
            eps_rings: Number of H3 rings to consider as neighborhood

        Returns:
            Dictionary containing cluster assignments and statistics

        Example:
            >>> clusters = analyzer.density_based_clustering('pickup_count', eps_rings=2)
            >>> print(f"Found {clusters['n_clusters']} clusters")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract valid data
        valid_cells = []
        values = []

        for cell in self.grid.cells:
            if (
                value_column in cell.properties
                and cell.properties[value_column] is not None
            ):
                valid_cells.append(cell)
                values.append(float(cell.properties[value_column]))

        if len(valid_cells) < 3:
            return {"error": "Insufficient data for clustering"}

        # Auto-calculate minimum density if not provided
        if min_density is None:
            min_density = cast(float, np.percentile(values, 75))

        # Perform clustering
        clusters = self._h3_dbscan(valid_cells, values, min_density, eps_rings)

        # Calculate statistics
        cluster_stats = self._calculate_cluster_stats(clusters, values)

        return {
            "clusters": clusters,
            "n_clusters": len(
                set(c["cluster_id"] for c in clusters if c["cluster_id"] != -1)
            ),
            "n_noise": len([c for c in clusters if c["cluster_id"] == -1]),
            "min_density_threshold": min_density,
            "eps_rings": eps_rings,
            "cluster_statistics": cluster_stats,
            "method": "H3-adapted DBSCAN",
            "reference": "https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2",
        }

    def _h3_dbscan(
        self, valid_cells: list, values: list[float], min_density: float, eps_rings: int
    ) -> list[dict[str, Any]]:
        """
        H3-adapted DBSCAN clustering algorithm.

        Args:
            valid_cells: List of valid H3 cells
            values: Corresponding density values
            min_density: Minimum density threshold
            eps_rings: Neighborhood ring size

        Returns:
            List of cluster assignments
        """
        clusters = []
        visited = set()
        cluster_id = 0

        cell_value_map = {cell.index: val for cell, val in zip(valid_cells, values)}

        for i, cell in enumerate(valid_cells):
            if cell.index in visited:
                continue

            visited.add(cell.index)

            # Get neighbors within eps_rings
            neighbors = self._get_h3_neighbors(cell.index, eps_rings, cell_value_map)

            # Check if this is a core point
            if values[i] >= min_density and len(neighbors) >= 2:
                # Start new cluster
                cluster_points = self._expand_cluster(
                    cell.index,
                    neighbors,
                    min_density,
                    eps_rings,
                    cell_value_map,
                    visited,
                )

                for point_idx in cluster_points:
                    clusters.append(
                        {
                            "cell_index": point_idx,
                            "cluster_id": cluster_id,
                            "is_core": point_idx == cell.index
                            or cell_value_map[point_idx] >= min_density,
                        }
                    )

                cluster_id += 1
            else:
                # Noise point
                clusters.append(
                    {
                        "cell_index": cell.index,
                        "cluster_id": -1,  # -1 indicates noise
                        "is_core": False,
                    }
                )

        return clusters

    def _get_h3_neighbors(
        self, cell_index: str, rings: int, cell_value_map: dict[str, float]
    ) -> list[str]:
        """
        Get H3 neighbors within specified number of rings.

        Args:
            cell_index: H3 cell index
            rings: Number of rings to include
            cell_value_map: Map of cell indices to values

        Returns:
            List of neighbor cell indices
        """
        neighbors = []

        try:
            neighbor_set = h3.grid_disk(cell_index, rings)
            # Remove self and filter to valid cells
            neighbors = [
                n for n in neighbor_set if n != cell_index and n in cell_value_map
            ]
        except Exception as e:
            logger.warning(f"Failed to get neighbors for {cell_index}: {e}")

        return neighbors

    def _expand_cluster(
        self,
        core_index: str,
        neighbors: list[str],
        min_density: float,
        eps_rings: int,
        cell_value_map: dict[str, float],
        visited: set,
    ) -> list[str]:
        """
        Expand cluster from core point.

        Args:
            core_index: Core point cell index
            neighbors: Initial neighbors
            min_density: Minimum density threshold
            eps_rings: Neighborhood ring size
            cell_value_map: Map of cell indices to values
            visited: Set of visited cells

        Returns:
            List of all points in the cluster
        """
        cluster_points = [core_index]
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)

                # Get neighbors of this neighbor
                neighbor_neighbors = self._get_h3_neighbors(
                    neighbor_idx, eps_rings, cell_value_map
                )

                # If this neighbor is also a core point, add its neighbors
                if (
                    neighbor_idx in cell_value_map
                    and cell_value_map[neighbor_idx] >= min_density
                    and len(neighbor_neighbors) >= 2
                ):
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)

            if neighbor_idx not in cluster_points:
                cluster_points.append(neighbor_idx)

            i += 1

        return cluster_points

    def _calculate_cluster_stats(
        self, clusters: list[dict], values: list[float]
    ) -> dict[str, Any]:
        """
        Calculate statistics for each cluster.

        Args:
            clusters: List of cluster assignments
            values: Original values

        Returns:
            Dictionary of cluster statistics
        """
        stats = {}

        # Group by cluster ID
        cluster_groups: dict[Any, list[dict[str, Any]]] = {}
        for cluster in clusters:
            cid = cluster["cluster_id"]
            if cid not in cluster_groups:
                cluster_groups[cid] = []
            cluster_groups[cid].append(cluster)

        # Calculate stats for each cluster
        for cid, group in cluster_groups.items():
            if cid == -1:  # Skip noise
                continue

            cluster_values = []
            for item in group:
                if "value" in item:
                    cluster_values.append(item["value"])
                elif "properties" in item and isinstance(item["properties"], dict):
                    for v in item["properties"].values():
                        if isinstance(v, (int, float)):
                            cluster_values.append(v)
                            break

            stats[f"cluster_{cid}"] = {
                "size": len(group),
                "core_points": len([item for item in group if item["is_core"]]),
                "border_points": len([item for item in group if not item["is_core"]]),
            }

        return stats

    def hierarchical_clustering(
        self, value_column: str, linkage_method: str = "ward"
    ) -> dict[str, Any]:
        """
        Perform hierarchical clustering on H3 grid cells.

        Uses H3 spatial relationships to inform the clustering process.

        Args:
            value_column: Column containing values for clustering
            linkage_method: Linkage method ('ward', 'complete', 'average')

        Returns:
            Dictionary containing hierarchical clustering results

        Example:
            >>> result = analyzer.hierarchical_clustering('density')
            >>> print(f"Optimal clusters: {result['optimal_clusters']}")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract valid data
        valid_data = []
        for cell in self.grid.cells:
            if (
                value_column in cell.properties
                and cell.properties[value_column] is not None
            ):
                valid_data.append(
                    {"cell": cell, "value": float(cell.properties[value_column])}
                )

        if len(valid_data) < 3:
            return {"error": "Insufficient data for hierarchical clustering"}

        # Create distance matrix incorporating spatial relationships
        distance_matrix = self._create_spatial_distance_matrix(valid_data)

        # Perform hierarchical clustering (simplified version)
        clusters = self._simple_hierarchical_clustering(valid_data, distance_matrix)

        return {
            "clusters": clusters,
            "method": f"Hierarchical clustering with {linkage_method} linkage",
            "distance_matrix_size": len(distance_matrix),
            "spatial_weighting": True,
        }

    def _create_spatial_distance_matrix(
        self, valid_data: list[dict]
    ) -> list[list[float]]:
        """
        Create distance matrix incorporating both value and spatial distances.

        Args:
            valid_data: List of valid cell data

        Returns:
            Distance matrix
        """
        n = len(valid_data)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Value distance
                value_dist = abs(valid_data[i]["value"] - valid_data[j]["value"])

                # Spatial distance (H3 grid distance)
                spatial_dist = self._get_h3_distance(
                    valid_data[i]["cell"].index, valid_data[j]["cell"].index
                )

                # Combined distance (weighted)
                combined_dist = 0.7 * value_dist + 0.3 * spatial_dist

                distance_matrix[i][j] = combined_dist
                distance_matrix[j][i] = combined_dist

        return distance_matrix

    def _get_h3_distance(self, cell1: str, cell2: str) -> float:
        """
        Get H3 grid distance between two cells.

        Args:
            cell1: First cell index
            cell2: Second cell index

        Returns:
            Grid distance (normalized)
        """
        try:
            return float(h3.grid_distance(cell1, cell2))
        except Exception as e:
            logger.warning(f"Failed to calculate H3 distance: {e}")

        return 1.0  # Default distance

    def _simple_hierarchical_clustering(
        self, valid_data: list[dict], distance_matrix: list[list[float]]
    ) -> list[dict]:
        """
        Simple hierarchical clustering implementation.

        Args:
            valid_data: List of valid cell data
            distance_matrix: Distance matrix

        Returns:
            List of cluster assignments
        """
        n = len(valid_data)
        clusters = [
            {"cell_index": data["cell"].index, "cluster_id": i}
            for i, data in enumerate(valid_data)
        ]

        # Simple agglomerative clustering (merge closest pairs)
        current_clusters = list(range(n))

        # Perform a few merge steps
        for _ in range(min(3, n - 1)):
            min_dist = float("inf")
            merge_i, merge_j = -1, -1

            # Find closest pair
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    if (
                        distance_matrix[current_clusters[i]][current_clusters[j]]
                        < min_dist
                    ):
                        min_dist = distance_matrix[current_clusters[i]][
                            current_clusters[j]
                        ]
                        merge_i, merge_j = i, j

            if merge_i != -1 and merge_j != -1:
                # Merge clusters
                cluster_id = min(current_clusters[merge_i], current_clusters[merge_j])

                # Update cluster assignments
                for cluster in clusters:
                    if (
                        cluster["cluster_id"] == current_clusters[merge_i]
                        or cluster["cluster_id"] == current_clusters[merge_j]
                    ):
                        cluster["cluster_id"] = cluster_id

                # Remove merged cluster
                current_clusters.pop(max(merge_i, merge_j))
                current_clusters.pop(min(merge_i, merge_j))
                current_clusters.append(cluster_id)

        return clusters

    """
    Clustering analysis for H3 grids.

    Provides methods for identifying clusters and patterns
    in H3 hexagonal grid data.
    """

    def __init__(self, grid: H3Grid) -> None:  # type: ignore[no-redef]
        """
        Initialize cluster analyzer for an H3Grid.

        Args:
            grid: H3Grid instance to analyze
        """
        self.grid = grid

    def simple_clustering(
        self, value_column: str, num_clusters: int = 3
    ) -> dict[str, Any]:
        """
        Perform simple k-means-style clustering on cell values.

        Args:
            value_column: Column name to cluster on
            num_clusters: Number of clusters

        Returns:
            Dictionary with clustering results
        """
        if not self.grid.cells:
            return {"error": "No cells to cluster"}

        # Extract values
        values = []
        for cell in self.grid.cells:
            value = cell.properties.get(value_column, 0)
            values.append(value)

        if not values:
            return {"error": f"No values found for column {value_column}"}

        # Simple clustering by value ranges
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            # All values are the same
            for cell in self.grid.cells:
                cell.properties[f"{value_column}_cluster"] = 0
            return {
                "num_clusters": 1,
                "cluster_centers": [min_val],
                "cells_clustered": len(self.grid.cells),
            }

        # Create clusters by dividing value range
        cluster_size = (max_val - min_val) / num_clusters
        cluster_centers = []

        for i in range(num_clusters):
            center = min_val + (i + 0.5) * cluster_size
            cluster_centers.append(center)

        # Assign clusters
        for cell in self.grid.cells:
            value = cell.properties.get(value_column, 0)
            cluster_id = min(num_clusters - 1, int((value - min_val) / cluster_size))
            cell.properties[f"{value_column}_cluster"] = cluster_id

        return {
            "num_clusters": num_clusters,
            "cluster_centers": cluster_centers,
            "cells_clustered": len(self.grid.cells),
            "value_range": (min_val, max_val),
        }


class H3DensityAnalyzer:
    """
    Density analysis for H3 grids.

    Provides methods for calculating density surfaces, kernel density estimation,
    and density-based pattern analysis within H3 hexagonal grids.
    Based on methods from UGRC's H3 analysis:
    https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/
    """

    def calculate_kernel_density(
        self,
        point_column: str | None = None,
        bandwidth_rings: int = 2,
        kernel_type: str = "gaussian",
    ) -> dict[str, Any]:
        """
        Calculate kernel density estimation using H3 spatial relationships.

        Based on methods from UGRC's address point analysis:
        https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/

        Args:
            point_column: Column containing point counts (uses cell count if None)
            bandwidth_rings: Number of H3 rings for kernel bandwidth
            kernel_type: Type of kernel ('gaussian', 'uniform', 'triangular')

        Returns:
            Dictionary containing density surface and statistics

        Example:
            >>> density = analyzer.calculate_kernel_density('address_count', bandwidth_rings=3)
            >>> print(f"Max density: {density['max_density']:.2f}")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Prepare data
        cell_data: list[dict[str, Any]] = []
        for cell in self.grid.cells:
            if point_column and point_column in cell.properties:
                value = (
                    float(cell.properties[point_column])
                    if cell.properties[point_column] is not None
                    else 0.0
                )
            else:
                value = 1.0  # Default: count each cell as 1

            cell_data.append({"cell": cell, "value": value})

        # Calculate kernel density for each cell
        density_results = []

        for target_data in cell_data:
            target_cell = target_data["cell"]

            # Get neighbors within bandwidth
            neighbors = self._get_neighbors_within_rings(
                target_cell.index, bandwidth_rings
            )

            # Calculate kernel density
            density = self._calculate_kernel_value(
                target_cell.index, neighbors, cell_data, bandwidth_rings, kernel_type
            )

            density_results.append(
                {
                    "cell_index": target_cell.index,
                    "density": density,
                    "neighbors_count": len(neighbors),
                }
            )

        # Calculate statistics
        densities = [r["density"] for r in density_results]

        stats = {
            "mean_density": float(np.mean(densities)),
            "std_density": float(np.std(densities)),
            "min_density": float(np.min(densities)),
            "max_density": float(np.max(densities)),
            "median_density": float(np.median(densities)),
        }

        return {
            "density_surface": density_results,
            "statistics": stats,
            "bandwidth_rings": bandwidth_rings,
            "kernel_type": kernel_type,
            "method": "H3 Kernel Density Estimation",
            "reference": "https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/",
        }

    def _get_neighbors_within_rings(self, cell_index: str, rings: int) -> list[str]:
        """
        Get all neighbors within specified number of rings.

        Args:
            cell_index: Center cell index
            rings: Number of rings to include

        Returns:
            List of neighbor cell indices
        """
        try:
            neighbor_set = h3.grid_disk(cell_index, rings)
            return sorted(neighbor_set)
        except Exception as e:
            raise ValueError(f"Failed to get H3 neighbors for {cell_index}: {e}") from e

    def _calculate_kernel_value(
        self,
        target_index: str,
        neighbors: list[str],
        cell_data: list[dict],
        bandwidth_rings: int,
        kernel_type: str,
    ) -> float:
        """
        Calculate kernel density value for a target cell.

        Args:
            target_index: Target cell index
            neighbors: List of neighbor cell indices
            cell_data: List of all cell data
            bandwidth_rings: Kernel bandwidth in rings
            kernel_type: Type of kernel function

        Returns:
            Kernel density value
        """
        density = 0.0
        cell_index_map = {data["cell"].index: data["value"] for data in cell_data}

        for neighbor_idx in neighbors:
            if neighbor_idx in cell_index_map:
                # Calculate distance in rings
                distance = self._get_ring_distance(target_index, neighbor_idx)

                # Apply kernel function
                weight = self._kernel_function(distance, bandwidth_rings, kernel_type)

                # Add weighted contribution
                density += weight * cell_index_map[neighbor_idx]

        return density

    def _get_ring_distance(self, cell1: str, cell2: str) -> int:
        """
        Get distance in H3 rings between two cells.

        Args:
            cell1: First cell index
            cell2: Second cell index

        Returns:
            Distance in rings
        """
        try:
            return cast(int, h3.grid_distance(cell1, cell2))
        except Exception as e:
            logger.warning(f"Failed to calculate ring distance: {e}")

        return 0 if cell1 == cell2 else 1

    def _kernel_function(
        self, distance: int, bandwidth: int, kernel_type: str
    ) -> float:
        """
        Apply kernel function based on distance.

        Args:
            distance: Distance in rings
            bandwidth: Kernel bandwidth
            kernel_type: Type of kernel

        Returns:
            Kernel weight
        """
        if distance > bandwidth:
            return 0.0

        normalized_distance = distance / bandwidth if bandwidth > 0 else 0

        if kernel_type == "gaussian":
            return math.exp(-0.5 * (normalized_distance * 3) ** 2)  # 3-sigma cutoff
        elif kernel_type == "uniform":
            return 1.0
        elif kernel_type == "triangular":
            return max(0, 1 - normalized_distance)
        else:
            return 1.0  # Default to uniform

    def analyze_density_patterns(self, value_column: str) -> dict[str, Any]:
        """
        Analyze density patterns and identify clusters, gaps, and gradients.

        Args:
            value_column: Column containing density values

        Returns:
            Dictionary containing pattern analysis results

        Example:
            >>> patterns = analyzer.analyze_density_patterns('population_density')
            >>> print(f"Found {len(patterns['high_density_clusters'])} high-density areas")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract valid data
        valid_data: list[dict[str, Any]] = []
        values: list[float] = []

        for cell in self.grid.cells:
            if (
                value_column in cell.properties
                and cell.properties[value_column] is not None
            ):
                value = float(cell.properties[value_column])
                valid_data.append({"cell": cell, "value": value})
                values.append(value)

        if len(valid_data) < 3:
            return {"error": "Insufficient data for pattern analysis"}

        # Calculate thresholds
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Identify patterns
        high_density_clusters = []
        low_density_gaps = []
        density_gradients = []

        for data in valid_data:
            cell = data["cell"]
            value = data["value"]

            # High density clusters (above 75th percentile)
            if value > q75:
                high_density_clusters.append(
                    {
                        "cell_index": cell.index,
                        "density": value,
                        "percentile_rank": self._calculate_percentile_rank(
                            value, values
                        ),
                    }
                )

            # Low density gaps (below 25th percentile)
            elif value < q25:
                low_density_gaps.append(
                    {
                        "cell_index": cell.index,
                        "density": value,
                        "percentile_rank": self._calculate_percentile_rank(
                            value, values
                        ),
                    }
                )

            # Calculate local gradient
            gradient = self._calculate_local_gradient(cell.index, valid_data)
            if abs(gradient) > std_val:  # Significant gradient
                density_gradients.append(
                    {"cell_index": cell.index, "gradient": gradient, "density": value}
                )

        return {
            "high_density_clusters": high_density_clusters,
            "low_density_gaps": low_density_gaps,
            "density_gradients": density_gradients,
            "thresholds": {"q25": q25, "q75": q75, "mean": mean_val, "std": std_val},
            "pattern_summary": {
                "n_high_density": len(high_density_clusters),
                "n_low_density": len(low_density_gaps),
                "n_significant_gradients": len(density_gradients),
            },
        }

    def _calculate_percentile_rank(self, value: float, values: list[float]) -> float:
        """
        Calculate percentile rank of a value.

        Args:
            value: Value to rank
            values: List of all values

        Returns:
            Percentile rank (0-100)
        """
        from scipy import stats

        return cast(float, stats.percentileofscore(values, value))

    def _calculate_local_gradient(
        self, cell_index: str, valid_data: list[dict]
    ) -> float:
        """
        Calculate local density gradient for a cell.

        Args:
            cell_index: Cell index
            valid_data: List of valid cell data

        Returns:
            Local gradient value
        """
        # Find cell value
        cell_value = None
        for data in valid_data:
            if data["cell"].index == cell_index:
                cell_value = data["value"]
                break

        if cell_value is None:
            return 0.0

        # Get neighbors. h3-py 4.x raises ValueError for malformed indices and
        # h3.H3BaseException subclasses for library errors (there is no
        # h3.CellError in this version).
        try:
            neighbors = h3.grid_disk(cell_index, 1)
            neighbors = [n for n in neighbors if n != cell_index]
        except (ValueError, h3.H3BaseException) as e:
            logger.warning(
                f"grid_disk failed for {cell_index}; using empty neighborhood in "
                f"spatial lag (local gradient); results may be degraded: {e}"
            )
            neighbors = []

        # Calculate average neighbor value
        neighbor_values = []
        for neighbor_idx in neighbors:
            for data in valid_data:
                if data["cell"].index == neighbor_idx:
                    neighbor_values.append(data["value"])
                    break

        if not neighbor_values:
            return 0.0

        avg_neighbor_value = sum(neighbor_values) / len(neighbor_values)
        return cast(float, cell_value - avg_neighbor_value)

    """
    Density analysis for H3 grids.

    Provides methods for analyzing density patterns and distributions
    within H3 hexagonal grids.
    """

    def __init__(self, grid: H3Grid) -> None:  # type: ignore[no-redef]
        """
        Initialize density analyzer for an H3Grid.

        Args:
            grid: H3Grid instance to analyze
        """
        self.grid = grid

    def calculate_density_surface(
        self, value_column: str, radius_cells: int = 2
    ) -> dict[str, Any]:
        """
        Calculate density surface using kernel density estimation.

        Args:
            value_column: Column name for density calculation
            radius_cells: Radius in cells for density calculation

        Returns:
            Dictionary with density surface results
        """
        if not self.grid.cells:
            return {"error": "No cells for density calculation"}

        # Kernel-style neighborhood density: each cell's value averaged over
        # its true H3 neighborhood (grid_disk of radius_cells, self included).
        cell_by_index = {cell.index: cell for cell in self.grid.cells}
        density_results: dict[str, float] = {}

        for cell in self.grid.cells:
            neighborhood = h3.grid_disk(cell.index, radius_cells)
            local_values = []
            for neighbor_index in neighborhood:
                neighbor = cell_by_index.get(neighbor_index)
                if neighbor is not None:
                    local_values.append(neighbor.properties.get(value_column, 0))
            local_density = (
                sum(local_values) / len(local_values) if local_values else 0.0
            )
            cell.properties[f"{value_column}_density"] = local_density
            density_results[cell.index] = local_density

        # Calculate overall statistics
        densities = list(density_results.values())

        return {
            "cells_processed": len(density_results),
            "density_range": (min(densities), max(densities)) if densities else (0, 0),
            "mean_density": sum(densities) / len(densities) if densities else 0,
            "radius_cells": radius_cells,
        }


class H3NetworkAnalyzer:
    """
    Network analysis for H3 grids.

    Provides methods for analyzing flow patterns, connectivity,
    and network-based relationships within H3 hexagonal grids.
    Based on methods from Uber's H3 usage for ride-sharing analytics.
    """

    def analyze_flow_patterns(
        self,
        origin_column: str,
        destination_column: str,
        flow_volume_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze flow patterns between H3 cells.

        Based on Uber's approach to analyzing ride patterns between hexagonal zones.

        Args:
            origin_column: Column containing origin H3 indices
            destination_column: Column containing destination H3 indices
            flow_volume_column: Column containing flow volumes (optional)

        Returns:
            Dictionary containing flow analysis results

        Example:
            >>> flows = analyzer.analyze_flow_patterns('pickup_h3', 'dropoff_h3', 'trip_count')
            >>> print(f"Analyzed {flows['total_flows']} flow connections")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract flow data
        flows = []
        cell_indices = {cell.index for cell in self.grid.cells}

        for cell in self.grid.cells:
            if (
                origin_column in cell.properties
                and destination_column in cell.properties
            ):
                origin = cell.properties[origin_column]
                destination = cell.properties[destination_column]

                if (
                    origin
                    and destination
                    and origin in cell_indices
                    and destination in cell_indices
                ):
                    volume = 1.0  # Default volume
                    if flow_volume_column and flow_volume_column in cell.properties:
                        volume = (
                            float(cell.properties[flow_volume_column])
                            if cell.properties[flow_volume_column]
                            else 1.0
                        )

                    flows.append(
                        {"origin": origin, "destination": destination, "volume": volume}
                    )

        if not flows:
            return {"error": "No valid flow data found"}

        # Analyze flow patterns
        flow_analysis = self._analyze_flows(flows)

        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(flows, cell_indices)

        return {
            "flows": flows,
            "total_flows": len(flows),
            "flow_analysis": flow_analysis,
            "network_metrics": network_metrics,
            "method": "H3 Flow Pattern Analysis",
        }

    def _analyze_flows(self, flows: list[dict]) -> dict[str, Any]:
        """
        Analyze flow patterns and identify key characteristics.

        Args:
            flows: List of flow records

        Returns:
            Flow analysis results
        """
        # Aggregate flows by origin-destination pairs
        od_matrix = {}
        total_volume = 0

        for flow in flows:
            od_pair = (flow["origin"], flow["destination"])
            if od_pair not in od_matrix:
                od_matrix[od_pair] = 0
            od_matrix[od_pair] += flow["volume"]
            total_volume += flow["volume"]

        # Find top flows
        sorted_flows = sorted(od_matrix.items(), key=lambda x: x[1], reverse=True)
        top_flows = [
            {
                "origin": pair[0],
                "destination": pair[1],
                "volume": volume,
                "percentage": (volume / total_volume) * 100 if total_volume > 0 else 0,
            }
            for pair, volume in sorted_flows[:10]
        ]

        # Calculate flow statistics
        volumes = list(od_matrix.values())
        flow_stats = {
            "mean_volume": float(np.mean(volumes)),
            "std_volume": float(np.std(volumes)),
            "max_volume": float(np.max(volumes)),
            "min_volume": float(np.min(volumes)),
        }

        return {
            "top_flows": top_flows,
            "unique_od_pairs": len(od_matrix),
            "total_volume": total_volume,
            "flow_statistics": flow_stats,
        }

    def _calculate_network_metrics(
        self, flows: list[dict], cell_indices: set
    ) -> dict[str, Any]:
        """
        Calculate network-based metrics for the H3 grid.

        Args:
            flows: List of flow records
            cell_indices: Set of all cell indices

        Returns:
            Network metrics
        """
        # Calculate in-degree and out-degree for each cell
        in_degree = {cell_idx: 0 for cell_idx in cell_indices}
        out_degree = {cell_idx: 0 for cell_idx in cell_indices}
        in_volume = {cell_idx: 0.0 for cell_idx in cell_indices}
        out_volume = {cell_idx: 0.0 for cell_idx in cell_indices}

        for flow in flows:
            origin = flow["origin"]
            destination = flow["destination"]
            volume = flow["volume"]

            if origin in out_degree:
                out_degree[origin] += 1
                out_volume[origin] += volume

            if destination in in_degree:
                in_degree[destination] += 1
                in_volume[destination] += volume

        # Find hubs (high degree nodes)
        total_degree = {
            cell_idx: in_degree[cell_idx] + out_degree[cell_idx]
            for cell_idx in cell_indices
        }

        sorted_by_degree = sorted(
            total_degree.items(), key=lambda x: x[1], reverse=True
        )
        hubs = [
            {
                "cell_index": cell_idx,
                "total_degree": degree,
                "in_degree": in_degree[cell_idx],
                "out_degree": out_degree[cell_idx],
                "in_volume": in_volume[cell_idx],
                "out_volume": out_volume[cell_idx],
            }
            for cell_idx, degree in sorted_by_degree[:10]
            if degree > 0
        ]

        # Calculate network density
        possible_connections = len(cell_indices) * (len(cell_indices) - 1)
        actual_connections = len(set((f["origin"], f["destination"]) for f in flows))
        network_density = (
            actual_connections / possible_connections if possible_connections > 0 else 0
        )

        return {
            "hubs": hubs,
            "network_density": network_density,
            "total_nodes": len(cell_indices),
            "total_connections": actual_connections,
            "average_degree": (
                sum(total_degree.values()) / len(cell_indices) if cell_indices else 0
            ),
        }

    def calculate_accessibility(
        self, impedance_column: str | None = None, max_rings: int = 5
    ) -> dict[str, Any]:
        """
        Calculate accessibility measures for each H3 cell.

        Measures how easily each cell can be reached from other cells,
        considering the H3 grid structure.

        Args:
            impedance_column: Column containing travel impedance (optional)
            max_rings: Maximum number of H3 rings to consider

        Returns:
            Dictionary containing accessibility measures

        Example:
            >>> access = analyzer.calculate_accessibility('travel_time', max_rings=3)
            >>> print(f"Most accessible cell: {access['most_accessible']['cell_index']}")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        accessibility_results: list[dict[str, Any]] = []

        for target_cell in self.grid.cells:
            # Calculate accessibility from this cell
            total_accessibility = 0.0
            reachable_cells = 0

            for ring in range(1, max_rings + 1):
                # Get cells at this ring distance
                ring_cells = self._get_cells_at_ring_distance(target_cell.index, ring)

                for ring_cell_idx in ring_cells:
                    # Find the corresponding cell
                    ring_cell = None
                    for cell in self.grid.cells:
                        if cell.index == ring_cell_idx:
                            ring_cell = cell
                            break

                    if ring_cell:
                        # Calculate impedance
                        if (
                            impedance_column
                            and impedance_column in ring_cell.properties
                        ):
                            impedance = (
                                float(ring_cell.properties[impedance_column])
                                if ring_cell.properties[impedance_column]
                                else ring
                            )
                        else:
                            impedance = ring  # Use ring distance as default impedance

                        # Add to accessibility (inverse of impedance)
                        if impedance > 0:
                            total_accessibility += 1.0 / impedance
                            reachable_cells += 1

            accessibility_results.append(
                {
                    "cell_index": target_cell.index,
                    "accessibility_score": total_accessibility,
                    "reachable_cells": reachable_cells,
                    "average_impedance": (
                        (max_rings * reachable_cells / total_accessibility)
                        if total_accessibility > 0
                        else float("inf")
                    ),
                }
            )

        # Find most and least accessible cells
        sorted_results = sorted(
            accessibility_results, key=lambda x: x["accessibility_score"], reverse=True
        )

        return {
            "accessibility_scores": accessibility_results,
            "most_accessible": sorted_results[0] if sorted_results else None,
            "least_accessible": sorted_results[-1] if sorted_results else None,
            "max_rings_analyzed": max_rings,
            "method": "H3 Grid-based Accessibility Analysis",
        }

    def _get_cells_at_ring_distance(
        self, center_index: str, ring_distance: int
    ) -> list[str]:
        """
        Get cells at exactly the specified ring distance.

        Args:
            center_index: Center cell index
            ring_distance: Ring distance

        Returns:
            List of cell indices at the specified ring distance
        """
        if not isinstance(ring_distance, int) or ring_distance < 1:
            raise ValueError("ring_distance must be a positive integer")
        try:
            return sorted(
                set(h3.grid_disk(center_index, ring_distance))
                - set(h3.grid_disk(center_index, ring_distance - 1))
            )
        except Exception as e:
            raise ValueError(f"Failed to get H3 ring cells: {e}") from e

    def detect_network_communities(self, flow_threshold: float = 0.1) -> dict[str, Any]:
        """
        Detect communities in the H3 network based on flow patterns.

        Uses a simplified community detection algorithm based on flow volumes.

        Args:
            flow_threshold: Minimum flow volume to consider for community detection

        Returns:
            Dictionary containing detected communities

        Example:
            >>> communities = analyzer.detect_network_communities(flow_threshold=0.05)
            >>> print(f"Found {len(communities['communities'])} communities")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Build adjacency matrix based on flows
        cell_indices = [cell.index for cell in self.grid.cells]
        adjacency: dict[str, set[str]] = {idx: set() for idx in cell_indices}

        # Add edges based on flow data (simplified approach)
        for cell in self.grid.cells:
            # Look for flow-related properties
            for prop_name, prop_value in cell.properties.items():
                if "flow" in prop_name.lower() and isinstance(prop_value, (int, float)):
                    if prop_value >= flow_threshold:
                        # Add spatial neighbors as connected
                        try:
                            neighbors = h3.grid_disk(cell.index, 1)
                            for neighbor in neighbors:
                                if neighbor != cell.index and neighbor in adjacency:
                                    adjacency[cell.index].add(neighbor)
                                    adjacency[neighbor].add(cell.index)
                        except Exception:
                            continue

        # Simple community detection using connected components
        communities: list[dict[str, Any]] = []
        visited: set[str] = set()

        for cell_idx in cell_indices:
            if cell_idx not in visited:
                # Find connected component
                community = self._find_connected_component(cell_idx, adjacency, visited)
                if len(community) > 1:  # Only include communities with multiple cells
                    communities.append(
                        {
                            "community_id": len(communities),
                            "cells": list(community),
                            "size": len(community),
                        }
                    )

        return {
            "communities": communities,
            "n_communities": len(communities),
            "flow_threshold": flow_threshold,
            "method": "Connected Components Community Detection",
        }

    def _find_connected_component(
        self,
        start_cell: str,
        adjacency: dict[str, set[str]],
        visited: set[str],
    ) -> set[str]:
        """
        Find connected component starting from a cell.

        Args:
            start_cell: Starting cell index
            adjacency: Adjacency dictionary
            visited: Set of visited cells

        Returns:
            Set of cells in the connected component
        """
        component = set()
        stack = [start_cell]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)

                # Add unvisited neighbors to stack
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return component

    """
    Network analysis for H3 grids.

    Provides methods for analyzing network properties and connectivity
    patterns within H3 hexagonal grids.
    """

    def __init__(self, grid: H3Grid) -> None:  # type: ignore[no-redef]
        """
        Initialize network analyzer for an H3Grid.

        Args:
            grid: H3Grid instance to analyze
        """
        self.grid = grid

    def analyze_connectivity_patterns(self) -> dict[str, Any]:
        """
        Analyze connectivity patterns in the grid.

        Returns:
            Dictionary with connectivity analysis results
        """
        if not self.grid.cells:
            return {"error": "No cells for connectivity analysis"}

        # Exact H3 adjacency: two grid cells are connected when they are
        # direct neighbors on the H3 lattice (grid_disk ring 1).
        grid_indices = {cell.index for cell in self.grid.cells}
        cell_connections: dict[str, int] = {}
        total_connections = 0

        for cell in self.grid.cells:
            neighbor_indices = set(h3.grid_disk(cell.index, 1)) - {cell.index}
            connections = len(neighbor_indices & grid_indices)
            cell_connections[cell.index] = connections
            total_connections += connections

        # Calculate network metrics
        if cell_connections:
            avg_connections = sum(cell_connections.values()) / len(cell_connections)
            max_connections = max(cell_connections.values())
            min_connections = min(cell_connections.values())
        else:
            avg_connections = max_connections = min_connections = 0

        return {
            "total_cells": len(self.grid.cells),
            "total_connections": total_connections
            // 2,  # Each connection counted twice
            "average_connections": avg_connections,
            "max_connections": max_connections,
            "min_connections": min_connections,
            "cell_connections": cell_connections,
        }


class H3TemporalAnalyzer:
    """
    Temporal analysis for H3 grids.

    Provides methods for analyzing temporal patterns, trends,
    and time-series data within H3 hexagonal grids.
    Based on methods from Helsinki bike sharing temporal analysis:
    https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2
    """

    def analyze_temporal_patterns(
        self,
        timestamp_column: str,
        value_column: str,
        temporal_resolution: str = "hour",
    ) -> dict[str, Any]:
        """
        Analyze temporal patterns in H3 grid data.

        Based on Helsinki bike sharing analysis showing peak usage times:
        https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values to analyze
            temporal_resolution: Temporal resolution ('hour', 'day', 'week', 'month')

        Returns:
            Dictionary containing temporal pattern analysis

        Example:
            >>> patterns = analyzer.analyze_temporal_patterns('timestamp', 'trip_count')
            >>> print(f"Peak hour: {patterns['peak_periods'][0]['period']}")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract temporal data
        temporal_data: list[dict[str, Any]] = []

        for cell in self.grid.cells:
            if (
                timestamp_column in cell.properties
                and value_column in cell.properties
                and cell.properties[timestamp_column] is not None
                and cell.properties[value_column] is not None
            ):
                try:
                    # Parse timestamp
                    timestamp_str = str(cell.properties[timestamp_column])
                    timestamp = self._parse_timestamp(timestamp_str)

                    if timestamp:
                        temporal_data.append(
                            {
                                "cell_index": cell.index,
                                "timestamp": timestamp,
                                "value": float(cell.properties[value_column]),
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse temporal data for cell {cell.index}: {e}"
                    )

        if not temporal_data:
            return {"error": "No valid temporal data found"}

        # Aggregate by temporal resolution
        aggregated_data = self._aggregate_by_temporal_resolution(
            temporal_data, temporal_resolution
        )

        # Analyze patterns
        patterns = self._analyze_patterns(aggregated_data, temporal_resolution)

        # Calculate statistics
        stats = self._calculate_temporal_stats(aggregated_data)

        return {
            "temporal_patterns": patterns,
            "aggregated_data": aggregated_data,
            "statistics": stats,
            "temporal_resolution": temporal_resolution,
            "data_points": len(temporal_data),
            "method": "H3 Temporal Pattern Analysis",
            "reference": "https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2",
        }

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """
        Parse timestamp string into datetime object.

        Args:
            timestamp_str: Timestamp string

        Returns:
            Parsed datetime object or None
        """
        try:
            # Try common timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            # Try parsing as ISO format
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return None

    def _aggregate_by_temporal_resolution(
        self, temporal_data: list[dict], resolution: str
    ) -> dict[int, Any]:
        """
        Aggregate data by temporal resolution.

        Args:
            temporal_data: List of temporal data points
            resolution: Temporal resolution

        Returns:
            Dictionary of aggregated data
        """
        aggregated: dict[int, Any] = {}

        for data_point in temporal_data:
            timestamp = data_point["timestamp"]
            value = data_point["value"]

            # Extract temporal key based on resolution
            if resolution == "hour":
                key = timestamp.hour
            elif resolution == "day":
                key = timestamp.weekday()  # 0=Monday, 6=Sunday
            elif resolution == "week":
                key = timestamp.isocalendar()[1]  # Week number
            elif resolution == "month":
                key = timestamp.month
            else:
                key = timestamp.hour  # Default to hour

            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)

        # Calculate averages
        for key in aggregated:
            values = aggregated[key]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "sum": float(np.sum(values)),
                "count": len(values),
                "std": float(np.std(values)),
            }

        return aggregated

    def _analyze_patterns(
        self, aggregated_data: dict, resolution: str
    ) -> dict[str, Any]:
        """
        Analyze temporal patterns in aggregated data.

        Args:
            aggregated_data: Aggregated temporal data
            resolution: Temporal resolution

        Returns:
            Pattern analysis results
        """
        if not aggregated_data:
            return {}

        # Find peak periods
        sorted_periods = sorted(
            aggregated_data.items(), key=lambda x: x[1]["mean"], reverse=True
        )

        peak_periods = []
        for period, stats in sorted_periods[:5]:  # Top 5 periods
            period_name = self._get_period_name(period, resolution)
            peak_periods.append(
                {
                    "period": period,
                    "period_name": period_name,
                    "mean_value": stats["mean"],
                    "total_value": stats["sum"],
                    "count": stats["count"],
                }
            )

        # Calculate temporal variability
        all_means = [stats["mean"] for stats in aggregated_data.values()]
        variability = (
            float(np.std(all_means)) / float(np.mean(all_means))
            if np.mean(all_means) > 0
            else 0
        )

        # Identify patterns
        pattern_type = self._identify_pattern_type(aggregated_data, resolution)

        return {
            "peak_periods": peak_periods,
            "temporal_variability": variability,
            "pattern_type": pattern_type,
            "total_periods": len(aggregated_data),
        }

    def _get_period_name(self, period: int, resolution: str) -> str:
        """
        Get human-readable name for temporal period.

        Args:
            period: Period number
            resolution: Temporal resolution

        Returns:
            Human-readable period name
        """
        if resolution == "hour":
            return f"{period:02d}:00"
        elif resolution == "day":
            days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            return days[period] if 0 <= period < 7 else f"Day {period}"
        elif resolution == "week":
            return f"Week {period}"
        elif resolution == "month":
            months = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            return months[period - 1] if 1 <= period <= 12 else f"Month {period}"
        else:
            return str(period)

    def _identify_pattern_type(self, aggregated_data: dict, resolution: str) -> str:
        """
        Identify the type of temporal pattern.

        Args:
            aggregated_data: Aggregated temporal data
            resolution: Temporal resolution

        Returns:
            Pattern type description
        """
        if not aggregated_data:
            return "No pattern"

        values = [stats["mean"] for stats in aggregated_data.values()]

        max_val = np.max(values)
        min_val = np.min(values)
        mean_val = np.mean(values)

        # Simple pattern classification
        if (max_val - min_val) / mean_val < 0.2:  # Low variability
            return "Uniform pattern"
        elif resolution == "hour":
            # Check for typical daily patterns
            morning_peak = any(
                6 <= period <= 9 and aggregated_data[period]["mean"] > mean_val * 1.2
                for period in aggregated_data
                if isinstance(period, int)
            )
            evening_peak = any(
                16 <= period <= 19 and aggregated_data[period]["mean"] > mean_val * 1.2
                for period in aggregated_data
                if isinstance(period, int)
            )

            if morning_peak and evening_peak:
                return "Commuter pattern (morning and evening peaks)"
            elif morning_peak:
                return "Morning peak pattern"
            elif evening_peak:
                return "Evening peak pattern"
            else:
                return "Irregular hourly pattern"
        elif resolution == "day":
            # Check for weekday/weekend patterns
            weekday_avg = (
                sum(
                    aggregated_data[d]["mean"] for d in range(5) if d in aggregated_data
                )
                / 5
            )
            weekend_avg = (
                sum(aggregated_data[d]["mean"] for d in [5, 6] if d in aggregated_data)
                / 2
            )

            if weekday_avg > weekend_avg * 1.2:
                return "Weekday-dominant pattern"
            elif weekend_avg > weekday_avg * 1.2:
                return "Weekend-dominant pattern"
            else:
                return "Balanced weekly pattern"
        else:
            return "Complex temporal pattern"

    def _calculate_temporal_stats(self, aggregated_data: dict) -> dict[str, Any]:
        """
        Calculate temporal statistics.

        Args:
            aggregated_data: Aggregated temporal data

        Returns:
            Temporal statistics
        """
        if not aggregated_data:
            return {}

        all_values = []
        total_count = 0
        total_sum = 0

        for stats in aggregated_data.values():
            all_values.append(stats["mean"])
            total_count += stats["count"]
            total_sum += stats["sum"]

        return {
            "overall_mean": float(np.mean(all_values)),
            "overall_std": float(np.std(all_values)),
            "overall_min": float(np.min(all_values)),
            "overall_max": float(np.max(all_values)),
            "total_observations": total_count,
            "total_value": total_sum,
        }

    def detect_temporal_anomalies(
        self,
        timestamp_column: str,
        value_column: str,
        method: str = "zscore",
        threshold: float = 2.0,
    ) -> dict[str, Any]:
        """
        Detect temporal anomalies in H3 grid data.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values to analyze
            method: Anomaly detection method ('zscore', 'iqr')
            threshold: Threshold for anomaly detection

        Returns:
            Dictionary containing detected anomalies

        Example:
            >>> anomalies = analyzer.detect_temporal_anomalies('timestamp', 'activity_level')
            >>> print(f"Found {len(anomalies['anomalies'])} temporal anomalies")
        """
        if not self.grid.cells:
            return {"error": "No cells in grid"}

        # Extract temporal data
        temporal_data: list[dict[str, Any]] = []

        for cell in self.grid.cells:
            if (
                timestamp_column in cell.properties
                and value_column in cell.properties
                and cell.properties[timestamp_column] is not None
                and cell.properties[value_column] is not None
            ):
                try:
                    timestamp_str = str(cell.properties[timestamp_column])
                    timestamp = self._parse_timestamp(timestamp_str)

                    if timestamp:
                        temporal_data.append(
                            {
                                "cell_index": cell.index,
                                "timestamp": timestamp,
                                "value": float(cell.properties[value_column]),
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse temporal data for cell {cell.index}: {e}"
                    )

        if len(temporal_data) < 10:
            return {"error": "Insufficient data for anomaly detection"}

        # Sort by timestamp
        temporal_data.sort(key=lambda x: x["timestamp"])

        # Detect anomalies
        if method == "zscore":
            anomalies = self._detect_zscore_anomalies(temporal_data, threshold)
        elif method == "iqr":
            anomalies = self._detect_iqr_anomalies(temporal_data, threshold)
        else:
            return {"error": f"Unknown anomaly detection method: {method}"}

        return {
            "anomalies": anomalies,
            "method": method,
            "threshold": threshold,
            "total_data_points": len(temporal_data),
            "anomaly_rate": len(anomalies) / len(temporal_data) if temporal_data else 0,
        }

    def _detect_zscore_anomalies(
        self, temporal_data: list[dict], threshold: float
    ) -> list[dict]:
        """
        Detect anomalies using Z-score method.

        Args:
            temporal_data: List of temporal data points
            threshold: Z-score threshold

        Returns:
            List of detected anomalies
        """
        values = [d["value"] for d in temporal_data]

        mean_val = np.mean(values)
        std_val = np.std(values)

        anomalies = []

        for data_point in temporal_data:
            if std_val > 0:
                zscore = abs(data_point["value"] - mean_val) / std_val
                if zscore > threshold:
                    anomalies.append(
                        {
                            "cell_index": data_point["cell_index"],
                            "timestamp": data_point["timestamp"].isoformat(),
                            "value": data_point["value"],
                            "zscore": zscore,
                            "expected_value": mean_val,
                            "anomaly_type": (
                                "high" if data_point["value"] > mean_val else "low"
                            ),
                        }
                    )

        return anomalies

    def _detect_iqr_anomalies(
        self, temporal_data: list[dict], multiplier: float
    ) -> list[dict]:
        """
        Detect anomalies using IQR method.

        Args:
            temporal_data: List of temporal data points
            multiplier: IQR multiplier for outlier detection

        Returns:
            List of detected anomalies
        """
        values = [d["value"] for d in temporal_data]

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        anomalies = []

        for data_point in temporal_data:
            value = data_point["value"]
            if value < lower_bound or value > upper_bound:
                anomalies.append(
                    {
                        "cell_index": data_point["cell_index"],
                        "timestamp": data_point["timestamp"].isoformat(),
                        "value": value,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "anomaly_type": "high" if value > upper_bound else "low",
                    }
                )

        return anomalies

    """
    Temporal analysis for H3 grids.

    Provides methods for analyzing temporal patterns and changes
    in H3 hexagonal grid data over time.
    """

    def __init__(self, grids: list[H3Grid]) -> None:  # type: ignore[no-redef]
        """
        Initialize temporal analyzer for multiple H3Grids.

        Args:
            grids: List of H3Grid instances representing time series
        """
        self.grids = grids
        self.timestamps = [grid.created_at for grid in grids]

    def analyze_temporal_trends(self, value_column: str) -> dict[str, Any]:
        """
        Analyze temporal trends in grid values.

        Args:
            value_column: Column name to analyze over time

        Returns:
            Dictionary with temporal trend analysis
        """
        if not self.grids:
            return {"error": "No grids for temporal analysis"}

        # Track values over time for each cell position
        temporal_data: dict[str, list[dict[str, Any]]] = {}

        for i, grid in enumerate(self.grids):
            timestamp = self.timestamps[i]

            for cell in grid.cells:
                cell_key = cell.index
                value = cell.properties.get(value_column, 0)

                if cell_key not in temporal_data:
                    temporal_data[cell_key] = []

                temporal_data[cell_key].append(
                    {"timestamp": timestamp, "value": value, "time_index": i}
                )

        # Calculate trends
        trend_analysis = {}

        for cell_key, time_series in temporal_data.items():
            if len(time_series) < 2:
                continue

            values = [point["value"] for point in time_series]

            # Simple trend calculation
            first_value = values[0]
            last_value = values[-1]
            trend = (
                "increasing"
                if last_value > first_value
                else "decreasing"
                if last_value < first_value
                else "stable"
            )

            trend_analysis[cell_key] = {
                "trend": trend,
                "change": last_value - first_value,
                "percent_change": (
                    ((last_value - first_value) / first_value * 100)
                    if first_value != 0
                    else 0
                ),
                "data_points": len(time_series),
            }

        # Overall statistics
        if trend_analysis:
            changes = [analysis["change"] for analysis in trend_analysis.values()]
            avg_change = sum(changes) / len(changes)

            trend_counts: dict[str, int] = {}
            for analysis in trend_analysis.values():
                trend = analysis["trend"]
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
        else:
            avg_change = 0
            trend_counts = {}

        return {
            "cells_analyzed": len(trend_analysis),
            "time_periods": len(self.grids),
            "average_change": avg_change,
            "trend_distribution": trend_counts,
            "cell_trends": trend_analysis,
        }

    def detect_anomalies(
        self, value_column: str, threshold_std: float = 2.0
    ) -> dict[str, Any]:
        """
        Detect temporal anomalies in grid values.

        Args:
            value_column: Column name to analyze
            threshold_std: Standard deviation threshold for anomaly detection

        Returns:
            Dictionary with anomaly detection results
        """
        if not self.grids:
            return {"error": "No grids for anomaly detection"}

        # Collect all values across time and space
        all_values = []
        for grid in self.grids:
            for cell in grid.cells:
                value = cell.properties.get(value_column, 0)
                all_values.append(value)

        if not all_values:
            return {"error": f"No values found for column {value_column}"}

        # Calculate statistics
        mean_value = sum(all_values) / len(all_values)

        std_value = np.std(all_values)

        # Define anomaly thresholds
        upper_threshold = mean_value + threshold_std * std_value
        lower_threshold = mean_value - threshold_std * std_value

        # Find anomalies
        anomalies = []

        for i, grid in enumerate(self.grids):
            timestamp = self.timestamps[i]

            for cell in grid.cells:
                value = cell.properties.get(value_column, 0)

                if value > upper_threshold or value < lower_threshold:
                    anomalies.append(
                        {
                            "timestamp": timestamp,
                            "time_index": i,
                            "cell_index": cell.index,
                            "value": value,
                            "anomaly_type": (
                                "high" if value > upper_threshold else "low"
                            ),
                            "deviation": (
                                abs(value - mean_value) / std_value
                                if std_value > 0
                                else 0
                            ),
                        }
                    )

        return {
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(all_values) if all_values else 0,
            "mean_value": mean_value,
            "std_value": std_value,
            "thresholds": (lower_threshold, upper_threshold),
            "anomalies": anomalies,
        }
