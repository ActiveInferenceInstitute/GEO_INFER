"""
Generic spatial analytics interface for GEO-INFER-SPACE.

This module defines the generic interface for spatial analytics operations
that can be implemented by different backends (H3, SRAI, etc.).
"""

from typing import Any, cast
import logging
import numpy as np

from .interfaces import UnsupportedSpatialOperationError

logger = logging.getLogger(__name__)


class SpatialAnalyticsInterface:
    """
    Generic interface for spatial analytics operations.

    This class provides a unified API for spatial analytics that can dispatch
    to different backends based on configuration.
    """

    def __init__(self, backend: str | None = None) -> None:
        from .dispatcher import get_backend_dispatcher

        self.dispatcher = get_backend_dispatcher()
        self.backend = backend

    def analyze_hotspots(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Analyze spatial hotspots in the data.

        Args:
            data: Spatial data with values to analyze
            **kwargs: Additional backend-specific parameters

        Returns:
            Hotspot analysis results
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_analytics_operation(
                "analyze_hotspots", data, backend=self.backend, **kwargs
            ),
        )

    def analyze_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze spatial context for a given location or area.

        Args:
            context: Spatial context information

        Returns:
            Analysis results for the spatial context
        """
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")

        result = {
            "backend": self.backend or self.dispatcher.get_default_backend("indexing"),
            "context": context,
            "status": "analyzed",
        }
        position = context.get("position")
        if position is None:
            return result

        try:
            latitude, longitude = position[:2]
            resolution = int(context.get("resolution", 8))
            result["cell"] = self.dispatcher.dispatch_indexing_operation(
                "latlng_to_cell",
                float(latitude),
                float(longitude),
                resolution,
                backend=self.backend,
            )
        except (TypeError, ValueError, IndexError) as exc:
            logger.debug("Unable to resolve spatial context to a cell: %s", exc)
            result["status"] = "unresolved"
            result["error"] = str(exc)
        return result

    def analyze_clusters(
        self, data: np.ndarray, method: str = "dbscan", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Analyze spatial clustering patterns in data.

        Args:
            data: Spatial data points
            method: Clustering method to use
            **kwargs: Additional clustering parameters

        Returns:
            Clustering analysis results
        """
        points = np.asarray(data, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "data must be an (n, 2) array of latitude/longitude points"
            )
        if method != "dbscan":
            raise ValueError("H3-backed clustering currently supports method='dbscan'")
        return self.cluster_points(
            [tuple(point) for point in points],
            **kwargs,
        )

    def find_hotspots(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Find spatial hotspots in data.

        Args:
            data: Spatial data with intensity values
            **kwargs: Additional hotspot detection parameters

        Returns:
            Hotspot detection results
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_analytics_operation(
                "find_hotspots", data, backend=self.backend, **kwargs
            ),
        )

    def compute_proximity(
        self, points: list[tuple[float, float]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Compute proximity analysis between points.

        Args:
            points: List of (lat, lng) coordinate pairs
            **kwargs: Additional backend-specific parameters

        Returns:
            Proximity analysis results
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_analytics_operation(
                "compute_proximity", points, backend=self.backend, **kwargs
            ),
        )

    def cluster_points(
        self, points: list[tuple[float, float]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Cluster spatial points.

        Args:
            points: List of (lat, lng) coordinate pairs
            **kwargs: Additional backend-specific parameters (e.g., num_clusters, method)

        Returns:
            Clustering results with cluster assignments
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_analytics_operation(
                "cluster_points", points, backend=self.backend, **kwargs
            ),
        )

    def interpolate_values(
        self,
        points: list[tuple[float, float, float]],
        target_points: list[tuple[float, float]] | None = None,
        resolution: int = 9,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Interpolate values across a spatial surface.

        Args:
            points: List of (lat, lng, value) tuples
            **kwargs: Additional backend-specific parameters (e.g., method, resolution)

        Returns:
            Interpolated surface data
        """
        backend_name = self.backend or self.dispatcher.get_default_backend("analytics")
        if backend_name != "h3":
            raise ValueError(
                "interpolate_values currently requires the H3 backend and native "
                "H3 cell inputs"
            )
        if not points:
            raise ValueError("points must contain at least one (lat, lng, value) tuple")
        if not 0 <= resolution <= 15:
            raise ValueError("resolution must be between 0 and 15")
        source_cells = []
        values = []
        for point in points:
            if len(point) != 3:
                raise ValueError(
                    "each source point must be (latitude, longitude, value)"
                )
            lat, lng, value = point
            source_cells.append(
                self.dispatcher.dispatch_indexing_operation(
                    "latlng_to_cell",
                    float(lat),
                    float(lng),
                    resolution,
                    backend=self.backend,
                )
            )
            values.append(float(value))

        target_cells = kwargs.pop("target_cells", None)
        if target_cells is None:
            target_points = target_points or [(p[0], p[1]) for p in points]
            target_cells = [
                self.dispatcher.dispatch_indexing_operation(
                    "latlng_to_cell",
                    float(lat),
                    float(lng),
                    resolution,
                    backend=self.backend,
                )
                for lat, lng in target_points
            ]
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_analytics_operation(
                "interpolate_values",
                source_cells,
                values,
                list(target_cells),
                backend=self.backend,
                **kwargs,
            ),
        )

    def analyze_network(
        self, edges: list[tuple[int, int, float]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Analyze spatial network structure.

        Args:
            edges: List of (source_id, target_id, weight) tuples
            **kwargs: Additional backend-specific parameters

        Returns:
            Network analysis results (centrality, connectivity, etc.)
        """
        backend = self.backend or self.dispatcher.get_default_backend("analytics")
        raise UnsupportedSpatialOperationError("analyze_network", backend)

    def detect_patterns(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Detect spatial patterns in the data.

        Args:
            data: Spatial data to analyze for patterns
            **kwargs: Additional backend-specific parameters

        Returns:
            Pattern detection results
        """
        backend = self.backend or self.dispatcher.get_default_backend("analytics")
        raise UnsupportedSpatialOperationError("detect_patterns", backend)

    def compute_density(
        self, points: list[tuple[float, float]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Compute spatial density.

        Args:
            points: List of (lat, lng) coordinate pairs
            **kwargs: Additional backend-specific parameters (e.g., bandwidth, resolution)

        Returns:
            Density analysis results
        """
        backend = self.backend or self.dispatcher.get_default_backend("analytics")
        raise UnsupportedSpatialOperationError("compute_density", backend)

    def analyze_accessibility(
        self,
        origins: list[tuple[float, float]],
        destinations: list[tuple[float, float]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze spatial accessibility between origins and destinations.

        Args:
            origins: List of origin (lat, lng) coordinates
            destinations: List of destination (lat, lng) coordinates
            **kwargs: Additional backend-specific parameters

        Returns:
            Accessibility analysis results
        """
        backend = self.backend or self.dispatcher.get_default_backend("analytics")
        raise UnsupportedSpatialOperationError("analyze_accessibility", backend)
