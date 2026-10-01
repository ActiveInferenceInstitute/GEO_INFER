"""
Generic geometric operations interface for GEO-INFER-SPACE.

This module defines the generic interface for geometric operations
that can be implemented by different backends (H3, SRAI, etc.).
"""

from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


class GeometricOperationsInterface:
    """
    Generic interface for geometric operations.

    This class provides a unified API for geometric operations that can dispatch
    to different backends based on configuration.
    """

    def __init__(self, backend: str | None = None) -> None:
        from .dispatcher import get_backend_dispatcher

        self.dispatcher = get_backend_dispatcher()
        self.backend = backend

    def buffer_geometry(
        self,
        geometry: dict[str, Any],
        distance: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a buffer around a geometry.

        Args:
            geometry: Input geometry as GeoJSON-like dictionary
            distance: Buffer distance
            **kwargs: Additional backend-specific parameters

        Returns:
            Buffered geometry
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_geometric_operation(
                "buffer_geometry",
                geometry,
                distance,
                backend=self.backend,
                **kwargs,
            ),
        )

    def calculate_area(self, geometry: dict[str, Any]) -> float:
        """
        Calculate the area of a geometry.

        Args:
            geometry: Input geometry as GeoJSON-like dictionary

        Returns:
            Area of the geometry
        """
        return cast(
            float,
            self.dispatcher.dispatch_geometric_operation(
                "calculate_area", geometry, backend=self.backend
            ),
        )

    def calculate_perimeter(self, geometry: dict[str, Any]) -> float:
        """
        Calculate the perimeter of a geometry.

        Args:
            geometry: Input geometry as GeoJSON-like dictionary

        Returns:
            Perimeter of the geometry
        """
        return cast(
            float,
            self.dispatcher.dispatch_geometric_operation(
                "calculate_perimeter", geometry, backend=self.backend
            ),
        )

    def calculate_centroid(self, geometry: dict[str, Any]) -> tuple[float, float]:
        """
        Calculate the centroid of a geometry.

        Args:
            geometry: Input geometry as GeoJSON-like dictionary

        Returns:
            Centroid coordinates as (lat, lng)
        """
        return cast(
            tuple[float, float],
            self.dispatcher.dispatch_geometric_operation(
                "calculate_centroid", geometry, backend=self.backend
            ),
        )

    def calculate_distance(self, geom1: dict[str, Any], geom2: dict[str, Any]) -> float:
        """
        Calculate distance between two geometries.

        Args:
            geom1: First geometry
            geom2: Second geometry

        Returns:
            Distance between geometries
        """
        return cast(
            float,
            self.dispatcher.dispatch_geometric_operation(
                "calculate_distance", geom1, geom2, backend=self.backend
            ),
        )

    def union_geometries(self, geometries: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Union multiple geometries.

        Args:
            geometries: List of geometries to union

        Returns:
            Union of all input geometries
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_geometric_operation(
                "union_geometries", geometries, backend=self.backend
            ),
        )

    def intersection_geometries(
        self, geom1: dict[str, Any], geom2: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate intersection of two geometries.

        Args:
            geom1: First geometry
            geom2: Second geometry

        Returns:
            Intersection of the two geometries
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_geometric_operation(
                "intersection_geometries", geom1, geom2, backend=self.backend
            ),
        )

    def difference_geometries(
        self, geom1: dict[str, Any], geom2: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate difference of two geometries.

        Args:
            geom1: First geometry ( minuend)
            geom2: Second geometry (subtrahend)

        Returns:
            Difference of the two geometries
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_geometric_operation(
                "difference_geometries", geom1, geom2, backend=self.backend
            ),
        )

    def contains_geometry(
        self, container: dict[str, Any], contained: dict[str, Any]
    ) -> bool:
        """
        Check if one geometry contains another.

        Args:
            container: Container geometry
            contained: Geometry to check for containment

        Returns:
            True if contained is within container
        """
        return cast(
            bool,
            self.dispatcher.dispatch_geometric_operation(
                "contains_geometry", container, contained, backend=self.backend
            ),
        )

    def intersects_geometry(self, geom1: dict[str, Any], geom2: dict[str, Any]) -> bool:
        """
        Check if two geometries intersect.

        Args:
            geom1: First geometry
            geom2: Second geometry

        Returns:
            True if geometries intersect
        """
        return cast(
            bool,
            self.dispatcher.dispatch_geometric_operation(
                "intersects_geometry", geom1, geom2, backend=self.backend
            ),
        )

    def transform_geometry(
        self, geometry: dict[str, Any], from_crs: str, to_crs: str
    ) -> dict[str, Any]:
        """
        Transform geometry from one coordinate reference system to another.

        Args:
            geometry: Input geometry
            from_crs: Source CRS (EPSG code or proj string)
            to_crs: Target CRS (EPSG code or proj string)

        Returns:
            Transformed geometry
        """
        return cast(
            dict[str, Any],
            self.dispatcher.dispatch_indexing_operation(
                "transform_geometry",
                geometry,
                from_crs,
                to_crs,
                backend=self.backend,
            ),
        )
