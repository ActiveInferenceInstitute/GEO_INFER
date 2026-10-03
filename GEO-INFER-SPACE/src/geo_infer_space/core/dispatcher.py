"""
Spatial backend dispatcher for GEO-INFER-SPACE.

This module provides the central dispatch system that routes spatial operations
to the appropriate backend (H3, SRAI, etc.) based on configuration and operation type.
"""

import logging
from typing import Any

from .interfaces import (
    SpatialBackendProtocol,
)

logger = logging.getLogger(__name__)


class SpatialBackendDispatcher:
    """
    Central dispatcher for spatial operations across different backends.

    This class manages backend registration, selection, and operation dispatch
    based on configuration and operation requirements.
    """

    def __init__(self) -> None:
        self.backends: dict[str, SpatialBackendProtocol] = {}
        self.default_backends: dict[str, str] = {}
        self.backend_capabilities: dict[str, dict[str, Any]] = {}
        self._load_backends()

    _STANDARD_OPERATION_TYPES = ("indexing", "geometric", "analytics")

    def _load_backends(self) -> None:
        """Load available spatial backends."""
        h3_backend = self._load_h3_backend()
        if h3_backend.is_available():
            self.register_backend("h3", h3_backend)
            for operation_type in self._STANDARD_OPERATION_TYPES:
                self.default_backends.setdefault(operation_type, "h3")
            logger.info("H3 backend loaded successfully")
        srai_backend = self._load_srai_backend()
        self.register_backend("srai", srai_backend)

    def _load_h3_backend(self) -> SpatialBackendProtocol:
        """Load the required H3 implementation; internal failures propagate."""
        from ..backends.h3 import H3Backend

        return H3Backend()

    def _load_srai_backend(self) -> SpatialBackendProtocol:
        """Load the SRAI interface, which reports an absent extra explicitly."""
        from ..backends.srai import SraiBackend

        return SraiBackend()

    def register_backend(self, name: str, backend: SpatialBackendProtocol) -> None:
        """Register a spatial backend."""
        self.backends[name] = backend
        self.backend_capabilities[name] = backend.get_capabilities()
        logger.info(f"Registered spatial backend: {name}")

    def get_backend(self, name: str) -> SpatialBackendProtocol | None:
        """Get a specific backend by name."""
        return self.backends.get(name)

    def get_available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return [
            name for name, backend in self.backends.items() if backend.is_available()
        ]

    def set_default_backend(self, operation_type: str, backend_name: str) -> None:
        """Set the default backend for a specific operation type."""
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' is not registered")
        self.default_backends[operation_type] = backend_name

    def get_default_backend(self, operation_type: str) -> str | None:
        """Get the registered default backend for an operation type.

        Returns the backend name previously registered through
        :meth:`set_default_backend` (or auto-registered when a backend
        loaded successfully), or ``None`` when no usable default has been
        registered. No implicit backend name is invented here; dispatch
        callers are responsible for raising a precise error when they need
        a default and none is available.
        """
        return self.default_backends.get(operation_type)

    def _resolve_backend_name(self, operation_type: str, backend: str | None) -> str:
        """Resolve the backend for a dispatch call or raise a precise error.

        Raises:
            ValueError: When no backend was given and no default is
                registered, or when the resolved backend is not registered.
                The message always lists the currently available backends.
        """
        backend_name = backend or self.get_default_backend(operation_type)
        available = ", ".join(self.get_available_backends()) or "none"
        if backend_name is None:
            raise ValueError(
                f"No default backend registered for {operation_type!r} "
                f"operations and no explicit backend given; "
                f"available backends: {available}"
            )
        if backend_name not in self.backends:
            raise ValueError(
                f"Backend '{backend_name}' is not available; "
                f"available backends: {available}"
            )
        return backend_name

    def dispatch_indexing_operation(
        self,
        operation: str,
        *args: Any,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a spatial indexing operation to the appropriate backend."""
        backend_name = self._resolve_backend_name("indexing", backend)

        backend_instance = self.backends[backend_name]

        # Check backend supports indexing (duck typing)
        if not hasattr(backend_instance, "latlng_to_cell"):
            raise ValueError(
                f"Backend '{backend_name}' does not support indexing operations"
            )

        # Map operation names to backend methods
        operation_map = {
            "latlng_to_cell": "latlng_to_cell",
            "cell_to_latlng": "cell_to_latlng",
            "polygon_to_cells": "polygon_to_cells",
            "get_neighbors": "get_cell_neighbors",
            "get_cells_within_radius": "get_cells_within_radius",
            "get_distance": "get_cell_distance",
            "compact_cells": "compact_cells",
            "uncompact_cells": "uncompact_cells",
            "get_cell_parent": "get_cell_parent",
            "get_cell_children": "get_cell_children",
            "get_cell_path": "get_cell_path",
            "get_cell_ring": "get_cell_ring",
            "get_cell_resolution": "get_cell_resolution",
            "get_cell_boundary": "get_cell_boundary",
            "get_cell_area": "get_cell_area",
            "cells_to_multipolygon": "cells_to_multipolygon",
        }

        if operation not in operation_map:
            raise ValueError(f"Unknown indexing operation: {operation}")

        method_name = operation_map[operation]
        method = getattr(backend_instance, method_name, None)
        if method is None:
            raise ValueError(
                f"Backend '{backend_name}' does not implement indexing operation "
                f"'{operation}'"
            )

        return method(*args, **kwargs)

    def dispatch_geometric_operation(
        self,
        operation: str,
        *args: Any,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a geometry operation to a backend that implements it."""
        backend_name = self._resolve_backend_name("geometric", backend)

        operation_map = {
            "buffer_geometry": "buffer_geometry",
            "calculate_area": "calculate_area",
            "calculate_perimeter": "calculate_perimeter",
            "calculate_centroid": "calculate_centroid",
            "calculate_distance": "calculate_distance",
            "union_geometries": "union_geometries",
            "intersection_geometries": "intersection_geometries",
            "difference_geometries": "difference_geometries",
            "contains_geometry": "contains_geometry",
            "intersects_geometry": "intersects_geometry",
            "transform_geometry": "transform_geometry",
        }
        if operation not in operation_map:
            raise ValueError(f"Unknown geometric operation: {operation}")

        method_name = operation_map[operation]
        method = getattr(self.backends[backend_name], method_name, None)
        if method is None:
            raise ValueError(
                f"Backend '{backend_name}' does not implement geometric operation "
                f"'{operation}'"
            )
        return method(*args, **kwargs)

    def dispatch_analytics_operation(
        self,
        operation: str,
        *args: Any,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a spatial analytics operation to the appropriate backend."""
        backend_name = self._resolve_backend_name("analytics", backend)

        backend_instance = self.backends[backend_name]

        # Map operation names to backend methods
        operation_map = {
            "analyze_hotspots": "analyze_hotspots",
            "find_hotspots": "analyze_hotspots",
            "compute_proximity": "compute_proximity",
            "find_clusters": "find_clusters",
            "calculate_density": "calculate_density",
            "spatial_join": "spatial_join",
            "interpolate_values": "interpolate_values",
            "cluster_points": "cluster_points",
        }

        if operation not in operation_map:
            raise ValueError(f"Unknown analytics operation: {operation}")

        method_name = operation_map[operation]
        method = getattr(backend_instance, method_name, None)
        if method is None:
            raise ValueError(
                f"Backend '{backend_name}' does not implement analytics operation "
                f"'{operation}'"
            )

        return method(*args, **kwargs)

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about all registered backends."""
        return {
            name: {
                "available": backend.is_available(),
                "capabilities": self.backend_capabilities.get(name, {}),
                "version": getattr(backend, "version", "unknown"),
            }
            for name, backend in self.backends.items()
        }


# Global dispatcher instance
_dispatcher: SpatialBackendDispatcher | None = None


def get_backend_dispatcher() -> SpatialBackendDispatcher:
    """Get the global backend dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = SpatialBackendDispatcher()
    return _dispatcher


def configure_backends(config: dict[str, Any]) -> None:
    """Configure backend defaults from configuration."""
    dispatcher = get_backend_dispatcher()

    if "default_backends" in config:
        for operation_type, backend_name in config["default_backends"].items():
            dispatcher.set_default_backend(operation_type, backend_name)

    logger.info("Backend configuration applied")


def reset_dispatcher() -> None:
    """Reset the global dispatcher (mainly for testing)."""
    global _dispatcher
    _dispatcher = None
