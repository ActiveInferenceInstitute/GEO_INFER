"""
Spatial Backend Interfaces for GEO-INFER-SPACE.

This module consolidates all protocol and interface definitions for spatial backends.
Backends implementing these interfaces can be registered with the dispatcher for
unified spatial operations across different spatial indexing systems.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpatialBackendProtocol(Protocol):
    """
    Protocol defining the base interface that all spatial backends must implement.

    This protocol ensures consistent behavior across H3, SRAI, and future backends.
    """

    @property
    def name(self) -> str:
        """Return the backend name identifier."""
        ...

    @property
    def version(self) -> str:
        """Return the backend version string."""
        ...

    def is_available(self) -> bool:
        """Check if the backend is available and functional."""
        ...

    def get_capabilities(self) -> dict[str, Any]:
        """
        Return the backend's capabilities as a structured dictionary.

        Returns:
            Dictionary with capability categories (indexing, analytics, geometric)
            and their supported operations.
        """
        ...


@runtime_checkable
class IndexingBackendProtocol(SpatialBackendProtocol, Protocol):
    """
    Protocol for spatial indexing backends.

    Defines the required methods for converting between coordinates and spatial cells,
    handling polygons, computing neighbors, and managing cell hierarchies.
    """

    def latlng_to_cell(self, lat: float, lng: float, resolution: int) -> str:
        """
        Convert latitude/longitude coordinates to a spatial index cell.

        Args:
            lat: Latitude coordinate (-90 to 90)
            lng: Longitude coordinate (-180 to 180)
            resolution: Resolution level for the spatial index (0-15 for H3)

        Returns:
            Spatial index cell identifier as a string

        Raises:
            ValueError: If coordinates are out of valid range
            RuntimeError: If backend is not available
        """
        ...

    def cell_to_latlng(self, cell: str) -> tuple[float, float]:
        """
        Convert a spatial index cell back to latitude/longitude coordinates.

        Args:
            cell: Spatial index cell identifier

        Returns:
            Tuple of (latitude, longitude) coordinates

        Raises:
            ValueError: If cell identifier is invalid
            RuntimeError: If backend is not available
        """
        ...

    def polygon_to_cells(self, polygon: dict[str, Any], resolution: int) -> list[str]:
        """
        Convert a polygon geometry to a list of spatial index cells.

        Args:
            polygon: Polygon geometry as GeoJSON-like dictionary
            resolution: Resolution level for the spatial index

        Returns:
            List of spatial index cell identifiers covering the polygon

        Raises:
            ValueError: If polygon format is invalid
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_neighbors(self, cell: str, k: int = 1) -> list[str]:
        """
        Get neighboring cells around a given cell.

        Args:
            cell: Central spatial index cell
            k: Number of rings of neighbors to return (default 1)

        Returns:
            List of neighboring cell identifiers

        Raises:
            ValueError: If cell identifier is invalid
            RuntimeError: If backend is not available
        """
        ...

    def get_cells_within_radius(self, cell: str, k: int = 1) -> list[str]:
        """Return all cells within ``k`` grid rings, excluding the center."""
        ...

    def get_cell_distance(self, cell1: str, cell2: str) -> int:
        """
        Calculate the grid distance between two spatial index cells.

        Args:
            cell1: First spatial index cell
            cell2: Second spatial index cell

        Returns:
            Distance between cells in grid units

        Raises:
            ValueError: If cell identifiers are invalid or incompatible
            RuntimeError: If backend is not available
        """
        ...

    def compact_cells(self, cells: list[str]) -> list[str]:
        """
        Compact a list of cells into a more efficient representation.

        Args:
            cells: List of spatial index cell identifiers

        Returns:
            Compacted list of cell identifiers at mixed resolutions
        """
        ...

    def uncompact_cells(self, compacted_cells: list[str], resolution: int) -> list[str]:
        """
        Uncompact cells back to individual cell identifiers at target resolution.

        Args:
            compacted_cells: Compacted cell identifiers
            resolution: Target resolution level

        Returns:
            List of individual cell identifiers at the target resolution
        """
        ...

    def get_cell_resolution(self, cell: str) -> int:
        """
        Get the resolution level of a spatial index cell.

        Args:
            cell: Spatial index cell identifier

        Returns:
            Resolution level (0-15 for H3)

        Raises:
            ValueError: If cell identifier is invalid
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_boundary(self, cell: str) -> list[tuple[float, float]]:
        """
        Get the boundary coordinates of a spatial index cell.

        Args:
            cell: Spatial index cell identifier

        Returns:
            List of (latitude, longitude) tuples forming the cell boundary

        Raises:
            ValueError: If cell identifier is invalid
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_area(self, cell: str) -> float:
        """
        Get the area of a spatial index cell in square kilometers.

        Args:
            cell: Spatial index cell identifier

        Returns:
            Area in km²

        Raises:
            ValueError: If cell identifier is invalid
            RuntimeError: If backend is not available
        """
        ...

    def cells_to_multipolygon(self, cells: list[str]) -> dict[str, Any]:
        """
        Convert a list of cells to a GeoJSON MultiPolygon geometry.

        Args:
            cells: List of spatial index cell identifiers

        Returns:
            GeoJSON-like dictionary with 'type' and 'coordinates'

        Raises:
            ValueError: If cell identifiers are invalid
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_parent(self, cell: str, resolution: int) -> str:
        """
        Get the parent of a cell at a coarser resolution.

        Args:
            cell: Spatial index cell identifier
            resolution: Target resolution (must be less than current cell resolution)

        Returns:
            Parent cell identifier

        Raises:
            ValueError: If resolutions are incompatible
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_children(self, cell: str, resolution: int) -> list[str]:
        """
        Get children of a cell at a finer resolution.

        Args:
            cell: Spatial index cell identifier
            resolution: Target resolution (must be greater than current cell resolution)

        Returns:
            List of child cell identifiers

        Raises:
            ValueError: If resolutions are incompatible
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_path(self, start_cell: str, end_cell: str) -> list[str]:
        """
        Get the path of cells between two cells.

        Args:
            start_cell: Start cell identifier
            end_cell: End cell identifier

        Returns:
            List of cell identifiers in the path (inclusive)

        Raises:
            ValueError: If cells are invalid or disconnected
            RuntimeError: If backend is not available
        """
        ...

    def get_cell_ring(self, cell: str, k: int) -> list[str]:
        """
        Get the ring of cells at distance k.

        Args:
            cell: Center cell identifier
            k: Distance in grid steps

        Returns:
            List of cell identifiers in the ring

        Raises:
            ValueError: If cell is invalid
            RuntimeError: If backend is not available
        """
        ...


@runtime_checkable
class AnalyticsBackendProtocol(SpatialBackendProtocol, Protocol):
    """
    Protocol for spatial analytics backends.

    Defines methods for spatial analysis operations like hotspot detection,
    proximity analysis, clustering, and spatial interpolation.
    """

    def analyze_hotspots(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze spatial hotspots in the data.

        Args:
            data: Dictionary containing 'cells' (list of cell IDs) and
                  'values' (corresponding numeric values)

        Returns:
            Dictionary with hotspot analysis results including identified
            hotspots, threshold used, and statistics

        Raises:
            ValueError: If data format is invalid
            RuntimeError: If backend is not available
        """
        ...

    def compute_proximity(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        """
        Compute proximity analysis between points.

        Args:
            points: List of (latitude, longitude) coordinate tuples

        Returns:
            Dictionary with proximity analysis results including
            pairwise distances and statistics

        Raises:
            ValueError: If points are invalid
            RuntimeError: If backend is not available
        """
        ...

    def find_clusters(
        self,
        cells: list[str],
        values: list[float],
        min_cluster_size: int = 3,
        distance_threshold: int = 1,
    ) -> dict[str, Any]:
        """
        Find spatial clusters of cells based on values and proximity.

        Args:
            cells: List of spatial cell identifiers
            values: Corresponding values for each cell
            min_cluster_size: Minimum number of cells to form a cluster
            distance_threshold: Maximum grid distance between cluster members

        Returns:
            Dictionary with:
                - clusters: List of cluster dictionaries with cells and stats
                - num_clusters: Number of clusters found
                - noise_cells: Cells not belonging to any cluster

        Raises:
            ValueError: If cells and values have different lengths
            RuntimeError: If backend is not available
        """
        ...

    def calculate_density(
        self, cells: list[str], values: list[float], kernel_radius: int = 1
    ) -> dict[str, Any]:
        """
        Calculate density values across cells using kernel smoothing.

        Args:
            cells: List of spatial cell identifiers
            values: Values at each cell location
            kernel_radius: Radius for kernel density estimation in grid steps

        Returns:
            Dictionary with:
                - densities: Dictionary mapping cell -> density value
                - statistics: Mean, max, min density values

        Raises:
            ValueError: If cells and values have different lengths
            RuntimeError: If backend is not available
        """
        ...

    def spatial_join(
        self, cells_a: list[str], cells_b: list[str], join_type: str = "intersects"
    ) -> dict[str, Any]:
        """
        Join two sets of cells based on spatial relationships.

        Args:
            cells_a: First set of spatial cell identifiers
            cells_b: Second set of spatial cell identifiers
            join_type: Type of join ('intersects', 'contains', 'within')

        Returns:
            Dictionary with:
                - matches: List of (cell_a, cell_b) pairs that satisfy the relationship
                - unmatched_a: Cells from A with no matches
                - unmatched_b: Cells from B with no matches

        Raises:
            ValueError: If join_type is invalid
            RuntimeError: If backend is not available
        """
        ...

    def interpolate_values(
        self,
        cells: list[str],
        values: list[float],
        target_cells: list[str],
        method: str = "idw",
    ) -> dict[str, Any]:
        """
        Interpolate values at target cell locations.

        Args:
            cells: List of cells with known values
            values: Known values at each cell location
            target_cells: Cells where values should be interpolated
            method: Interpolation method ('idw', 'nearest', 'linear')

        Returns:
            Dictionary with:
                - interpolated: Dictionary mapping target_cell -> interpolated value
                - method: Method used
                - source_count: Number of source cells used

        Raises:
            ValueError: If cells and values have different lengths
            RuntimeError: If backend is not available
        """
        ...


class H3UnavailableError(RuntimeError):
    """
    Raised when H3 library is required but not available.

    This error indicates that the H3 library needs to be installed
    to perform the requested operation.
    """

    def __init__(self, operation: str = "spatial operation") -> None:
        super().__init__(
            f"H3 library is required for {operation}. Install it with: pip install h3"
        )
        self.operation = operation


class SRAIUnavailableError(RuntimeError):
    """
    Raised when SRAI library is required but not available.

    This error indicates that the SRAI library needs to be installed
    to perform the requested operation.
    """

    def __init__(self, operation: str = "spatial operation") -> None:
        super().__init__(
            f"SRAI library is required for {operation}. "
            "Install it with: pip install srai"
        )
        self.operation = operation


class BackendNotAvailableError(RuntimeError):
    """
    Raised when a requested backend is not available.
    """

    def __init__(self, backend_name: str, available_backends: list[str]) -> None:
        available_str = ", ".join(available_backends) if available_backends else "none"
        super().__init__(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available_str}"
        )
        self.backend_name = backend_name
        self.available_backends = available_backends


class UnsupportedSpatialOperationError(ValueError):
    """Raised when a public spatial facade operation has no backend support."""

    def __init__(self, operation: str, backend: str) -> None:
        super().__init__(
            f"Spatial operation '{operation}' is not supported by backend '{backend}'"
        )
        self.operation = operation
        self.backend = backend
