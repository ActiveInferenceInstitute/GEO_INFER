"""Optional float64 GPU numeric distances and authoritative host H3 topology.

Accelerators load only on a capability probe or numeric GPU request. NumPy CPU
execution is always available without accelerator packages. Availability
(``is_accelerator_available``) means a usable float64 GPU, not installation.
"""

from .gpu_acceleration import (
    AcceleratorUnavailableError,
    DEFAULT_CHUNK_SIZE,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
    euclidean_distance_kernel,
    get_available_backends,
    get_backend_diagnostics,
    gpu_spatial_join_by_distance,
    h3_grid_distance_kernel,
    is_accelerator_available,
    pairwise_haversine_kernel,
    spatial_join_kernel,
)

__version__ = "0.4.0"


__all__ = [
    "AcceleratorUnavailableError",
    "DEFAULT_CHUNK_SIZE",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "euclidean_distance_kernel",
    "get_available_backends",
    "get_backend_diagnostics",
    "gpu_spatial_join_by_distance",
    "h3_grid_distance_kernel",
    "is_accelerator_available",
    "pairwise_haversine_kernel",
    "spatial_join_kernel",
]
