"""Enhanced tensor operations for spatial AI models.

Provides spatial tensor operations that bridge core linear algebra
with AI/ML model requirements, including distance tensors,
spatial convolution kernels, and adjacency tensor construction.
"""

import numpy as np
from typing import Any, cast
import logging

# Import core tensor ops under a different name to avoid collision
from geo_infer_math.core.linalg_tensor import TensorOperations as _CoreTensorOperations

logger = logging.getLogger(__name__)


class SpatialTensorOperations:
    """Enhanced tensor operations for spatial AI models.

    Wraps core tensor operations and adds spatial-specific
    functionality for use in neural network and ML pipelines.
    """

    def __init__(self) -> None:
        """Initialize with core tensor operations backend."""
        self._core_ops = _CoreTensorOperations()
        logger.debug("SpatialTensorOperations initialized")

    def spatial_tensor_operation(
        self,
        tensor: np.ndarray,
        operation: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Dispatch a named operation to the core tensor backend.

        Args:
            tensor: Input tensor.
            operation: Name of the core TensorOperations method to call.
            **kwargs: Forwarded to the backend method.

        Returns:
            Result tensor.

        Raises:
            AttributeError: If the requested operation does not exist.
        """
        if not hasattr(self._core_ops, operation):
            raise AttributeError(
                f"Core TensorOperations has no method '{operation}'. "
                f"Available: {[m for m in dir(self._core_ops) if not m.startswith('_')]}"
            )
        logger.debug(
            "Dispatching tensor operation '%s' shape=%s", operation, tensor.shape
        )
        return cast(np.ndarray, getattr(self._core_ops, operation)(tensor, **kwargs))

    def compute_distance_tensor(
        self,
        coordinates: np.ndarray,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """Compute pairwise distance tensor from coordinates.

        Args:
            coordinates: (N, D) array of spatial coordinates.
            metric: Distance metric — 'euclidean' or 'manhattan'.

        Returns:
            (N, N) distance matrix.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        n = coordinates.shape[0]
        logger.debug("Computing distance tensor for %d points, metric=%s", n, metric)

        if metric == "euclidean":
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=-1))
        elif metric == "manhattan":
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            distances = np.sum(np.abs(diff), axis=-1)
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. Use 'euclidean' or 'manhattan'."
            )

        return cast(np.ndarray, distances)

    def build_adjacency_tensor(
        self,
        coordinates: np.ndarray,
        threshold: float | None = None,
        k_nearest: int | None = None,
    ) -> np.ndarray:
        """Build a spatial adjacency matrix from coordinates.

        Exactly one of ``threshold`` or ``k_nearest`` must be provided.

        Args:
            coordinates: (N, D) array of spatial coordinates.
            threshold: Distance threshold for adjacency.
            k_nearest: Number of nearest neighbours for adjacency.

        Returns:
            (N, N) binary adjacency matrix.
        """
        if (threshold is None) == (k_nearest is None):
            raise ValueError("Provide exactly one of 'threshold' or 'k_nearest'.")

        distances = self.compute_distance_tensor(coordinates)
        n = distances.shape[0]

        if threshold is not None:
            adjacency = (distances <= threshold).astype(np.float64)
            np.fill_diagonal(adjacency, 0.0)
            logger.debug(
                "Adjacency tensor (threshold=%.4f): %d edges",
                threshold,
                int(adjacency.sum()),
            )
        else:
            assert k_nearest is not None
            adjacency = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                indices = np.argsort(distances[i])
                # Skip self (index 0 after sort)
                neighbours = indices[1 : k_nearest + 1]
                adjacency[i, neighbours] = 1.0
            logger.debug(
                "Adjacency tensor (k=%d): %d edges", k_nearest, int(adjacency.sum())
            )

        return adjacency

    def spatial_convolution_kernel(
        self,
        size: int,
        kernel_type: str = "gaussian",
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Generate a 2-D spatial convolution kernel.

        Args:
            size: Kernel size (must be odd).
            kernel_type: 'gaussian', 'laplacian', or 'mean'.
            sigma: Standard deviation for Gaussian kernel.

        Returns:
            (size, size) kernel array, normalised to sum to 1 for
            'gaussian' and 'mean'.
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        center = size // 2
        y, x = np.ogrid[-center : center + 1, -center : center + 1]

        if kernel_type == "gaussian":
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel /= kernel.sum()
        elif kernel_type == "laplacian":
            kernel = np.zeros((size, size), dtype=np.float64)
            kernel[center, center] = -4.0
            if center > 0:
                kernel[center - 1, center] = 1.0
                kernel[center + 1, center] = 1.0
                kernel[center, center - 1] = 1.0
                kernel[center, center + 1] = 1.0
        elif kernel_type == "mean":
            kernel = np.ones((size, size), dtype=np.float64) / (size * size)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        logger.debug("Generated %s kernel size=%d", kernel_type, size)
        return cast(np.ndarray, kernel)
