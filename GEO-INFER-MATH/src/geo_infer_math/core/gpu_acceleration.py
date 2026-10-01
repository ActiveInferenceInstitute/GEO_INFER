"""
GPU Acceleration Module

This module provides GPU-accelerated implementations of core mathematical
operations for improved performance on large geospatial datasets.
"""

import numpy as np
from typing import Any, cast
import logging
import warnings

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """GPU acceleration manager for geospatial computations."""

    def __init__(self) -> None:
        """Initialize GPU accelerator."""
        self.gpu_available = self._check_gpu_availability()
        self.backends = self._detect_available_backends()

        if self.gpu_available:
            logger.info(
                f"GPU acceleration available with backends: {list(self.backends.keys())}"
            )
        else:
            logger.debug(
                "No GPU acceleration backends available; using CPU implementations"
            )

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Check for CuPy (NVIDIA GPU support)
            try:
                import importlib

                importlib.import_module("cupy")  # import for availability probe

                self.cupy_available = True
                logger.debug("CuPy (NVIDIA GPU) backend available")
            except ImportError:
                self.cupy_available = False
                logger.debug("CuPy not available")

            # Check for PyTorch (GPU support)
            try:
                import torch

                self.torch_available = (
                    torch.cuda.is_available() if torch.cuda.is_available() else False
                )
                if self.torch_available:
                    logger.debug("PyTorch GPU backend available")
            except ImportError:
                self.torch_available = False
                logger.debug("PyTorch not available")

            # Check for TensorFlow (GPU support)
            try:
                import tensorflow as tf

                self.tensorflow_available = (
                    len(tf.config.list_physical_devices("GPU")) > 0
                )
                if self.tensorflow_available:
                    logger.debug("TensorFlow GPU backend available")
            except ImportError:
                self.tensorflow_available = False
                logger.debug("TensorFlow not available")

            return (
                self.cupy_available or self.torch_available or self.tensorflow_available
            )

        except Exception as e:
            logger.debug(f"GPU availability check failed: {e}")
            return False

    def _detect_available_backends(self) -> dict[str, Any]:
        """Detect available GPU backends."""
        backends = {}

        try:
            import cupy as cp

            backends["cupy"] = cp
        except ImportError:
            pass

        try:
            import torch

            backends["torch"] = torch
        except ImportError:
            pass

        try:
            import tensorflow as tf

            backends["tensorflow"] = tf
        except ImportError:
            pass

        return backends

    def accelerate_matrix_operations(
        self, matrices: list[np.ndarray], operation: str = "multiply"
    ) -> list[np.ndarray]:
        """
        Accelerate matrix operations using GPU.

        Args:
            matrices: List of matrices to operate on
            operation: Operation type ('multiply', 'add', 'transpose', etc.)

        Returns:
            List of results
        """
        if not self.gpu_available:
            warnings.warn("GPU acceleration not available, using CPU")
            return self._cpu_matrix_operations(matrices, operation)

        # Use CuPy for NVIDIA GPU acceleration
        if self.backends.get("cupy") is not None:
            return self._cupy_matrix_operations(matrices, operation)

        # Use PyTorch for GPU acceleration
        elif self.backends.get("torch") is not None:
            return self._torch_matrix_operations(matrices, operation)

        # Fallback to CPU
        return self._cpu_matrix_operations(matrices, operation)

    def _cupy_matrix_operations(
        self, matrices: list[np.ndarray], operation: str
    ) -> list[np.ndarray]:
        """CuPy-based matrix operations."""
        import cupy as cp

        results = []

        if operation == "multiply":
            # Matrix multiplication
            if len(matrices) == 2:
                A_gpu = cp.asarray(matrices[0])
                B_gpu = cp.asarray(matrices[1])
                C_gpu = cp.matmul(A_gpu, B_gpu)
                results.append(cp.asnumpy(C_gpu))

        elif operation == "add":
            # Matrix addition
            if len(matrices) == 2:
                A_gpu = cp.asarray(matrices[0])
                B_gpu = cp.asarray(matrices[1])
                C_gpu = A_gpu + B_gpu
                results.append(cp.asnumpy(C_gpu))

        elif operation == "transpose":
            # Matrix transpose
            for matrix in matrices:
                A_gpu = cp.asarray(matrix)
                C_gpu = cp.transpose(A_gpu)
                results.append(cp.asnumpy(C_gpu))

        elif operation == "inverse":
            # Matrix inverse
            for matrix in matrices:
                A_gpu = cp.asarray(matrix)
                C_gpu = cp.linalg.inv(A_gpu)
                results.append(cp.asnumpy(C_gpu))

        return results

    def _torch_matrix_operations(
        self, matrices: list[np.ndarray], operation: str
    ) -> list[np.ndarray]:
        """PyTorch-based matrix operations."""
        import torch

        results = []

        if operation == "multiply":
            # Matrix multiplication
            if len(matrices) == 2:
                A_gpu = torch.from_numpy(matrices[0]).cuda()
                B_gpu = torch.from_numpy(matrices[1]).cuda()
                C_gpu = torch.matmul(A_gpu, B_gpu)
                results.append(C_gpu.cpu().numpy())

        elif operation == "add":
            # Matrix addition
            if len(matrices) == 2:
                A_gpu = torch.from_numpy(matrices[0]).cuda()
                B_gpu = torch.from_numpy(matrices[1]).cuda()
                C_gpu = A_gpu + B_gpu
                results.append(C_gpu.cpu().numpy())

        elif operation == "transpose":
            # Matrix transpose
            for matrix in matrices:
                A_gpu = torch.from_numpy(matrix).cuda()
                C_gpu = torch.t(A_gpu)
                results.append(C_gpu.cpu().numpy())

        return results

    def _cpu_matrix_operations(
        self, matrices: list[np.ndarray], operation: str
    ) -> list[np.ndarray]:
        """CPU-based matrix operations (fallback)."""
        results = []

        if operation == "multiply":
            # Matrix multiplication
            if len(matrices) == 2:
                results.append(np.dot(matrices[0], matrices[1]))

        elif operation == "add":
            # Matrix addition
            if len(matrices) == 2:
                results.append(matrices[0] + matrices[1])

        elif operation == "transpose":
            # Matrix transpose
            for matrix in matrices:
                results.append(matrix.T)

        elif operation == "inverse":
            # Matrix inverse
            for matrix in matrices:
                results.append(np.linalg.inv(matrix))

        return results

    def accelerate_distance_calculations(
        self, points1: np.ndarray, points2: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Accelerate distance matrix calculations using GPU.

        Args:
            points1: First set of points
            points2: Second set of points (optional)

        Returns:
            Distance matrix
        """
        p2 = points1 if points2 is None else points2

        if not self.gpu_available:
            warnings.warn("GPU acceleration not available, using CPU")
            return self._cpu_distance_calculation(points1, p2)

        # Use CuPy for distance calculations
        if self.backends.get("cupy") is not None:
            return self._cupy_distance_calculation(points1, p2)

        # Use PyTorch for distance calculations
        elif self.backends.get("torch") is not None:
            return self._torch_distance_calculation(points1, p2)

        # Fallback to CPU
        return self._cpu_distance_calculation(points1, p2)

    def _cupy_distance_calculation(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """CuPy-based distance calculation."""
        import cupy as cp

        points1_gpu = cp.asarray(points1)
        points2_gpu = cp.asarray(points2)

        # Vectorized distance calculation
        diff = points1_gpu[:, cp.newaxis, :] - points2_gpu[cp.newaxis, :, :]
        distances_gpu = cp.sqrt(cp.sum(diff**2, axis=2))

        return cast(np.ndarray, np.asarray(cp.asnumpy(distances_gpu)))

    def _torch_distance_calculation(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """PyTorch-based distance calculation."""
        import torch

        points1_gpu = torch.from_numpy(points1).cuda()
        points2_gpu = torch.from_numpy(points2).cuda()

        # Vectorized distance calculation
        diff = points1_gpu.unsqueeze(1) - points2_gpu.unsqueeze(0)
        distances_gpu = torch.sqrt(torch.sum(diff**2, dim=2))

        return distances_gpu.cpu().numpy()

    def _cpu_distance_calculation(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """CPU-based distance calculation (fallback)."""
        # Use scipy's optimized distance calculation
        from scipy.spatial.distance import cdist

        return cast(np.ndarray, np.asarray(cdist(points1, points2, metric="euclidean")))

    def accelerate_spatial_interpolation(
        self,
        known_points: np.ndarray,
        known_values: np.ndarray,
        query_points: np.ndarray,
        method: str = "idw",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Accelerate spatial interpolation using GPU.

        Args:
            known_points: Known point coordinates
            known_values: Known values
            query_points: Query points for interpolation
            method: Interpolation method
            **kwargs: Additional parameters

        Returns:
            Interpolated values
        """
        if not self.gpu_available:
            warnings.warn("GPU acceleration not available, using CPU")
            return self._cpu_spatial_interpolation(
                known_points, known_values, query_points, method, **kwargs
            )

        # Use CuPy for spatial interpolation
        if self.backends.get("cupy") is not None:
            return self._cupy_spatial_interpolation(
                known_points, known_values, query_points, method, **kwargs
            )

        # Use PyTorch for spatial interpolation
        elif self.backends.get("torch") is not None:
            return self._torch_spatial_interpolation(
                known_points, known_values, query_points, method, **kwargs
            )

        # Fallback to CPU
        return self._cpu_spatial_interpolation(
            known_points, known_values, query_points, method, **kwargs
        )

    def _cupy_spatial_interpolation(
        self,
        known_points: np.ndarray,
        known_values: np.ndarray,
        query_points: np.ndarray,
        method: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """CuPy-based spatial interpolation."""
        import cupy as cp

        if method == "idw":
            power = kwargs.get("power", 2)

            # Convert to GPU arrays
            known_points_gpu = cp.asarray(known_points)
            known_values_gpu = cp.asarray(known_values)
            query_points_gpu = cp.asarray(query_points)

            # Calculate distances
            diff = (
                known_points_gpu[:, cp.newaxis, :] - query_points_gpu[cp.newaxis, :, :]
            )
            distances_gpu = cp.sqrt(cp.sum(diff**2, axis=2))

            # IDW weights
            weights_gpu = 1.0 / (distances_gpu + 1e-10) ** power

            # Normalize weights
            weight_sums = cp.sum(weights_gpu, axis=0)
            weights_gpu = weights_gpu / weight_sums[cp.newaxis, :]

            # Calculate interpolated values
            interpolated_gpu = cp.sum(
                weights_gpu * known_values_gpu[:, cp.newaxis], axis=0
            )

            return np.asarray(cp.asnumpy(interpolated_gpu))

        else:
            # Fallback for unsupported methods
            return self._cpu_spatial_interpolation(
                known_points, known_values, query_points, method, **kwargs
            )

    def _torch_spatial_interpolation(
        self,
        known_points: np.ndarray,
        known_values: np.ndarray,
        query_points: np.ndarray,
        method: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """PyTorch-based spatial interpolation."""
        import torch

        if method == "idw":
            power = kwargs.get("power", 2)

            # Convert to GPU tensors
            known_points_gpu = torch.from_numpy(known_points).cuda()
            known_values_gpu = torch.from_numpy(known_values).cuda()
            query_points_gpu = torch.from_numpy(query_points).cuda()

            # Calculate distances
            diff = known_points_gpu.unsqueeze(1) - query_points_gpu.unsqueeze(0)
            distances_gpu = torch.sqrt(torch.sum(diff**2, dim=2))

            # IDW weights
            weights_gpu = 1.0 / (distances_gpu + 1e-10) ** power

            # Normalize weights
            weight_sums = torch.sum(weights_gpu, dim=0)
            weights_gpu = weights_gpu / weight_sums.unsqueeze(0)

            # Calculate interpolated values
            interpolated_gpu = torch.sum(
                weights_gpu * known_values_gpu.unsqueeze(1), dim=0
            )

            return interpolated_gpu.cpu().numpy()

        else:
            # Fallback for unsupported methods
            return self._cpu_spatial_interpolation(
                known_points, known_values, query_points, method, **kwargs
            )

    def _cpu_spatial_interpolation(
        self,
        known_points: np.ndarray,
        known_values: np.ndarray,
        query_points: np.ndarray,
        method: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """CPU-based spatial interpolation (fallback)."""
        from geo_infer_math.core.numerical_methods import SpatialInterpolator

        interpolator = SpatialInterpolator(method=method)
        interpolator.fit(known_points, known_values, **kwargs)
        return interpolator.predict(query_points)

    def accelerate_clustering(
        self,
        data: np.ndarray,
        coordinates: np.ndarray,
        method: str = "kmeans",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Accelerate clustering operations using GPU.

        Args:
            data: Feature data
            coordinates: Spatial coordinates
            method: Clustering method
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        if not self.gpu_available:
            warnings.warn("GPU acceleration not available, using CPU")
            return self._cpu_clustering(data, coordinates, method, **kwargs)

        # Use CuPy for clustering
        if self.backends.get("cupy") is not None:
            return self._cupy_clustering(data, coordinates, method, **kwargs)

        # Use PyTorch for clustering
        elif self.backends.get("torch") is not None:
            return self._torch_clustering(data, coordinates, method, **kwargs)

        # Fallback to CPU
        return self._cpu_clustering(data, coordinates, method, **kwargs)

    def _cupy_clustering(
        self, data: np.ndarray, coordinates: np.ndarray, method: str, **kwargs: Any
    ) -> dict[str, Any]:
        """CuPy-based clustering."""
        # For now, use CPU fallback for clustering
        # CuPy doesn't have built-in clustering algorithms like scikit-learn
        return self._cpu_clustering(data, coordinates, method, **kwargs)

    def _torch_clustering(
        self, data: np.ndarray, coordinates: np.ndarray, method: str, **kwargs: Any
    ) -> dict[str, Any]:
        """PyTorch-based clustering."""
        # For now, use CPU fallback for clustering
        # Would require implementing custom clustering algorithms
        return self._cpu_clustering(data, coordinates, method, **kwargs)

    def _cpu_clustering(
        self, data: np.ndarray, coordinates: np.ndarray, method: str, **kwargs: Any
    ) -> dict[str, Any]:
        """CPU-based clustering (fallback)."""
        from geo_infer_math.models.clustering import spatial_clustering_analysis

        return cast(
            dict[str, Any],
            spatial_clustering_analysis(data, coordinates, method=method, **kwargs),
        )

    def get_performance_info(self) -> dict[str, Any]:
        """
        Get information about GPU acceleration performance.

        Returns:
            Dictionary with performance information
        """
        info: dict[str, Any] = {
            "gpu_available": self.gpu_available,
            "backends": list(self.backends.keys()),
            "memory_info": {},
        }

        mem_dict: dict[str, Any] = info["memory_info"]
        if self.backends.get("cupy") is not None:
            try:
                import cupy as cp

                mem_info = cp.get_default_memory_pool().get_limit()
                mem_dict["cupy_limit"] = mem_info
            except Exception:
                pass

        if self.backends.get("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    mem_dict["torch_allocated"] = torch.cuda.memory_allocated(device)
                    mem_dict["torch_reserved"] = torch.cuda.memory_reserved(device)
            except Exception:
                pass

        return info

    def benchmark_acceleration(
        self, test_data: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """
        Benchmark GPU vs CPU performance.

        Args:
            test_data: Test data for benchmarking

        Returns:
            Performance comparison results
        """
        import time

        results: dict[str, dict[str, Any]] = {
            "matrix_multiplication": {},
            "distance_calculation": {},
            "spatial_interpolation": {},
        }

        # Matrix multiplication benchmark
        if "matrix_a" in test_data and "matrix_b" in test_data:
            matrices = [test_data["matrix_a"], test_data["matrix_b"]]

            # CPU timing
            start_time = time.time()
            _cpu_result = self._cpu_matrix_operations(matrices, "multiply")
            cpu_time = time.time() - start_time
            results["matrix_multiplication"]["cpu_time"] = cpu_time

            # GPU timing
            if self.gpu_available:
                start_time = time.time()
                _gpu_result = self.accelerate_matrix_operations(matrices, "multiply")
                gpu_time = time.time() - start_time
                results["matrix_multiplication"]["gpu_time"] = gpu_time
                results["matrix_multiplication"]["speedup"] = (
                    cpu_time / gpu_time if gpu_time > 0 else float("inf")
                )

        # Distance calculation benchmark
        if "points1" in test_data and "points2" in test_data:
            # CPU timing
            start_time = time.time()
            _cpu_distance_res = self._cpu_distance_calculation(
                test_data["points1"], test_data["points2"]
            )
            cpu_time = time.time() - start_time
            results["distance_calculation"]["cpu_time"] = cpu_time

            # GPU timing
            if self.gpu_available:
                start_time = time.time()
                _gpu_distance_res = self.accelerate_distance_calculations(
                    test_data["points1"], test_data["points2"]
                )
                gpu_time = time.time() - start_time
                results["distance_calculation"]["gpu_time"] = gpu_time
                results["distance_calculation"]["speedup"] = (
                    cpu_time / gpu_time if gpu_time > 0 else float("inf")
                )

        return results


# Global GPU accelerator instance
gpu_accelerator = GPUAccelerator()


# Convenience functions
def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return gpu_accelerator.gpu_available


def get_gpu_info() -> dict[str, Any]:
    """Get GPU acceleration information."""
    return gpu_accelerator.get_performance_info()


def benchmark_gpu_performance(test_data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Benchmark GPU vs CPU performance."""
    return gpu_accelerator.benchmark_acceleration(test_data)


# GPU-accelerated versions of common operations
def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU-accelerated matrix multiplication."""
    matrices: list[np.ndarray] = [a, b]
    return gpu_accelerator.accelerate_matrix_operations(matrices, "multiply")[0]


def gpu_distance_matrix(
    points1: np.ndarray, points2: np.ndarray | None = None
) -> np.ndarray:
    """GPU-accelerated distance matrix calculation."""
    if points2 is not None:
        points_list: list[np.ndarray] = [points1, points2]
        return gpu_accelerator.accelerate_distance_calculations(
            points_list[0], points_list[1]
        )
    return gpu_accelerator.accelerate_distance_calculations(points1)


def gpu_spatial_interpolation(
    known_points: np.ndarray,
    known_values: np.ndarray,
    query_points: np.ndarray,
    method: str = "idw",
    **kwargs: Any,
) -> np.ndarray:
    """GPU-accelerated spatial interpolation."""
    return gpu_accelerator.accelerate_spatial_interpolation(
        known_points, known_values, query_points, method, **kwargs
    )


__all__ = [
    "GPUAccelerator",
    "gpu_accelerator",
    "is_gpu_available",
    "get_gpu_info",
    "benchmark_gpu_performance",
    "gpu_matrix_multiply",
    "gpu_distance_matrix",
    "gpu_spatial_interpolation",
]
