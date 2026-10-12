"""
Parallel Processing Module

This module provides utilities for parallel computation and distributed processing
of geospatial data and mathematical operations.
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from typing import Any, cast
from collections.abc import Callable, Iterable
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)

# Global settings
DEFAULT_NUM_WORKERS = min(mp.cpu_count(), 8)  # Limit to 8 workers by default
MAX_CHUNK_SIZE = 10000  # Maximum chunk size for memory efficiency
MEMORY_THRESHOLD_MB = 1000  # Memory threshold for adaptive chunk sizing


def parallel_compute(
    func: Callable,
    data: list | np.ndarray,
    num_workers: int | None = None,
    chunk_size: int | None = None,
    use_processes: bool = True,
    max_memory_mb: float | None = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Apply a function to data in parallel with adaptive chunk sizing and memory monitoring.

    Args:
        func: Function to apply
        data: Input data (list or array)
        num_workers: Number of worker processes/threads
        chunk_size: Size of data chunks for each worker (auto-calculated if None)
        use_processes: Whether to use processes (True) or threads (False)
        max_memory_mb: Maximum memory per chunk in MB
        **kwargs: Additional arguments to pass to func

    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = get_optimal_worker_count(len(data))

    # Estimate memory usage and calculate optimal chunk size
    if chunk_size is None:
        chunk_size = _calculate_optimal_chunk_size(data, num_workers, max_memory_mb)

    # Convert to list if necessary
    data_list: list[Any]
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = list(data)

    # Split data into chunks with memory-aware sizing
    chunks = _create_memory_aware_chunks(data_list, chunk_size, max_memory_mb)

    logger.info(
        f"Processing {len(data_list)} items in {len(chunks)} chunks with {num_workers} workers"
    )

    # Create partial function with additional arguments
    partial_func = partial(func, **kwargs)

    # Choose executor type and submit tasks
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results = []
    start_time = time.time()

    with executor_class(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_chunk = {
            executor.submit(partial_func, chunk): i for i, chunk in enumerate(chunks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                chunk_idx = future_to_chunk[future]
                results.append((chunk_idx, result))

                # Log progress
                completed = len(results)
                if completed % max(1, len(chunks) // 10) == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{len(chunks)} chunks, {rate:.1f} chunks/s"
                    )

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                raise

    # Sort results by chunk index to maintain order
    results.sort(key=lambda x: x[0])
    ordered_results = [result for _, result in results]

    # Flatten results if function returns multiple values per chunk
    if ordered_results and isinstance(ordered_results[0], list):
        # Function returns one result per input item
        flat_results = []
        for result_list in ordered_results:
            flat_results.extend(result_list)
        return flat_results
    else:
        # Function returns one result per chunk
        return ordered_results


def _calculate_optimal_chunk_size(
    data: list | np.ndarray,
    num_workers: int,
    max_memory_mb: float | None = None,
) -> int:
    """Calculate optimal chunk size based on data size and memory constraints."""
    data_size = len(data)

    if max_memory_mb is None:
        max_memory_mb = MEMORY_THRESHOLD_MB

    # Estimate memory per item (rough heuristic)
    if isinstance(data, np.ndarray):
        bytes_per_item = (
            data.itemsize * data.shape[1] if data.ndim > 1 else data.itemsize
        )
    else:
        bytes_per_item = 100  # Assume 100 bytes per item for lists

    # Calculate chunk size to stay within memory limits
    max_items_per_worker = int(
        (max_memory_mb * 1024 * 1024) / (bytes_per_item * num_workers)
    )

    # Balance between memory efficiency and parallelization efficiency
    chunk_size = min(max_items_per_worker, max(1, data_size // num_workers))

    return chunk_size


def _create_memory_aware_chunks(
    data: list, chunk_size: int, max_memory_mb: float | None = None
) -> list[list]:
    """Create chunks that respect memory constraints."""
    if not data:
        return []

    # For very small datasets, don't chunk
    if len(data) <= chunk_size * 2:
        return [data]

    # Create chunks
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


def parallel_map(
    func: Callable,
    iterable: Iterable,
    num_workers: int | None = None,
    use_processes: bool = True,
) -> list[Any]:
    """
    Parallel version of map function.

    Args:
        func: Function to apply
        iterable: Input iterable
        num_workers: Number of workers
        use_processes: Whether to use processes

    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with executor_class(max_workers=num_workers) as executor:
        results = list(executor.map(func, iterable))

    return results


def parallel_matrix_operation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray | None = None,
    operation: str = "multiply",
    num_workers: int | None = None,
) -> np.ndarray:
    """
    Perform parallel matrix operations.

    Args:
        matrix_a: First matrix
        matrix_b: Second matrix (optional)
        operation: Operation type ('multiply', 'add', 'subtract', 'elementwise_multiply')
        num_workers: Number of workers

    Returns:
        Result matrix
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    if operation == "multiply":
        if matrix_b is None:
            raise ValueError("Matrix B required for multiplication")

        # Parallel matrix multiplication
        return parallel_matrix_multiply(matrix_a, matrix_b, num_workers)

    elif operation == "add":
        if matrix_b is None:
            raise ValueError("Matrix B required for addition")
        return cast(np.ndarray, matrix_a + matrix_b)

    elif operation == "subtract":
        if matrix_b is None:
            raise ValueError("Matrix B required for subtraction")
        return cast(np.ndarray, matrix_a - matrix_b)

    elif operation == "elementwise_multiply":
        if matrix_b is None:
            raise ValueError("Matrix B required for elementwise multiplication")
        return cast(np.ndarray, matrix_a * matrix_b)

    else:
        raise ValueError(f"Unknown operation: {operation}")


def parallel_matrix_multiply(
    matrix_a: np.ndarray, matrix_b: np.ndarray, num_workers: int | None = None
) -> np.ndarray:
    """
    Parallel matrix multiplication.

    Args:
        matrix_a: First matrix
        matrix_b: Second matrix
        num_workers: Number of workers

    Returns:
        Result matrix
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    m, k = matrix_a.shape
    k2, n = matrix_b.shape

    if k != k2:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    # For small matrices, use standard multiplication
    if m * n < 1000000:  # Less than 1M elements
        return cast(np.ndarray, np.dot(matrix_a, matrix_b))

    result = np.zeros((m, n))

    # Split computation by rows
    chunk_size = max(1, m // num_workers)

    def multiply_chunk(start_row: int, end_row: int) -> np.ndarray:
        return cast(np.ndarray, np.dot(matrix_a[start_row:end_row], matrix_b))

    chunks = [(i, min(i + chunk_size, m)) for i in range(0, m, chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(multiply_chunk, start, end) for start, end in chunks]

        for i, future in enumerate(as_completed(futures)):
            start_row = chunks[i][0]
            end_row = chunks[i][1]
            result[start_row:end_row] = future.result()

    return result


def parallel_distance_matrix(
    points_a: np.ndarray,
    points_b: np.ndarray | None = None,
    metric: str = "euclidean",
    num_workers: int | None = None,
) -> np.ndarray:
    """
    Compute distance matrix in parallel.

    Args:
        points_a: First set of points
        points_b: Second set of points (optional)
        metric: Distance metric
        num_workers: Number of workers

    Returns:
        Distance matrix
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    if points_b is None:
        points_b = points_a

    n_a, n_dims = points_a.shape
    n_b = points_b.shape[0]

    # For small matrices, use scipy's cdist
    if n_a * n_b < 1000000:  # Less than 1M elements
        from scipy.spatial.distance import cdist

        return cast(np.ndarray, cdist(points_a, points_b, metric=metric))

    # Parallel computation for large matrices
    result = np.zeros((n_a, n_b))

    chunk_size = max(1, n_a // num_workers)

    def distance_chunk(start_row: int, end_row: int) -> np.ndarray:
        chunk_points = points_a[start_row:end_row]
        from scipy.spatial.distance import cdist

        return cast(np.ndarray, cdist(chunk_points, points_b, metric=metric))

    chunks = [(i, min(i + chunk_size, n_a)) for i in range(0, n_a, chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(distance_chunk, start, end) for start, end in chunks]

        for i, future in enumerate(as_completed(futures)):
            start_row = chunks[i][0]
            end_row = chunks[i][1]
            result[start_row:end_row] = future.result()

    return result


def parallel_spatial_interpolation(
    known_points: np.ndarray,
    known_values: np.ndarray,
    query_points: np.ndarray,
    method: str = "idw",
    num_workers: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Perform spatial interpolation in parallel.

    Args:
        known_points: Known point coordinates
        known_values: Known point values
        query_points: Query point coordinates
        method: Interpolation method
        num_workers: Number of workers
        **kwargs: Method-specific parameters

    Returns:
        Interpolated values
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    n_query = len(query_points)

    # For small datasets, use serial computation
    if n_query < 1000:
        from ..core.numerical_methods import SpatialInterpolator

        interpolator = SpatialInterpolator(method=method)
        interpolator.fit(known_points, known_values, **kwargs)
        return interpolator.predict(query_points)

    # Parallel computation for large datasets
    chunk_size = max(1, n_query // num_workers)

    def interpolate_chunk(chunk_indices: np.ndarray) -> np.ndarray:
        chunk_points = query_points[chunk_indices]
        from ..core.numerical_methods import SpatialInterpolator

        interpolator = SpatialInterpolator(method=method)
        interpolator.fit(known_points, known_values, **kwargs)
        return interpolator.predict(chunk_points)

    chunks = [
        np.arange(i, min(i + chunk_size, n_query))
        for i in range(0, n_query, chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(interpolate_chunk, chunk) for chunk in chunks]

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # Combine results in correct order
    final_result = np.zeros(n_query)
    for i, chunk in enumerate(chunks):
        final_result[chunk] = results[i]

    return final_result


def parallel_statistical_analysis(
    data: np.ndarray,
    analysis_func: Callable,
    num_workers: int | None = None,
    **kwargs: Any,
) -> Any:
    """
    Perform statistical analysis in parallel.

    Args:
        data: Input data
        analysis_func: Analysis function
        num_workers: Number of workers
        **kwargs: Additional arguments

    Returns:
        Analysis result
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    # For simple analyses, use serial computation
    if len(data) < 10000:
        return analysis_func(data, **kwargs)

    # Split data into chunks for parallel processing
    chunk_size = max(1, len(data) // num_workers)
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    def analyze_chunk(chunk: np.ndarray) -> Any:
        return analysis_func(chunk, **kwargs)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_chunk, chunk) for chunk in chunks]

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # Combine results (this depends on the specific analysis function)
    # For now, return the first result as an example
    return results[0] if results else None


def get_optimal_worker_count(
    data_size: int, operation_complexity: str = "medium"
) -> int:
    """
    Determine optimal number of workers based on data size and operation complexity.

    Args:
        data_size: Size of the data
        operation_complexity: Complexity level ('low', 'medium', 'high')

    Returns:
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()

    # Complexity multipliers
    complexity_multiplier = {"low": 1.0, "medium": 0.7, "high": 0.5}

    multiplier = complexity_multiplier.get(operation_complexity, 0.7)

    # Base calculation
    if data_size < 1000:
        return 1
    elif data_size < 10000:
        return min(2, cpu_count)
    elif data_size < 100000:
        return min(int(cpu_count * multiplier), cpu_count)
    else:
        return min(int(cpu_count * multiplier * 0.8), cpu_count)


def parallel_file_processing(
    file_list: list[str],
    processing_func: Callable,
    num_workers: int | None = None,
    file_batch_size: int = 1,
) -> list[Any]:
    """
    Process multiple files in parallel.

    Args:
        file_list: List of file paths
        processing_func: Function to process each file
        num_workers: Number of workers
        file_batch_size: Number of files per batch

    Returns:
        List of processing results
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    # Group files into batches
    file_batches = [
        file_list[i : i + file_batch_size]
        for i in range(0, len(file_list), file_batch_size)
    ]

    def process_batch(batch: list[str]) -> list[Any]:
        return [processing_func(file_path) for file_path in batch]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in file_batches]

        results = []
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)

    return results


def memory_efficient_parallel(
    func: Callable,
    data: list | np.ndarray,
    max_memory_mb: float = 1000.0,
    num_workers: int | None = None,
) -> list[Any]:
    """
    Memory-efficient parallel processing.

    Args:
        func: Function to apply
        data: Input data
        max_memory_mb: Maximum memory usage in MB
        num_workers: Number of workers

    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    # Estimate memory usage per item
    if isinstance(data, np.ndarray):
        item_size_mb: float = float(
            data.itemsize * data.shape[1] if len(data.shape) > 1 else data.itemsize
        )
        item_size_mb /= 1024 * 1024  # Convert to MB
    else:
        item_size_mb = 0.001  # Assume 1KB per item

    # Calculate optimal chunk size based on memory constraints
    max_items_per_chunk = int(max_memory_mb / item_size_mb / num_workers)
    chunk_size = min(max_items_per_chunk, max(1, len(data) // num_workers))

    logger.info(f"Using chunk size {chunk_size} for memory-efficient processing")

    return parallel_compute(func, data, num_workers=num_workers, chunk_size=chunk_size)


__all__ = [
    "parallel_compute",
    "parallel_map",
    "parallel_matrix_operation",
    "parallel_matrix_multiply",
    "parallel_distance_matrix",
    "parallel_spatial_interpolation",
    "parallel_statistical_analysis",
    "get_optimal_worker_count",
    "parallel_file_processing",
    "memory_efficient_parallel",
    "DEFAULT_NUM_WORKERS",
    "MAX_CHUNK_SIZE",
]
