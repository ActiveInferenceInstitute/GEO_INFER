"""
Performance optimization utilities for geospatial art generation.
"""

import logging
import os
import time
import functools
from typing import Any, cast
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Performance optimization utilities for geospatial art generation.

    This class provides tools for optimizing the performance of art generation
    processes through caching, parallel processing, and resource management.
    """

    def __init__(self, cache_dir: str = ".art_cache", max_cache_size: int = 100):
        """
        Initialize performance optimizer.

        Args:
            cache_dir: Directory for caching generated results
            max_cache_size: Maximum number of cached items
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache: dict[str, Any] = {}
        self.cache_timestamps: dict[str, float] = {}
        self.cache_sizes: dict[str, int] = {}

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Load existing cache metadata
        self._load_cache_metadata()

    def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")

        if os.path.exists(metadata_file):
            try:
                import json

                with open(metadata_file) as f:
                    metadata = json.load(f)

                self.cache = metadata.get("cache", {})
                self.cache_timestamps = metadata.get("timestamps", {})
                self.cache_sizes = metadata.get("sizes", {})

                # Clean up old cache entries
                self._cleanup_cache()

            except Exception:
                # Reset cache if metadata is corrupted
                self.cache = {}
                self.cache_timestamps = {}
                self.cache_sizes = {}

    def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")

        try:
            import json

            metadata = {
                "cache": self.cache,
                "timestamps": self.cache_timestamps,
                "sizes": self.cache_sizes,
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception:
            # Cache metadata persistence is best-effort; keep serving from memory.
            logger.warning("Failed to save cache metadata to %s", metadata_file)

    def _cleanup_cache(self) -> None:
        """Clean up old cache entries to maintain size limits."""
        if len(self.cache) <= self.max_cache_size:
            return

        # Sort by timestamp (oldest first)
        sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])

        # Remove oldest entries until under limit
        while len(self.cache) > self.max_cache_size:
            oldest_key = sorted_items.pop(0)[0]
            self._remove_cache_entry(oldest_key)

    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry."""
        if key in self.cache:
            # Remove file if it exists
            cache_file = self.cache[key]
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except Exception as exc:
                    logger.warning(
                        "Could not remove cache file %s: %s", cache_file, exc
                    )

            # Remove from memory
            del self.cache[key]
            del self.cache_timestamps[key]
            del self.cache_sizes[key]

    def get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for function call."""
        import hashlib

        # Create a string representation of the call
        call_str = f"{func_name}:{args}:{sorted(kwargs.items())}"

        # Hash the string
        return hashlib.md5(call_str.encode()).hexdigest()

    def cached_execution(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        cache_key: str | None = None,
    ) -> Any:
        """
        Execute a function with caching.

        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            cache_key: Custom cache key (auto-generated if None)

        Returns:
            Function result (cached or newly computed)
        """
        kwargs = kwargs or {}

        if cache_key is None:
            cache_key = self.get_cache_key(func.__name__, args, kwargs)

        # Check cache first
        if cache_key in self.cache:
            cache_file = self.cache[cache_key]

            try:
                # Load cached result
                if cache_file.endswith(".npy"):
                    return np.load(cache_file)
                else:
                    # For other formats, return the file path
                    return cache_file

            except Exception:
                # Cache file corrupted, remove it
                self._remove_cache_entry(cache_key)

        # Execute function and cache result
        try:
            result = func(*args, **kwargs)

            # Cache the result
            if isinstance(result, np.ndarray):
                # Save numpy arrays
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
                np.save(cache_file, result)

                self.cache[cache_key] = cache_file
                self.cache_timestamps[cache_key] = time.time()
                self.cache_sizes[cache_key] = result.nbytes

            elif hasattr(result, "save"):  # PIL Image or matplotlib figure
                # Save images
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.png")
                result.savefig(cache_file, dpi=150, bbox_inches="tight")

                self.cache[cache_key] = cache_file
                self.cache_timestamps[cache_key] = time.time()
                self.cache_sizes[cache_key] = os.path.getsize(cache_file)

                # Return the file path instead of the object
                return cache_file

            else:
                # Return result as-is for non-cacheable types
                return result

            # Save metadata
            self._save_cache_metadata()

            return result

        except Exception as e:
            # Remove failed cache entry
            if cache_key in self.cache:
                self._remove_cache_entry(cache_key)
            raise e

    def parallel_execution(
        self,
        func: Callable,
        parameter_sets: list[dict],
        max_workers: int | None = None,
        progress_callback: Callable | None = None,
    ) -> list[Any]:
        """
        Execute a function in parallel with different parameter sets.

        Args:
            func: Function to execute
            parameter_sets: List of parameter dictionaries
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback for progress updates

        Returns:
            List of function results
        """
        results = [None] * len(parameter_sets)

        if max_workers is None:
            max_workers = min(len(parameter_sets), os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, **params): i
                for i, params in enumerate(parameter_sets)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]

                try:
                    results[index] = future.result()

                    if progress_callback:
                        progress_callback(index + 1, len(parameter_sets))

                except Exception as e:
                    logger.error("Error in parallel execution %d: %s", index, e)
                    results[index] = None

        return results

    def benchmark_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark a function's performance.

        Args:
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing statistics
        """
        kwargs = kwargs or {}

        times = []

        for _ in range(iterations):
            start_time = time.time()
            try:
                func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.warning("Benchmark iteration failed: %s", e)
                continue

        if not times:
            return {"error": "All benchmark iterations failed"}

        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "iterations": len(times),
        }

    def optimize_resolution(
        self,
        target_time: float = 1.0,
        min_resolution: int = 100,
        max_resolution: int = 2000,
        test_function: Callable | None = None,
        test_args: tuple = (),
        test_kwargs: dict | None = None,
    ) -> int:
        """
        Find optimal resolution for target execution time.

        Args:
            target_time: Target execution time in seconds
            min_resolution: Minimum resolution to test
            max_resolution: Maximum resolution to test
            test_function: Function to test (defaults to a simple computation)
            test_args: Arguments for test function
            test_kwargs: Keyword arguments for test function

        Returns:
            Optimal resolution
        """
        test_kwargs = test_kwargs or {}

        if test_function is None:
            # Simple test function
            def test_function(resolution: int) -> Any:
                arr = np.random.rand(resolution, resolution)
                return np.sum(arr)

        assert test_function is not None
        left, right = min_resolution, max_resolution
        best_resolution = min_resolution

        while left <= right:
            mid = (left + right) // 2

            # Test current resolution
            try:
                start_time = time.time()
                test_function(mid, *test_args, **test_kwargs)
                execution_time = time.time() - start_time

                if execution_time <= target_time:
                    best_resolution = mid
                    left = mid + 1
                else:
                    right = mid - 1

            except Exception:
                # If test fails, try smaller resolution
                right = mid - 1

        return best_resolution

    def memory_efficient_processing(
        self,
        data: np.ndarray,
        chunk_size: int = 1000,
        process_function: Callable | None = None,
    ) -> np.ndarray:
        """
        Process large arrays in chunks to manage memory usage.

        Args:
            data: Input array to process
            chunk_size: Size of chunks to process
            process_function: Function to apply to each chunk

        Returns:
            Processed array
        """
        if process_function is None:
            # Default: just return the data (no processing needed)
            return data

        # Determine chunk dimensions
        if data.ndim == 2:
            height, width = data.shape
            chunks = []

            # Process in horizontal strips
            for i in range(0, height, chunk_size):
                end_i = min(i + chunk_size, height)
                chunk = data[i:end_i, :]
                processed_chunk = process_function(chunk)
                chunks.append(processed_chunk)

            return np.vstack(chunks)

        elif data.ndim == 3:
            depth, height, width = data.shape
            chunks = []

            # Process in depth slices
            for i in range(0, depth, chunk_size):
                end_i = min(i + chunk_size, depth)
                chunk = data[i:end_i, :, :]
                processed_chunk = process_function(chunk)
                chunks.append(processed_chunk)

            return np.concatenate(chunks, axis=0)

        else:
            # For other dimensions, process as-is
            return cast(np.ndarray, process_function(data))

    def create_performance_report(self) -> dict[str, Any]:
        """
        Create a comprehensive performance report.

        Returns:
            Dictionary with performance metrics
        """
        report = {
            "cache_stats": {
                "entries": len(self.cache),
                "total_size_bytes": sum(self.cache_sizes.values()),
                "max_entries": self.max_cache_size,
            },
            "system_info": {
                "cpu_count": os.cpu_count(),
                "available_memory": self._get_available_memory(),
            },
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except ImportError:
            # Fallback for systems without psutil
            return 0

    def _generate_recommendations(self) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Cache recommendations
        cache_usage = len(self.cache) / max(self.max_cache_size, 1)
        if cache_usage < 0.5:
            recommendations.append(
                "Consider increasing cache size for better performance"
            )
        elif cache_usage > 0.9:
            recommendations.append(
                "Cache is nearly full, consider increasing max_cache_size"
            )

        # Memory recommendations
        if hasattr(self, "_get_available_memory"):
            available_mem = self._get_available_memory()
            if available_mem < 1024 * 1024 * 1024:  # Less than 1GB
                recommendations.append("Low available memory detected")

        return recommendations


def cache_result(cache_optimizer: PerformanceOptimizer) -> Callable:
    """
    Decorator for caching function results.

    Args:
        cache_optimizer: PerformanceOptimizer instance

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cache_optimizer.cached_execution(func, args, kwargs)

        return wrapper

    return decorator


def parallel_map(
    func: Callable, items: list[Any], max_workers: int | None = None
) -> list[Any]:
    """
    Apply a function to a list of items in parallel.

    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of worker threads

    Returns:
        List of results
    """
    optimizer = PerformanceOptimizer()

    parameter_sets = [{"item": item} for item in items]

    def wrapper(item: Any) -> Any:
        return func(item)

    return optimizer.parallel_execution(wrapper, parameter_sets, max_workers)


def time_execution(func: Callable) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Decorated function that returns (result, execution_time)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time

    return wrapper
