"""
Decorators Module

This module provides decorators for common functionality used throughout
the GEO-INFER-MATH library, including caching, validation, timing, and logging.
"""

import functools
import inspect
import time
import logging
from typing import Any
from collections.abc import Callable
import numpy as np

logger = logging.getLogger(__name__)


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for caching function results.

    Args:
        func: Function to memoize

    Returns:
        Memoized function
    """
    cache: dict[Any, Any] = {}

    @functools.wraps(func)
    def memoized_func(*args: Any, **kwargs: Any) -> Any:
        # Create a hashable key from arguments
        key = (args, tuple(sorted(kwargs.items())))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    # Add cache clearing method
    memoized_func.clear_cache = lambda: cache.clear()
    memoized_func.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}

    return memoized_func


def memoize_with_expiry(expiry_seconds: float) -> Callable:
    """
    Memoization decorator with time-based cache expiry.

    Args:
        expiry_seconds: Cache expiry time in seconds

    Returns:
        Memoization decorator
    """

    def decorator(func: Callable) -> Callable:
        cache: dict[Any, Any] = {}
        timestamps: dict[Any, float] = {}

        @functools.wraps(func)
        def memoized_func(*args: Any, **kwargs: Any) -> Any:
            # Create a hashable key from arguments
            key = (args, tuple(sorted(kwargs.items())))

            # Check if cached result exists and is not expired
            current_time = time.time()
            if key in cache:
                if current_time - timestamps[key] < expiry_seconds:
                    return cache[key]
                else:
                    # Remove expired entry
                    del cache[key]
                    del timestamps[key]

            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = current_time

            return result

        # Add cache management methods
        def clear_cache() -> None:
            cache.clear()
            timestamps.clear()

        def cache_info() -> dict[str, Any]:
            current_time = time.time()
            valid_entries = sum(
                1 for t in timestamps.values() if current_time - t < expiry_seconds
            )
            return {
                "size": len(cache),
                "valid_entries": valid_entries,
                "expired_entries": len(cache) - valid_entries,
            }

        memoized_func.clear_cache = clear_cache
        memoized_func.cache_info = cache_info

        return memoized_func

    return decorator


def validate_input(**validators: Any) -> Callable:
    """
    Input validation decorator.

    Args:
        **validators: Validation functions for specific parameters

    Returns:
        Validation decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def validated_func(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    val = bound_args.arguments[param_name]
                    try:
                        validator(val)
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}': {e}"
                        ) from e

            return func(*args, **kwargs)

        return validated_func

    return decorator


def log_execution(level: int = logging.INFO) -> Callable:
    """
    Logging decorator for function execution.

    Args:
        level: Logging level

    Returns:
        Logging decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def logged_func(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__

            # Log function call
            logger.log(level, f"Executing {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}")
                raise

        return logged_func

    return decorator


def time_execution(func: Callable) -> Callable:
    """
    Timing decorator for measuring function execution time.

    Args:
        func: Function to time

    Returns:
        Timed function
    """

    @functools.wraps(func)
    def timed_func(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

        return result

    return timed_func


def requires_positive_values(*param_names: str) -> Callable:
    """
    Decorator to ensure specified parameters contain only positive values.

    Args:
        *param_names: Parameter names to check

    Returns:
        Validation decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def validated_func(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    if isinstance(value, (int, float)):
                        if value <= 0:
                            raise ValueError(
                                f"Parameter '{param_name}' must be positive, got {value}"
                            )
                    elif isinstance(value, np.ndarray):
                        if np.any(value <= 0):
                            raise ValueError(
                                f"All values in parameter '{param_name}' must be positive"
                            )
                    elif isinstance(value, (list, tuple)):
                        if any(v <= 0 for v in value):
                            raise ValueError(
                                f"All values in parameter '{param_name}' must be positive"
                            )

            return func(*args, **kwargs)

        return validated_func

    return decorator


def requires_finite_values(*param_names: str) -> Callable:
    """
    Decorator to ensure specified parameters contain only finite values.

    Args:
        *param_names: Parameter names to check

    Returns:
        Validation decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def validated_func(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    if isinstance(value, (int, float)):
                        if not np.isfinite(value):
                            raise ValueError(
                                f"Parameter '{param_name}' must be finite, got {value}"
                            )
                    elif isinstance(value, np.ndarray):
                        if not np.all(np.isfinite(value)):
                            raise ValueError(
                                f"All values in parameter '{param_name}' must be finite"
                            )
                    elif isinstance(value, (list, tuple)):
                        if not all(np.isfinite(v) for v in value):
                            raise ValueError(
                                f"All values in parameter '{param_name}' must be finite"
                            )

            return func(*args, **kwargs)

        return validated_func

    return decorator


def handle_exceptions(return_value: Any = None) -> Callable:
    """
    Exception handling decorator.

    Args:
        return_value: Value to return on exception

    Returns:
        Exception handling decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def exception_handled_func(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return return_value

        return exception_handled_func

    return decorator


def requires_numpy_arrays(*param_names: str) -> Callable:
    """
    Decorator to ensure specified parameters are numpy arrays.

    Args:
        *param_names: Parameter names to convert

    Returns:
        Conversion decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def array_func(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert specified parameters to numpy arrays
            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    if not isinstance(value, np.ndarray):
                        try:
                            bound_args.arguments[param_name] = np.array(value)
                        except Exception as e:
                            raise ValueError(
                                f"Cannot convert parameter '{param_name}' to numpy array: {e}"
                            )

            return func(*bound_args.args, **bound_args.kwargs)

        return array_func

    return decorator


def cache_results(cache_dict: dict[Any, Any] | None = None) -> Callable:
    """
    External cache decorator using a provided dictionary.

    Args:
        cache_dict: Dictionary to use for caching (optional)

    Returns:
        Caching decorator
    """
    target_cache: dict[Any, Any] = cache_dict if cache_dict is not None else {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def cached_func(*args: Any, **kwargs: Any) -> Any:
            # Create a hashable key from arguments
            key = (args, tuple(sorted(kwargs.items())))

            if key not in target_cache:
                target_cache[key] = func(*args, **kwargs)

            return target_cache[key]

        # Add cache management methods
        def clear_cache() -> None:
            target_cache.clear()

        def get_cache_size() -> int:
            return len(target_cache)

        cached_func.clear_cache = clear_cache
        cached_func.cache_size = get_cache_size

        return cached_func

    return decorator


def validate_output(output_validator: Callable) -> Callable:
    """
    Output validation decorator.

    Args:
        output_validator: Function to validate output

    Returns:
        Output validation decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def validated_output_func(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            try:
                output_validator(result)
            except Exception as e:
                raise ValueError(f"Output validation failed: {e}") from e

            return result

        return validated_output_func

    return decorator


def retry_on_failure(
    max_retries: int = 3,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    delay: float = 0.1,
) -> Callable:
    """
    Retry decorator for handling transient failures.

    Args:
        max_retries: Maximum number of retries
        exceptions: Exception types to catch
        delay: Delay between retries

    Returns:
        Retry decorator
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def retry_func(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry failed without exception")

        return retry_func

    return decorator


__all__ = [
    "memoize",
    "memoize_with_expiry",
    "validate_input",
    "log_execution",
    "time_execution",
    "requires_positive_values",
    "requires_finite_values",
    "handle_exceptions",
    "requires_numpy_arrays",
    "cache_results",
    "validate_output",
    "retry_on_failure",
]
