"""
Validation Utilities

This module provides validation decorators and utilities for
mathematical operations.
"""

import functools
from typing import Any
from collections.abc import Callable

import numpy as np
import logging

from geo_infer_math.utils.exceptions import (
    InvalidDistributionError,
    NumericalError,
    ValidationError,
    SpatialError,
)

logger = logging.getLogger(__name__)


def _validate_probability_array(probabilities: Any) -> bool:
    """Validate a probability array and return ``True`` when it is valid."""
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise InvalidDistributionError("Probabilities must be finite and non-empty")
    if np.any(values < 0):
        raise InvalidDistributionError("Probabilities must be non-negative")
    total = float(np.sum(values))
    if not np.isclose(total, 1.0, rtol=1e-10):
        if total > 0:
            logger.warning(
                "Probabilities do not sum to 1; callers should normalize them"
            )
        else:
            raise InvalidDistributionError("Probabilities must sum to a positive value")
    return True


def validate_probabilities(value: Any) -> Any:
    """Validate probabilities directly or validate arguments in a decorator.

    Passing an array validates it and returns ``True``. Passing a callable
    returns a decorator that validates every ndarray argument before the call.
    """
    if not callable(value):
        return _validate_probability_array(value)

    func = value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, np.ndarray):
                _validate_probability_array(arg)
        return func(*args, **kwargs)

    return wrapper


def _validate_coordinate_array(coordinates: Any) -> bool:
    """Validate an array of two- or three-dimensional finite coordinates."""
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise SpatialError(
            f"Coordinates must be a 2D array with 2 or 3 columns, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise SpatialError("Coordinates contain NaN or Inf values")
    return True


def validate_coordinates(value: Any) -> Any:
    """Validate coordinates directly or validate arguments in a decorator."""
    if not callable(value):
        return _validate_coordinate_array(value)

    func = value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, np.ndarray) and arg.ndim == 2:
                _validate_coordinate_array(arg)
        return func(*args, **kwargs)

    return wrapper


def _validate_numerical_value(value: Any) -> bool:
    """Validate scalar or array-like numerical input."""
    values = np.asarray(value)
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise NumericalError(f"Invalid numerical value: {value}")
    return True


def validate_numerical(value: Any) -> Any:
    """Validate numerical values directly or validate arguments in a decorator."""
    if not callable(value):
        return _validate_numerical_value(value)

    func = value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, (int, float, complex, np.number, np.ndarray)):
                _validate_numerical_value(arg)
        return func(*args, **kwargs)

    return wrapper


def validate_shape(expected_shape: tuple[int, ...], axis: int = 0) -> Callable:
    """
    Decorator to validate array shapes.

    Args:
        expected_shape: Expected shape tuple
        axis: Which argument to check (0 = first, 1 = second, etc.)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) > axis:
                arg = args[axis]
                if isinstance(arg, np.ndarray):
                    if arg.shape != expected_shape:
                        raise ValueError(
                            f"Expected shape {expected_shape}, got {arg.shape}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_range(param_name: str, min_val: float, max_val: float) -> Callable:
    """
    Decorator to validate parameter ranges.

    Args:
        param_name: Name of parameter to validate
        min_val: Minimum value
        max_val: Maximum value
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check in kwargs first
            if param_name in kwargs:
                val = kwargs[param_name]
                if not (min_val <= val <= max_val):
                    raise ValueError(
                        f"{param_name} must be between {min_val} and {max_val}, got {val}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_matrix(matrix: Any, *, square: bool = False) -> bool:
    """Validate a finite two-dimensional numeric matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or (square and values.shape[0] != values.shape[1]):
        raise ValidationError(f"Expected a 2D matrix, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NumericalError("Matrix contains NaN or Inf values")
    return True


def validate_weights_matrix(weights: Any) -> bool:
    """Validate a finite, square, non-negative spatial weights matrix."""
    validate_matrix(weights, square=True)
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0):
        raise ValidationError("Spatial weights must be non-negative")
    return True


def validate_values_array(values: Any) -> bool:
    """Validate a non-empty, one-dimensional finite values array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError("Values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise NumericalError("Values contain NaN or Inf values")
    return True


def validate_bounds(bounds: Any) -> bool:
    """Validate bounds ordered as ``(min_x, min_y, max_x, max_y)``."""
    values = np.asarray(bounds, dtype=float)
    if values.size != 4 or not np.all(np.isfinite(values)):
        raise ValidationError("Bounds must contain four finite numeric values")
    min_x, min_y, max_x, max_y = values.ravel()
    if min_x > max_x or min_y > max_y:
        raise ValidationError("Bounds minima must not exceed their maxima")
    return True


def validate_function_input(function: Any) -> bool:
    """Validate that an analysis callback is callable."""
    if not callable(function):
        raise ValidationError("Function input must be callable")
    return True


def validate_spatial_autocorrelation_params(values: Any, weights: Any) -> bool:
    """Validate values and weights used by spatial autocorrelation methods."""
    validate_values_array(values)
    validate_weights_matrix(weights)
    if np.asarray(values).size != np.asarray(weights).shape[0]:
        raise ValidationError("Values and weights must describe the same observations")
    return True


def validate_interpolation_params(points: Any, values: Any) -> bool:
    """Validate point coordinates and values used by interpolation."""
    validate_coordinates(points)
    validate_values_array(values)
    if np.asarray(points).shape[0] != np.asarray(values).size:
        raise ValidationError("Points and values must have the same length")
    return True


def validate_clustering_params(data: Any, n_clusters: int | None = None) -> bool:
    """Validate clustering data and an optional requested cluster count."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValidationError("Clustering data must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise NumericalError("Clustering data contains NaN or Inf values")
    if n_clusters is not None and (n_clusters < 1 or n_clusters > values.shape[0]):
        raise ValidationError("n_clusters must be between 1 and the number of rows")
    return True


def validate_tensor_data(data: Any, expected_ndim: int | None = None) -> bool:
    """Validate finite tensor-like numeric data."""
    values = np.asarray(data)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValidationError("Tensor data must be a non-empty numeric array")
    if not np.all(np.isfinite(values)):
        raise NumericalError("Tensor data contains NaN or Inf values")
    if expected_ndim is not None and values.ndim != expected_ndim:
        raise ValidationError(
            f"Tensor data must have {expected_ndim} dimensions, got {values.ndim}"
        )
    return True


def handle_validation_errors(func: Callable) -> Callable:
    """Translate ordinary input errors into :class:`ValidationError`."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc

    return wrapper


def handle_numerical_errors(func: Callable) -> Callable:
    """Translate numerical failures into :class:`NumericalError`."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except NumericalError:
            raise
        except (FloatingPointError, OverflowError) as exc:
            raise NumericalError(str(exc)) from exc

    return wrapper
