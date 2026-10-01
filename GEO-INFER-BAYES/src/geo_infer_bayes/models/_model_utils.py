"""Shared validation and posterior helpers for Bayesian model implementations."""

from __future__ import annotations

from math import lgamma, log, pi
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
from ..utils.rng import SeedLike, resolve_rng


def observations_from(data: Any) -> np.ndarray:
    """Return finite one-dimensional observations from a supported data object."""
    if isinstance(data, Mapping):
        for key in ("observations", "y", "values"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError("Data must contain one of: observations, y, or values")
    observations = np.asarray(data, dtype=float).reshape(-1)
    if observations.size == 0:
        raise ValueError("At least one observation is required")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations must be finite")
    return observations


def features_from(X: Any) -> np.ndarray:
    """Return finite numeric features as a two-dimensional array."""
    features = np.asarray(X, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("X must contain at least one one- or two-dimensional row")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    return features


def signal_from(X: Any) -> np.ndarray:
    """Reduce one or more features to a stable scalar signal per row."""
    return np.asarray(np.mean(features_from(X), axis=1), dtype=float)


def posterior_values(posterior: Any) -> Mapping[str, Any]:
    """Get posterior samples from a mapping or PosteriorAnalysis-like object."""
    if posterior is None:
        return {}
    values = (
        posterior.get("samples")
        if isinstance(posterior, Mapping)
        else getattr(posterior, "samples", None)
    )
    if not isinstance(values, Mapping):
        raise TypeError("posterior must expose a mapping named samples")
    return values


def posterior_vector(posterior: Any, key: str, limit: int | None = None) -> np.ndarray:
    """Return flattened finite samples for one posterior parameter."""
    values = posterior_values(posterior)
    if key not in values:
        return np.array([], dtype=float)
    vector = np.asarray(values[key], dtype=float).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if limit is not None:
        vector = vector[:limit]
    return np.asarray(vector, dtype=float)


def scalar_parameter(theta: Mapping[str, Any], key: str, default: float) -> float:
    """Read a finite scalar parameter, accepting a one-element array."""
    value = np.asarray(theta.get(key, default), dtype=float).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"Parameter {key!r} must be a finite scalar")
    return float(value[0])


def parameter_array(theta: Mapping[str, Any], key: str, default: Any) -> np.ndarray:
    """Read a finite parameter array or scalar."""
    value = np.asarray(theta.get(key, default), dtype=float)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"Parameter {key!r} must contain finite values")
    return value.reshape(-1)


def gaussian_log_likelihood(
    observations: np.ndarray, predictions: np.ndarray, scale: float
) -> float:
    """Compute a Gaussian log likelihood with a strictly positive scale."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Likelihood scale must be finite and greater than zero")
    residuals = observations - np.asarray(predictions, dtype=float)
    return float(-0.5 * np.sum((residuals / scale) ** 2 + log(2.0 * pi * scale**2)))


def log_prior_from_parameters(
    parameters: Mapping[str, Any], theta: Mapping[str, Any]
) -> float:
    """Evaluate the scalar/array priors declared by a model."""
    result = 0.0
    for name, specification in parameters.items():
        if name not in theta:
            continue
        values = parameter_array(theta, name, [])
        prior = specification.get("prior")
        hyperparams = specification.get("hyperparams", {})
        if prior == "normal":
            mu = float(hyperparams.get("mu", 0.0))
            sigma = float(hyperparams.get("sigma", 1.0))
            if sigma <= 0:
                raise ValueError(f"Prior scale for {name!r} must be positive")
            result += float(
                np.sum(
                    -0.5 * ((values - mu) / sigma) ** 2 - log(sigma * np.sqrt(2.0 * pi))
                )
            )
        elif prior == "half_normal":
            sigma = float(hyperparams.get("sigma", 1.0))
            if sigma <= 0 or np.any(values <= 0):
                return -np.inf
            result += float(
                np.sum(np.log(2.0 / (pi * sigma**2)) / 2 - values**2 / (2.0 * sigma**2))
            )
        elif prior == "gamma":
            shape = float(hyperparams.get("shape", 1.0))
            scale = float(hyperparams.get("scale", 1.0))
            if shape <= 0 or scale <= 0 or np.any(values <= 0):
                return -np.inf
            result += float(
                np.sum(
                    (shape - 1.0) * np.log(values)
                    - values / scale
                    - lgamma(shape)
                    - shape * log(scale)
                )
            )
        elif prior == "inverse_gamma":
            alpha = float(hyperparams.get("alpha", 1.0))
            beta = float(hyperparams.get("beta", 1.0))
            if alpha <= 0 or beta <= 0 or np.any(values <= 0):
                return -np.inf
            result += float(
                np.sum(
                    alpha * log(beta)
                    - lgamma(alpha)
                    - (alpha + 1.0) * np.log(values)
                    - beta / values
                )
            )
        elif prior == "log_normal":
            mu = float(hyperparams.get("mu", 0.0))
            sigma = float(hyperparams.get("sigma", 1.0))
            if sigma <= 0 or np.any(values <= 0):
                return -np.inf
            result += float(
                np.sum(
                    -0.5 * ((np.log(values) - mu) / sigma) ** 2
                    - np.log(values * sigma * np.sqrt(2.0 * pi))
                )
            )
        elif prior == "uniform":
            low = float(hyperparams["low"])
            high = float(hyperparams["high"])
            if np.any((values < low) | (values > high)):
                return -np.inf
            result -= values.size * log(high - low)
        else:
            raise ValueError(f"Unsupported prior distribution {prior!r} for {name!r}")
    return float(result)


def predictive_samples(
    mean: np.ndarray,
    scale: np.ndarray | float,
    samples: int,
    seed: SeedLike = None,
) -> np.ndarray:
    """Draw validated predictive samples around a model prediction.

    Args:
        mean: Predictive mean, one entry per prediction point.
        scale: Predictive standard deviation, scalar or aligned to ``mean``.
        samples: Number of draws to return.
        seed: Seed or generator for the draws; see
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

    Returns:
        Array of shape ``(samples, len(mean))``.
    """
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    mean = np.asarray(mean, dtype=float).reshape(-1)
    scale = np.asarray(scale, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Predictive scale must be finite and greater than zero")
    rng = resolve_rng(seed)
    return rng.normal(loc=mean, scale=scale, size=(samples, mean.size))


def posterior_draw_indices(
    posterior: Any, samples: int, names: Sequence[str]
) -> np.ndarray:
    """Choose which posterior draws a prediction should average over.

    Two things this gets right that an ad-hoc ``range(min(samples, ...))``
    does not. First, the number of available draws is the length of a parameter
    *array*, not ``len(posterior.samples)`` -- that is the count of parameter
    names, so using it silently collapses a 4000-draw posterior to a handful of
    draws. Second, the chosen draws are spread evenly across the chain instead
    of taken from the front, which would otherwise weight the least-converged
    part of the run most heavily.

    Args:
        posterior: Object exposing a ``samples`` mapping of parameter name to
            draws along the leading axis.
        samples: Maximum number of draws to use. Must be positive.
        names: Parameter names the caller needs; the draw count is the minimum
            across them, so a ragged posterior cannot index out of bounds.

    Returns:
        Integer indices into the draw axis, ascending, of length
        ``min(samples, available)``.

    Raises:
        ValueError: If ``samples`` is not positive, if ``names`` is empty, if a
            name is missing from the posterior, or if no draws are available.
    """
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    if not names:
        raise ValueError("names must not be empty")

    draw_counts = []
    for name in names:
        if name not in posterior.samples:
            raise ValueError(f"posterior has no samples for parameter {name!r}")
        draw_counts.append(len(np.asarray(posterior.samples[name])))
    available = min(draw_counts)
    if available == 0:
        raise ValueError("posterior contains no usable parameter samples")

    return np.linspace(0, available - 1, num=min(int(samples), available), dtype=int)
