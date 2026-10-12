"""
Probabilistic evaluation metrics for distributional predictions.

In geospatial and environmental modeling the quantity of interest is a
*predictive distribution*, not a point forecast. A 50-meter estimate with a
huge interval is very different from the same estimate with a tight, well
calibrated one. This module provides the standard strictly-proper scoring
rules and calibration diagnostics used to judge those distributions:

* :func:`crps` and :func:`crps_gaussian` -- the Continuous Ranked Probability
  Score, a strictly proper skill score that jointly rewards a sharp predictive
  centered on the observation.
* :func:`pinball_loss` -- the quantile (pinball) loss minimized by a quantile
  regression forecaster.
* :func:`empirical_coverage`, :func:`coverage_calibration_error`,
  :func:`interval_score` -- whether a stated interval actually contains the
  observation at the expected rate, and the Winkler interval score that
  balances width against coverage.
* :func:`pit_values` / :func:`pit_gaussian` and
  :func:`pit_uniformity_statistic` -- the Probability Integral Transform, the
  standard calibration check: a well-calibrated predictive renders every
  held-out observation uniformly distributed.
* :func:`log_predictive_density` /
  :func:`log_predictive_density_gaussian` -- the average log density of held-out
  observations under the predictive, a legitimate scoring rule.
* :func:`evaluate_predictive` and :func:`evaluate_gaussian` -- convenience
  summaries that assemble the above into a single dictionary.

Every function validates its inputs (finite, aligned, admissible levels) before
computing anything, so a silent shape mistake surfaces as a loud error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "crps",
    "crps_pointwise",
    "crps_gaussian",
    "pinball_loss",
    "empirical_coverage",
    "coverage_calibration_error",
    "interval_score",
    "pit_values",
    "pit_gaussian",
    "pit_uniformity_statistic",
    "log_predictive_density",
    "log_predictive_density_pointwise",
    "log_predictive_density_gaussian",
    "evaluate_predictive",
    "evaluate_gaussian",
]

_TWO_PI = 2.0 * np.pi
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _as_observations(observations: Any) -> np.ndarray:
    obs = np.asarray(observations, dtype=float).reshape(-1)
    if obs.size == 0:
        raise ValueError("observations must not be empty")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observations must be finite")
    return obs


def _as_samples(samples: Any, n_obs: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            "predictive_samples must be a two-dimensional array of shape (n_draws, n_points)"
        )
    if arr.shape[1] != n_obs:
        raise ValueError(
            "predictive_samples points must match observations length "
            f"(got {arr.shape[1]}, expected {n_obs})"
        )
    if arr.shape[0] < 1:
        raise ValueError("predictive_samples must contain at least one draw")
    if not np.all(np.isfinite(arr)):
        raise ValueError("predictive_samples must be finite")
    return arr


def _check_level(level: float, name: str = "level") -> float:
    level = float(level)
    if not np.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return level


def _gaussian_parameters(
    observations: np.ndarray, mean: Any, std: Any
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.broadcast_to(np.asarray(mean, dtype=float).reshape(-1), observations.shape)
    sg = np.broadcast_to(np.asarray(std, dtype=float).reshape(-1), observations.shape)
    if np.any(sg <= 0) or not np.all(np.isfinite(sg)) or not np.all(np.isfinite(mu)):
        raise ValueError("mean and std must be finite with strictly positive std")
    return mu, sg


def _crps_per_point(observations: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS per observed point from sorted draws."""
    n_draws = draws.shape[0]
    ordered = np.sort(draws, axis=0)  # (n_draws, n_points)
    below = ordered < observations[None, :]
    absolute = np.where(
        below, observations[None, :] - ordered, ordered - observations[None, :]
    ).sum(axis=0)
    index = np.arange(n_draws, dtype=float)[:, None]
    # Closed form for the (canonical, ordered) pairwise term. ``ordered *
    # (2*index - n_draws + 1)`` sums the unordered pair costs; the ordered
    # double-count makes the CRPS second term exactly ``pair_cost / N**2``.
    pair_cost = np.sum(ordered * (2.0 * index - n_draws + 1.0), axis=0)
    return np.asarray(absolute / n_draws - pair_cost / (n_draws * n_draws), dtype=float)


def crps(observations: Any, predictive_samples: Any) -> float:
    """Continuous Ranked Probability Score from predictive draws.

    Uses the exact empirical CRPS, computed on sorted draws with the closed
    form for the pairwise draw-cost term. A sharp predictive centered on the
    observation scores near zero; a diffuse predictive scores roughly its
    average absolute error. This estimator converges to the continuous CRPS as
    the draw count grows, so it can be asserted against
    :func:`crps_gaussian` when the predictive is Gaussian.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predictive_samples : array-like of shape (n_draws, n_points)
        Posterior predictive draws, one column per observed point.

    Returns
    -------
    float
        Mean CRPS over the observation points (lower is better).
    """
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    return float(np.mean(_crps_per_point(obs, draws)))


def crps_pointwise(observations: Any, predictive_samples: Any) -> np.ndarray:
    """Per-point CRPS, one value per observation (lower is better)."""
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    return _crps_per_point(obs, draws)


def crps_gaussian(observations: Any, mean: Any, std: Any) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution.

    For ``Z = (y - mu) / sigma`` the CRPS is
    ``sigma * (Z * (2 * Phi(Z) - 1) + 2 * phi(Z) - 1/sqrt(pi))``. This is the
    exact value the sample estimator converges to as the draw count grows, so
    it is the figure to assert against in unit tests of the sample form.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    mean, std : array-like of shape (n_points,) or scalar
        Predictive mean and standard deviation, aligned to ``observations``.

    Returns
    -------
    float
        Mean Gaussian CRPS (lower is better).
    """
    obs = _as_observations(observations)
    mu, sg = _gaussian_parameters(obs, mean, std)
    z = (obs - mu) / sg
    phi = _TWO_PI ** (-0.5) * np.exp(-0.5 * z**2)
    per_point = sg * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * phi - _INV_SQRT_PI)
    return float(np.mean(per_point))


def pinball_loss(
    observations: Any, predicted_quantile: Any, quantile_level: float = 0.5
) -> float:
    """Quantile (pinball) loss for a predicted conditional quantile.

    Lower values are better; the optimal value of ``quantile_level`` for a
    forecaster is the true quantile of the predictive at that probability.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predicted_quantile : array-like of shape (n_points,)
        The predicted value at the target quantile.
    quantile_level : float in (0, 1), default=0.5
        The nominal quantile level of ``predicted_quantile``.

    Returns
    -------
    float
        Mean pinball loss (lower is better).
    """
    q = _check_level(quantile_level, "quantile_level")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile_level must be strictly between zero and one")
    obs = _as_observations(observations)
    pred = np.broadcast_to(
        np.asarray(predicted_quantile, dtype=float).reshape(-1), obs.shape
    )
    if not np.all(np.isfinite(pred)):
        raise ValueError("predicted_quantile must be finite")
    residual = obs - pred
    loss = np.where(residual >= 0.0, q * residual, (q - 1.0) * residual)
    return float(np.mean(loss))


def _aligned_interval(
    observations: Any, lower: Any, upper: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = _as_observations(observations)
    lo = np.broadcast_to(np.asarray(lower, dtype=float).reshape(-1), obs.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float).reshape(-1), obs.shape)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("lower and upper bounds must be finite")
    if np.any(hi < lo):
        raise ValueError("upper bounds must not be below lower bounds")
    return obs, lo, hi


def empirical_coverage(observations: Any, lower: Any, upper: Any) -> float:
    """Fraction of observations inside ``[lower, upper]``.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    lower, upper : array-like of shape (n_points,) or scalar
        Interval bounds, aligned to ``observations``.

    Returns
    -------
    float
        Empirical coverage fraction in [0, 1].
    """
    obs, lo, hi = _aligned_interval(observations, lower, upper)
    return float(np.mean((obs >= lo) & (obs <= hi)))


def coverage_calibration_error(
    observations: Any, lower: Any, upper: Any, expected_level: float = 0.95
) -> float:
    """Absolute deviation of observed coverage from the nominal ``expected_level``.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    lower, upper : array-like of shape (n_points,) or scalar
        Interval bounds.
    expected_level : float in [0, 1], default=0.95
        The nominal coverage the interval claims.

    Returns
    -------
    float
        ``abs(empirical_coverage - expected_level)`` in [0, 1]; zero is a
        perfectly calibrated interval.
    """
    expected = _check_level(expected_level, "expected_level")
    return abs(empirical_coverage(observations, lower, upper) - expected)


def interval_score(
    observations: Any, lower: Any, upper: Any, expected_level: float = 0.95
) -> float:
    """Winkler interval score, balancing interval width against coverage.

    For an interval ``[l, u]`` claiming a nominal coverage ``expected_level``
    the score per point is ``(u - l)`` when the observation is inside, and
    ``(u - l) + (2 / level) * (l - y)`` or ``(u - l) + (2 / level) * (y - u)``
    when it falls outside. Lower is better: a wider interval pays a constant
    penalty even when it covers everything, and a sharp interval that misses
    the observation pays a heavy miss penalty.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    lower, upper : array-like of shape (n_points,) or scalar
        Interval bounds aligned to ``observations``.
    expected_level : float, default=0.95
        Nominal coverage the interval is meant to provide, in ``(0, 1)``.

    Returns
    -------
    float
        Mean Winkler interval score (lower is better).
    """
    level = _check_level(expected_level, "expected_level")
    if not 0.0 < level <= 1.0:
        raise ValueError("expected_level must be in (0, 1]")
    obs, lo, hi = _aligned_interval(observations, lower, upper)
    width = hi - lo
    below = obs < lo
    above = obs > hi
    miss_penalty = np.zeros_like(obs)
    miss_penalty[below] = (2.0 / level) * (lo[below] - obs[below])
    miss_penalty[above] = (2.0 / level) * (obs[above] - hi[above])
    return float(np.mean(width + miss_penalty))


def pit_values(observations: Any, predictive_samples: Any) -> np.ndarray:
    """Probability Integral Transform values from predictive draws.

    For each observation ``y`` the PIT is the fraction of predictive draws
    strictly less than ``y``. A calibrated predictive renders these values
    uniformly distributed over ``[0, 1]``.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predictive_samples : array-like of shape (n_draws, n_points)
        Posterior predictive draws, one column per observed point.

    Returns
    -------
    ndarray of shape (n_points,)
        PIT values in ``[0, 1]``.
    """
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    return np.asarray(
        np.sum(draws < obs[None, :], axis=0) / draws.shape[0], dtype=float
    )


def pit_gaussian(observations: Any, mean: Any, std: Any) -> np.ndarray:
    """PIT values for a Gaussian predictive: ``Phi((y - mu) / sigma)``."""
    obs = _as_observations(observations)
    mu, sg = _gaussian_parameters(obs, mean, std)
    return np.asarray(stats.norm.cdf((obs - mu) / sg), dtype=float)


def pit_uniformity_statistic(
    observations: Any, predictive_samples: Any, n_bins: int = 10
) -> float:
    """Max absolute deviation of the PIT histogram from a uniform expectation.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predictive_samples : array-like of shape (n_draws, n_points)
        Predictive draws aligned to ``observations``.
    n_bins : int, default=10
        Number of histogram bins over ``[0, 1]``.

    Returns
    -------
    float
        The largest absolute difference between the bin's share of PIT values
        and the uniform share ``1 / n_bins``. Zero signals perfect uniformity.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    obs = _as_observations(observations)
    _as_samples(predictive_samples, obs.size)
    pits = pit_values(obs, predictive_samples)
    hist, _ = np.histogram(pits, bins=n_bins, range=(0.0, 1.0))
    expected_share = obs.size / n_bins
    counts = np.asarray(hist, dtype=float)
    # Largest absolute deviation, expressed as a proportion of the sample.
    return float(np.max(np.abs(counts - expected_share)) / obs.size)


def log_predictive_density(observations: Any, predictive_samples: Any) -> float:
    """Mean log predictive density of held-out observations.

    For each observed point a kernel density estimate is fit to that column of
    posterior draws and the log density of the observation under it is
    returned, averaged across points. This is a legitimate anonymous scoring
    rule: better models assign held-out observations higher probability.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predictive_samples : array-like of shape (n_draws, n_points)
        Predictive draws, one column per observation. Kernel estimation needs
        several draws per column; below a handful the estimate is unstable.

    Returns
    -------
    float
        Mean log predictive density (higher is better).
    """
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    return float(np.mean(log_predictive_density_pointwise(obs, draws)))


def log_predictive_density_pointwise(
    observations: Any, predictive_samples: Any
) -> np.ndarray:
    """Per-point log predictive density, one value per observation."""
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    if draws.shape[0] < 3:
        raise ValueError("log_predictive_density needs at least three draws per point")
    log_densities = np.empty(obs.size, dtype=float)
    for i in range(obs.size):
        kde = stats.gaussian_kde(draws[:, i])
        log_densities[i] = float(kde.logpdf(np.asarray([obs[i]]))[0])
    return log_densities


def log_predictive_density_gaussian(observations: Any, mean: Any, std: Any) -> float:
    """Average Gaussian log density ``-0.5 * ((y-mu)/std)^2 - log(std) - c``."""
    obs = _as_observations(observations)
    mu, sg = _gaussian_parameters(obs, mean, std)
    per_point = -0.5 * ((obs - mu) / sg) ** 2 - np.log(sg) - 0.5 * np.log(_TWO_PI)
    return float(np.mean(per_point))


def evaluate_predictive(
    observations: Any,
    predictive_samples: Any,
    level: float = 0.95,
) -> dict[str, float]:
    """One-call distributional evaluation from predictive draws.

    Parameters
    ----------
    observations : array-like of shape (n_points,)
        Observed values.
    predictive_samples : array-like of shape (n_draws, n_points)
        Predictive draws, one column per observation.
    level : float, default=0.95
        Nominal coverage level for the interval diagnostics, in ``(0, 1)``.

    Returns
    -------
    dict
        CRPS, mean absolute error, log predictive density, coverage,
        calibration deviation, interval score and PIT uniformity of the draws.

    Raises
    ------
    ValueError
        If ``level`` is outside ``(0, 1)``.
    """
    interval_level = _check_level(level, "level")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    obs = _as_observations(observations)
    draws = _as_samples(predictive_samples, obs.size)
    tail = (1.0 - interval_level) / 2.0
    lower = np.percentile(draws, 100.0 * tail, axis=0)
    upper = np.percentile(draws, 100.0 * (1.0 - tail), axis=0)
    mean = np.mean(draws, axis=0)
    return {
        "crps": crps(obs, draws),
        "mean_absolute_error": float(np.mean(np.abs(obs - mean))),
        "log_predictive_density": log_predictive_density(obs, draws),
        "coverage": empirical_coverage(obs, lower, upper),
        "coverage_deviation": coverage_calibration_error(
            obs, lower, upper, interval_level
        ),
        "interval_score": interval_score(obs, lower, upper, interval_level),
        "pit_uniformity": pit_uniformity_statistic(obs, draws),
    }


def evaluate_gaussian(
    observations: Any, mean: Any, std: Any, level: float = 0.95
) -> dict[str, float]:
    """Evaluate a Gaussian predictive ``(mean, std)`` against observations."""
    interval_level = _check_level(level, "level")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    obs = _as_observations(observations)
    mu, sg = _gaussian_parameters(obs, mean, std)
    tail = (1.0 - interval_level) / 2.0
    lower = mu - stats.norm.ppf(1.0 - tail) * sg
    upper = mu + stats.norm.ppf(1.0 - tail) * sg
    return {
        "crps": crps_gaussian(obs, mu, sg),
        "mean_absolute_error": float(np.mean(np.abs(obs - mu))),
        "log_predictive_density": log_predictive_density_gaussian(obs, mu, sg),
        "coverage": empirical_coverage(obs, lower, upper),
        "coverage_deviation": coverage_calibration_error(
            obs, lower, upper, interval_level
        ),
        "interval_score": interval_score(obs, lower, upper, interval_level),
        "pit_uniformity": float(np.max(np.abs(pit_gaussian(obs, mu, sg) - 0.5))),
    }
