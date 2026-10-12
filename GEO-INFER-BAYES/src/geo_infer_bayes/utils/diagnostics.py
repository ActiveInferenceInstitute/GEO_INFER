"""Convergence and efficiency diagnostics for Bayesian samples."""

from typing import Any

import numpy as np


def _as_chains(values: np.ndarray) -> np.ndarray:
    """Normalize one parameter to ``(chains, draws)`` without hiding empties."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("posterior samples must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("posterior samples must be finite")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def _effective_sample_size(chains: np.ndarray) -> float:
    """Estimate ESS using the initial-positive-sequence autocorrelation sum."""
    n_chains, n_draws = chains.shape
    if n_draws < 2:
        return float(n_chains * n_draws)

    autocorrelations = []
    for chain in chains:
        centered = chain - np.mean(chain)
        variance = float(np.dot(centered, centered) / n_draws)
        if variance == 0:
            autocorrelations.append(np.zeros(n_draws - 1))
            continue
        correlations = np.correlate(centered, centered, mode="full")[n_draws - 1 :]
        autocorrelations.append(correlations[1:] / (n_draws * variance))

    mean_autocorrelation = np.mean(autocorrelations, axis=0)
    positive_sum = 0.0
    for correlation in mean_autocorrelation:
        if correlation <= 0:
            break
        positive_sum += float(correlation)
    total_draws = n_chains * n_draws
    return float(total_draws / max(1.0, 1.0 + 2.0 * positive_sum))


def _r_hat(chains: np.ndarray) -> float:
    """Return split-chain R-hat, or NaN when multiple chains are unavailable."""
    n_chains, n_draws = chains.shape
    if n_chains < 2 or n_draws < 2:
        return float("nan")
    within_chain = np.mean(np.var(chains, axis=1, ddof=1))
    if within_chain == 0:
        return (
            1.0
            if np.allclose(np.mean(chains, axis=1), np.mean(chains))
            else float("inf")
        )
    between_chain = n_draws * np.var(np.mean(chains, axis=1), ddof=1)
    variance_hat = ((n_draws - 1) / n_draws) * within_chain + (between_chain / n_draws)
    return float(np.sqrt(max(variance_hat / within_chain, 0.0)))


def mcmc_diagnostics(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute posterior summaries, effective sample size, and R-hat.

    One-dimensional arrays are treated as a single chain.  Pass a two-
    dimensional ``(chain, draw)`` array when a meaningful R-hat is required.
    """
    diagnostics: dict[str, Any] = {}
    for parameter, values in samples.items():
        chains = _as_chains(values)
        flattened = chains.reshape(-1)
        diagnostics[parameter] = {
            "mean": float(np.mean(flattened)),
            "std": float(np.std(flattened)),
            "median": float(np.median(flattened)),
            "q25": float(np.percentile(flattened, 25)),
            "q75": float(np.percentile(flattened, 75)),
            "min": float(np.min(flattened)),
            "max": float(np.max(flattened)),
            "ess": _effective_sample_size(chains),
            "r_hat": _r_hat(chains),
        }
    return diagnostics


def convergence_metrics(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute R-hat, Monte Carlo standard error, and Geweke statistics."""
    metrics: dict[str, Any] = {}
    for parameter, values in samples.items():
        chains = _as_chains(values)
        flattened = chains.reshape(-1)
        n = len(flattened)
        split = max(1, n // 2)
        first = flattened[:split]
        last = flattened[split:]
        var_first = float(np.var(first, ddof=1)) if len(first) > 1 else 0.0
        var_last = float(np.var(last, ddof=1)) if len(last) > 1 else 0.0
        denominator = np.sqrt(
            var_first / max(len(first), 1) + var_last / max(len(last), 1)
        )
        geweke_z = (
            float((np.mean(first) - np.mean(last)) / denominator)
            if denominator > 0
            else 0.0
        )
        ess = _effective_sample_size(chains)
        metrics[parameter] = {
            "potential_scale_reduction": _r_hat(chains),
            "monte_carlo_se": float(np.std(flattened) / np.sqrt(max(ess, 1.0))),
            "geweke_z": geweke_z,
        }
    return metrics
