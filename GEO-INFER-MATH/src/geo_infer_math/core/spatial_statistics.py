"""
Spatial Statistics Module

This module provides functions and classes for analyzing spatial patterns,
autocorrelation, and distributions in geospatial data.
"""

import numpy as np
from typing import Any, cast
from dataclasses import dataclass
from math import erfc, sqrt

from geo_infer_math.utils.rng import resolve_rng


def _generate_weights(coords: np.ndarray, include_self: bool = False) -> np.ndarray:
    """
    Generate a spatial weights matrix from coordinates.

    Args:
        coords: Array of coordinates (n x 2)
        include_self: Whether to include self-weights on the diagonal

    Returns:
        Spatial weights matrix (n x n), row-standardized
    """

    # Vectorized distance calculation for better performance
    coords_i = coords[:, np.newaxis, :]  # Shape: (n, 1, 2)
    coords_j = coords[np.newaxis, :, :]  # Shape: (1, n, 2)

    # Calculate pairwise distances
    distances = np.sqrt(np.sum((coords_i - coords_j) ** 2, axis=2))

    # Create weights matrix (inverse distance)
    epsilon = 1e-10
    weights = 1.0 / (distances + epsilon)
    np.fill_diagonal(weights, 0)  # No self-weights initially

    # Row standardize for better numerical stability
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = weights / row_sums

    if include_self:
        np.fill_diagonal(weights, 1.0)

    return cast(np.ndarray, weights)


def _cliff_ord_terms(weights_matrix: np.ndarray) -> tuple[float, float, float]:
    """Compute the S0, S1, and S2 connectivity terms of Cliff & Ord (1973).

    S0 = sum of all off-diagonal weights.
    S1 = 1/2 * sum over ordered pairs of (w_ij + w_ji)^2.
    S2 = sum over locations of (row sum_i + column sum_i)^2.
    """
    W = np.asarray(weights_matrix, dtype=np.float64)
    W_offdiag = W.copy()
    np.fill_diagonal(W_offdiag, 0.0)
    s0 = float(np.sum(W_offdiag))
    s1 = 0.5 * float(np.sum((W_offdiag + W_offdiag.T) ** 2))
    s2 = float(np.sum((np.sum(W_offdiag, axis=1) + np.sum(W_offdiag, axis=0)) ** 2))
    return s0, s1, s2


def morans_i_variance(values: np.ndarray, weights_matrix: np.ndarray) -> float:
    """Variance of Moran's I under the randomization assumption.

    Implements the standard Cliff & Ord (1973) randomization variance using
    the S0/S1/S2 connectivity terms and the sample kurtosis of the data:

        Var(I) = [ n((n^2 - 3n + 3) S1 - n S2 + 3 S0^2)
                   - b2 ((n^2 - n) S1 - 2 n S2 + 6 S0^2) ]
                 / [ (n - 1)(n - 2)(n - 3) S0^2 ]  -  1/(n - 1)^2

    where b2 = n * sum(z^4) / (sum(z^2))^2 is the moment kurtosis of the
    centered values. Requires n >= 4.

    Args:
        values: Attribute values at the n locations.
        weights_matrix: n x n spatial weights matrix (diagonal ignored).

    Returns:
        Randomization-assumption variance of Moran's I.

    Raises:
        ValueError: If fewer than 4 locations are supplied, shapes mismatch,
            or S0 is zero.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    W = np.asarray(weights_matrix, dtype=np.float64)
    n = len(values)
    if W.shape[0] != n or W.shape[1] != n:
        raise ValueError(f"Weights matrix shape {W.shape} does not match {n} values")
    if n < 4:
        raise ValueError(
            "Moran's I randomization variance requires at least 4 locations"
        )
    s0, s1, s2 = _cliff_ord_terms(W)
    if s0 == 0.0:
        raise ValueError("Spatial weights matrix has zero total weight")
    z = values - np.mean(values)
    sum_sq = float(np.sum(z**2))
    if sum_sq == 0.0:
        raise ValueError("Values are constant; variance of Moran's I is undefined")
    b2 = n * float(np.sum(z**4)) / (sum_sq**2)
    numerator = n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2) - b2 * (
        (n**2 - n) * s1 - 2 * n * s2 + 6 * s0**2
    )
    denominator = (n - 1) * (n - 2) * (n - 3) * s0**2
    variance = numerator / denominator - 1.0 / (n - 1) ** 2
    return float(variance)


@dataclass
class SpatialDescriptiveStats:
    """Container for spatial descriptive statistics."""

    mean: float
    median: float
    stdev: float
    variance: float
    min_value: float
    max_value: float
    centroid: tuple[float, float]
    dispersion: float
    skewness: float
    kurtosis: float


class MoranI:
    """
    Implementation of Moran's I statistic for spatial autocorrelation.

    Moran's I measures the spatial autocorrelation (clustering or similarity)
    of values across geographic locations.
    """

    def __init__(self, weights_matrix: np.ndarray | None = None):
        """
        Initialize MoranI calculator.

        Args:
            weights_matrix: Spatial weights matrix defining relationships
                            between locations
        """
        self.weights_matrix = weights_matrix

    def compute(
        self, values: np.ndarray, coords: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Compute Moran's I statistic.

        Args:
            values: Array of values at each location
            coords: Optional array of coordinates if weights_matrix
                    is not provided

        Returns:
            Dictionary containing Moran's I statistic and p-value
        """
        if self.weights_matrix is None:
            if coords is not None:
                self.weights_matrix = _generate_weights(coords)
            else:
                raise ValueError("weights_matrix must be set or coords provided")

        assert self.weights_matrix is not None
        # Input validation
        if len(values) != self.weights_matrix.shape[0]:
            raise ValueError(
                f"Values array length ({len(values)}) must match "
                f"weights matrix size ({self.weights_matrix.shape[0]})"
            )

        if len(values) <= 1:
            raise ValueError("Moran's I requires at least 2 data points")

        # Handle constant values case
        if np.std(values) == 0:
            return {
                "I": 0.0,
                "expected_I": -1.0 / (len(values) - 1),
                "var_I": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
            }

        n = len(values)
        z = (values - np.mean(values)) / np.std(values)

        w_sum = np.sum(self.weights_matrix)
        z_outer = np.outer(z, z)
        numerator = np.sum(z_outer * self.weights_matrix)
        denominator = np.sum(z**2)

        I_val = (n / w_sum) * (numerator / denominator)
        expected_I = -1.0 / (n - 1)

        var_I = morans_i_variance(values, self.weights_matrix)

        z_score_val = (I_val - expected_I) / np.sqrt(var_I) if var_I > 0 else 0.0
        p_value = erfc(abs(z_score_val) / sqrt(2))

        return {
            "I": I_val,
            "expected_I": expected_I,
            "var_I": var_I,
            "z_score": z_score_val,
            "p_value": p_value,
        }


class GearysC:
    """
    Implementation of Geary's C statistic for spatial autocorrelation.

    Geary's C is a measure of spatial autocorrelation that uses
    differences between neighbouring values. Values range from 0 to 2:
    - C < 1: positive spatial autocorrelation
    - C = 1: no spatial autocorrelation (random)
    - C > 1: negative spatial autocorrelation
    """

    def __init__(
        self,
        weights_matrix: np.ndarray | None = None,
        rng: Any | None = None,
        n_permutations: int = 200,
    ) -> None:
        """
        Initialize GearysC calculator.

        Args:
            weights_matrix: Spatial weights matrix
            rng: Optional seed or np.random.Generator used for the
                permutational variance estimate (resolved via
                ``resolve_rng``; a fixed seed keeps the estimate
                deterministic).
            n_permutations: Number of random value permutations used to
                estimate Var(C).
        """
        self.weights_matrix = weights_matrix
        self._rng = resolve_rng(rng)
        self.n_permutations = n_permutations

    def _permutational_variance(self, values: np.ndarray) -> float:
        """Estimate Var(C) by random value permutations (randomization).

        Recomputes Geary's C on ``n_permutations`` random permutations of the
        observed values and returns the sample variance of the resulting
        statistic. This is the permutational (randomization-assumption)
        variance of Geary's C.
        """
        W = np.asarray(self.weights_matrix, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        z_centered = values - np.mean(values)
        var_z = np.sum(z_centered**2) / (n - 1)
        w_sum = np.sum(W)

        def _c(vals: np.ndarray) -> float:
            diff_sq = (vals[:, np.newaxis] - vals[np.newaxis, :]) ** 2
            return float((np.sum(W * diff_sq) / (2.0 * w_sum)) / var_z)

        perm_stats = np.empty(self.n_permutations, dtype=np.float64)
        for idx in range(self.n_permutations):
            perm_stats[idx] = _c(self._rng.permutation(values))
        return float(np.var(perm_stats, ddof=1))

    def compute(
        self, values: np.ndarray, coords: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Compute Geary's C statistic.

        Args:
            values: Array of values at each location
            coords: Optional coordinates to generate weights

        Returns:
            Dictionary with C, expected_C, var_C, z_score, p_value
        """
        if self.weights_matrix is None:
            if coords is not None:
                self.weights_matrix = _generate_weights(coords)
            else:
                raise ValueError("weights_matrix must be set or coords provided")

        n = len(values)
        if len(values) != self.weights_matrix.shape[0]:
            raise ValueError(
                f"Values length ({n}) must match weights "
                f"({self.weights_matrix.shape[0]})"
            )
        if n < 2:
            raise ValueError("Geary's C requires at least 2 data points")

        if np.std(values) == 0:
            return {
                "C": 1.0,
                "expected_C": 1.0,
                "var_C": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
            }

        z = values - np.mean(values)
        W = self.weights_matrix
        w_sum = np.sum(W)

        # Geary's C numerator: sum of weighted squared differences
        diff_sq = (z[:, np.newaxis] - z[np.newaxis, :]) ** 2
        numerator = np.sum(W * diff_sq)

        # Variance term
        var_z = np.sum(z**2) / (n - 1)
        C = (numerator / (2 * w_sum)) / var_z

        # Permutational variance of C under the randomization assumption
        expected_C = 1.0
        var_C = self._permutational_variance(values)

        z_score_val = (C - expected_C) / np.sqrt(var_C) if var_C > 0 else 0.0
        p_value = erfc(abs(z_score_val) / sqrt(2))

        return {
            "C": C,
            "expected_C": expected_C,
            "var_C": var_C,
            "z_score": z_score_val,
            "p_value": p_value,
        }


class GetisOrd:
    """
    Class-based interface for the Getis-Ord G* statistic (hot-spot analysis).

    Getis-Ord G* identifies statistically significant hot spots (high-value
    clusters) and cold spots (low-value clusters) in spatial data.
    """

    def __init__(self, weights_matrix: np.ndarray | None = None):
        """
        Initialize GetisOrd calculator.

        Args:
            weights_matrix: Spatial weights matrix (should include
                            self-weights for G* calculation)
        """
        self.weights_matrix = weights_matrix

    def compute(
        self, values: np.ndarray, coords: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Compute Getis-Ord G* statistics.

        Args:
            values: Array of attribute values at each location
            coords: Optional array of coordinates (n×2). If provided and
                    no weights_matrix was set at init, generates an inverse-distance
                    weight matrix with self-weights (diagonal = 1 for G*).

        Returns:
            Dictionary containing:
                'local_g'  – Local G* statistic for each location (numpy array)
                'z_scores' – Z-scores for each location (numpy array)
                'global_g' – Global Getis-Ord G statistic (float)
        """
        if self.weights_matrix is None:
            if coords is not None:
                self.weights_matrix = _generate_weights(coords, include_self=True)
            else:
                raise ValueError("weights_matrix must be set or coords provided")

        n = len(values)
        if n != self.weights_matrix.shape[0]:
            raise ValueError(
                f"Values array length ({n}) must match weights matrix size "
                f"({self.weights_matrix.shape[0]})"
            )

        W = self.weights_matrix
        sum_x = np.sum(values)
        mean_x = np.mean(values)
        sum_x_sq = np.sum(values**2)
        s = np.sqrt((sum_x_sq / n) - mean_x**2)

        g_star = np.zeros(n)
        z_scores = np.zeros(n)

        for i in range(n):
            w_i = W[i]
            sum_w = np.sum(w_i)
            sum_wx = np.sum(w_i * values)

            numerator = sum_wx - mean_x * sum_w
            denom = (
                s * np.sqrt((n * np.sum(w_i**2) - sum_w**2) / (n - 1)) if n > 1 else 0.0
            )

            if sum_x > 0:
                g_star[i] = sum_wx / sum_x
            if denom > 0:
                z_scores[i] = numerator / denom

        # Global Getis-Ord G
        total_weights = np.sum(W)
        if total_weights > 0 and sum_x > 0:
            global_g_val = np.sum(W * np.outer(values, values)) / (
                total_weights * sum_x
            )
        else:
            global_g_val = 0.0

        return {
            "local_g": g_star,
            "z_scores": z_scores,
            "global_g": global_g_val,
        }


def getis_ord_g(values: np.ndarray, weights_matrix: np.ndarray) -> dict[str, Any]:
    """
    Calculate Getis-Ord G* statistic for hot spot analysis.

    Args:
        values: Array of values at each location
        weights_matrix: Spatial weights matrix

    Returns:
        Dictionary with G* statistics for each location and global G
    """
    n = len(values)
    sum_x = np.sum(values)
    mean_x = np.mean(values)
    sum_x_sq = np.sum(values**2)
    s = np.sqrt((sum_x_sq / n) - (mean_x**2))

    g_star = np.zeros(n)
    z_scores = np.zeros(n)

    for i in range(n):
        w_i = weights_matrix[i]
        sum_w = np.sum(w_i)
        sum_wx = np.sum(w_i * values)

        numerator = sum_wx - mean_x * sum_w
        ss = s * np.sqrt((n * sum_w**2 - sum_w**2) / (n - 1))

        if ss > 0:
            g_star[i] = sum_wx / sum_x
            z_scores[i] = numerator / ss

    total_weights = np.sum(weights_matrix)
    global_g = np.sum(weights_matrix * np.outer(values, values)) / (
        total_weights * sum_x
    )

    return {
        "local_g": g_star,
        "z_scores": z_scores,
        "global_g": global_g,
    }


def ripley_k(
    points: np.ndarray,
    distances: list[float],
    area: float,
    boundary_correction: bool = True,
) -> dict[str, np.ndarray]:
    """
    Calculate Ripley's K function for point pattern analysis.

    Args:
        points: Array of point coordinates (n x 2)
        distances: List of distances at which to calculate K
        area: Total area of the study region
        boundary_correction: Whether to apply edge correction

    Returns:
        Dictionary with K function values and L function transform
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (n x 2) array of coordinates")
    if area <= 0:
        raise ValueError("area must be positive")
    n_points = points.shape[0]
    if n_points < 2:
        return {
            "distances": np.array(distances),
            "k_function": np.zeros(len(distances)),
            "l_function": np.zeros(len(distances)) - np.array(distances),
        }

    k_values = np.zeros(len(distances))
    l_values = np.zeros(len(distances))

    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    for i, r in enumerate(distances):
        count = np.sum(dist_matrix <= r) - n_points
        k = (area * count) / (n_points * (n_points - 1))
        k_values[i] = k
        l_values[i] = np.sqrt(k / np.pi) - r

    return {
        "distances": np.array(distances),
        "k_function": k_values,
        "l_function": l_values,
    }


def semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    lag_distances: list[float],
    tolerance: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Calculate empirical semivariogram.

    Args:
        coords: Array of coordinates (n x 2)
        values: Array of values at each location
        lag_distances: List of lag distances at which to calculate semivariance
        tolerance: Tolerance for binning point pairs by distance

    Returns:
        Dictionary with semivariogram values
    """
    n_points = coords.shape[0]
    n_lags = len(lag_distances)

    semivariance = np.zeros(n_lags)
    count = np.zeros(n_lags, dtype=int)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            for k, lag in enumerate(lag_distances):
                if abs(dist - lag) <= tolerance:
                    sq_diff = (values[i] - values[j]) ** 2
                    semivariance[k] += sq_diff
                    count[k] += 1
                    break

    valid_lags = count > 0
    semivariance[valid_lags] = semivariance[valid_lags] / (2 * count[valid_lags])

    return {
        "lag_distances": np.array(lag_distances),
        "semivariance": semivariance,
        "count": count,
    }


def spatial_descriptive_statistics(
    coords: np.ndarray, values: np.ndarray
) -> SpatialDescriptiveStats:
    """
    Calculate spatial descriptive statistics.

    Args:
        coords: Array of coordinates (n x 2)
        values: Array of values at each location

    Returns:
        SpatialDescriptiveStats object with calculated statistics
    """
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    var_val = np.var(values)
    min_val = np.min(values)
    max_val = np.max(values)

    total_weight = np.sum(values)
    if total_weight > 0:
        centroid_x = np.sum(coords[:, 0] * values) / total_weight
        centroid_y = np.sum(coords[:, 1] * values) / total_weight
    else:
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])

    centroid = np.array([centroid_x, centroid_y])
    distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
    dispersion = np.mean(distances)

    diff = values - mean_val
    skewness = np.sum(diff**3) / (len(values) * std_val**3)
    kurtosis = np.sum(diff**4) / (len(values) * std_val**4) - 3

    return SpatialDescriptiveStats(
        mean=mean_val,
        median=median_val,
        stdev=std_val,
        variance=var_val,
        min_value=min_val,
        max_value=max_val,
        centroid=(centroid_x, centroid_y),
        dispersion=dispersion,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def spatial_entropy(values: np.ndarray, bins: int = 10) -> float:
    """
    Calculate spatial entropy of a distribution.

    Args:
        values: Array of values
        bins: Number of bins for histogram

    Returns:
        Entropy value
    """
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def local_indicators_spatial_association(
    values: np.ndarray, weights_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Calculate Local Indicators of Spatial Association (LISA).

    Args:
        values: Array of values at each location
        weights_matrix: Spatial weights matrix

    Returns:
        Dictionary with LISA statistics for each location
    """
    n = len(values)
    z = (values - np.mean(values)) / np.std(values)

    lisa = np.zeros(n)
    expected_i = -1.0 / (n - 1)
    var_i = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.zeros(n)

    for i in range(n):
        w_i = weights_matrix[i]
        sum_w = np.sum(w_i)

        if sum_w > 0:
            w_std = w_i / sum_w
            lisa[i] = z[i] * np.sum(w_std * z)

            b2 = np.sum(z**4) / n
            s1 = np.sum(w_std**2)
            var_i[i] = s1 * (n - b2) / (n - 1)

            z_scores[i] = (lisa[i] - expected_i) / np.sqrt(var_i[i])
            p_values[i] = 2 * (1 - np.abs(np.clip(z_scores[i], -8, 8) / 8))

    classifications = np.zeros(n, dtype=int)
    significant = p_values <= 0.05

    for i in range(n):
        if not significant[i]:
            continue
        if z[i] > 0:
            classifications[i] = 1 if lisa[i] > 0 else 3
        else:
            classifications[i] = 2 if lisa[i] > 0 else 4

    return {
        "lisa": lisa,
        "z_scores": z_scores,
        "p_values": p_values,
        "classifications": classifications,
        "significant": significant,
    }


__all__ = [
    "SpatialDescriptiveStats",
    "MoranI",
    "GearysC",
    "GetisOrd",
    "morans_i_variance",
    "getis_ord_g",
    "ripley_k",
    "semivariogram",
    "spatial_descriptive_statistics",
    "spatial_entropy",
    "local_indicators_spatial_association",
]
