"""
Mathematical utilities for active inference models.

This module provides mathematical functions for computing
information-theoretic quantities, probability distributions,
and Active Inference specific calculations.
"""

import logging
from typing import cast
from collections.abc import Callable

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """
    Compute softmax transformation of input array.

    Args:
        x: Input array
        temperature: Temperature parameter (higher = more uniform)
        axis: Axis along which to compute softmax

    Returns:
        Softmax-transformed array
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("softmax input must not be empty")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if np.any(~np.isfinite(values)):
        raise ValueError("softmax input must contain only finite values")
    if values.ndim == 0:
        return np.ones_like(values, dtype=float)

    if (
        not isinstance(axis, (int, np.integer))
        or not -values.ndim <= axis < values.ndim
    ):
        raise ValueError(
            f"axis {axis} is invalid for an array with {values.ndim} dimensions"
        )
    axis = int(axis) % values.ndim
    if values.shape[axis] == 0:
        raise ValueError("softmax cannot normalize an empty axis")

    # Subtract the maximum before scaling so large EFE magnitudes remain
    # finite while preserving the exact normalized result.
    x_scaled = (values - np.max(values, axis=axis, keepdims=True)) / temperature
    exp_x = np.exp(x_scaled)
    denominator = np.sum(exp_x, axis=axis, keepdims=True)
    return cast(np.ndarray, exp_x / denominator)


def normalize_distribution(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize array to form a probability distribution.

    Args:
        x: Input array
        axis: Axis along which to normalize

    Returns:
        Normalized probability distribution
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("Distribution values must be finite")

    if values.ndim == 0:
        return np.asarray(1.0, dtype=float)

    # Ensure non-negative values while retaining the historical clipping
    # behaviour used by callers that pass small numerical negatives.
    values = np.maximum(values, 0.0)
    sum_x = np.sum(values, axis=axis, keepdims=True)
    normalized = np.divide(
        values,
        sum_x,
        out=np.zeros_like(values),
        where=sum_x != 0,
    )

    # A zero vector is not a probability distribution.  Use the least
    # informative valid distribution rather than silently returning zeros.
    width = values.shape[axis]
    if width == 0:
        return normalized
    uniform = np.full_like(values, 1.0 / width)
    return np.where(sum_x == 0, uniform, normalized)


def categorical_posterior(
    prior_beliefs: np.ndarray,
    observation: np.ndarray,
    likelihood_matrix: np.ndarray,
) -> np.ndarray:
    """Compute a numerically stable categorical Bayes posterior.

    ``likelihood_matrix`` is interpreted as ``P(observation | state)`` with
    observations on rows and states on columns.  Observation values may be
    non-negative hard or soft counts.  The likelihood product is evaluated in
    log space, so large count totals cannot underflow the normalization step.

    Args:
        prior_beliefs: Prior state distribution.
        observation: Non-negative observation/count vector.
        likelihood_matrix: Observation model with shape
            ``(observation_dim, state_dim)``.

    Returns:
        A finite, non-negative state distribution whose entries sum to one.

    Raises:
        ValueError: If an input has an invalid shape/value or no state has
            non-zero posterior support.
    """
    prior = np.asarray(prior_beliefs, dtype=float).reshape(-1)
    obs = np.asarray(observation, dtype=float).reshape(-1)
    likelihood = np.asarray(likelihood_matrix, dtype=float)

    if prior.size == 0:
        raise ValueError("Categorical priors must not be empty")
    if obs.size == 0:
        raise ValueError("Categorical observations must not be empty")
    if likelihood.shape != (obs.size, prior.size):
        raise ValueError(
            "Categorical likelihood matrix must have shape "
            f"({obs.size}, {prior.size}), got {likelihood.shape}"
        )
    if not np.all(np.isfinite(prior)):
        raise ValueError("Categorical priors must contain finite values")
    if not np.all(np.isfinite(obs)):
        raise ValueError("Categorical observations must contain finite values")
    if not np.all(np.isfinite(likelihood)):
        raise ValueError("Categorical likelihoods must contain finite values")
    if np.any(prior < 0):
        raise ValueError("Categorical priors must be non-negative")
    if np.any(obs < 0):
        raise ValueError("Categorical observations must be non-negative")
    if np.any(likelihood < 0):
        raise ValueError("Categorical likelihoods must be non-negative")

    prior_total = float(np.sum(prior))
    if prior_total <= 0:
        raise ValueError("Categorical priors must have positive total mass")
    prior = prior / prior_total

    column_totals = np.sum(likelihood, axis=0)
    if np.any(column_totals <= 0):
        raise ValueError("Each categorical likelihood column needs positive mass")
    likelihood = likelihood / column_totals

    with np.errstate(divide="ignore", invalid="ignore"):
        log_likelihood = np.sum(
            np.where(
                obs[:, np.newaxis] > 0,
                obs[:, np.newaxis] * np.log(likelihood),
                0.0,
            ),
            axis=0,
        )
        log_prior = np.full(prior.shape, -np.inf, dtype=float)
        positive_prior = prior > 0
        log_prior[positive_prior] = np.log(prior[positive_prior])
        log_posterior = log_likelihood + log_prior

    maximum = float(np.max(log_posterior))
    if not np.isfinite(maximum):
        raise ValueError(
            "Categorical observation has zero posterior support under the "
            "current prior and likelihood model"
        )

    posterior = np.exp(log_posterior - maximum)
    posterior_total = float(np.sum(posterior))
    if not np.isfinite(posterior_total) or posterior_total <= 0:
        raise ValueError("Categorical posterior could not be normalized")
    return np.asarray(posterior / posterior_total, dtype=float)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence D(p||q)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )
    if p.size == 0:
        raise ValueError("KL divergence inputs must not be empty")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("KL divergence inputs must be finite")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("KL divergence inputs must be non-negative")

    p_total = float(np.sum(p))
    q_total = float(np.sum(q))
    if p_total <= 0 or q_total <= 0:
        raise ValueError("KL divergence inputs must have positive total mass")
    p = p / p_total
    q = q / q_total
    q_safe = np.maximum(q, epsilon)
    positive = p > 0
    divergence = np.sum(p[positive] * np.log(p[positive] / q_safe[positive]))
    return float(max(0.0, divergence))


def entropy(p: np.ndarray, base: float | str = "e") -> float:
    """
    Compute entropy of a probability distribution.

    Args:
        p: Probability distribution
        base: Logarithm base ('e', 2, 10, or float)

    Returns:
        Entropy value
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return 0.0
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("entropy input must be finite and non-negative")
    total = float(np.sum(p))
    if total <= 0:
        raise ValueError("entropy input must have positive total mass")
    p = p / total

    # Filter out zero probabilities
    p_nonzero = p[p > 0]

    if len(p_nonzero) == 0:
        return 0.0

    # Compute entropy
    if base == "e":
        entropy_value = np.sum(p_nonzero * np.log(p_nonzero))
    elif base == 2:
        entropy_value = np.sum(p_nonzero * np.log2(p_nonzero))
    elif base == 10:
        entropy_value = np.sum(p_nonzero * np.log10(p_nonzero))
    else:
        entropy_value = np.sum(p_nonzero * (np.log(p_nonzero) / np.log(float(base))))

    return float(-entropy_value)


def mutual_information(joint: np.ndarray) -> float:
    """
    Compute mutual information from joint probability distribution.

    Args:
        joint: Joint probability distribution (2D array)

    Returns:
        Mutual information value
    """
    joint = np.asarray(joint)

    # Marginal distributions
    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)

    # Mutual information
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))

    return mi


def precision_weighted_error(
    mean: np.ndarray, target: np.ndarray, precision: np.ndarray
) -> float:
    """
    Compute precision-weighted prediction error.

    Args:
        mean: Predicted mean
        target: Target values
        precision: Precision matrix

    Returns:
        Precision-weighted error
    """
    error = target - mean

    if precision.ndim == 0 or (precision.ndim == 1 and len(precision) == 1):
        # Scalar precision
        return float(precision * np.sum(error**2))
    elif precision.ndim == 1:
        # Diagonal precision
        return float(np.sum(precision * error**2))
    else:
        # Full precision matrix
        return float(error.T @ precision @ error)


def gaussian_log_likelihood(
    x: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> float:
    """
    Compute log likelihood of Gaussian distribution.

    Args:
        x: Observed values
        mean: Mean of distribution
        precision: Precision matrix (inverse covariance)

    Returns:
        Log likelihood
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    precision = np.asarray(precision, dtype=float)
    if x.shape != mean.shape or x.size == 0:
        raise ValueError("x and mean must be non-empty vectors with the same shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(mean)):
        raise ValueError("x and mean must be finite")

    # Residual
    residual = x - mean

    # Dimensionality
    d = len(x)

    if precision.ndim == 0 or (precision.ndim == 1 and len(precision) == 1):
        # Scalar precision
        scalar_precision = float(precision.reshape(-1)[0])
        if not np.isfinite(scalar_precision) or scalar_precision <= 0:
            raise ValueError("Gaussian precision must be finite and positive")
        log_det_precision = d * np.log(scalar_precision)
        quadratic_form = scalar_precision * np.sum(residual**2)
    elif precision.ndim == 1:
        # Diagonal precision
        if (
            precision.shape != (d,)
            or not np.all(np.isfinite(precision))
            or np.any(precision <= 0)
        ):
            raise ValueError("Diagonal Gaussian precision must be finite and positive")
        log_det_precision = np.sum(np.log(precision))
        quadratic_form = np.sum(precision * residual**2)
    else:
        # Full precision matrix
        if precision.shape != (d, d) or not np.all(np.isfinite(precision)):
            raise ValueError("Full Gaussian precision must be a finite square matrix")
        if not np.allclose(precision, precision.T, atol=1e-10):
            raise ValueError("Full Gaussian precision must be symmetric")
        sign, log_det_precision = np.linalg.slogdet(precision)
        if sign <= 0 or not np.isfinite(log_det_precision):
            raise ValueError("Gaussian precision must be positive definite")
        quadratic_form = residual.T @ precision @ residual

    # Log likelihood
    log_likelihood = 0.5 * (log_det_precision - d * np.log(2 * np.pi) - quadratic_form)

    return float(log_likelihood)


def categorical_log_likelihood(
    observations: np.ndarray, probabilities: np.ndarray
) -> float:
    """
    Compute log likelihood for categorical distribution.

    Args:
        observations: Observed counts or one-hot encoded observations
        probabilities: Category probabilities

    Returns:
        Log likelihood
    """
    observations = np.asarray(observations, dtype=float).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if observations.shape != probabilities.shape or observations.size == 0:
        raise ValueError(
            "observations and probabilities must have the same non-empty shape"
        )
    if not np.all(np.isfinite(observations)) or np.any(observations < 0):
        raise ValueError("observations must be finite and non-negative")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")

    # Ensure probabilities are valid
    probabilities = probabilities / total
    probabilities = np.maximum(probabilities, 1e-10)

    # Compute log likelihood
    return float(np.sum(observations * np.log(probabilities)))


def dirichlet_kl_divergence(alpha1: np.ndarray, alpha2: np.ndarray) -> float:
    """
    Compute KL divergence between two Dirichlet distributions.

    Args:
        alpha1: Parameters of first Dirichlet distribution
        alpha2: Parameters of second Dirichlet distribution

    Returns:
        KL divergence
    """
    from scipy.special import gammaln, digamma

    alpha1 = np.asarray(alpha1)
    alpha2 = np.asarray(alpha2)

    # Sum of parameters
    sum_alpha1 = np.sum(alpha1)
    sum_alpha2 = np.sum(alpha2)

    # KL divergence formula for Dirichlet distributions
    kl = (
        gammaln(sum_alpha1)
        - gammaln(sum_alpha2)
        + np.sum(gammaln(alpha2) - gammaln(alpha1))
        + np.sum((alpha1 - alpha2) * (digamma(alpha1) - digamma(sum_alpha1)))
    )

    return float(kl)


def sample_categorical(
    probabilities: np.ndarray, n_samples: int = 1, random_state: int | None = None
) -> np.ndarray:
    """
    Sample from categorical distribution.

    Args:
        probabilities: Category probabilities
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        Sampled indices
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("probabilities must not be empty")
    if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    probabilities = probabilities / total
    rng = np.random.default_rng(random_state)
    return rng.choice(len(probabilities), size=int(n_samples), p=probabilities)


def compute_free_energy_categorical(
    beliefs: np.ndarray, observations: np.ndarray, prior: np.ndarray | None = None
) -> float:
    """
    Compute variational free energy for categorical models.

    Args:
        beliefs: Current belief distribution
        observations: Observed data
        prior: Prior distribution (uniform if None)

    Returns:
        Free energy value
    """
    beliefs = np.asarray(beliefs)
    observations = np.asarray(observations)

    if prior is None:
        prior = np.ones_like(beliefs) / len(beliefs)

    # Accuracy term (expected log likelihood)
    accuracy = np.sum(beliefs * np.log(observations + 1e-10))

    # Complexity term (KL divergence from prior)
    complexity = kl_divergence(beliefs, prior)

    # Free energy = Complexity - Accuracy
    return cast(float, complexity - accuracy)


def compute_expected_free_energy(
    beliefs: np.ndarray, preferences: np.ndarray, exploration_bonus: float = 0.1
) -> float:
    """
    Compute expected free energy for policy evaluation.

    Args:
        beliefs: Current belief distribution
        preferences: Prior preferences
        exploration_bonus: Exploration bonus weight

    Returns:
        Expected free energy
    """
    beliefs = np.asarray(beliefs)
    preferences = np.asarray(preferences)

    # Epistemic value (information gain)
    epistemic_value = entropy(beliefs)

    # Pragmatic value (preference satisfaction)
    pragmatic_value = -np.sum(beliefs * np.log(preferences + 1e-10))

    # Expected free energy
    expected_free_energy = pragmatic_value - exploration_bonus * epistemic_value

    return float(expected_free_energy)


def numerical_gradient(
    func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.

    Args:
        func: Function to differentiate
        x: Point at which to compute gradient
        h: Step size

    Returns:
        Gradient vector
    """
    x = np.asarray(x)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    return grad


def stable_log_sum_exp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Args:
        x: Input array
        axis: Axis along which to compute

    Returns:
        Stable log-sum-exp
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        empty_shape = np.sum(values, axis=axis, keepdims=True).shape
        return np.full(empty_shape, -np.inf, dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError("log-sum-exp input must not contain NaN values")

    x_max = np.max(values, axis=axis, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = x_max + np.log(
            np.sum(np.exp(values - x_max), axis=axis, keepdims=True)
        )
    return np.where(np.isneginf(x_max), -np.inf, result)


def matrix_log_det(matrix: np.ndarray) -> float:
    """
    Compute log determinant of a matrix safely.

    Args:
        matrix: Input matrix

    Returns:
        Log determinant; -inf when the determinant is non-positive
        (log-det is undefined for det <= 0).
    """
    try:
        det = float(np.linalg.det(matrix))
        if det <= 0:
            logger.debug(
                "matrix_log_det: non-positive determinant %r; returning -inf", det
            )
            return -np.inf
        return float(np.log(det))
    except (np.linalg.LinAlgError, ValueError):
        # Singular or non-finite input: fall back to eigenvalues, keeping the
        # same non-positive-determinant semantics as the direct path.
        eigenvals = np.linalg.eigvals(matrix)
        eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
        if len(eigenvals) == 0:
            return -np.inf
        logger.debug(
            "matrix_log_det: eigenvalue fallback used (det <= 0 or singular); "
            "returning log of pseudo-determinant over %d positive eigenvalues",
            len(eigenvals),
        )
        return float(np.sum(np.log(eigenvals)))


# Additional analysis functions for pattern detection


def detect_stationarity(data: np.ndarray, window_size: int = 10) -> dict[str, float]:
    """
    Detect stationarity in time series data.

    Args:
        data: Time series data
        window_size: Window size for analysis

    Returns:
        Dictionary with stationarity metrics
    """
    data = np.asarray(data)

    if len(data) < 2 * window_size:
        return {"is_stationary": False, "stationarity_score": 0.0}

    # Compute rolling statistics
    n_windows = len(data) - window_size + 1
    rolling_means = []
    rolling_vars = []

    for i in range(n_windows):
        window = data[i : i + window_size]
        rolling_means.append(np.mean(window))
        rolling_vars.append(np.var(window))

    # Compute variation in rolling statistics
    mean_variation = np.var(rolling_means) / (np.mean(rolling_means) ** 2 + 1e-8)
    var_variation = np.var(rolling_vars) / (np.mean(rolling_vars) + 1e-8)

    # Simple stationarity score (lower is more stationary)
    stationarity_score = 1.0 / (1.0 + mean_variation + var_variation)
    is_stationary = mean_variation < 0.1 and var_variation < 0.1

    return {
        "is_stationary": is_stationary,
        "stationarity_score": float(stationarity_score),
        "mean_variation": float(mean_variation),
        "variance_variation": float(var_variation),
    }


def detect_periodicity(
    data: np.ndarray, min_period: int = 2
) -> dict[str, bool | float | int]:
    """
    Detect periodic patterns in data.

    Args:
        data: Time series data
        min_period: Minimum period to consider

    Returns:
        Dictionary with periodicity information
    """
    data = np.asarray(data)

    if len(data) < 2 * min_period:
        return {"is_periodic": False, "period": 0, "strength": 0.0}

    # Autocorrelation analysis
    max_lag = min(len(data) // 2, 50)  # Limit for computational efficiency
    autocorr_list = []

    for lag in range(1, max_lag):
        if lag < len(data):
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorr_list.append(corr if not np.isnan(corr) else 0.0)
        else:
            autocorr_list.append(0.0)

    autocorr = np.array(autocorr_list)

    # Find peaks in autocorrelation
    if len(autocorr) > min_period:
        peaks, _ = find_peaks(autocorr[min_period - 1 :], height=0.3)

        if len(peaks) > 0:
            # Most prominent peak
            best_peak_idx = np.argmax(autocorr[min_period - 1 :][peaks])
            period = peaks[best_peak_idx] + min_period
            strength = autocorr[min_period - 1 :][peaks[best_peak_idx]]

            return {
                "is_periodic": True,
                "period": int(period),
                "strength": float(strength),
                "autocorr_peaks": peaks + min_period,
            }

    return {"is_periodic": False, "period": 0, "strength": 0.0}


def assess_complexity(data: np.ndarray) -> dict[str, float]:
    """
    Assess complexity of data using multiple metrics.

    Args:
        data: Input data array

    Returns:
        Dictionary with complexity metrics
    """
    data = np.asarray(data)

    if data.ndim > 1:
        # For multi-dimensional data, analyze each dimension
        complexities = []
        for dim in range(data.shape[1]):
            dim_complexity = assess_complexity(data[:, dim])
            complexities.append(dim_complexity["overall_complexity"])
        return {"overall_complexity": float(np.mean(complexities))}

    if len(data) < 3:
        return {"overall_complexity": 0.0}

    # 1. Entropy-based complexity
    # Discretize data for entropy calculation
    try:
        hist, _ = np.histogram(data, bins=min(10, len(data) // 2))
        hist = hist + 1  # Add pseudocount
        probs = hist / np.sum(hist)
        entropy_complexity = entropy(probs) / np.log(len(probs))
    except (ValueError, TypeError):
        logger.debug(
            "entropy complexity fallback: entropy(probs) failed",
            exc_info=True,
        )
        entropy_complexity = 0.0

    # 2. Variation complexity
    variation_complexity = np.std(data) / (np.mean(np.abs(data)) + 1e-8)
    variation_complexity = min(1.0, variation_complexity)

    # 3. Autocorrelation complexity
    if len(data) > 1:
        try:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            autocorr_complexity = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
        except (ValueError, TypeError):
            logger.debug(
                "autocorrelation complexity fallback: corrcoef failed",
                exc_info=True,
            )
            autocorr_complexity = 0.5
    else:
        autocorr_complexity = 0.0
    if len(data) > 2:
        try:
            trend_coef = np.polyfit(range(len(data)), data, 1)[0]
            trend_complexity = min(1.0, abs(trend_coef) / (np.std(data) + 1e-8))
        except (ValueError, TypeError):
            logger.debug(
                "trend complexity fallback: polyfit failed",
                exc_info=True,
            )
            trend_complexity = 0.0
    else:
        trend_complexity = 0.0

    # Overall complexity (weighted average)
    weights = [0.3, 0.3, 0.2, 0.2]
    components = [
        entropy_complexity,
        variation_complexity,
        autocorr_complexity,
        trend_complexity,
    ]
    overall_complexity = sum(w * c for w, c in zip(weights, components))

    return {
        "overall_complexity": float(overall_complexity),
        "entropy_complexity": float(entropy_complexity),
        "variation_complexity": float(variation_complexity),
        "autocorr_complexity": float(autocorr_complexity),
        "trend_complexity": float(trend_complexity),
    }


def compute_prediction_accuracy(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """
    Compute various prediction accuracy metrics.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        Dictionary with accuracy metrics
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))

    # R-squared (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    # Correlation coefficient
    try:
        correlation = np.corrcoef(predictions, targets)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except (ValueError, TypeError):
        logger.debug(
            "correlation fallback: corrcoef failed",
            exc_info=True,
        )
        correlation = 0.0
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "correlation": float(correlation),
    }


def compute_information_gain(prior_entropy: float, posterior_entropy: float) -> float:
    """
    Compute information gain from prior to posterior.

    Args:
        prior_entropy: Entropy before observation
        posterior_entropy: Entropy after observation

    Returns:
        Information gain (reduction in entropy)
    """
    return float(max(0.0, prior_entropy - posterior_entropy))


def compute_surprise(
    observation: np.ndarray, predicted_distribution: np.ndarray, sigma: float = 0.1
) -> float:
    """
    Compute surprise of an observation given predicted distribution.

    Args:
        observation: Observed outcome (can be one-hot, index, or continuous)
        predicted_distribution: Predicted probability distribution
        sigma: Standard deviation for continuous observations (default: 0.1)

    Returns:
        Surprise value (negative log probability)
    """
    observation = np.asarray(observation, dtype=float)
    predicted_distribution = np.asarray(predicted_distribution, dtype=float).reshape(-1)
    if predicted_distribution.size == 0:
        raise ValueError("predicted_distribution must not be empty")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation must be finite")
    if not np.all(np.isfinite(predicted_distribution)) or np.any(
        predicted_distribution < 0
    ):
        raise ValueError("predicted_distribution must be finite and non-negative")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")
    total = float(np.sum(predicted_distribution))
    if total <= 0:
        raise ValueError("predicted_distribution must have positive total mass")

    # Ensure valid distribution
    predicted_distribution = predicted_distribution / total
    predicted_distribution = np.maximum(predicted_distribution, 1e-10)
    predicted_distribution /= np.sum(predicted_distribution)

    # For one-hot encoded observations (categorical)
    if (
        observation.ndim == 1
        and len(observation) == len(predicted_distribution)
        and np.allclose(np.sum(observation), 1.0)
    ):
        if np.any(observation < 0):
            raise ValueError("categorical observations must be non-negative")
        prob = np.sum(observation * predicted_distribution)
        return float(-np.log(prob + 1e-10))

    # For single index observation
    elif observation.ndim == 0 or (observation.ndim == 1 and len(observation) == 1):
        idx = int(observation.item() if hasattr(observation, "item") else observation)
        if 0 <= idx < len(predicted_distribution):
            prob = predicted_distribution[idx]
            return float(-np.log(prob + 1e-10))
        else:
            return 10.0  # High surprise for invalid index

    # For continuous/multi-dimensional observations
    else:
        # Compute surprise based on distance from mean prediction
        mean_pred = np.mean(predicted_distribution)
        obs_mean = np.mean(observation)

        # Gaussian surprise approximation
        diff = (obs_mean - mean_pred) ** 2
        surprise = diff / (2 * sigma**2) + 0.5 * np.log(2 * np.pi * sigma**2)

        return float(max(0.0, surprise))


def assess_convergence(
    sequence: np.ndarray, window_size: int = 10, threshold: float = 1e-3
) -> dict[str, bool | float | int]:
    """
    Assess convergence of a sequence.

    Args:
        sequence: Input sequence
        window_size: Window size for convergence check
        threshold: Convergence threshold

    Returns:
        Dictionary with convergence information
    """
    sequence = np.asarray(sequence)

    if len(sequence) < window_size:
        return {
            "converged": False,
            "convergence_step": -1,
            "final_variance": np.var(sequence),
        }

    # Check convergence using moving variance
    for i in range(window_size, len(sequence)):
        window = sequence[i - window_size : i]
        variance = np.var(window)

        if variance < threshold:
            return {
                "converged": True,
                "convergence_step": i,
                "final_variance": float(variance),
                "convergence_rate": float(1.0 / (i + 1)),
            }

    return {
        "converged": False,
        "convergence_step": -1,
        "final_variance": float(np.var(sequence[-window_size:])),
        "convergence_rate": 0.0,
    }


def sample_dirichlet(
    alpha: np.ndarray, random_state: int | np.random.Generator | None = None
) -> np.ndarray:
    """Sample from a Dirichlet distribution using an isolated RNG stream."""
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    return rng.dirichlet(alpha)
