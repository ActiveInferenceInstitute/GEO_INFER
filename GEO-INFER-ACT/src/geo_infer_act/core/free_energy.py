"""
Free energy calculation for active inference models.

This module implements variational free energy calculations for different
types of active inference models, including categorical and Gaussian models.

References:
    - Friston, K. (2010). The free-energy principle: a unified brain theory?
    - Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference
    - Formal analogue: fep_lean topic fep-001 (variational free energy bound),
      fep-002 (variational evidence bound via KL divergence)

The fep_lean topic ids are correspondence-of-constructs references into a
separate Lean formalization catalogue, canonically mapped in
`fep_lean/specs/geo-infer-notation-bridge/data/notation-map.yaml` and
documented in `GEO-INFER-ACT/docs/fep_lean_notation_bridge.md`. They state
no verification relationship between this numerical implementation and the
Lean proofs.
"""

from typing import Any, cast
import logging

import numpy as np

from geo_infer_act.core.types import FreeEnergyBreakdown
from geo_infer_act.utils.math import kl_divergence, softmax

logger = logging.getLogger(__name__)

EPSILON = 1e-12

# Preferences may be a plain vector or the structured dict shape produced by
# helpers such as ``hazard_policy_prior``; ``_preferences_to_vector`` lowers
# both into a belief-aligned vector before use.
PreferenceInput = np.ndarray | dict[str, Any]


def _coerce_probability_vector(
    values: np.ndarray | list | tuple,
    target_length: int | None = None,
    *,
    use_softmax: bool = False,
) -> np.ndarray:
    """Return a finite normalized probability vector with optional alignment."""
    vector = np.asarray(values, dtype=float).reshape(-1)

    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Probability vectors must be nonempty and finite")
    if target_length is not None and len(vector) != target_length:
        raise ValueError(f"Probability vector must have length {target_length}")

    if use_softmax:
        vector = softmax(vector)
    else:
        if np.any(vector < 0):
            raise ValueError("Probability vectors must be nonnegative")
        total = float(np.sum(vector))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Probability vectors must have finite positive mass")
        vector = vector / total

    return vector


def validate_spd_precision(name: str, matrix: np.ndarray) -> None:
    """Validate one precision matrix as finite, symmetric, positive definite.

    Shared by ``FreeEnergyCalculator``, ``VariationalInference``, and
    ``BayesianBeliefUpdate`` so the Gaussian surface enforces one contract.

    Args:
        name: Parameter name used in error messages.
        matrix: Candidate precision matrix.

    Raises:
        ValueError: If the matrix is non-finite, asymmetric, or not
            positive definite.
    """
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be finite and symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _preferences_to_vector(preferences: Any, target_length: int) -> np.ndarray:
    """Normalize supported preference shapes to a vector aligned with beliefs."""
    if isinstance(preferences, dict):
        for key in ("states", "observations", "preferences"):
            if preferences.get(key) is not None:
                preferences = preferences[key]
                break
        else:
            preferences = np.ones(target_length) / max(target_length, 1)
    return _coerce_probability_vector(preferences, target_length)


def compute_policy_expected_free_energy(
    beliefs: np.ndarray,
    policy: dict[str, Any],
    preferences: np.ndarray | dict[str, Any] | None = None,
    return_breakdown: bool = False,
) -> float | FreeEnergyBreakdown:
    """Compute expected free energy for one policy (single shared implementation).

    This is the canonical expected-free-energy (EFE) evaluator for discrete
    policies: :meth:`FreeEnergyCalculator.compute_expected_free_energy` and
    :meth:`PolicySelector.compute_expected_free_energy` both delegate here so
    the two classes cannot drift.

    The score decomposes as

        G = temporal_discount * pragmatic_value
            - exploration_bonus * epistemic_value
            + risk + ambiguity

    where the epistemic term is the KL divergence between the policy's
    expected posterior and its predictive prior when the policy supplies one
    (``expected_posterior`` / ``posterior_beliefs``), and the predictive
    entropy otherwise.

    Args:
        beliefs: Current beliefs.
        policy: Policy to evaluate. Supported diagnostic keys include
            ``expected_free_energy`` for externally supplied scores,
            ``predicted_beliefs`` or ``expected_observation`` for
            policy-conditioned predictive distributions,
            ``expected_posterior``/``posterior_beliefs`` for the
            policy-conditioned posterior, ``exploration_bonus``,
            ``risk_preference``, ``ambiguity``, and ``temporal_discount``.
        preferences: Prior preferences; either a vector or a structured dict
            with ``states``/``observations``/``preferences`` keys.
        return_breakdown: When true, return the decomposed
            :class:`FreeEnergyBreakdown` instead of the scalar.

    Returns:
        Expected free energy value, or a decomposed result object when
        ``return_breakdown`` is true.

    Raises:
        ValueError: If ``beliefs`` is empty or cannot be coerced to a
            normalized probability vector.

    Notes:
        ``temporal_discount`` defaults to ``1.0`` (no temporal discounting of
        the pragmatic term) at this single definition site; the historical
        ``PolicySelector`` default of ``0.9`` was an accidental drift, not a
        per-class contract (no test or caller pins it).  Policies that need a
        discount pass ``temporal_discount`` explicitly, as
        ``PolicySelector._create_default_policies`` already does.
    """
    if "expected_free_energy" in policy:
        expected_free_energy = float(policy["expected_free_energy"])
        if not np.isfinite(expected_free_energy):
            raise ValueError("expected_free_energy must be finite")
        if return_breakdown:
            return FreeEnergyBreakdown(
                free_energy=expected_free_energy,
                metadata={"policy_supplied_expected_free_energy": True},
            )
        return expected_free_energy

    beliefs = _coerce_probability_vector(beliefs)
    if beliefs.size == 0:
        raise ValueError("belief and preference vectors must not be empty")

    if "predicted_beliefs" in policy:
        predictive = _coerce_probability_vector(
            policy["predicted_beliefs"], len(beliefs)
        )
    elif "expected_observation" in policy:
        predictive = _coerce_probability_vector(
            policy["expected_observation"], len(beliefs)
        )
    else:
        predictive = beliefs

    entropy = float(-np.sum(predictive * np.log(predictive + EPSILON)))
    if "expected_posterior" in policy or "posterior_beliefs" in policy:
        expected_posterior = _coerce_probability_vector(
            cast(
                Any,
                policy.get("expected_posterior", policy.get("posterior_beliefs")),
            ),
            len(beliefs),
        )
        # Information gain is the KL divergence between the expected
        # posterior and the predictive prior. Policies without an
        # expected posterior use entropy as their exploration term.
        epistemic_value = kl_divergence(expected_posterior, predictive)
    else:
        expected_posterior = None
        epistemic_value = entropy

    if preferences is not None:
        preference_vector = _preferences_to_vector(preferences, len(predictive))
        pragmatic_value = float(
            -np.sum(predictive * np.log(preference_vector + EPSILON))
        )
    elif "expected_observation" in policy:
        # Without stated preferences an expected observation is scored
        # against the maximally uninformative uniform preference vector.
        uniform_preferences = np.ones_like(predictive) / len(predictive)
        pragmatic_value = float(
            -np.sum(predictive * np.log(uniform_preferences + EPSILON))
        )
    else:
        pragmatic_value = 0.0

    exploration_bonus = float(policy.get("exploration_bonus", 0.1))
    risk_preference = float(policy.get("risk_preference", 0.0))
    temporal_discount = float(policy.get("temporal_discount", 1.0))
    ambiguity = float(policy.get("ambiguity", 0.0))
    if not all(
        np.isfinite(value)
        for value in (exploration_bonus, risk_preference, temporal_discount, ambiguity)
    ):
        raise ValueError("Policy weights must be finite")

    risk = float(risk_preference * np.var(predictive))
    expected_free_energy = float(
        temporal_discount * pragmatic_value
        - exploration_bonus * epistemic_value
        + risk
        + ambiguity
    )

    if return_breakdown:
        return FreeEnergyBreakdown(
            free_energy=expected_free_energy,
            entropy=entropy,
            pragmatic_value=pragmatic_value,
            epistemic_value=epistemic_value,
            risk=risk,
            ambiguity=ambiguity,
            metadata={
                "model_type": "expected_policy",
                "predictive_beliefs": predictive.copy(),
                "expected_posterior": (
                    expected_posterior.copy()
                    if expected_posterior is not None
                    else None
                ),
                "epistemic_value_source": (
                    "expected_posterior_kl"
                    if expected_posterior is not None
                    else "predictive_entropy"
                ),
                "temporal_discount": temporal_discount,
                "exploration_bonus": exploration_bonus,
            },
        )
    return expected_free_energy


class FreeEnergyCalculator:
    """
    Calculator for variational free energy in active inference models.

    The free energy serves as a cost function that agents minimize through
    perception (belief updating) and action (policy selection).

    References:
        - Friston, K. (2010). The free-energy principle: a unified brain theory?
        - Formal analogue: fep_lean topic fep-002 (variational evidence bound
          via KL divergence)
    """

    def __init__(self) -> None:
        """Initialize the free energy calculator with default configuration."""
        self.last_computed_energy: float = 0.0
        self.computation_count: int = 0

    def compute_categorical_free_energy(
        self,
        beliefs: np.ndarray,
        observations: np.ndarray,
        preferences: np.ndarray | None = None,
        return_breakdown: bool = False,
    ) -> float | FreeEnergyBreakdown:
        """
        Compute variational free energy for categorical models.

        The free energy F is decomposed into accuracy (expected log-likelihood)
        and complexity (KL divergence from prior):

        F[q(s), o] = E_q[log q(s)] - E_q[log p(o,s)]
                   = D_KL[q(s)||p(s)] - E_q[log p(o|s)]
                   = Complexity - Accuracy

        Where:
        - q(s) is the variational posterior (beliefs)
        - p(s) is the prior
        - p(o|s) is the likelihood of observations given states
        - D_KL is the Kullback-Leibler divergence

        Mathematical Foundation:
        The free energy provides an upper bound on the negative log evidence:
        -log p(o) ≤ F[q(s), o]

        Minimizing free energy simultaneously:
        1. Maximizes model evidence (Occam's principle)
        2. Minimizes prediction error (Darwinian imperative)

        Args:
            beliefs: Current variational posterior q(s) over hidden states
            observations: Observed data vector o
            preferences: Prior preferences C (log prior probabilities)

        Returns:
            Free energy value F[q(s), o], or a decomposed result object when
            ``return_breakdown`` is true.

        References:
            - Friston, K. (2010). The free-energy principle: a unified brain theory?
            - Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference
        """
        beliefs = np.asarray(beliefs)
        if beliefs.dtype == object:
            try:
                beliefs = np.stack([np.asarray(b, dtype=float) for b in beliefs.flat])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "object-dtype beliefs must contain numeric values that "
                    "stack into one numeric array"
                ) from exc
        else:
            beliefs = beliefs.astype(float)
        beliefs = _coerce_probability_vector(beliefs)

        entropy = float(-np.sum(beliefs * np.log(beliefs + EPSILON)))

        if preferences is not None:
            prior = _coerce_probability_vector(preferences, len(beliefs))
        else:
            prior = np.ones_like(beliefs) / len(beliefs)
        complexity = float(
            np.sum(beliefs * (np.log(beliefs + EPSILON) - np.log(prior + EPSILON)))
        )

        obs_prob = _coerce_probability_vector(
            observations, len(beliefs), use_softmax=True
        )
        accuracy = float(np.sum(beliefs * np.log(obs_prob + EPSILON)))

        free_energy = float(complexity - accuracy)
        self.last_computed_energy = free_energy
        self.computation_count += 1

        if return_breakdown:
            return FreeEnergyBreakdown(
                free_energy=free_energy,
                accuracy=accuracy,
                complexity=complexity,
                entropy=entropy,
                metadata={"model_type": "categorical"},
            )
        return free_energy

    def compute_gaussian_free_energy(
        self,
        mean: np.ndarray,
        precision: np.ndarray,
        observations: np.ndarray,
        prior_mean: np.ndarray | None = None,
        prior_precision: np.ndarray | None = None,
        *,
        observation_matrix: np.ndarray | None = None,
        observation_precision: np.ndarray | None = None,
    ) -> float:
        """
        Compute free energy for Gaussian models.

        Args:
            mean: Current belief mean
            precision: Current belief precision matrix
            observations: Observed data
            prior_mean: Prior mean
            prior_precision: Prior precision matrix

        Returns:
            Free energy value
        """
        mean = np.asarray(mean, dtype=float).reshape(-1)
        precision = np.asarray(precision, dtype=float)
        observations = np.asarray(observations, dtype=float).reshape(-1)
        if mean.size == 0 or observations.size == 0:
            raise ValueError("mean and observations must be non-empty vectors")
        if precision.shape != (mean.size, mean.size):
            raise ValueError("precision must be square with one row per state")
        if prior_mean is None:
            prior_mean = np.zeros_like(mean)
        else:
            prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
        if prior_precision is None:
            prior_precision = np.eye(len(mean))
        else:
            prior_precision = np.asarray(prior_precision, dtype=float)
        if prior_mean.shape != mean.shape or prior_precision.shape != precision.shape:
            raise ValueError(
                "prior mean and precision must match the belief dimensions"
            )
        for name, matrix in (
            ("precision", precision),
            ("prior_precision", prior_precision),
        ):
            validate_spd_precision(name, matrix)
        if not all(
            np.all(np.isfinite(value)) for value in (mean, observations, prior_mean)
        ):
            raise ValueError("mean, prior_mean and observations must be finite")
        observation_matrix = np.asarray(
            np.eye(observations.size, mean.size)
            if observation_matrix is None
            else observation_matrix,
            dtype=float,
        )
        observation_precision = np.asarray(
            np.eye(observations.size)
            if observation_precision is None
            else observation_precision,
            dtype=float,
        )
        if observation_matrix.shape != (observations.size, mean.size) or not np.all(
            np.isfinite(observation_matrix)
        ):
            raise ValueError(
                "observation_matrix must be finite with shape (observations, states)"
            )
        if observation_precision.shape != (observations.size, observations.size):
            raise ValueError("observation_precision must match observations")
        validate_spd_precision("observation_precision", observation_precision)
        covariance = np.linalg.solve(precision, np.eye(mean.size))

        # Complexity term (KL divergence from prior), using log-determinants
        # instead of determinants so well-conditioned large matrices remain
        # finite.
        _, logdet_prior = np.linalg.slogdet(prior_precision)
        _, logdet_precision = np.linalg.slogdet(precision)
        complexity = 0.5 * (
            np.trace(prior_precision @ covariance)
            + (mean - prior_mean).T @ prior_precision @ (mean - prior_mean)
            - len(mean)
            + logdet_precision
            - logdet_prior
        )

        # Expected negative log likelihood includes posterior uncertainty.
        residual = observations - observation_matrix @ mean
        accuracy = 0.5 * (
            observations.size * np.log(2 * np.pi)
            - np.linalg.slogdet(observation_precision)[1]
            + residual.T @ observation_precision @ residual
            + np.trace(
                observation_precision
                @ observation_matrix
                @ covariance
                @ observation_matrix.T
            )
        )

        free_energy = float(complexity + accuracy)
        self.last_computed_energy = free_energy
        self.computation_count += 1

        return free_energy

    def compute_expected_free_energy(
        self,
        beliefs: np.ndarray,
        policy: dict[str, Any],
        preferences: PreferenceInput | None = None,
        return_breakdown: bool = False,
    ) -> float | FreeEnergyBreakdown:
        """
        Compute expected free energy for policy evaluation.

        Delegates to the module-level
        :func:`compute_policy_expected_free_energy` — the single shared
        implementation also used by ``PolicySelector.compute_expected_free_energy``
        — and records the score in the calculator's bookkeeping counters.

        Args:
            beliefs: Current beliefs
            policy: Policy to evaluate. Supported diagnostic keys include
                ``expected_free_energy`` for externally supplied scores,
                ``predicted_beliefs`` or ``expected_observation`` for
                policy-conditioned predictive distributions,
                ``exploration_bonus``, ``risk_preference``, ``ambiguity``,
                and ``temporal_discount`` (default ``1.0``, no temporal
                discounting; see the shared implementation's Notes).
            preferences: Prior preferences

        Returns:
            Expected free energy value, or a decomposed result object when
            ``return_breakdown`` is true.
        """
        policy_supplied = isinstance(policy, dict) and "expected_free_energy" in policy
        result = compute_policy_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=return_breakdown
        )
        if not policy_supplied:
            value = result.free_energy if return_breakdown else result
            self.last_computed_energy = float(value)
            self.computation_count += 1
        return result

    def compute(
        self,
        beliefs: np.ndarray | dict,
        observations: np.ndarray | None = None,
        preferences: np.ndarray | None = None,
        model_type: str = "categorical",
    ) -> float:
        """General free energy compute dispatching."""
        if model_type == "categorical":
            if isinstance(beliefs, dict):
                beliefs_arr = beliefs.get("states", beliefs.get("mean"))
            else:
                beliefs_arr = beliefs
            if not isinstance(beliefs_arr, np.ndarray):
                raise ValueError(
                    "categorical beliefs must be an array or a dict with "
                    "'states' or 'mean'"
                )
            obs = (
                observations
                if observations is not None
                else np.ones_like(beliefs_arr) / len(beliefs_arr)
            )
            return cast(
                float,
                self.compute_categorical_free_energy(beliefs_arr, obs, preferences),
            )
        elif model_type == "gaussian":
            if not isinstance(beliefs, dict):
                raise ValueError(
                    "gaussian beliefs must be a dict with 'mean' and 'precision'"
                )
            mean = beliefs.get("mean", beliefs)
            precision = beliefs.get("precision", np.eye(len(mean)))
            obs = observations if observations is not None else np.zeros_like(mean)
            prior_mean = (
                preferences.get("mean") if isinstance(preferences, dict) else None
            )
            prior_prec = (
                preferences.get("precision") if isinstance(preferences, dict) else None
            )
            return self.compute_gaussian_free_energy(
                mean, precision, obs, prior_mean, prior_prec
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
