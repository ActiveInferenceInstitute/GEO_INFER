"""
Policy selection for active inference models.

This module implements policy selection mechanisms based on expected
free energy minimization and other active inference principles.

References:
    - Friston, K. (2010). The free-energy principle: a unified brain theory?
    - Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., &
      Pezzulo, G. (2017). Active inference: a process theory
    - Formal analogue: fep_lean topic fep-021 (EFE epistemic-pragmatic
      balance), fep-008 (finite policy objective minimizer)

The fep_lean topic ids are correspondence-of-constructs references into a
separate Lean formalization catalogue, canonically mapped in
`fep_lean/specs/geo-infer-notation-bridge/data/notation-map.yaml` and
documented in `GEO-INFER-ACT/docs/fep_lean_notation_bridge.md`. They state
no verification relationship between this numerical implementation and the
Lean proofs.
"""

from typing import Any
import logging

import numpy as np

from geo_infer_act.core.free_energy import (
    _coerce_probability_vector,
    compute_policy_expected_free_energy,
)
from geo_infer_act.core.types import FreeEnergyBreakdown, PolicyEvaluation
from geo_infer_act.utils.math import softmax

logger = logging.getLogger(__name__)
EPSILON = 1e-12

# Preferences may be a plain vector or the structured dict shape produced by
# helpers such as ``hazard_policy_prior``; ``_preferences_to_vector`` lowers
# both into a belief-aligned vector before use.
PreferenceInput = np.ndarray | dict[str, Any]


def _normalize_vector(values: Any, target_length: int | None = None) -> np.ndarray:
    """Normalize a finite belief/preference vector (raises on empty input)."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("belief and preference vectors must not be empty")
    return _coerce_probability_vector(vector, target_length)


def _policy_to_dict(policy: Any) -> dict[str, Any]:
    """Represent arbitrary action/policy inputs as policy dictionaries."""
    if isinstance(policy, dict):
        return policy
    return {"action": policy, "exploration_bonus": 0.1}


class PolicySelector:
    """
    Policy selector for active inference models.

    Selects actions/policies based on expected free energy minimization,
    balancing exploration (epistemic value) and exploitation (pragmatic value).

    References:
        - Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference
        - Formal analogue: fep_lean topic fep-028 (support-aware finite softmax
          policy), fep-031 (finite Boltzmann-Gibbs weights for the inverse
          temperature), fep-008 (finite policy objective minimizer)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        selection_mode: str = "sample",
        random_seed: int | None = None,
    ):
        """
        Initialize the policy selector.

        Args:
            temperature: Temperature parameter for policy selection
            selection_mode: ``sample`` for stochastic selection or
                ``deterministic`` for lowest expected free energy.
            random_seed: Optional seed for reproducible stochastic selection.
        """
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive")
        self.temperature = float(temperature)
        if selection_mode not in {"sample", "deterministic"}:
            raise ValueError("selection_mode must be 'sample' or 'deterministic'")
        self.selection_mode = selection_mode
        self.rng = np.random.default_rng(random_seed)

    def select_policy(
        self,
        beliefs: np.ndarray,
        policies: list[dict[str, Any]],
        preferences: PreferenceInput | None = None,
    ) -> dict[str, Any]:
        """
        Select a policy based on expected free energy.

        Args:
            beliefs: Current belief distribution
            policies: List of available policies
            preferences: Prior preferences

        Returns:
            Selected policy and associated information
        """
        evaluation = self.evaluate_policy_set(beliefs, policies, preferences)
        policies = evaluation["policies"]
        expected_free_energies = evaluation["expected_free_energies"]
        policy_probs = evaluation["probabilities"]

        if self.selection_mode == "deterministic":
            selected_idx = int(evaluation["best_policy_idx"])
        else:
            selected_idx = int(self.rng.choice(len(policies), p=policy_probs))
        selected_policy = policies[selected_idx]
        selected_evaluation = evaluation["evaluations"][selected_idx]

        return {
            "policy": selected_policy,
            "probability": float(policy_probs[selected_idx]),
            "expected_free_energy": float(expected_free_energies[selected_idx]),
            "all_probabilities": policy_probs,
            "all_free_energies": expected_free_energies,
            "selected_index": selected_idx,
            "evaluation": selected_evaluation,
            "evaluations": evaluation["evaluations"],
        }

    def _create_default_policies(self, n_states: int) -> list[dict[str, Any]]:
        """
        Create default policies for exploration.

        Args:
            n_states: Number of states in the model

        Returns:
            List of default policies
        """
        policies = []

        # Create diverse policies with different characteristics
        for i in range(5):  # Create 5 default policies
            policy = {
                "id": i,
                "exploration_bonus": float(self.rng.uniform(0.0, 0.5)),
                "temporal_discount": float(self.rng.uniform(0.8, 1.0)),
                "risk_preference": float(self.rng.uniform(-0.2, 0.2)),
                "type": "exploration" if i < 3 else "exploitation",
            }
            policies.append(policy)

        return policies

    def compute_expected_free_energy(
        self,
        beliefs: np.ndarray,
        policy: dict[str, Any],
        preferences: PreferenceInput | None = None,
        return_breakdown: bool = False,
    ) -> float | FreeEnergyBreakdown:
        """
        Compute expected free energy for a policy.

        Delegates to the shared module-level implementation in
        ``geo_infer_act.core.free_energy`` so this selector and
        ``FreeEnergyCalculator`` produce identical expected free energies
        for identical inputs (pinned by the parity tests).

        Args:
            beliefs: Current beliefs
            policy: Policy to evaluate; non-dict policies are wrapped as
                ``{"action": policy, "exploration_bonus": 0.1}``.
            preferences: Prior preferences

        Returns:
            Expected free energy value, or a decomposed result object when
            ``return_breakdown`` is true.
        """
        policy = _policy_to_dict(policy)
        return compute_policy_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=return_breakdown
        )

    def compute_policy_precision(
        self, expected_free_energies: np.ndarray, baseline_precision: float = 1.0
    ) -> float:
        """
        Compute precision parameter for policy distribution.

        Args:
            expected_free_energies: Array of EFE values
            baseline_precision: Baseline precision value

        Returns:
            Computed precision parameter
        """
        expected_free_energies = np.asarray(
            expected_free_energies, dtype=float
        ).reshape(-1)
        if expected_free_energies.size == 0:
            raise ValueError("expected_free_energies must not be empty")
        if not np.all(np.isfinite(expected_free_energies)):
            raise ValueError("expected_free_energies must be finite")
        if not np.isfinite(baseline_precision) or baseline_precision <= 0:
            raise ValueError("baseline_precision must be finite and positive")

        # Adaptive precision based on policy differentiation
        efe_range = np.max(expected_free_energies) - np.min(expected_free_energies)

        if efe_range > 1e-6:
            # Higher precision when policies are well-differentiated
            precision = baseline_precision * (1.0 + efe_range)
        else:
            # Lower precision when policies are similar
            precision = baseline_precision * 0.5

        return float(precision)

    def compose_policy_posterior(
        self,
        expected_free_energies: np.ndarray,
        precision: float | None = None,
        prior: np.ndarray | None = None,
        prior_temperature: float = 1.0,
    ) -> dict[str, Any]:
        """
        Compose the policy posterior q(pi) = softmax(-gamma * G)` from raw EFE
        scores, with an optional E-based habit prior (softmax-normalised).

        When ``precision`` is omitted it is inferred adaptively from the
        spread of EFE scores (well-separated policies get sharper selection).
        The returned dict carries the posterior, the inferred
        ``precision``, and the diversity used to infer it, so callers
        can audit how sharply the exploration/exploitation trade-off
        was resolved.

        Args:
            expected_free_energies: Raw EFE per policy (lower = preferred).
            precision: Optional explicit inverse-temperature precision.
            prior: Optional E-vector of prior policy probabilities (habits).
            prior_temperature: Temperature applied to ``prior`` before mixing.

        Returns:
            Dict with ``posterior``, ``precision``, ``efe_scores``,
            ``diversity`` and ``prior`` keys.
        """
        scores = np.asarray(expected_free_energies, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ValueError("expected_free_energies must not be empty")
        if not np.all(np.isfinite(scores)):
            raise ValueError("expected_free_energies must be finite")
        if prior is not None:
            prior = _normalize_vector(prior, len(scores))
        gamma = (
            float(precision)
            if precision is not None
            else float(self.compute_policy_precision(scores))
        )
        logits = -gamma * scores
        if prior is not None:
            logits = logits + np.log(np.clip(prior, EPSILON, 1.0)) / prior_temperature
        posterior = softmax(logits, temperature=1.0)
        return {
            "posterior": posterior,
            "precision": float(gamma),
            "expected_free_energies": scores.tolist(),
            "diversity": float(np.std(scores)),
            "prior": prior.tolist() if prior is not None else None,
        }

    def decompose_efe(
        self,
        beliefs: np.ndarray,
        policies: list[dict[str, Any]],
        preferences: PreferenceInput | None = None,
    ) -> dict[str, Any]:
        """
        Decompose a policy set into its epistemic (information-gain) and
        pragmatic (preference-alignment) contributions and report which term
        dominates each candidate.

        This provides the explanatory half of the EFE decomposition: a policy
        with a large subtractive epistemic term is exploration-driven,
        whereas one whose pragmatic term dominates is exploitation-driven.

        Args:
            beliefs: Current belief distribution.
            policies: Candidate policies to evaluate.
            preferences: Optional prior preferences.

        Returns:
            A dict with ``policies``, ``epistemic_values``,
            ``pragmatic_values``, ``efe_scores``, ``dominance`` (per-policy
            label) and ``exploration_share`` (fraction of policies whose
            epistemic term dominates).
        """
        belief_vector = _normalize_vector(beliefs)
        if not policies:
            policies = self._create_default_policies(len(belief_vector))
        epistemic_values: list[float] = []
        pragmatic_values: list[float] = []
        efe_scores: list[float] = []
        dominance: list[str] = []
        for policy in policies:
            breakdown = self.compute_expected_free_energy(
                belief_vector, policy, preferences, return_breakdown=True
            )
            assert isinstance(breakdown, FreeEnergyBreakdown)
            efe_scores.append(breakdown.free_energy)
            epistemic_values.append(breakdown.epistemic_value)
            pragmatic_values.append(breakdown.pragmatic_value)
            dominance.append(
                "epistemic"
                if breakdown.epistemic_value >= abs(breakdown.pragmatic_value)
                else "pragmatic"
            )
        exploration_share = (
            float(np.mean([int(item == "epistemic") for item in dominance]))
            if dominance
            else 0.0
        )
        return {
            "policies": policies,
            "efe_scores": efe_scores,
            "epistemic_values": epistemic_values,
            "pragmatic_values": pragmatic_values,
            "dominance": dominance,
            "exploration_share": exploration_share,
            "best_index": int(np.argmin(efe_scores)) if efe_scores else -1,
        }

    def evaluate_policy_set(
        self,
        beliefs: np.ndarray,
        policies: list[dict[str, Any]],
        preferences: PreferenceInput | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a set of policies without selection.

        Args:
            beliefs: Current beliefs
            policies: List of policies to evaluate
            preferences: Prior preferences

        Returns:
            Policy evaluation results
        """
        beliefs = _normalize_vector(beliefs)
        if not policies:
            policies = self._create_default_policies(len(beliefs))

        breakdowns = []

        for policy in policies:
            breakdown = self.compute_expected_free_energy(
                beliefs,
                policy,
                preferences,
                return_breakdown=True,
            )
            assert isinstance(breakdown, FreeEnergyBreakdown)
            breakdowns.append(breakdown)

        expected_free_energies = np.array([b.free_energy for b in breakdowns])
        epistemic_values = np.array([b.epistemic_value for b in breakdowns])
        pragmatic_values = np.array([b.pragmatic_value for b in breakdowns])
        risks = np.array([b.risk for b in breakdowns])
        ambiguities = np.array([b.ambiguity for b in breakdowns])

        # Compute policy probabilities
        policy_probs = softmax(-expected_free_energies, temperature=self.temperature)
        evaluations = [
            PolicyEvaluation(
                policy=policy,
                expected_free_energy=float(expected_free_energies[idx]),
                probability=float(policy_probs[idx]),
                index=idx,
                epistemic_value=float(epistemic_values[idx]),
                pragmatic_value=float(pragmatic_values[idx]),
                risk=float(risks[idx]),
                ambiguity=float(ambiguities[idx]),
                metadata={"breakdown": breakdowns[idx]},
            )
            for idx, policy in enumerate(policies)
        ]

        return {
            "policies": policies,
            "expected_free_energies": expected_free_energies,
            "epistemic_values": epistemic_values,
            "pragmatic_values": pragmatic_values,
            "probabilities": policy_probs,
            "best_policy_idx": int(np.argmin(expected_free_energies)),
            "diversity": float(np.std(expected_free_energies)),
            "evaluations": evaluations,
        }

    def select_action(
        self,
        beliefs: np.ndarray,
        available_actions: list[Any],
        generative_model: Any = None,
    ) -> Any:
        """
        Select a single action based on current beliefs.

        Args:
            beliefs: Current belief state
            available_actions: List of available actions
            generative_model: Optional generative model for context

        Returns:
            Selected action
        """
        if not available_actions:
            return None

        if len(available_actions) == 1:
            return available_actions[0]

        policies = []
        for action in available_actions:
            if isinstance(action, dict):
                policy = dict(action)
                policy.setdefault("action", action.get("id", action))
            else:
                policy = {"action": action}
            policies.append(policy)

        result = self.select_policy(beliefs, policies)
        selected_policy = result["policy"]
        return (
            selected_policy.get("action", selected_policy)
            if isinstance(selected_policy, dict)
            else selected_policy
        )
