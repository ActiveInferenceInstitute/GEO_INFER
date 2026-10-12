"""Categorical expected-free-energy terms matching pymdp 1.0.3 exactly.

For a one-step policy ``a`` with predicted states ``q(s'|a) = B[:, :, a] q(s)``
and predicted outcomes ``q(o|a) = A q(s'|a)``:

- ``risk(a) = KL[q(o|a) || softmax(C)]``
- ``ambiguity(a) = sum_s' q(s'|a) H[A[:, s']]``
- ``efe(a) = risk(a) + ambiguity(a)``

pymdp 1.0.3 scores ``neg_efe(a) = sum_o q(o|a) C(o) + H[q(o|a)] - ambiguity(a)``
(expected utility plus state information gain). Therefore

``efe(a) = -neg_efe(a) + log sum_o exp C(o)``

for every belief, with no full-support hypothesis on predicted states or
outcomes: only the preference law ``softmax(C)`` must be positive, and it
always is. The fep_lean ``cascadia-geo-act-proof`` slice proves this identity
for finite laws; this module evaluates both sides numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable, Sequence

import numpy as np


def log_partition(preferences: Sequence[float]) -> float:
    """Stable ``log sum exp C``: the normalizer of the preference law."""
    values = np.asarray(preferences, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Preferences must be a nonempty finite vector")
    peak = float(values.max())
    return peak + float(np.log(np.exp(values - peak).sum()))


def preference_law(preferences: Sequence[float]) -> np.ndarray:
    """Return ``softmax(C)``, a strictly positive probability vector."""
    values = np.asarray(preferences, dtype=float)
    return np.exp(values - log_partition(values))


def entropy(law: Sequence[float]) -> float:
    """Shannon entropy in nats with ``0 log 0 = 0``."""
    values = np.asarray(law, dtype=float)
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum())


@dataclass(frozen=True)
class PolicyTerms:
    """Expected-free-energy terms of one one-step policy."""

    action: int
    predicted_states: np.ndarray
    predicted_outcomes: np.ndarray
    risk: float
    ambiguity: float
    pragmatic_cost: float
    epistemic_value: float
    expected_free_energy: float
    pymdp_neg_efe: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe scalar terms (predictions omitted)."""
        return {
            "action": self.action,
            "risk": self.risk,
            "ambiguity": self.ambiguity,
            "pragmatic_cost": self.pragmatic_cost,
            "epistemic_value": self.epistemic_value,
            "expected_free_energy": self.expected_free_energy,
            "pymdp_neg_efe": self.pymdp_neg_efe,
        }


def one_step_policy_terms(
    belief: Sequence[float],
    likelihood: np.ndarray,
    transition: np.ndarray,
    preferences: Sequence[float],
) -> list[PolicyTerms]:
    """Evaluate risk, ambiguity and both EFE forms for every one-step policy.

    ``likelihood`` is ``A[observation, state]``; ``transition`` is
    ``B[next_state, previous_state, action]``.
    """
    q = np.asarray(belief, dtype=float)
    A = np.asarray(likelihood, dtype=float)
    B = np.asarray(transition, dtype=float)
    C = np.asarray(preferences, dtype=float)
    if A.ndim != 2 or B.ndim != 3 or B.shape[:2] != (A.shape[1], A.shape[1]):
        raise ValueError("Expected A[obs, state] and B[next, previous, action]")
    if q.shape != (A.shape[1],) or C.shape != (A.shape[0],):
        raise ValueError("Belief and preferences must match A")
    law = preference_law(C)
    state_entropy = np.array([entropy(A[:, state]) for state in range(A.shape[1])])
    terms = []
    for action in range(B.shape[2]):
        predicted_states = B[:, :, action] @ q
        predicted_outcomes = A @ predicted_states
        positive = predicted_outcomes > 0
        risk = float(
            (
                predicted_outcomes[positive]
                * (np.log(predicted_outcomes[positive]) - np.log(law[positive]))
            ).sum()
        )
        ambiguity = float(predicted_states @ state_entropy)
        pragmatic = float(-(predicted_outcomes * np.log(law)).sum())
        outcome_entropy = entropy(predicted_outcomes)
        terms.append(
            PolicyTerms(
                action=action,
                predicted_states=predicted_states,
                predicted_outcomes=predicted_outcomes,
                risk=risk,
                ambiguity=ambiguity,
                pragmatic_cost=pragmatic,
                epistemic_value=outcome_entropy - ambiguity,
                expected_free_energy=risk + ambiguity,
                pymdp_neg_efe=float(predicted_outcomes @ C)
                + outcome_entropy
                - ambiguity,
            )
        )
    return terms


def support(law: Sequence[float]) -> frozenset[int]:
    """Indices carrying strictly positive mass."""
    return frozenset(
        int(index) for index in np.flatnonzero(np.asarray(law, dtype=float) > 0)
    )


def light_cone_violations(
    beliefs: Iterable[Sequence[float]],
    allowed: Iterable[Iterable[int]],
) -> list[tuple[int, list[int]]]:
    """Return ``(step, indices)`` where belief mass lies outside the allowed set."""
    violations = []
    for step, (belief, cone) in enumerate(zip(beliefs, allowed)):
        outside = sorted(support(belief) - set(cone))
        if outside:
            violations.append((step, outside))
    return violations
