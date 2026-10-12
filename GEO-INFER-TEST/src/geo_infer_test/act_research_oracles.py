"""Independent finite references for ACT's default four-state research profile.

These checks use NumPy probability arithmetic, never ACT or pymdp scoring code.
They cover the declared one-step, unit-precision, uniform-policy-prior profile;
arbitrary environmental observations need not visit distinct policy regions.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence
from numbers import Integral

import numpy as np


def research_profile_reference() -> tuple[np.ndarray, ...]:
    """Return independently specified A, B, C and D for the default profile."""
    likelihood = np.full((4, 4), 1 / 75, dtype=float)
    np.fill_diagonal(likelihood, 24 / 25)
    transitions = np.full((4, 4, 4), 1 / 50, dtype=float)
    for source in range(4):
        transitions[source, source, 0] = 47 / 50
        for action in range(1, 4):
            transitions[source, source, action] = 3 / 5
            transitions[(source + action) % 4, source, action] = 9 / 25
    return (
        likelihood,
        transitions,
        np.array([-0.45, -0.15, 0.65, 0.35]),
        np.array([0.20, 0.25, 0.30, 0.25]),
    )


def _probability_vector(values: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if (
        vector.shape != (4,)
        or not np.isfinite(vector).all()
        or np.any(vector < 0)
        or not np.isclose(vector.sum(), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(f"{label} must be a normalized four-state probability vector")
    return vector


def research_posterior_reference(
    observation: Sequence[float],
    *,
    prior: Sequence[float] | None = None,
) -> np.ndarray:
    """Condition categorical frequency evidence: q(s) ∝ D(s) ∏ A(o,s)^v(o)."""
    likelihood, _, _, default_prior = research_profile_reference()
    evidence = _probability_vector(observation, "observation")
    initial = _probability_vector(default_prior if prior is None else prior, "prior")
    positive = initial > 0
    log_weights = np.full(4, -np.inf)
    log_weights[positive] = (
        np.log(initial[positive]) + np.log(likelihood[:, positive]).T @ evidence
    )
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def research_policy_reference(beliefs: Sequence[float]) -> dict[str, Any]:
    """Evaluate B→A outcomes, expected utility and state-information gain.

    Negative EFE is E[C(o)] + H(o) - E[H(A(:,s))]. The default profile uses
    gamma=1 and uniform E, so its policy posterior is softmax(negative EFE).
    """
    likelihood, transitions, preferences, _ = research_profile_reference()
    posterior = _probability_vector(beliefs, "policy beliefs")
    likelihood_entropy = -np.sum(likelihood * np.log(likelihood), axis=0)
    negative_efe = []
    for action in range(4):
        future_states = transitions[:, :, action] @ posterior
        outcomes = likelihood @ future_states
        information = -float(outcomes @ np.log(outcomes)) - float(
            future_states @ likelihood_entropy
        )
        negative_efe.append(float(outcomes @ preferences) + information)
    scores = np.asarray(negative_efe)
    weights = np.exp(scores - scores.max())
    return {
        "negative_expected_free_energy": scores,
        "action_posterior": weights / weights.sum(),
        "selected_action_index": int(np.argmax(scores)),
    }


def assert_research_policy(
    *,
    beliefs: Sequence[float],
    action_posterior: Sequence[float],
    negative_expected_free_energy: Sequence[float],
    selected_action_index: int,
    label: str,
) -> None:
    """Reject wrong scores, posteriors or selection even if actions vary."""
    scores = np.asarray(negative_expected_free_energy, dtype=float)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError(f"{label}: negative EFE must be four finite scores")
    posterior = _probability_vector(action_posterior, f"{label}: policy posterior")
    if (
        isinstance(selected_action_index, (bool, np.bool_))
        or not isinstance(selected_action_index, Integral)
        or not 0 <= selected_action_index < 4
    ):
        raise ValueError(f"{label}: selected action must be an integer in 0..3")
    expected = research_policy_reference(beliefs)
    np.testing.assert_allclose(
        scores,
        expected["negative_expected_free_energy"],
        rtol=0,
        atol=1e-6,
        err_msg=f"{label}: one-step negative EFE",
    )
    np.testing.assert_allclose(
        posterior,
        expected["action_posterior"],
        rtol=0,
        atol=1e-6,
        err_msg=f"{label}: policy posterior",
    )
    assert selected_action_index == expected["selected_action_index"], (
        f"{label}: selected action differs from analytical optimum"
    )
