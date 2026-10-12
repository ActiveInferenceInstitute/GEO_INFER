"""Risk/ambiguity EFE terms and their exact relation to pymdp 1.0.4 scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geo_infer_act.core.efe_decomposition import (
    entropy,
    light_cone_violations,
    log_partition,
    one_step_policy_terms,
    preference_law,
)
from geo_infer_act.utils.pymdp_adapter import run_pymdp_step

ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "examples/cascadia/data/smith_river_homing.geo-infer.json"
)


@pytest.fixture(scope="module")
def matrices() -> dict[str, np.ndarray]:
    """A, B, C of the committed Cascadia homing artifact."""
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))["matrices"]
    return {name: np.asarray(data[name], dtype=float) for name in ("A", "B", "C")}


def test_preference_law_is_stable_and_normalized() -> None:
    """softmax(C) sums to one and log Z is shift-equivariant for large values."""
    C = np.array([1000.0, 1001.0, 999.0])
    law = preference_law(C)
    assert np.isclose(law.sum(), 1.0) and np.all(law > 0)
    assert np.isclose(log_partition(C), 1000.0 + log_partition(C - 1000.0))
    assert entropy([1.0, 0.0]) == 0.0


def test_efe_is_risk_plus_ambiguity_and_pragmatic_minus_epistemic(matrices) -> None:
    """Both EFE forms agree on a sparse spatial belief."""
    belief = np.zeros(matrices["A"].shape[1])
    belief[[0, 1, 7]] = [0.5, 0.25, 0.25]
    for term in one_step_policy_terms(
        belief, matrices["A"], matrices["B"], matrices["C"]
    ):
        assert np.isclose(term.expected_free_energy, term.risk + term.ambiguity)
        assert np.isclose(
            term.expected_free_energy, term.pragmatic_cost - term.epistemic_value
        )
        assert term.risk >= 0 and term.ambiguity >= 0
        assert np.isclose(term.predicted_outcomes.sum(), 1.0)


def test_pymdp_negative_efe_equals_log_partition_minus_efe(matrices) -> None:
    """pymdp's own neg-EFE equals log sum exp C minus (risk + ambiguity)."""
    rng = np.random.default_rng(11)
    states = matrices["A"].shape[1]
    for _ in range(3):
        prior = np.zeros(states)
        chosen = rng.choice(states, size=4, replace=False)
        prior[chosen] = rng.dirichlet(np.ones(4))
        observation = int(np.argmax(matrices["A"] @ prior))
        result = run_pymdp_step(
            observation=np.eye(matrices["A"].shape[0])[observation],
            observation_model=matrices["A"],
            transition_model=matrices["B"],
            preferences=matrices["C"],
            prior=prior,
            policy_prior=np.full(4, 0.25),
            action_count=4,
            strict=True,
        )
        posterior = (
            matrices["A"][observation] * prior / (matrices["A"][observation] @ prior)
        )
        terms = one_step_policy_terms(
            posterior, matrices["A"], matrices["B"], matrices["C"]
        )
        reported = np.asarray(result.negative_expected_free_energy, dtype=float)
        predicted = log_partition(matrices["C"]) - np.array(
            [term.expected_free_energy for term in terms]
        )
        np.testing.assert_allclose(reported, predicted, atol=1e-5)


def test_light_cone_violations_report_mass_outside_the_cone() -> None:
    """Mass outside an allowed index set is reported with its step."""
    beliefs = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]
    allowed = [{0}, {0, 1}, {0, 1}]
    assert light_cone_violations(beliefs, allowed) == [(2, [2])]
