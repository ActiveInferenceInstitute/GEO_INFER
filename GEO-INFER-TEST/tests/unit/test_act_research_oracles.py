"""Finite analytic references reject wrong policies without demanding diversity."""

import numpy as np
import pytest

from geo_infer_test.act_research_oracles import (
    assert_research_policy,
    research_policy_reference,
    research_posterior_reference,
    research_profile_reference,
)


def test_onehot_posterior_has_independently_calculated_weights():
    # Observation0: unnormalized weights are D * A[0,:].
    weights = np.array([24 / 125, 1 / 300, 1 / 250, 1 / 300])
    np.testing.assert_allclose(
        research_posterior_reference([1, 0, 0, 0]), weights / weights.sum(), atol=1e-15
    )


def test_declared_probe_states_visit_three_known_optimal_actions():
    chosen = [
        research_policy_reference(research_posterior_reference(observation))[
            "selected_action_index"
        ]
        for observation in np.eye(4)
    ]
    # State2 prefers action1: its additional information gain outweighs staying.
    assert chosen == [2, 1, 1, 3]


def test_uniform_evidence_preserves_the_declared_prior():
    _, _, _, prior = research_profile_reference()
    np.testing.assert_allclose(research_posterior_reference([0.25] * 4), prior)


@pytest.mark.parametrize("field", ["scores", "posterior", "selected"])
def test_policy_oracle_rejects_corruption_even_when_other_fields_are_valid(field):
    beliefs = research_posterior_reference([1, 0, 0, 0])
    expected = research_policy_reference(beliefs)
    scores = expected["negative_expected_free_energy"].copy()
    posterior = expected["action_posterior"].copy()
    selected = expected["selected_action_index"]
    if field == "scores":
        scores += 0.01  # Selection and softmax stay the same; scores are wrong.
    elif field == "posterior":
        posterior = np.roll(posterior, 1)
    else:
        selected = (selected + 1) % 4
    with pytest.raises(AssertionError):
        assert_research_policy(
            beliefs=beliefs,
            action_posterior=posterior,
            negative_expected_free_energy=scores,
            selected_action_index=selected,
            label="deliberately corrupted record",
        )


@pytest.mark.parametrize(
    "values", [[1, 0, 0], [np.nan, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 0, 0]]
)
def test_references_reject_invalid_probability_vectors(values):
    with pytest.raises(ValueError):
        research_posterior_reference(values)
    with pytest.raises(ValueError):
        research_policy_reference(values)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("negative_expected_free_energy", 0.0),
        ("negative_expected_free_energy", [[0.0] * 4]),
        ("negative_expected_free_energy", [np.inf, 0, 0, 0]),
        ("negative_expected_free_energy", [np.nan, 0, 0, 0]),
        ("action_posterior", 0.25),
        ("action_posterior", [[0.25] * 4]),
        ("action_posterior", [np.nan, 0, 0, 1]),
        ("action_posterior", [-0.1, 0.3, 0.4, 0.4]),
        ("action_posterior", [0.2] * 4),
        ("selected_action_index", True),
        ("selected_action_index", np.bool_(True)),
        ("selected_action_index", 1.0),
        ("selected_action_index", "1"),
        ("selected_action_index", -1),
        ("selected_action_index", 4),
    ],
)
def test_policy_oracle_rejects_malformed_diagnostics(field, invalid):
    # Observation1's actual optimum is 1, so True/1.0 would otherwise compare
    # equal and obscure a corrupt selected-action representation.
    beliefs = research_posterior_reference([0, 1, 0, 0])
    fields = research_policy_reference(beliefs)
    fields[field] = invalid
    with pytest.raises(ValueError):
        assert_research_policy(beliefs=beliefs, label="malformed diagnostic", **fields)
