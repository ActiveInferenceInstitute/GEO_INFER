from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from geo_infer_act.core.gnn_factored_batch import FactoredPolicyEvaluator
from geo_infer_act.core.gnn_factored_contract import (
    FactoredGNNArtifact,
    infer_factored_step,
)


def artifact(*, horizon: int = 1, policies: int = 2) -> FactoredGNNArtifact:
    raw = Path(__file__).with_name("factored_example.json").read_text()
    source = json.loads(raw)
    source["control_factors"][0]["actions"] = [
        f"action_{index}" for index in range(policies)
    ]
    source["transitions"][0]["probabilities"] = np.take(
        np.asarray(source["transitions"][0]["probabilities"]),
        [index % 2 for index in range(policies)],
        axis=-1,
    ).tolist()
    source["policies"] = [
        [[index, 0] for _ in range(horizon)] for index in range(policies)
    ]
    source["policy_prior"] = [1 / policies] * policies
    return FactoredGNNArtifact.from_dict(
        {
            **source,
            "schema_version": "gnn-geo-infer/factored/1",
            "model_type": "categorical_factored",
            "time": {"step_seconds": 60},
            "provenance": {
                "producer": "GNN explicit factored JSON exporter",
                "source_kind": "explicit_factored_json",
                "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            },
        }
    )


def test_batch_matches_exact_observation_tree_and_preserves_inputs() -> None:
    """Compare correlated joint beliefs with the exact observation-tree evaluator."""
    model = artifact()
    posteriors = np.random.default_rng(73).dirichlet(np.ones(6), size=40)
    original = posteriors.copy()
    evaluator = FactoredPolicyEvaluator(model)
    result = evaluator.evaluate(posteriors)
    for index, posterior in enumerate(posteriors):
        reference = infer_factored_step(model, [0, 0], posterior=posterior)
        np.testing.assert_allclose(
            result.expected_free_energy[index],
            reference["expected_free_energy"],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            result.policy_posterior[index],
            reference["policy_posterior"],
            atol=1e-12,
            rtol=0,
        )
        assert result.selected_policy[index] == reference["selected_policy"]
        assert result.selected_action[index] == tuple(reference["selected_action"])
    np.testing.assert_array_equal(posteriors, original)
    np.testing.assert_allclose(
        result.expected_free_energy,
        result.expected_preference_cost - result.information_gain,
        atol=1e-14,
    )
    assert np.min(result.information_gain) >= -1e-12
    assert (
        result.backend == "geo-infer-exact-joint"
        and result.artifact_sha256 == model.digest
    )
    with pytest.raises(ValueError):
        result.expected_free_energy.setflags(write=True)
    with pytest.raises(ValueError):
        evaluator._emissions.setflags(write=True)


def test_batch_accepts_degenerate_support_and_zero_policy_prior() -> None:
    """Preserve exact zero support and exclude policies with zero prior mass."""
    data = artifact().to_dict()
    data["policy_prior"] = [0, 1]
    result = FactoredPolicyEvaluator(FactoredGNNArtifact.from_dict(data)).evaluate(
        np.eye(6)
    )
    np.testing.assert_array_equal(result.policy_posterior, np.tile([0, 1], (6, 1)))
    assert result.selected_policy == (1,) * 6


@pytest.mark.parametrize(
    "condition",
    [
        "scalar",
        "width",
        "empty",
        "negative",
        "nan",
        "normalization",
        "boolean",
        "strings",
        "rows",
        "work",
    ],
)
def test_batch_refuses_invalid_beliefs_and_excessive_work(condition: str) -> None:
    """Reject malformed beliefs and bounded-work violations before evaluation."""
    model, beliefs = artifact(), np.full((2, 6), 1 / 6)
    if condition == "scalar":
        beliefs = np.array(0.2)
    elif condition == "width":
        beliefs = np.full((2, 3), 1 / 3)
    elif condition == "empty":
        beliefs = np.zeros((0, 6))
    elif condition == "negative":
        beliefs[0, :2] = [-0.1, 0.1 + 1 / 3]
    elif condition == "nan":
        beliefs[0, 0] = np.nan
    elif condition == "normalization":
        beliefs[0, 0] = 1
    elif condition == "boolean":
        beliefs = np.zeros((1, 6), dtype=bool)
    elif condition == "strings":
        beliefs = np.full((1, 6), "0.1666")
    elif condition == "rows":
        beliefs = np.full((30001, 6), 1 / 6)
    else:
        model, beliefs = artifact(policies=20), np.full((30000, 6), 1 / 6)
    with pytest.raises(ValueError):
        FactoredPolicyEvaluator(model).evaluate(beliefs)


def test_horizon_and_artifact_type_are_explicit_boundaries() -> None:
    """Reject unsupported planning horizons and nonfactored input artifacts."""
    with pytest.raises(ValueError, match="one-step"):
        FactoredPolicyEvaluator(artifact(horizon=2))
    with pytest.raises(TypeError):
        FactoredPolicyEvaluator({})


def test_aggregated_preference_overflow_is_rejected() -> None:
    """Fail explicitly when valid local preferences overflow their aggregate."""
    data = artifact().to_dict()
    for modality in data["modalities"]:
        modality["preferences"] = [1e308] * len(modality["preferences"])
    with pytest.raises(ValueError, match="finite numeric range"):
        FactoredPolicyEvaluator(FactoredGNNArtifact.from_dict(data))
