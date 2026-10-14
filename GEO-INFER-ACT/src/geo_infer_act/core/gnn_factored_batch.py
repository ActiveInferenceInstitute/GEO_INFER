"""Bounded exact one-step policy evaluation across a batch of joint beliefs.

The joint state and observation spaces are enumerated from the GNN artifact.
Batching changes arithmetic organization, not dependency axes or the objective.
Longer policies remain the responsibility of infer_factored_step.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from geo_infer_act.core.gnn_factored_contract import (
    FactoredGNNArtifact,
    MAX_POLICY_WORK,
)

FloatArray = NDArray[np.float64]


def _immutable(values: FloatArray) -> FloatArray:
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


@dataclass(frozen=True, slots=True)
class FactoredPolicyBatch:
    expected_free_energy: FloatArray
    information_gain: FloatArray
    expected_preference_cost: FloatArray
    policy_posterior: FloatArray
    selected_policy: tuple[int, ...]
    selected_action: tuple[tuple[int, ...], ...]
    artifact_sha256: str
    backend: str = "geo-infer-exact-joint"


class FactoredPolicyEvaluator:
    def __init__(self, artifact: FactoredGNNArtifact) -> None:
        if not isinstance(artifact, FactoredGNNArtifact):
            raise TypeError("artifact must be a FactoredGNNArtifact")
        data = artifact.to_dict()
        if any(len(policy) != 1 for policy in data["policies"]):
            raise ValueError("Batch evaluation requires explicit one-step policies")
        self.artifact_sha256: str = artifact.digest
        self._actions: tuple[tuple[int, ...], ...] = tuple(
            tuple(policy[0]) for policy in data["policies"]
        )
        self._policy_prior = _immutable(
            np.asarray(data["policy_prior"], dtype=np.float64)
        )
        states = list(
            itertools.product(
                *(range(len(factor["states"])) for factor in data["state_factors"])
            )
        )
        outcomes = list(
            itertools.product(
                *(range(len(modality["outcomes"])) for modality in data["modalities"])
            )
        )
        a = [
            np.asarray(modality["likelihood"], dtype=np.float64)
            for modality in data["modalities"]
        ]
        b = [
            np.asarray(transition["probabilities"], dtype=np.float64)
            for transition in data["transitions"]
        ]
        emissions = np.ones((len(outcomes), len(states)))
        preferences = np.zeros(len(outcomes))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for oi, outcome in enumerate(outcomes):
                    preferences[oi] = sum(
                        modality["preferences"][outcome[mi]]
                        for mi, modality in enumerate(data["modalities"])
                    )
                    for si, state in enumerate(states):
                        for mi, modality in enumerate(data["modalities"]):
                            emissions[oi, si] *= a[mi][
                                (outcome[mi],)
                                + tuple(
                                    state[axis] for axis in modality["dependencies"]
                                )
                            ]
                transitions = []
                for action in self._actions:
                    matrix = np.ones((len(states), len(states)))
                    for current_i, current in enumerate(states):
                        for next_i, following in enumerate(states):
                            for fi, spec in enumerate(data["transitions"]):
                                matrix[next_i, current_i] *= b[fi][
                                    (following[fi],)
                                    + tuple(
                                        current[axis] for axis in spec["dependencies"]
                                    )
                                    + (action[spec["control_factor"]],)
                                ]
                    transitions.append(_immutable(matrix))
                conditional_entropy = -np.sum(xlogy(emissions, emissions), axis=0)
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError(
                    "Factored batch compilation exceeds finite numeric range"
                ) from exc
        if not np.all(np.isfinite(preferences)) or not np.all(
            np.isfinite(conditional_entropy)
        ):
            raise ValueError(
                "Aggregated preferences or entropy exceed finite numeric range"
            )
        self._emissions = _immutable(emissions)
        self._preferences = _immutable(preferences)
        self._conditional_entropy = _immutable(
            np.asarray(conditional_entropy, dtype=np.float64)
        )
        self._transitions = tuple(transitions)

    def evaluate(self, posteriors: FloatArray) -> FactoredPolicyBatch:
        if (
            not isinstance(posteriors, np.ndarray)
            or posteriors.ndim != 2
            or posteriors.dtype.kind not in "fiu"
            or not 1 <= len(posteriors) <= 30000
            or posteriors.shape[1] != self._emissions.shape[1]
        ):
            raise ValueError(
                "Batch beliefs must be a nonempty numeric N by joint-state array"
            )
        if (
            not np.all(np.isfinite(posteriors))
            or np.any(posteriors < 0)
            or not np.allclose(posteriors.sum(axis=1), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                "Every batch belief must be finite, nonnegative and normalized"
            )
        count, states = posteriors.shape
        if (
            count * len(self._actions) * states * (states + len(self._emissions))
            > MAX_POLICY_WORK
        ):
            raise ValueError(
                "Batch policy evaluation exceeds the bounded work contract"
            )
        costs = np.empty((count, len(self._actions)))
        information = np.empty_like(costs)
        preference_costs = np.empty_like(costs)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for policy, transition in enumerate(self._transitions):
                    predicted = posteriors @ transition.T
                    outcomes = predicted @ self._emissions.T
                    gain = (
                        -np.sum(xlogy(outcomes, outcomes), axis=1)
                        - predicted @ self._conditional_entropy
                    )
                    preference = -outcomes @ self._preferences
                    costs[:, policy] = preference - gain
                    information[:, policy], preference_costs[:, policy] = (
                        gain,
                        preference,
                    )
                supported = self._policy_prior > 0
                scores = -costs[:, supported] + np.log(self._policy_prior[supported])
                maximum = np.max(scores, axis=1, keepdims=True)
                weights = np.exp(scores - maximum)
                probabilities = np.zeros_like(costs)
                probabilities[:, supported] = weights / weights.sum(
                    axis=1, keepdims=True
                )
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError(
                    "Factored batch inference exceeds finite numeric range"
                ) from exc
        if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Factored batch inference returned nonfinite policies")
        selected = tuple(map(int, np.argmax(probabilities, axis=1)))
        return FactoredPolicyBatch(
            expected_free_energy=_immutable(costs),
            information_gain=_immutable(information),
            expected_preference_cost=_immutable(preference_costs),
            policy_posterior=_immutable(probabilities),
            selected_policy=selected,
            selected_action=tuple(self._actions[index] for index in selected),
            artifact_sha256=self.artifact_sha256,
        )
