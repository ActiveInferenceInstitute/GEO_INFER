"""Bounded exact categorical factor/modal interchange and policy reference.

This independent contract does not silently flatten GNN v1 or approximate
correlated joint beliefs with products of marginals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "gnn-geo-infer/factored/1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 1_000_000
MAX_JOINT_STATES = 256
MAX_POLICY_WORK = 20_000_000


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{label} requires exactly {expected}")


def _integer(value, upper, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"{label} must be an integer in [0, {upper})")
    return value


def _labels(values, label):
    if (
        not isinstance(values, list)
        or not values
        or len(values) > 256
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must contain 1..256 unique nonempty labels")
    return len(values)


def _shape(value, dimensions, label):
    if not dimensions:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must contain finite numeric values")
        return
    if not isinstance(value, list) or len(value) != dimensions[0]:
        raise ValueError(f"{label} requires shape {dimensions}")
    for child in value:
        _shape(child, dimensions[1:], label)


def validate_factored_artifact(data):
    """Validate dependency axes and bounds before allocating numerical tensors."""
    _keys(
        data,
        "schema_version model_type model_name state_factors control_factors modalities transitions initial_joint policies policy_prior time provenance",
        "Artifact",
    )
    if (
        data["schema_version"] != CONTRACT_VERSION
        or data["model_type"] != "categorical_factored"
    ):
        raise ValueError("Unsupported factored contract version or model type")
    if not isinstance(data["model_name"], str) or not data["model_name"]:
        raise ValueError("model_name must be nonempty")
    _keys(data["time"], "step_seconds", "time")
    seconds = data["time"]["step_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("step_seconds must be finite and positive")
    _keys(data["provenance"], "producer source_kind source_sha256", "provenance")
    origin = data["provenance"]
    if (
        origin["source_kind"] != "explicit_factored_json"
        or not isinstance(origin["producer"], str)
        or not origin["producer"]
        or not isinstance(origin["source_sha256"], str)
        or len(origin["source_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in origin["source_sha256"])
    ):
        raise ValueError("Invalid structured-source provenance")
    dimensions = []
    for section, labels in (
        ("state_factors", "states"),
        ("control_factors", "actions"),
    ):
        factors = data[section]
        if not isinstance(factors, list) or not 1 <= len(factors) <= 8:
            raise ValueError(f"{section} must contain 1..8 factors")
        sizes, ids = [], []
        for factor in factors:
            _keys(factor, f"id {labels}", section)
            ids.append(factor["id"])
            sizes.append(_labels(factor[labels], labels))
        _labels(ids, section)
        dimensions.append(sizes)
    states, controls = dimensions
    joint_states = math.prod(states)
    if joint_states > MAX_JOINT_STATES:
        raise ValueError("Joint state budget exceeded")
    entries = joint_states * joint_states + joint_states
    arrays = [("initial_joint", data["initial_joint"], [joint_states], True)]
    modalities = data["modalities"]
    transitions = data["transitions"]
    if not isinstance(modalities, list) or not 1 <= len(modalities) <= 8:
        raise ValueError("modalities must contain 1..8 observation modalities")
    if not isinstance(transitions, list) or len(transitions) != len(states):
        raise ValueError("One transition tensor is required per state factor")
    outcome_sizes, modality_ids = [], []
    for index, modality in enumerate(modalities):
        _keys(modality, "id outcomes dependencies likelihood preferences", "modality")
        modality_ids.append(modality["id"])
        count = _labels(modality["outcomes"], "outcomes")
        outcome_sizes.append(count)
        arrays.append((f"C[{index}]", modality["preferences"], [count], False))
    _labels(modality_ids, "modality IDs")
    for label, items in (("A", modalities), ("B", transitions)):
        for index, item in enumerate(items):
            if label == "B":
                _keys(item, "dependencies control_factor probabilities", "transition")
            deps = item["dependencies"]
            if not isinstance(deps, list) or len(deps) > len(states):
                raise ValueError(f"{label} dependencies must list state factor indices")
            for dep in deps:
                _integer(dep, len(states), "dependency")
            if len(set(deps)) != len(deps):
                raise ValueError("Dependency axes must be unique")
            shape = [outcome_sizes[index] if label == "A" else states[index]]
            shape += [states[dep] for dep in deps]
            if label == "B":
                control = _integer(
                    item["control_factor"], len(controls), "control_factor"
                )
                shape.append(controls[control])
            arrays.append(
                (
                    f"{label}[{index}]",
                    item["likelihood" if label == "A" else "probabilities"],
                    shape,
                    True,
                )
            )
    outcomes = math.prod(outcome_sizes)
    entries += outcomes * joint_states
    policies = data["policies"]
    if not isinstance(policies, list) or not 1 <= len(policies) <= 256:
        raise ValueError("policies must explicitly enumerate 1..256 policies")
    horizon = len(policies[0]) if isinstance(policies[0], list) else 0
    if not 1 <= horizon <= 8:
        raise ValueError("Policy horizon must be 1..8")
    for policy in policies:
        if not isinstance(policy, list) or len(policy) != horizon:
            raise ValueError("All policies must have the same horizon")
        for action in policy:
            if not isinstance(action, list) or len(action) != len(controls):
                raise ValueError("Every policy action must specify each control factor")
            for value, count in zip(action, controls):
                _integer(value, count, "action")
    if len({json.dumps(p) for p in policies}) != len(policies):
        raise ValueError("Policies must be unique")
    arrays.append(("E", data["policy_prior"], [len(policies)], True))
    nodes, level_nodes = 0, 1
    for _ in range(horizon):
        nodes += level_nodes
        if (
            len(policies) * nodes * joint_states * joint_states * outcomes
            > MAX_POLICY_WORK
        ):
            raise ValueError("Exact policy observation-tree work budget exceeded")
        level_nodes *= outcomes
    entries += sum(math.prod(shape) for _, _, shape, _ in arrays)
    if entries > MAX_ENTRIES:
        raise ValueError("Factored matrix entry budget exceeded")
    for label, value, shape, _ in arrays:
        _shape(value, shape, label)
    # Numerical conversion follows every dimension, budget, and nesting check.
    for label, value, _, probability in arrays:
        array = np.asarray(value, dtype=float)
        if probability and (
            np.any(array < 0)
            or not np.allclose(array.sum(axis=0), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                f"{label} must be nonnegative and stochastic along axis zero"
            )
    return states, controls, outcome_sizes


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class FactoredGNNArtifact:
    """Immutable snapshot of the explicit bounded categorical joint contract."""

    _json: str

    def __post_init__(self):
        if (
            not isinstance(self._json, str)
            or len(self._json.encode()) > MAX_SOURCE_BYTES
        ):
            raise ValueError("Factored artifact exceeds four MiB")
        validate_factored_artifact(
            json.loads(self._json, object_pairs_hook=_unique_object)
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("Factored artifact exceeds four MiB")
        return cls(raw.decode("utf-8"))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def digest(self):
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _entropy(probabilities):
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def _infer_factored_step(artifact, observation, *, prior=None, posterior=None):
    """Condition once and evaluate explicit policies by exact observation trees.

    Future factor transitions and observation modalities are conditionally
    independent given their explicitly ordered dependencies. Joint beliefs
    retain correlations. Policy costs are negative expected log preferences
    minus expected state information gain, summed over the finite policy.
    Future observations are marginalized exactly, conditioning each branch.
    """
    if not isinstance(artifact, FactoredGNNArtifact):
        raise TypeError("artifact must be a FactoredGNNArtifact")
    data = artifact.to_dict()
    sizes = [len(f["states"]) for f in data["state_factors"]]
    outcome_sizes = [len(m["outcomes"]) for m in data["modalities"]]
    state_tuples = list(itertools.product(*(range(size) for size in sizes)))
    outcome_tuples = list(itertools.product(*(range(size) for size in outcome_sizes)))
    count = len(state_tuples)
    if not isinstance(observation, list) or len(observation) != len(outcome_sizes):
        raise ValueError("One integer observation is required per modality")
    for value, size in zip(observation, outcome_sizes):
        _integer(value, size, "observation")
    supplied_prior = (
        posterior
        if posterior is not None
        else (data["initial_joint"] if prior is None else prior)
    )
    if isinstance(supplied_prior, np.ndarray):
        if supplied_prior.shape != (count,):
            raise ValueError("prior must be a normalized joint-state vector")
    else:
        _shape(supplied_prior, [count], "prior")
    q = np.asarray(supplied_prior, dtype=float)
    if (
        q.shape != (count,)
        or not np.all(np.isfinite(q))
        or np.any(q < 0)
        or not np.isclose(q.sum(), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("prior must be a normalized joint-state vector")
    a_arrays = [np.asarray(m["likelihood"]) for m in data["modalities"]]
    b_arrays = [np.asarray(t["probabilities"]) for t in data["transitions"]]
    emissions = np.ones((len(outcome_tuples), count))
    preferences = np.zeros(len(outcome_tuples))
    for oi, outcome in enumerate(outcome_tuples):
        preferences[oi] = sum(
            m["preferences"][outcome[mi]] for mi, m in enumerate(data["modalities"])
        )
        if not np.isfinite(preferences[oi]):
            raise ValueError("Aggregated log preferences exceed finite numeric range")
        for si, state in enumerate(state_tuples):
            for mi, modality in enumerate(data["modalities"]):
                index = (outcome[mi],) + tuple(
                    state[d] for d in modality["dependencies"]
                )
                emissions[oi, si] *= a_arrays[mi][index]
    if posterior is None:
        likelihood = emissions[outcome_tuples.index(tuple(observation))]
        evidence = float(likelihood @ q)
        if evidence <= 0:
            raise ValueError("Observation has zero evidence under the joint prior")
        posterior = likelihood * q / evidence
    else:
        # Policy-only evaluation must not assimilate an observation twice.
        posterior = q
        evidence = None

    def transition(action):
        result = np.ones((count, count))
        for current_i, current in enumerate(state_tuples):
            for next_i, next_state in enumerate(state_tuples):
                for fi, spec in enumerate(data["transitions"]):
                    index = (
                        (next_state[fi],)
                        + tuple(current[d] for d in spec["dependencies"])
                        + (action[spec["control_factor"]],)
                    )
                    result[next_i, current_i] *= b_arrays[fi][index]
        return result

    def expected_cost(beliefs, policy, time_index):
        predicted = transition(policy[time_index]) @ beliefs
        outcome_probabilities = emissions @ predicted
        expected_entropy = 0.0
        future = 0.0
        for oi, probability in enumerate(outcome_probabilities):
            if probability <= 0:
                continue
            conditioned = emissions[oi] * predicted / probability
            expected_entropy += probability * _entropy(conditioned)
            if time_index + 1 < len(policy):
                future += probability * expected_cost(
                    conditioned, policy, time_index + 1
                )
        information_gain = _entropy(predicted) - expected_entropy
        cost = float(-outcome_probabilities @ preferences - information_gain + future)
        if not math.isfinite(cost):
            raise ValueError("Policy cost exceeds finite numeric range")
        return cost

    efe = np.array([expected_cost(posterior, p, 0) for p in data["policies"]])
    if not np.all(np.isfinite(efe)):
        raise ValueError("Expected free energy exceeds finite numeric range")
    e = np.asarray(data["policy_prior"])
    scores = np.full(len(e), -np.inf)
    supported = e > 0
    scores[supported] = -efe[supported] + np.log(e[supported])
    if not np.all(np.isfinite(scores[supported])):
        raise ValueError("Supported policy scores exceed finite numeric range")
    # Subtraction of opposite, finite float extremes may produce -inf. That
    # represents a probability below floating-point range, not invalid EFE.
    # Scalar subtraction avoids a NumPy overflow warning; the maximum score
    # always contributes exp(0)=1, so normalization cannot lose all support.
    largest = float(scores[supported].max())
    weights = np.zeros(len(e))
    for index in np.flatnonzero(supported):
        difference = float(scores[index]) - largest
        weights[index] = 0.0 if difference < -745 else math.exp(difference)
    policy_posterior = weights / weights.sum()
    if not np.all(np.isfinite(policy_posterior)):
        raise ValueError("Policy posterior must remain finite")
    selected = int(np.argmax(policy_posterior))
    action = data["policies"][selected][0]
    return {
        "posterior": posterior.tolist(),
        "evidence": evidence,
        "free_energy": -math.log(evidence) if evidence is not None else None,
        "policy_posterior": policy_posterior.tolist(),
        "expected_free_energy": efe.tolist(),
        "selected_policy": selected,
        "selected_action": action,
        "next_prior": (transition(action) @ posterior).tolist(),
        "backend": "geo-infer-exact-joint",
        "artifact_sha256": artifact.digest,
    }


def infer_factored_step(artifact, observation, *, prior=None, posterior=None):
    """Run exact joint inference, failing if finite inputs overflow arithmetic.

    See ``_infer_factored_step`` for the filtering and finite-policy objective.
    Unsupported numerical scale raises ValueError and never returns NaN or an
    action selected from nonfinite policy scores.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _infer_factored_step(
                artifact, observation, prior=prior, posterior=posterior
            )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Factored inference exceeds finite numeric range") from exc
