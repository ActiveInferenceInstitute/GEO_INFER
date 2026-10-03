"""Bounded exact joint adapter for domain-model categorical factor arrays.

Runtime A modalities may carry full factor axes or their C-order flattening.
B factors use (next, current, action); C contains log preference utilities.
Shared controls require one declared num_controls entry. Joint posteriors are
retained across observations, including correlations between factors.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np

from .gnn_factored_contract import FactoredGNNArtifact, MAX_ENTRIES, MAX_JOINT_STATES


def is_factored_model(model: Any) -> bool:
    """Recognize an explicit sequence of state-factor transition tensors."""
    transition = getattr(model, "transition_model", None)
    sequence = isinstance(transition, (list, tuple)) or (
        isinstance(transition, np.ndarray) and transition.dtype == object
    )
    return bool(
        sequence
        and len(transition)
        and all(np.asarray(value).ndim == 3 for value in transition)
    )


def build_runtime_artifact(model: Any) -> FactoredGNNArtifact:
    """Validate domain matrices and lower them to the exact joint contract."""
    parameters = model.parameters
    transitions = [np.asarray(value, dtype=float) for value in model.transition_model]
    if not 1 <= len(transitions) <= 8 or any(
        value.ndim != 3 or value.shape[0] != value.shape[1] or value.shape[0] < 1
        for value in transitions
    ):
        raise ValueError("Factored B requires square (next, current, action) tensors")
    sizes = [value.shape[0] for value in transitions]
    joint_size = math.prod(sizes)
    if joint_size > MAX_JOINT_STATES:
        raise ValueError("Joint state budget exceeded")
    controls = parameters.get("num_controls", parameters.get("action_dim"))
    if controls is None:
        controls = [value.shape[2] for value in transitions]
    controls = [controls] if isinstance(controls, (int, np.integer)) else list(controls)
    if not 1 <= len(controls) <= 8:
        raise ValueError("Factored controls require 1..8 control factors")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
        for value in controls
    ):
        raise ValueError("Factored control counts must be positive integers")
    controls = [int(value) for value in controls]
    control_factors = parameters.get("control_factor_indices")
    if control_factors is None:
        control_factors = (
            [0] * len(sizes) if len(controls) == 1 else list(range(len(sizes)))
        )
    if len(control_factors) != len(sizes) or any(
        type(value) is not int or not 0 <= value < len(controls)
        for value in control_factors
    ):
        raise ValueError(
            "control_factor_indices must declare one control per state factor"
        )
    if any(
        value.shape[2] != controls[control]
        for value, control in zip(transitions, control_factors)
    ):
        raise ValueError("B action axes must match their declared control counts")
    likelihoods = [np.asarray(value, dtype=float) for value in model.observation_model]
    preferences = [np.asarray(value, dtype=float) for value in model.preferences]
    if not 1 <= len(likelihoods) <= 8 or len(likelihoods) != len(preferences):
        raise ValueError("One preference vector is required per observation modality")
    if (
        sum(value.size for value in transitions + likelihoods + preferences)
        + joint_size**2
        + joint_size
        > MAX_ENTRIES
    ):
        raise ValueError("Factored matrix entry budget exceeded")
    modalities = []
    for index, (likelihood, preference) in enumerate(zip(likelihoods, preferences)):
        if likelihood.ndim == 2 and likelihood.shape[1] == joint_size:
            likelihood = likelihood.reshape((likelihood.shape[0], *sizes))
        if likelihood.shape[1:] != tuple(sizes):
            raise ValueError("A must preserve the full declared state-factor order")
        modalities.append(
            dict(
                id=f"modality_{index}",
                outcomes=[str(value) for value in range(likelihood.shape[0])],
                dependencies=list(range(len(sizes))),
                likelihood=likelihood.tolist(),
                preferences=preference.tolist(),
            )
        )
    initial = parameters.get("initial_joint")
    if initial is None:
        factors = parameters.get("D", model.beliefs.get("states"))
        if factors is None or len(factors) != len(sizes):
            raise ValueError(
                "One initial belief distribution is required per state factor"
            )
        initial = np.array([1.0])
        for factor, size in zip(factors, sizes):
            factor = np.asarray(factor, dtype=float)
            if factor.shape != (size,):
                raise ValueError("D must match each state-factor dimension")
            initial = np.kron(initial, factor)
        initial = initial.tolist()
    policies = parameters.get("policies")
    if policies is None:
        horizon = parameters.get(
            "policy_horizon", parameters.get("inference_horizon", 1)
        )
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or not 1 <= horizon <= 8
        ):
            raise ValueError("Factored policy horizon must be an integer in 1..8")
        if math.prod(controls) > 256:
            raise ValueError(
                "Factored action budget exceeded; supply explicit bounded policies"
            )
        policies = [
            [list(action) for _ in range(horizon)]
            for action in itertools.product(*(range(count) for count in controls))
        ]
    data = dict(
        schema_version="gnn-geo-infer/factored/1",
        model_type="categorical_factored",
        model_name=model.model_id or "runtime_factored",
        state_factors=[
            dict(id=f"factor_{index}", states=[str(value) for value in range(size)])
            for index, size in enumerate(sizes)
        ],
        control_factors=[
            dict(id=f"control_{index}", actions=[str(value) for value in range(size)])
            for index, size in enumerate(controls)
        ],
        modalities=modalities,
        transitions=[
            dict(
                dependencies=[index],
                control_factor=control,
                probabilities=value.tolist(),
            )
            for index, (value, control) in enumerate(zip(transitions, control_factors))
        ],
        initial_joint=initial,
        policies=policies,
        policy_prior=parameters.get("E", [1 / len(policies)] * len(policies)),
        time=dict(step_seconds=parameters.get("step_seconds", 1.0)),
    )
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    data["provenance"] = dict(
        producer="GEO-INFER-ACT runtime factor adapter",
        source_kind="explicit_factored_json",
        source_sha256=digest,
    )
    return FactoredGNNArtifact.from_dict(data)


def marginal_beliefs(joint: Any, sizes: list[int]) -> dict[str, Any]:
    """Return owned factor marginals together with their correlated joint."""
    joint = np.asarray(joint, dtype=float).reshape(tuple(sizes))
    factors = [
        joint.sum(
            axis=tuple(axis for axis in range(len(sizes)) if axis != index)
        ).copy()
        for index in range(len(sizes))
    ]
    return {"states": factors, "joint": joint.reshape(-1).copy()}
