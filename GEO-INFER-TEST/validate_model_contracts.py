#!/usr/bin/env python3
"""Execute deterministic model-contract checks for representative GEO-INFER models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any
from collections.abc import Callable

import numpy as np

from geo_infer_act.models.base import CategoricalModel, GaussianModel
from geo_infer_act.models.climate import ClimateModel
from geo_infer_act.models.ecological import EcologicalModel
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.models.resource import ResourceModel
from geo_infer_act.models.urban import UrbanModel
from geo_infer_test.testing import (
    assert_finite,
    assert_probability,
    assert_same_finite_values,
    assert_stochastic_matrix,
)


def _finite_tree(value: Any, *, name: str) -> None:
    """Validate nested model outputs without coercing object arrays prematurely."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, name=f"{name}[{index}]")
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in enumerate(value.flat):
            _finite_tree(item, name=f"{name}[{index}]")
    else:
        assert_finite(value, name=name)


def _record(records: list[dict[str, Any]], name: str, output: Any) -> None:
    """Record a finite model output with stable shape metadata."""
    _finite_tree(output, name=name)
    records.append({"name": name, "finite": True})


def audit_model_contracts(seed: int = 42, *, strict: bool = False) -> dict[str, Any]:
    """Run shape, probability, stochasticity, replay, and reset checks."""
    records: list[dict[str, Any]] = []

    categorical = CategoricalModel(state_dim=3, obs_dim=3)
    initial_categorical = categorical.beliefs.copy()
    categorical.set_transition_matrix(
        np.array([[3.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0]])
    )
    categorical.set_likelihood_matrix(
        np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    )
    assert_stochastic_matrix(
        categorical.transition_matrix, axis=1, name="categorical.transition"
    )
    assert_stochastic_matrix(
        categorical.likelihood_matrix, axis=0, name="categorical.likelihood"
    )
    posterior = categorical.update_beliefs(
        np.array([10**9, 10**9 + 1, 10**9 + 2], dtype=float)
    )
    assert_probability(posterior, name="categorical.posterior")
    _record(records, "categorical.update_beliefs", posterior)
    _record(records, "categorical.step", categorical.step())
    categorical.reset()
    assert_same_finite_values(
        categorical.beliefs, initial_categorical, name="categorical.reset"
    )

    gaussian = GaussianModel(state_dim=2, obs_dim=2)
    initial_gaussian = gaussian.belief_mean.copy()
    gaussian.set_transition_model(np.eye(2), Q=np.eye(2) * 0.01)
    gaussian.set_observation_model(np.eye(2), R=np.eye(2) * 0.01)
    gaussian_output = gaussian.update_beliefs(np.array([1.0, -1.0]))
    _record(records, "gaussian.update_beliefs", gaussian_output)
    _record(records, "gaussian.step", gaussian.step(np.array([0.25])))
    gaussian.reset()
    assert_same_finite_values(
        gaussian.belief_mean, initial_gaussian, name="gaussian.reset"
    )

    factories: list[tuple[str, Callable[[], Any]]] = [
        ("climate", lambda: ClimateModel(random_seed=seed)),
        ("ecological", lambda: EcologicalModel(random_seed=seed)),
        ("urban", lambda: UrbanModel(n_agents=2, n_locations=3, random_seed=seed)),
        (
            "resource",
            lambda: ResourceModel(n_resources=2, n_locations=3, random_seed=seed),
        ),
        (
            "multi_agent",
            lambda: MultiAgentModel(n_agents=2, n_resources=2, random_seed=seed),
        ),
    ]
    for name, factory in factories:
        first = factory()
        second = factory()
        first_output = first.step()
        second_output = second.step()
        _record(records, f"{name}.step", first_output)
        _finite_tree(second_output, name=f"{name}.replay")
        if name == "resource":
            assert_same_finite_values(
                first_output[0]["resource_distribution"],
                second_output[0]["resource_distribution"],
                name=f"{name}.seed_replay",
            )
            before_reset = first._initial_resource_distribution.copy()
            first.reset()
            assert_same_finite_values(
                first.resource_distribution, before_reset, name=f"{name}.reset"
            )
        elif name == "urban":
            assert_same_finite_values(
                first.resource_levels,
                second.resource_levels,
                name=f"{name}.seed_replay",
            )
            first.reset()
            assert_same_finite_values(
                first.resource_levels,
                first._initial_resource_levels,
                name=f"{name}.reset",
            )
        elif name == "multi_agent":
            assert_same_finite_values(
                first.resource_distribution,
                second.resource_distribution,
                name=f"{name}.seed_replay",
            )
            first.reset()
            assert_same_finite_values(
                first.resource_distribution,
                first._initial_resource_distribution,
                name=f"{name}.reset",
            )
        else:
            _finite_tree(first.current_beliefs, name=f"{name}.beliefs")
            _finite_tree(second.current_beliefs, name=f"{name}.replay_beliefs")

    result = {"seed": seed, "models_checked": len(records), "records": records}
    if strict and result["models_checked"] < 7:
        raise AssertionError("strict model audit did not exercise every required model")
    return result


def main() -> int:
    """Run model contracts and emit machine-readable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit_model_contracts(args.seed, strict=args.strict)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
