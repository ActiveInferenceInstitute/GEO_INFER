#!/usr/bin/env python3
"""GEO-INFER-ACT module orchestrator.

Runs one documented end-to-end ACT operation on synthetic data: build a
small categorical active-inference generative model (3 hidden states, 3
observations, 3 actions with explicit A/B/C/D matrices), attach it to an
``ActiveInferenceModel``, and run six seeded perceive-act steps, reporting
beliefs, selected actions, and free energy per step. All work goes through
the real ``geo_infer_act`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_act import ActiveInferenceModel, GenerativeModel

    n_states, n_obs, n_actions = 3, 3, 3

    # Observation model A: near-identity likelihood with mild confusability.
    A = 0.85 * np.eye(n_obs) + 0.15 / n_obs
    # Transition model B (next state x current state): column-stochastic,
    # with a stay-put bias under the stay action.
    B = np.zeros((n_states, n_states))
    for current in range(n_states):
        B[current, current] += 0.6
        B[(current + 1) % n_states, current] += 0.3
        B[(current + 2) % n_states, current] += 0.1

    # Prior preferences C: favor observation 2 (the "goal" observation).
    C = {"observations": np.array([0.15, 0.15, 0.70])}
    # Initial beliefs D: uniform over hidden states.
    D = np.ones(n_states) / n_states

    generative_model = GenerativeModel(
        model_type="categorical",
        parameters={
            "state_dim": n_states,
            "obs_dim": n_obs,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "random_seed": 7,
        },
    )
    agent = ActiveInferenceModel(
        model_type="categorical",
        allow_local_pymdp_fallback=True,
        random_seed=7,
    )
    agent.set_generative_model(generative_model)

    rng = np.random.default_rng(7)
    available_actions: list[int] = list(range(n_actions))
    steps: list[dict[str, Any]] = []
    for _ in range(6):
        true_state = int(rng.integers(0, n_states))
        observation = np.zeros(n_obs, dtype=float)
        observation[true_state] = 1.0
        result = agent.step(
            observation,
            available_actions=available_actions,
            return_result=True,
        )
        beliefs = np.asarray(result.beliefs["states"], dtype=float).reshape(-1)
        steps.append(
            {
                "observation_state": true_state,
                "selected_action": int(result.action),
                "free_energy": float(result.free_energy),
                "posterior_beliefs": [float(b) for b in beliefs],
            }
        )

    final_beliefs = np.asarray(agent.current_beliefs["states"], dtype=float).reshape(-1)
    return {
        "operation": "active_inference_perceive_act_loop",
        "model": {"states": n_states, "observations": n_obs, "actions": n_actions},
        "n_steps": len(steps),
        "steps": steps,
        "final_belief_max_state": int(np.argmax(final_beliefs)),
        "final_belief_entropy": float(
            -np.sum(final_beliefs * np.log(final_beliefs + 1e-12))
        ),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ACT", _operation))
