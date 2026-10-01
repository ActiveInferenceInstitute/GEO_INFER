#!/usr/bin/env python3
"""GEO-INFER-AGENT module orchestrator.

Runs one documented end-to-end agent operation on synthetic data: drive an
``ActiveInferenceAgent`` state through a perception-action loop on a small
synthetic environment — observe one-hot sensor readings with noise, infer
the hidden state, select actions by expected free energy, and let the
generative model learn the transition structure. All work goes through the
real ``geo_infer_agent`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_agent import ActiveInferenceState

    rng = np.random.default_rng(42)

    # Synthetic environment: 4 hidden locations, each emitting a noisy
    # one-hot observation. Actions shift the hidden location by 0, 1, or 2
    # steps (mod 4). The agent prefers observing location 2.
    n_states = 4
    n_actions = 3
    shifts = (0, 1, 2)
    preferred_location = 2

    def _observe(location: int) -> np.ndarray:
        observation = np.zeros(n_states)
        observation[location] = 1.0
        return observation + rng.normal(0.0, 0.05, n_states)

    def _transition(location: int, action: int) -> int:
        return (location + shifts[action]) % n_states

    state = ActiveInferenceState(
        state_dimensions=n_states,
        observation_dimensions=n_states,
        control_dimensions=n_actions,
    )
    preferred_obs = np.zeros(n_states)
    preferred_obs[preferred_location] = 1.0
    state.update_preferences(preferred_obs)

    location = 0
    free_energies = []
    for _ in range(30):
        observation = _observe(location)
        belief = state.update_with_observation(observation)
        action = state.generative_model.select_action(belief, planning_horizon=2)
        efe = state.generative_model.expected_free_energy(
            belief, action, planning_horizon=2
        )
        free_energies.append(float(efe))
        reward = float(preferred_obs @ observation)
        state.record_action(action, reward)
        location = _transition(location, action)

    n_early = len(free_energies) // 3
    final_belief = state.current_state_belief

    return {
        "operation": "active_inference_perception_action_loop",
        "n_steps": len(state.action_history),
        "n_actions": n_actions,
        "preferred_location": preferred_location,
        "total_reward": round(float(state.total_reward), 4),
        "mean_free_energy_first_third": round(
            float(np.mean(free_energies[:n_early])), 4
        ),
        "mean_free_energy_last_third": round(
            float(np.mean(free_energies[-n_early:])), 4
        ),
        "mean_prediction_error": round(float(np.mean(state.prediction_errors)), 6),
        "final_belief": [round(float(p), 4) for p in final_belief],
        "final_belief_argmax": int(np.argmax(final_belief)),
        "action_history_tail": [
            int(entry["action"]) for entry in state.action_history[-10:]
        ],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("AGENT", _operation))
