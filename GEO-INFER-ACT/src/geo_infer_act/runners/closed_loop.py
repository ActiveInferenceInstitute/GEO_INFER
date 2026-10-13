"""Closed-loop simulation of a ``gnn-geo-infer/1`` categorical artifact.

By default the process uses the agent artifact's A and B. An explicit process
artifact can instead supply those laws on the same ordered carrier and clock.
Each step samples a process cue, runs :func:`run_gnn_inference` with the agent's
predictive prior, and samples the process transition under the selected action.
A final replay of the complete observation record
through ``run_gnn_inference`` with the artifact's own D must reproduce every
action and belief exactly; any disagreement raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any

import numpy as np

from geo_infer_act.core.efe_decomposition import (
    entropy,
    log_partition,
    one_step_policy_terms,
)
from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference


class ClosedLoopInferenceError(ValueError):
    """Retain sampled process evidence when the agent rejects an observation."""

    def __init__(
        self, error: ValueError, states: list[int], observations: list[int]
    ) -> None:
        super().__init__(str(error))
        self.true_states = tuple(states)
        self.observations = tuple(observations)


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    """Replayed contract trace plus the sampled true states and observations."""

    trace: dict[str, Any]
    true_states: tuple[int, ...]
    observations: tuple[int, ...]
    process_seed: int
    agent_seed: int
    process_artifact_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record of the run."""
        result = {
            "process_seed": self.process_seed,
            "agent_seed": self.agent_seed,
            "true_states": list(self.true_states),
            "observations": list(self.observations),
            "trace": self.trace,
        }
        if self.process_artifact_sha256 is not None:
            result["process_artifact_sha256"] = self.process_artifact_sha256
        return result


def _timestamp(start: datetime, step: int, seconds: float) -> str:
    return (start + timedelta(seconds=step * seconds)).isoformat()


def run_closed_loop(
    artifact: GNNArtifact,
    *,
    initial_state: int,
    steps: int,
    process_seed: int,
    agent_seed: int = 0,
    start: datetime | None = None,
    process_artifact: GNNArtifact | None = None,
) -> ClosedLoopResult:
    """Run the artifact's real agent against a compatible GNN-defined process.

    Process A/B generate cues and transitions. Agent A/B/C/D/E govern inference
    and action. The explicit initial state overrides process D; process C/E are
    unused. Observation/action indices must share meanings: artifact schema 1
    carries dimensions but no labels for those alphabets.
    """
    data = artifact.to_dict()
    process = artifact if process_artifact is None else process_artifact
    process_data = process.to_dict()
    if any(data[key] != process_data[key] for key in ("dimensions", "space", "time")):
        raise ValueError("process carrier, dimensions and clock must be compatible")
    states = data["dimensions"]["states"]
    observations_count = data["dimensions"]["observations"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(initial_state) is not int or not 0 <= initial_state < states:
        raise ValueError("initial_state must be an integer in range")
    start = start or datetime(2026, 10, 1, tzinfo=UTC)
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    seconds = data["time"]["step_seconds"]
    A = np.asarray(process_data["matrices"]["A"], dtype=float)
    B = np.asarray(process_data["matrices"]["B"], dtype=float)
    rng = np.random.default_rng(process_seed)
    prior = list(data["matrices"]["D"])
    true_state = initial_state
    true_states: list[int] = []
    observed: list[int] = []
    online: list[dict[str, Any]] = []
    for step in range(steps):
        observation = int(rng.choice(observations_count, p=A[:, true_state]))
        true_states.append(true_state)
        observed.append(observation)
        step_data = dict(data, matrices=dict(data["matrices"], D=prior))
        try:
            result = run_gnn_inference(
                GNNArtifact.from_dict(step_data),
                [
                    {
                        "timestamp": _timestamp(start, step, seconds),
                        "observation": observation,
                    }
                ],
                random_seed=agent_seed + step,
            )["steps"][0]
        except ValueError as error:
            raise ClosedLoopInferenceError(error, true_states, observed) from error
        online.append(result)
        prior = result["next_prior"]
        true_state = int(rng.choice(states, p=B[:, true_state, result["action"]]))
    records = [
        {"timestamp": _timestamp(start, step, seconds), "observation": value}
        for step, value in enumerate(observed)
    ]
    trace = run_gnn_inference(artifact, records, random_seed=agent_seed)
    for step, (live, replay) in enumerate(zip(online, trace["steps"])):
        if live["action"] != replay["action"] or not (
            np.array_equal(live["posterior"], replay["posterior"])
            and np.array_equal(live["next_prior"], replay["next_prior"])
        ):
            raise RuntimeError(
                f"Closed-loop step {step} disagrees with contract replay"
            )
    return ClosedLoopResult(
        trace=trace,
        true_states=tuple(true_states),
        observations=tuple(observed),
        process_seed=process_seed,
        agent_seed=agent_seed,
        process_artifact_sha256=(
            process.digest if process_artifact is not None else None
        ),
    )


def trace_diagnostics(
    artifact: GNNArtifact, result: ClosedLoopResult
) -> list[dict[str, Any]]:
    """Per-step belief, evidence and EFE diagnostics for a closed-loop run.

    Each record reports belief entropy, posterior mass on the true state, the
    contract's free energy (negative log evidence), every policy's risk,
    ambiguity and EFE, and the largest absolute gap between
    ``efe - log sum exp C`` and pymdp's own negative-EFE report.
    """
    data = artifact.to_dict()
    A = np.asarray(data["matrices"]["A"], dtype=float)
    B = np.asarray(data["matrices"]["B"], dtype=float)
    C = np.asarray(data["matrices"]["C"], dtype=float)
    log_z = log_partition(C)
    records = []
    for step, entry in enumerate(result.trace["steps"]):
        posterior = np.asarray(entry["posterior"], dtype=float)
        terms = one_step_policy_terms(posterior, A, B, C)
        reported = np.asarray(
            entry["backend"]["negative_expected_free_energy"], dtype=float
        )
        predicted = np.array([term.pymdp_neg_efe for term in terms])
        records.append(
            {
                "step": step,
                "timestamp": entry["timestamp"],
                "true_state": result.true_states[step],
                "observation": entry["observation"],
                "action": entry["action"],
                "prior_entropy": entropy(entry["prior"]),
                "posterior_entropy": entropy(posterior),
                "posterior_true_state": float(posterior[result.true_states[step]]),
                "map_state": int(np.argmax(posterior)),
                "free_energy": entry["free_energy"],
                "policy_posterior": list(entry["policy_posterior"]),
                "policies": [term.to_dict() for term in terms],
                "log_partition": log_z,
                "pymdp_neg_efe": reported.tolist(),
                "pymdp_identity_gap": float(np.max(np.abs(reported - predicted))),
            }
        )
    return records
