"""GNN/GEO-INFER v1 categorical artifact validation and inference.

GNN owns notation extraction. This module consumes data only, with explicit
single-factor matrix axes, state order and fixed physical time intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from collections.abc import Iterable

import numpy as np

CONTRACT_VERSION = "gnn-geo-infer/1"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_MATRIX_ENTRIES = 1_000_000


def _keys(value: Any, names: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{context} requires exactly {sorted(names)}")


def _shape_matches(value: Any, shape: tuple[int, ...]) -> bool:
    """Inspect JSON nesting before allocating a dense numeric array."""
    if not shape:
        return type(value) in (int, float)
    return (
        isinstance(value, list)
        and len(value) == shape[0]
        and all(_shape_matches(child, shape[1:]) for child in value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class GNNArtifact:
    """Validated artifact snapshot. Methods return independent mutable copies."""

    _json: str

    def __post_init__(self) -> None:
        if len(self._json.encode("utf-8")) > MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact exceeds byte limit")
        data = json.loads(self._json, object_pairs_hook=_unique_object)
        _validate(data)
        object.__setattr__(
            self,
            "_json",
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GNNArtifact:
        """Validate a model without repairing probabilities or dimensions."""
        return cls(json.dumps(data, allow_nan=False))

    @classmethod
    def load(cls, path: str | Path) -> GNNArtifact:
        """Read bounded UTF-8 JSON; reject duplicate keys and executable formats."""
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_ARTIFACT_BYTES + 1)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact exceeds byte limit")
        return cls(raw.decode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Return a copy preserving matrix values and state ordering."""
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        """SHA-256 of canonical sorted compact JSON encoded as UTF-8."""
        return hashlib.sha256(self._json.encode()).hexdigest()

    def write(self, path: str | Path) -> None:
        """Write the validated canonical artifact as JSON."""
        Path(path).write_text(self._json + "\n", encoding="utf-8")


def _validate(data: dict[str, Any]) -> None:
    _keys(
        data,
        {
            "schema_version",
            "model_type",
            "model_name",
            "dimensions",
            "matrices",
            "space",
            "time",
            "provenance",
        },
        "artifact",
    )
    if (
        data["schema_version"] != CONTRACT_VERSION
        or data["model_type"] != "categorical"
    ):
        raise ValueError("Unsupported GNN contract version or model type")
    if not isinstance(data["model_name"], str) or not data["model_name"].strip():
        raise ValueError("model_name must be nonempty")
    dimensions = data["dimensions"]
    _keys(dimensions, {"states", "observations", "actions"}, "dimensions")
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
        raise ValueError("Dimensions must be positive integers")
    s, o, u = (dimensions[name] for name in ("states", "observations", "actions"))
    if o * s + s * s * u + o + s + u > MAX_MATRIX_ENTRIES:
        raise ValueError("Artifact exceeds dense matrix entry budget")
    shapes = {"A": (o, s), "B": (s, s, u), "C": (o,), "D": (s,), "E": (u,)}
    _keys(data["matrices"], set(shapes), "matrices")
    for name, shape in shapes.items():
        if not _shape_matches(data["matrices"][name], shape):
            raise ValueError(f"{name} must contain JSON numbers with shape {shape}")
        raw = np.asarray(data["matrices"][name])
        if raw.dtype.kind not in "ifu":
            raise ValueError(f"{name} must contain JSON numbers")
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if name != "C" and (
            np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=0), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                f"{name} must be nonnegative and stochastic along axis zero"
            )
    _keys(data["space"], {"kind", "state_ids"}, "space")
    ids = data["space"]["state_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != s
        or any(not isinstance(x, str) or not x for x in ids)
        or len(set(ids)) != s
    ):
        raise ValueError("state_ids must uniquely label every state in matrix order")
    if data["space"]["kind"] == "h3":
        import h3

        if (
            any(
                not h3.is_valid_cell(x) or h3.int_to_str(h3.str_to_int(x)) != x
                for x in ids
            )
            or len({h3.get_resolution(x) for x in ids}) != 1
        ):
            raise ValueError("H3 state IDs must be canonical cells at one resolution")
    elif data["space"]["kind"] != "categorical":
        raise ValueError("Unsupported space kind")
    _keys(data["time"], {"step_seconds"}, "time")
    seconds = data["time"]["step_seconds"]
    if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    _keys(data["provenance"], {"producer", "source_sha256"}, "provenance")
    if (
        not isinstance(data["provenance"]["producer"], str)
        or not data["provenance"]["producer"]
    ):
        raise ValueError("producer must be nonempty")
    if not isinstance(data["provenance"]["source_sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", data["provenance"]["source_sha256"]
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def run_gnn_inference(
    artifact: GNNArtifact,
    observations: Iterable[dict[str, Any]],
    *,
    random_seed: int = 0,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    """Condition once per observation, select a one-step policy, propagate once.

    Observation records contain exactly ``timestamp`` and integer ``observation``.
    The prior at the first timestamp is D. The selected action at t produces the
    prior at t+1 via B[:, :, action] @ posterior. Requires GEO-INFER-TIME only
    when this runner is invoked; loading an artifact does not import it.
    """
    from itertools import islice
    from geo_infer_time.core.inference_schedule import inference_schedule
    from geo_infer_act.utils.pymdp_adapter import run_pymdp_step

    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    records = list(islice(observations, max_steps + 1))
    if len(records) > max_steps:
        raise ValueError("Observation count exceeds max_steps")
    data = artifact.to_dict()
    o = data["dimensions"]["observations"]
    for record in records:
        _keys(record, {"timestamp", "observation"}, "observation record")
        if type(record["observation"]) is not int or not 0 <= record["observation"] < o:
            raise ValueError("Observation index is out of range")
    timestamps = inference_schedule(
        (r["timestamp"] for r in records),
        step_seconds=data["time"]["step_seconds"],
        max_steps=max_steps,
    )
    matrices = {k: np.asarray(v, dtype=float) for k, v in data["matrices"].items()}
    prior = matrices["D"].copy()
    trace = []
    for step, (record, timestamp) in enumerate(zip(records, timestamps)):
        observed = record["observation"]
        evidence = float(matrices["A"][observed] @ prior)
        if evidence <= 0:
            raise ValueError(
                f"Observation at step {step} has zero probability under the model"
            )
        result = run_pymdp_step(
            observation=np.eye(1, o, observed).reshape(-1),
            observation_model=matrices["A"],
            transition_model=matrices["B"],
            preferences=matrices["C"],
            prior=prior,
            policy_prior=matrices["E"],
            action_count=data["dimensions"]["actions"],
            random_seed=random_seed + step,
            strict=True,
        )
        # Exact categorical conditioning retains structural zeros; pymdp owns policy evaluation.
        posterior = matrices["A"][observed] * prior / evidence
        if not np.allclose(result.beliefs, posterior, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "pymdp posterior disagrees with the artifact's categorical update"
            )
        next_prior = matrices["B"][:, :, result.selected_action_index] @ posterior
        trace.append(
            {
                "timestamp": timestamp.isoformat(),
                "observation": observed,
                "prior": prior.tolist(),
                "posterior": posterior.tolist(),
                "next_prior": next_prior.tolist(),
                "action": result.selected_action_index,
                "policy_posterior": result.policy_posterior.tolist(),
                "free_energy": -float(np.log(evidence)),
                "backend": result.to_metadata(),
            }
        )
        prior = next_prior
    return {
        "schema_version": CONTRACT_VERSION,
        "artifact_sha256": artifact.digest,
        "state_ids": data["space"]["state_ids"],
        "random_seed": random_seed,
        "steps": trace,
    }
