"""Strict discrete-time linear Gaussian GNN/GEO-INFER v2 data contract.

Initial beliefs describe the state at the first measurement timestamp. Each
record conditions once, then its explicit control predicts the next interval.
This runner performs Gaussian filtering; it does not choose a control policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import json
import re
from typing import Any
from collections.abc import Iterable

import numpy as np

from .gnn_contract import (
    GNNArtifact,
    MAX_ARTIFACT_BYTES,
    _keys,
    _shape_matches,
    _unique_object,
)

CONTRACT_VERSION = "gnn-geo-infer/2"
MAX_MATRIX_ENTRIES = 1_000_000


def validate_gaussian_artifact(data: dict[str, Any]) -> None:
    """Reject unsupported axes, generators, invalid covariances and missing units."""
    _keys(
        data,
        {
            "schema_version",
            "model_type",
            "model_name",
            "dimensions",
            "matrices",
            "initial_belief",
            "units",
            "time",
            "provenance",
        },
        "artifact",
    )
    if (
        data["schema_version"] != CONTRACT_VERSION
        or data["model_type"] != "linear_gaussian"
    ):
        raise ValueError("Unsupported Gaussian contract version or model type")
    if not isinstance(data["model_name"], str) or not data["model_name"].strip():
        raise ValueError("model_name must be nonempty")
    dims = data["dimensions"]
    _keys(dims, {"states", "observations", "controls"}, "dimensions")
    if any(type(v) is not int or v < 1 for v in dims.values()):
        raise ValueError("Dimensions must be positive integers")
    n, m, k = (dims[x] for x in ("states", "observations", "controls"))
    if 3 * n * n + n * k + m * n + m * m + n > MAX_MATRIX_ENTRIES:
        raise ValueError("Artifact exceeds dense matrix entry budget")
    shapes = {"F": (n, n), "G": (n, k), "H": (m, n), "Q": (n, n), "R": (m, m)}
    _keys(data["matrices"], set(shapes), "matrices")
    _keys(data["initial_belief"], {"mean", "covariance"}, "initial_belief")
    values = dict(
        data["matrices"],
        mean=data["initial_belief"]["mean"],
        covariance=data["initial_belief"]["covariance"],
    )
    arrays = {}
    for name, shape in dict(shapes, mean=(n,), covariance=(n, n)).items():
        if not _shape_matches(values[name], shape):
            raise ValueError(f"{name} requires JSON numbers with shape {shape}")
        array = np.asarray(values[name], dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        arrays[name] = array
    for name in ("Q", "R", "covariance"):
        array = arrays[name]
        if not np.allclose(array, array.T, rtol=0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(array * 0.5 + array.T * 0.5)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"{name} covariance spectrum could not be validated"
            ) from exc
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError(f"{name} covariance spectrum must be finite")
        if np.min(eigenvalues) < 0 or (name != "Q" and np.min(eigenvalues) <= 0):
            raise ValueError(
                f"{name} must be positive {'semidefinite' if name == 'Q' else 'definite'}"
            )
    _keys(data["units"], set(dims), "units")
    for name, size in dims.items():
        units = data["units"][name]
        if (
            not isinstance(units, list)
            or len(units) != size
            or any(not isinstance(x, str) or not x.strip() for x in units)
        ):
            raise ValueError(f"units.{name} must label each coordinate")
    _keys(data["time"], {"domain", "step_seconds"}, "time")
    seconds = data["time"]["step_seconds"]
    if data["time"]["domain"] != "discrete":
        raise ValueError(
            "Only explicit discrete-time transitions are supported; generators require discretization"
        )
    if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    _keys(data["provenance"], {"producer", "source_sha256"}, "provenance")
    provenance = data["provenance"]
    if (
        not isinstance(provenance["producer"], str)
        or not provenance["producer"].strip()
    ):
        raise ValueError("producer must be nonempty")
    if not isinstance(provenance["source_sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", provenance["source_sha256"]
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class GaussianGNNArtifact(GNNArtifact):
    """Immutable validated v2 snapshot with bounded JSON IO and canonical digest."""

    def __post_init__(self) -> None:
        if len(self._json.encode("utf-8")) > MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact exceeds byte limit")
        data = json.loads(self._json, object_pairs_hook=_unique_object)
        validate_gaussian_artifact(data)
        object.__setattr__(
            self,
            "_json",
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


def run_gaussian_gnn_inference(
    artifact: GaussianGNNArtifact,
    observations: Iterable[dict[str, Any]],
    *,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    """Filter observed vectors once and propagate each supplied control once.

    Records have exactly timestamp, observation and control. Controls are physical
    vectors, not categorical action IDs. F, G and Q already describe one declared
    time interval; the runner never Euler-discretizes them or rescales their units.
    """
    from geo_infer_time.core.inference_schedule import inference_schedule

    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    records = list(islice(observations, max_steps + 1))
    if len(records) > max_steps:
        raise ValueError("Observation count exceeds max_steps")
    data = artifact.to_dict()
    n, m, k = (data["dimensions"][x] for x in ("states", "observations", "controls"))
    for record in records:
        _keys(record, {"timestamp", "observation", "control"}, "observation record")
        for field, size in (("observation", m), ("control", k)):
            if not _shape_matches(record[field], (size,)) or not np.all(
                np.isfinite(record[field])
            ):
                raise ValueError(f"{field} requires a finite vector of length {size}")
    stamps = inference_schedule(
        (r["timestamp"] for r in records),
        step_seconds=data["time"]["step_seconds"],
        max_steps=max_steps,
    )
    F, G, H, Q, R = (
        np.asarray(data["matrices"][x], dtype=float) for x in ("F", "G", "H", "Q", "R")
    )
    mean = np.asarray(data["initial_belief"]["mean"], dtype=float)
    covariance = np.asarray(data["initial_belief"]["covariance"], dtype=float)
    trace = []
    for record, stamp in zip(records, stamps):
        y, u = np.asarray(record["observation"]), np.asarray(record["control"])
        innovation = y - H @ mean
        innovation_cov = H @ covariance @ H.T + R
        if not np.all(np.isfinite(innovation)) or not np.all(
            np.isfinite(innovation_cov)
        ):
            raise ValueError("Gaussian innovation overflowed the finite numeric domain")
        gain = np.linalg.solve(innovation_cov, H @ covariance).T
        posterior_mean = mean + gain @ innovation
        residual = np.eye(n) - gain @ H
        posterior_cov = residual @ covariance @ residual.T + gain @ R @ gain.T
        posterior_cov = posterior_cov * 0.5 + posterior_cov.T * 0.5
        negative_log_evidence = 0.5 * (
            m * np.log(2 * np.pi)
            + np.linalg.slogdet(innovation_cov)[1]
            + innovation @ np.linalg.solve(innovation_cov, innovation)
        )
        next_mean = F @ posterior_mean + G @ u
        next_cov = F @ posterior_cov @ F.T + Q
        next_cov = next_cov * 0.5 + next_cov.T * 0.5
        if not all(
            np.all(np.isfinite(x))
            for x in (
                posterior_mean,
                posterior_cov,
                next_mean,
                next_cov,
                negative_log_evidence,
            )
        ):
            raise ValueError("Gaussian update overflowed the finite numeric domain")
        trace.append(
            dict(
                timestamp=stamp.isoformat(),
                observation=record["observation"],
                control=record["control"],
                prior_mean=mean.tolist(),
                prior_covariance=covariance.tolist(),
                posterior_mean=posterior_mean.tolist(),
                posterior_covariance=posterior_cov.tolist(),
                next_prior_mean=next_mean.tolist(),
                next_prior_covariance=next_cov.tolist(),
                negative_log_evidence=float(negative_log_evidence),
            )
        )
        mean, covariance = next_mean, next_cov
    return dict(
        schema_version=CONTRACT_VERSION,
        artifact_sha256=artifact.digest,
        units=data["units"],
        steps=trace,
    )
