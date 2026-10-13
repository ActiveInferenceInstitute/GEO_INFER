"""Finite descriptive scores of recorded observation-time posteriors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from geo_infer_act.models.river_homing_study import Condition

if TYPE_CHECKING:
    from geo_infer_act.runners.study_visualization import StudyView

__all__ = [
    "StepDiagnostics",
    "StudyDiagnostics",
    "diagnose_frame",
    "diagnostic_document",
]
NORMALIZATION_TOLERANCE: Final = 1e-8


class DiagnosticInputError(ValueError):
    """A score input violates the finite categorical contract."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"invalid diagnostic input: {field}")


class StepDiagnostics(BaseModel):
    """Null surprisal represents infinite surprisal at zero true-state mass."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    posterior_entropy_nats: float = Field(ge=0, allow_inf_nan=False)
    policy_entropy_nats: float = Field(ge=0, allow_inf_nan=False)
    true_state_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    true_state_surprisal_nats: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    brier_score: float = Field(ge=0, le=2 + 4e-8, allow_inf_nan=False)


class RunDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    condition: Condition
    process_seed: int
    steps: tuple[StepDiagnostics, ...]


class StudyDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: str = "geo-infer-act/cascadia-diagnostics/1"
    manifest_sha256: str
    runs: tuple[RunDiagnostics, ...]
    native_claim_ready: bool = False


def _entropy(probabilities: tuple[float, ...], field: str) -> float:
    if (
        not probabilities
        or any(
            type(p) is bool or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or not math.isclose(
            math.fsum(probabilities), 1, rel_tol=0, abs_tol=NORMALIZATION_TOLERANCE
        )
    ):
        raise DiagnosticInputError(field)
    return math.fsum(-p * math.log(p) for p in probabilities if p > 0)


def diagnose_frame(
    posterior: tuple[float, ...],
    true_state: int,
    policy_posterior: tuple[float, ...],
    *,
    elapsed_seconds: float,
) -> StepDiagnostics:
    """Score recorded belief against the state that generated this step's cue.

    Vectors are checked at absolute tolerance 1e-8 and are never renormalized.
    Brier is the sum of squared errors over the full state carrier, without
    division by its cardinality. This is a posterior score, not held-out skill.
    """
    entropy = _entropy(posterior, "posterior")
    policy_entropy = _entropy(policy_posterior, "policy_posterior")
    if type(true_state) is not int or not 0 <= true_state < len(posterior):
        raise DiagnosticInputError("true_state")
    mass = posterior[true_state]
    return StepDiagnostics(
        elapsed_seconds=elapsed_seconds,
        posterior_entropy_nats=entropy,
        policy_entropy_nats=policy_entropy,
        true_state_probability=mass,
        true_state_surprisal_nats=-math.log(mass) if mass > 0 else None,
        brier_score=math.fsum(
            (p - (1 if i == true_state else 0)) ** 2 for i, p in enumerate(posterior)
        ),
    )


def diagnostic_document(view: StudyView) -> StudyDiagnostics:
    """Project qualified display records into a portable, manifest-bound report."""
    return StudyDiagnostics(
        manifest_sha256=view.manifest_sha256,
        runs=tuple(
            RunDiagnostics(
                condition=run.outcome.condition,
                process_seed=run.outcome.process_seed,
                steps=run.diagnostics,
            )
            for run in view.runs
        ),
    )
