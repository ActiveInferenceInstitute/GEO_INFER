"""Typed conditions, censoring and numerical gates for Cascadia seed studies."""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
import math
from typing import assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geo_infer_act.models.river_homing import RiverHomingParameters


class Condition(StrEnum):
    REFERENCE = "reference"
    UNIFORM_HABIT = "uniform_habit"
    UNINFORMATIVE_ODOR = "uninformative_odor"


def condition_parameters(
    reference: RiverHomingParameters, condition: Condition
) -> RiverHomingParameters:
    """Change exactly the declared assumption, retaining exact dyadic tables."""
    match condition:
        case Condition.REFERENCE:
            return reference
        case Condition.UNIFORM_HABIT:
            return replace(reference, habit=(Fraction(1, 4),) * 4)
        case Condition.UNINFORMATIVE_ODOR:
            return replace(
                reference,
                odor_likelihood=((Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)),) * 3,
            )
        case unreachable:
            assert_never(unreachable)


class StudyDesign(BaseModel):
    """Bounded, ordered seed grid; the same grid is used for every condition."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    seeds: tuple[int, ...] = tuple(range(8))
    conditions: tuple[Condition, ...] = tuple(Condition)
    steps: int = Field(default=48, ge=1, le=512)
    agent_seed: int = Field(default=0, ge=0, le=2**32 - 513)

    @model_validator(mode="after")
    def bounded_grid(self) -> StudyDesign:
        """Reject duplicate, empty or excessive studies before creating output."""
        if not self.seeds or len(self.seeds) > 64:
            raise ValueError("seeds must contain 1 to 64 integers")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be nonempty and unique")
        if len(self.seeds) * len(self.conditions) * self.steps > 8192:
            raise ValueError("study exceeds the 8192 observed-step budget")
        return self


class RunMeasures(BaseModel):
    """Selected fields parsed from the reference example's analysis contract."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    steps: int = Field(ge=1)
    reached_home_step: int | None
    entered_natal_step: int | None
    home_lower_bound_steps: int = Field(ge=0)
    final_posterior_home: float = Field(ge=0, le=1)
    max_pymdp_identity_gap: float = Field(ge=0, allow_inf_nan=False)
    min_decision_margin: float = Field(ge=0, allow_inf_nan=False)
    prior_light_cone_violations: list[int]
    posterior_support_violations: list[int]
    home_prior_mass_before_bound: float = Field(ge=0, allow_inf_nan=False)
    mean_free_energy: float = Field(allow_inf_nan=False)

    def qualification_problems(self) -> tuple[str, ...]:
        """Numerical qualification never asserts that the animal must arrive."""
        problems = []
        if self.prior_light_cone_violations:
            problems.append("prior exceeds the graph light cone")
        if self.posterior_support_violations:
            problems.append("posterior exceeds prior support")
        if self.home_prior_mass_before_bound != 0:
            problems.append("home prior has mass before the graph lower bound")
        if self.max_pymdp_identity_gap >= 1e-5:
            problems.append("pymdp identity gap is at least 1e-5")
        for event in (self.reached_home_step, self.entered_natal_step):
            if event is not None and not 0 <= event < self.steps:
                problems.append("arrival event is outside the observed horizon")
        if (
            self.reached_home_step is not None
            and self.reached_home_step < self.home_lower_bound_steps
        ):
            problems.append("home arrival precedes the graph lower bound")
        return tuple(problems)


class SeedOutcome(BaseModel):
    """One retained trajectory, including arrivals censored at the last observation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    condition: Condition
    process_seed: int
    measures: RunMeasures
    run_directory: str
    qualified: bool
    problems: tuple[str, ...]


class ConditionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    condition: Condition
    runs: int
    home_arrivals: int
    home_censored: int
    arrival_fraction: float
    restricted_mean_first_home_steps: float
    max_pymdp_identity_gap: float
    min_decision_margin: float


def summarize(outcomes: tuple[SeedOutcome, ...]) -> tuple[ConditionSummary, ...]:
    """Report all seeds; censoring contributes the observed horizon, never zero.

    The restricted mean is the empirical average min(T_home, steps - 1).
    It is not a mean arrival time among successful trajectories.
    """
    summaries = []
    for condition in dict.fromkeys(item.condition for item in outcomes):
        selected = [item.measures for item in outcomes if item.condition == condition]
        arrivals = sum(item.reached_home_step is not None for item in selected)
        restricted = [
            item.reached_home_step
            if item.reached_home_step is not None
            else item.steps - 1
            for item in selected
        ]
        summaries.append(
            ConditionSummary(
                condition=condition,
                runs=len(selected),
                home_arrivals=arrivals,
                home_censored=len(selected) - arrivals,
                arrival_fraction=arrivals / len(selected),
                restricted_mean_first_home_steps=math.fsum(restricted) / len(selected),
                max_pymdp_identity_gap=max(
                    item.max_pymdp_identity_gap for item in selected
                ),
                min_decision_margin=min(item.min_decision_margin for item in selected),
            )
        )
    return tuple(summaries)
