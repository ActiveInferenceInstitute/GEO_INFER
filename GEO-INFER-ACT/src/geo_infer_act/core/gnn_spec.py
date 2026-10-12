"""Exact GNN v1 Markdown for single-factor categorical active-inference models.

GNN owns notation extraction and the ``gnn-geo-infer/1`` producer. This module
writes the *source* side: a GNN Markdown document whose ``InitialParameterization``
carries explicit A, B, C, D and E literals in the axis order the GNN strict
extractor verifies (``A[observation, state]``, ``B[next_state, previous_state,
action]``). Probabilities are exact fractions; when every denominator is a
power of two the decimal literals are exact binary64 values, so the GNN
artifact, the GEO-INFER consumer and a Lean proof read identical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math
from typing import Mapping, Sequence

DEFAULT_ONTOLOGY = {
    "A": "LikelihoodMatrix",
    "B": "TransitionMatrix",
    "C": "LogPreferenceVector",
    "D": "PriorOverHiddenStates",
    "E": "Habit",
    "G": "ExpectedFreeEnergy",
    "s": "HiddenState",
    "s_prime": "NextHiddenState",
    "o": "Observation",
    "π": "PolicyVector",
    "u": "Action",
    "t": "Time",
}


def is_dyadic(value: Fraction) -> bool:
    """Return True when the reduced denominator is a power of two."""
    denominator = Fraction(value).denominator
    return denominator & (denominator - 1) == 0


def exact_decimal(value: Fraction) -> str:
    """Render a dyadic fraction as its exact terminating decimal literal."""
    value = Fraction(value)
    if not is_dyadic(value):
        raise ValueError(f"{value} has no exact binary decimal literal")
    with localcontext() as context:
        context.prec = 80
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if "." not in text:
        text += ".0"
    text = text.rstrip("0")
    return text + "0" if text.endswith(".") else text


def _float_literal(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Preference literals must be finite")
    text = repr(float(value))
    return text if "." in text or "e" in text else text + ".0"


def _check_distribution(values: Sequence[Fraction], context: str) -> None:
    if any(value < 0 for value in values) or sum(values) != 1:
        raise ValueError(f"{context} must be a nonnegative exact distribution")


@dataclass(frozen=True)
class CategoricalGNNSpec:
    """One-factor categorical model with exact probabilities and GNN metadata."""

    model_name: str
    annotation: str
    A: tuple[tuple[Fraction, ...], ...]
    B: tuple[tuple[tuple[Fraction, ...], ...], ...]
    C: tuple[float, ...]
    D: tuple[Fraction, ...]
    E: tuple[Fraction, ...]
    step_seconds: int
    action_labels: tuple[str, ...]
    space_kind: str = "categorical"
    equations: str = (
        "State inference conditions the predictive prior on the observation "
        "likelihood; action selection minimizes expected free energy over one-step "
        "policies under the habit prior."
    )
    footer: str = ""
    parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observations, states = len(self.A), len(self.D)
        actions = len(self.E)
        if not self.model_name.strip() or "\n" in self.model_name:
            raise ValueError("model_name must be one nonempty line")
        if any(len(row) != states for row in self.A):
            raise ValueError("A must have shape [observation, state]")
        if len(self.B) != states or any(
            len(plane) != states or any(len(cell) != actions for cell in plane)
            for plane in self.B
        ):
            raise ValueError("B must have shape [next_state, previous_state, action]")
        if len(self.C) != observations or len(self.action_labels) != actions:
            raise ValueError("C and action labels must match the model dimensions")
        for state in range(states):
            _check_distribution([row[state] for row in self.A], f"A[:, {state}]")
            for action in range(actions):
                _check_distribution(
                    [self.B[nxt][state][action] for nxt in range(states)],
                    f"B[:, {state}, {action}]",
                )
        _check_distribution(self.D, "D")
        _check_distribution(self.E, "E")
        if self.space_kind not in {"categorical", "h3"}:
            raise ValueError("space_kind must be 'categorical' or 'h3'")
        if (
            isinstance(self.step_seconds, bool)
            or not isinstance(self.step_seconds, int)
            or self.step_seconds < 1
        ):
            raise ValueError("step_seconds must be a positive integer")

    @property
    def dimensions(self) -> dict[str, int]:
        """Return states, observations and actions counts."""
        return {
            "states": len(self.D),
            "observations": len(self.A),
            "actions": len(self.E),
        }

    def float_matrices(self) -> dict[str, list]:
        """Return the A-E matrices as nested float lists in GNN axis order."""
        return {
            "A": [[float(value) for value in row] for row in self.A],
            "B": [
                [[float(value) for value in cell] for cell in plane] for plane in self.B
            ],
            "C": [float(value) for value in self.C],
            "D": [float(value) for value in self.D],
            "E": [float(value) for value in self.E],
        }

    def render(self) -> str:
        """Return GNN v1 Markdown accepted by the GNN strict POMDP extractor."""
        n, m, u = (
            self.dimensions["states"],
            self.dimensions["observations"],
            self.dimensions["actions"],
        )

        def vector(values: Sequence[str]) -> str:
            return "(" + ", ".join(values) + ")"

        a_rows = ",\n".join(
            "  " + vector([exact_decimal(value) for value in row]) for row in self.A
        )
        b_planes = ",\n".join(
            "  (\n"
            + ",\n".join(
                "    " + vector([exact_decimal(value) for value in cell])
                for cell in plane
            )
            + "\n  )"
            for plane in self.B
        )
        parameter_lines = "\n".join(
            f"{key}: {value}"
            for key, value in {
                "num_hidden_states": str(n),
                "num_obs": str(m),
                "num_actions": str(u),
                "b_tensor_order": "next_state_previous_state_action",
                "action_labels": ",".join(self.action_labels),
                **dict(self.parameters),
            }.items()
        )
        ontology = "\n".join(
            f"{key}={value}" for key, value in DEFAULT_ONTOLOGY.items()
        )
        return f"""# GNN Example: {self.model_name}
# GNN Version: 1.0

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
{self.model_name}

## ModelAnnotation
{self.annotation.strip()}

## StateSpaceBlock
A[{m},{n},type=float]      # Likelihood matrix: observations by hidden states
B[{n},{n},{u},type=float]  # Transition tensor: next_state, previous_state, action
C[{m},type=float]          # Log-preferences over observations
D[{n},type=float]          # Prior over initial hidden state
E[{u},type=float]          # Habit prior over one-step policies

s[{n},1,type=float]        # Hidden state distribution over H3 cells
s_prime[{n},1,type=float]  # Next hidden state distribution
o[{m},1,type=int]          # Current observation
π[{u},type=float]          # Policy distribution over actions
u[1,type=int]              # Action index
G[π,type=float]            # Expected free energy per policy
t[1,type=int]              # Discrete time step

## Connections
D>s
s-A
A-o
s-B
B>u
u>s_prime
C>G
E>π
G>π
π>u
s>s_prime

## InitialParameterization
A={{
{a_rows}
}}

B={{
{b_planes}
}}

C={{{vector([_float_literal(value) for value in self.C])}}}

D={{{vector([exact_decimal(value) for value in self.D])}}}

E={{{vector([exact_decimal(value) for value in self.E])}}}

## Equations
{self.equations.strip()}

## Time
Time=t
Dynamic
Discrete
StepSeconds={self.step_seconds}
SpaceKind={self.space_kind}

## ActInfOntologyAnnotation
{ontology}

## ModelParameters
{parameter_lines}

## Footer
{self.footer.strip() or self.model_name}
"""

    def sha256(self) -> str:
        """SHA-256 of the rendered UTF-8 source, matching GNN provenance."""
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()
