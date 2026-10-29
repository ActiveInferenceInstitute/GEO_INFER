"""Exact categorical river-homing models over typed H3 flow networks.

A returning migrant (for example an adult salmon) occupies one cell of an
:class:`~geo_infer_act.core.h3_flow_network.H3FlowNetwork`. Each step it senses
a noisy habitat cue and a noisy natal-odor cue, then takes one of four one-step
actions:

``hold``
    stay, with a declared chance of drifting one cell downstream;
``downstream``
    move to the successor cell;
``upstream``
    continue up the current stream;
``turn``
    enter the stream joining at the current cell.

Habitat classes, odor levels and every probability are declared modeling
assumptions applied to real network geometry; they are not fitted to fish
movement data. All probabilities are exact dyadic fractions, so the generated
GNN source, the ``gnn-geo-infer/1`` artifact and the FEP_Formal proof read the
same numbers without rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Any
from collections.abc import Mapping, Sequence

import h3

from geo_infer_act.core.gnn_spec import CategoricalGNNSpec, is_dyadic
from geo_infer_act.core.h3_flow_network import H3FlowNetwork

HABITATS = ("marine", "estuary", "river", "creek")
ODORS = ("none", "faint", "strong")
ACTIONS = ("hold", "downstream", "upstream", "turn")

_F = Fraction
DEFAULT_HABITAT_LIKELIHOOD = (
    (_F(3, 4), _F(1, 4), _F(0), _F(0)),
    (_F(1, 8), _F(3, 4), _F(1, 8), _F(0)),
    (_F(0), _F(1, 8), _F(3, 4), _F(1, 8)),
    (_F(0), _F(0), _F(1, 4), _F(3, 4)),
)
DEFAULT_ODOR_LIKELIHOOD = (
    (_F(7, 8), _F(1, 8), _F(0)),
    (_F(1, 8), _F(3, 4), _F(1, 8)),
    (_F(0), _F(1, 8), _F(7, 8)),
)


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Probabilities cannot be booleans")
    if isinstance(value, float):
        raise ValueError("Write probabilities as exact fractions such as '3/4'")
    result = Fraction(value)
    if not 0 <= result <= 1 or not is_dyadic(result):
        raise ValueError(f"{value!r} must be a dyadic probability in [0, 1]")
    return result


def _matrix(
    rows: Sequence[Sequence[Any]], shape: tuple[int, int]
) -> tuple[tuple[Fraction, ...], ...]:
    result = tuple(tuple(_fraction(value) for value in row) for row in rows)
    if len(result) != shape[0] or any(len(row) != shape[1] for row in result):
        raise ValueError(f"Expected a {shape[0]}x{shape[1]} likelihood table")
    if any(sum(row) != 1 for row in result):
        raise ValueError("Each likelihood row (true class) must sum to one")
    return result


@dataclass(frozen=True)
class RiverHomingParameters:
    """Declared cue, movement, preference and timing assumptions."""

    estuary_hops: int = 2
    strong_plume_hops: int = 1
    habitat_likelihood: tuple[tuple[Fraction, ...], ...] = DEFAULT_HABITAT_LIKELIHOOD
    odor_likelihood: tuple[tuple[Fraction, ...], ...] = DEFAULT_ODOR_LIKELIHOOD
    hold_drift: Fraction = _F(1, 4)
    downstream_move: Fraction = _F(3, 4)
    upstream_move: Fraction = _F(1, 2)
    turn_move: Fraction = _F(1, 2)
    odor_preference: tuple[float, float, float] = (-2.0, 1.0, 2.0)
    home_preference: float = 4.0
    initial_sea: Fraction = _F(3, 4)
    habit: tuple[Fraction, ...] = (_F(1, 8), _F(1, 8), _F(1, 2), _F(1, 4))
    step_seconds: int = 21600

    def __post_init__(self) -> None:
        _matrix(self.habitat_likelihood, (len(HABITATS), len(HABITATS)))
        _matrix(self.odor_likelihood, (len(ODORS), len(ODORS)))
        for value in (
            self.hold_drift,
            self.downstream_move,
            self.upstream_move,
            self.turn_move,
            self.initial_sea,
        ):
            _fraction(value)
        habit = tuple(_fraction(value) for value in self.habit)
        if len(habit) != len(ACTIONS) or sum(habit) != 1:
            raise ValueError(
                "habit must be an exact distribution over the four actions"
            )
        if self.estuary_hops < 0 or self.strong_plume_hops < 0:
            raise ValueError("Hop counts must be nonnegative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RiverHomingParameters:
        """Build parameters from YAML-style data with fraction strings."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown river-homing parameters: {sorted(unknown)}")
        values: dict[str, Any] = dict(data)
        if "habitat_likelihood" in values:
            values["habitat_likelihood"] = _matrix(values["habitat_likelihood"], (4, 4))
        if "odor_likelihood" in values:
            values["odor_likelihood"] = _matrix(values["odor_likelihood"], (3, 3))
        for key in (
            "hold_drift",
            "downstream_move",
            "upstream_move",
            "turn_move",
            "initial_sea",
        ):
            if key in values:
                values[key] = _fraction(values[key])
        if "habit" in values:
            values["habit"] = tuple(_fraction(value) for value in values["habit"])
        if "odor_preference" in values:
            values["odor_preference"] = tuple(
                float(value) for value in values["odor_preference"]
            )
        return cls(**values)


@dataclass(frozen=True)
class RiverHomingModel:
    """Compiled exact model: typed cells, cue fields and A-E matrices."""

    network: H3FlowNetwork
    parameters: RiverHomingParameters
    habitat: tuple[int, ...]
    odor: tuple[int, ...]
    A: tuple[tuple[Fraction, ...], ...]
    B: tuple[tuple[tuple[Fraction, ...], ...], ...]
    C: tuple[float, ...]
    D: tuple[Fraction, ...]
    E: tuple[Fraction, ...]
    landmarks: Mapping[str, str] = field(default_factory=dict)

    @property
    def observation_labels(self) -> tuple[str, ...]:
        """Observation labels ``habitat|odor`` in matrix order."""
        return tuple(f"{h}|{o}" for h in HABITATS for o in ODORS)

    @property
    def action_labels(self) -> tuple[str, ...]:
        """Action labels in matrix order."""
        return ACTIONS

    def denominators(self) -> dict[str, int]:
        """Least common power-of-two denominator of each probability table."""
        tables = {
            "A": [value for row in self.A for value in row],
            "B": [value for plane in self.B for cell in plane for value in cell],
            "D": list(self.D),
            "E": list(self.E),
        }
        return {
            name: lcm(*(value.denominator for value in values))
            for name, values in tables.items()
        }

    def gnn_spec(
        self, *, model_name: str, annotation: str, footer: str = ""
    ) -> CategoricalGNNSpec:
        """Return the exact GNN source specification of this model."""
        return CategoricalGNNSpec(
            model_name=model_name,
            annotation=annotation,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            step_seconds=self.parameters.step_seconds,
            action_labels=ACTIONS,
            space_kind="h3",
            footer=footer,
            parameters={
                "h3_resolution": str(self.network.resolution),
                "observation_labels": ",".join(self.observation_labels),
                "home_cell": self.landmarks["home"],
                "sea_cell": self.landmarks["sea"],
            },
        )

    def domain_record(self) -> dict[str, Any]:
        """JSON-safe sidecar binding cells, cue fields and exact tables."""
        index = self.network.index()

        def text(value: Fraction) -> str:
            return f"{value.numerator}/{value.denominator}"

        return {
            "schema_version": "geo-infer-act/river-homing/1",
            "network": self.network.to_dict(),
            "network_sha256": self.network.digest,
            "h3_adjacency": {
                cell: list(neighbors)
                for cell, neighbors in self.network.h3_adjacency().items()
            },
            "hop_distance": self.network.hop_distance(),
            "landmarks": {
                name: {"cell": cell, "index": index[cell]}
                for name, cell in self.landmarks.items()
            },
            "habitat": [HABITATS[value] for value in self.habitat],
            "odor": [ODORS[value] for value in self.odor],
            "observation_labels": list(self.observation_labels),
            "action_labels": list(ACTIONS),
            "denominators": self.denominators(),
            "exact": {
                "A": [[text(value) for value in row] for row in self.A],
                "B": [
                    [[text(value) for value in cell] for cell in plane]
                    for plane in self.B
                ],
                "D": [text(value) for value in self.D],
                "E": [text(value) for value in self.E],
            },
            "C": list(self.C),
            "step_seconds": self.parameters.step_seconds,
        }


def compile_river_homing(
    network: H3FlowNetwork, parameters: RiverHomingParameters | None = None
) -> RiverHomingModel:
    """Compile habitat/odor fields and exact A-E matrices for a typed network.

    Habitat: the sea cell is marine; main-stem cells within ``estuary_hops`` of
    the outlet and slough cells are estuary; other main-stem cells are river;
    natal and tributary cells are creek. Odor: natal cells, the natal
    confluence and ``strong_plume_hops`` main-stem cells below it are strong;
    the remaining plume down to the sea is faint; every other cell has none.
    """
    parameters = parameters or RiverHomingParameters()
    cells = network.cells
    index = network.index()
    hops = network.hop_distance()
    sea = network.terminal()
    outlet = network.upstream[sea]
    if outlet is None:
        raise ValueError("The sea cell must have an outlet upstream")
    natal_confluence = next(
        cell
        for cell in cells
        if network.side[cell] is not None
        and network.roles[network.side[cell]] == "natal"
    )
    home = natal_confluence
    while network.upstream[home] is not None or network.side[home] is not None:
        home = (
            network.side[home] if home == natal_confluence else network.upstream[home]
        )

    habitat: list[int] = []
    for cell in cells:
        role = network.roles[cell]
        if role == "sea":
            habitat.append(0)
        elif role == "slough" or (
            role == "main" and hops[cell] - 1 <= parameters.estuary_hops
        ):
            habitat.append(1)
        elif role == "main":
            habitat.append(2)
        else:
            habitat.append(3)

    plume: list[str] = []
    current: str | None = natal_confluence
    while current is not None:
        plume.append(current)
        current = network.successor[current]
    odor = [0] * len(cells)
    for position, cell in enumerate(plume):
        odor[index[cell]] = 2 if position <= parameters.strong_plume_hops else 1
    for cell in cells:
        if network.roles[cell] == "natal":
            odor[index[cell]] = 2

    habitat_table, odor_table = (
        parameters.habitat_likelihood,
        parameters.odor_likelihood,
    )
    A = tuple(
        tuple(
            habitat_table[habitat[state]][h] * odor_table[odor[state]][d]
            for state in range(len(cells))
        )
        for h in range(len(HABITATS))
        for d in range(len(ODORS))
    )

    n, u = len(cells), len(ACTIONS)
    B = [[[_F(0)] * u for _ in range(n)] for _ in range(n)]

    def put(target: str | None, source: str, action: int, mass: Fraction) -> None:
        B[index[source if target is None else target]][index[source]][action] += mass

    for cell in cells:
        downstream = network.successor[cell]
        put(cell, cell, 0, 1 - (parameters.hold_drift if downstream else 0))
        if downstream:
            put(downstream, cell, 0, parameters.hold_drift)
        put(cell, cell, 1, 1 - (parameters.downstream_move if downstream else 0))
        if downstream:
            put(downstream, cell, 1, parameters.downstream_move)
        for action, target, move in (
            (2, network.upstream[cell], parameters.upstream_move),
            (3, network.side[cell], parameters.turn_move),
        ):
            put(cell, cell, action, 1 - (move if target else 0))
            if target:
                put(target, cell, action, move)

    C = tuple(
        parameters.home_preference
        if HABITATS[h] == "creek" and ODORS[d] == "strong"
        else parameters.odor_preference[d]
        for h in range(len(HABITATS))
        for d in range(len(ODORS))
    )
    D = [_F(0)] * n
    D[index[sea]] = parameters.initial_sea
    D[index[outlet]] += 1 - parameters.initial_sea
    landmarks = {
        "sea": sea,
        "outlet": outlet,
        "natal_confluence": natal_confluence,
        "home": home,
    }
    for cell in cells:
        if (
            network.roles[cell] == "tributary"
            and network.successor[cell] is not None
            and network.roles[network.successor[cell]] == "main"
        ):
            landmarks.setdefault("decoy_confluence", network.successor[cell])
    model = RiverHomingModel(
        network=network,
        parameters=parameters,
        habitat=tuple(habitat),
        odor=tuple(odor),
        A=A,
        B=tuple(tuple(tuple(cell) for cell in plane) for plane in B),
        C=C,
        D=tuple(D),
        E=tuple(parameters.habit),
        landmarks=landmarks,
    )
    if not all(h3.are_neighbor_cells(a, b) for _, a, b in network.relation_pairs()):
        raise ValueError("Every movement relation must join H3 neighbors")
    return model
