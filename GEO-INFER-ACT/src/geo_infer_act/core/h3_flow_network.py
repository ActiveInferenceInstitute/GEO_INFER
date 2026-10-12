"""Typed directed flow networks on one H3 resolution.

Hydrography lines digitized in the direction of flow (NHDPlus HR ``flowdir``
equal to 1) are traced onto H3 cells. Consecutive cells along one line become
directed flow edges; a step between non-neighbor cells is filled with the H3
grid path, so every edge joins H3 neighbors. :class:`H3FlowNetwork` then fixes
a typed tree toward one terminal cell with three relations:

``successor``
    the unique downstream neighbor of each cell (``None`` only at the terminal
    sea cell);
``upstream``
    the continuation of the same stream (``None`` at a selected headwater);
``side``
    the first cell of a joining stream (``None`` without a selected confluence).

Every relation joins H3 neighbors and is checked when the network is built.
Any transition kernel supported on the identity and these relations is
therefore local for H3 adjacency, which is the hypothesis of the light-cone
theorem in the fep_lean ``cascadia-geo-act-proof`` slice.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import h3

COASTLINE_FTYPE = 566
ROLES = ("sea", "main", "natal", "tributary", "slough")
RELATIONS = ("successor", "upstream", "side")


def _check_cell(cell: Any) -> str:
    if (
        not isinstance(cell, str)
        or not h3.is_valid_cell(cell)
        or h3.int_to_str(h3.str_to_int(cell)) != cell
    ):
        raise ValueError(f"Expected a canonical H3 cell string, got {cell!r}")
    return cell


def line_parts(geometry: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    """Return the lon/lat vertex lists of a GeoJSON LineString or MultiLineString."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "LineString":
        parts = [coordinates]
    elif kind == "MultiLineString":
        parts = list(coordinates or [])
    else:
        raise ValueError(f"Expected LineString or MultiLineString, got {kind!r}")
    result: list[list[tuple[float, float]]] = []
    for part in parts:
        vertices = [(float(x), float(y)) for x, y, *_ in part]
        if len(vertices) < 2 or not all(
            math.isfinite(x) and math.isfinite(y) for x, y in vertices
        ):
            raise ValueError("Line parts need at least two finite vertices")
        result.append(vertices)
    return result


def trace_cells(
    vertices: Sequence[tuple[float, float]],
    resolution: int,
    *,
    spacing_degrees: float = 2e-4,
) -> list[str]:
    """Return the ordered H3 cells a lon/lat polyline visits, neighbor-filled.

    Segments are sampled by linear interpolation in longitude/latitude every
    ``spacing_degrees``. Repeated cells collapse; a jump between non-neighbor
    cells is filled with ``h3.grid_path_cells`` so consecutive cells are always
    H3 neighbors.
    """
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must lie in 0..15")
    if not math.isfinite(spacing_degrees) or spacing_degrees <= 0:
        raise ValueError("spacing_degrees must be finite and positive")
    cells: list[str] = []

    def visit(lng: float, lat: float) -> None:
        cell = h3.latlng_to_cell(lat, lng, resolution)
        if cells and cells[-1] == cell:
            return
        if cells and not h3.are_neighbor_cells(cells[-1], cell):
            cells.extend(h3.grid_path_cells(cells[-1], cell)[1:-1])
        cells.append(cell)

    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:]):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing_degrees))
        for index in range(steps):
            fraction = index / steps
            visit(x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0))
    visit(*vertices[-1])
    return cells


@dataclass(frozen=True)
class FlowCellGraph:
    """Directed H3 cell graph traced from flow-ordered hydrography lines."""

    resolution: int
    edges: frozenset[tuple[str, str]]
    discharge: Mapping[str, float]
    stream_names: Mapping[str, Mapping[str, int]]

    @classmethod
    def from_features(
        cls,
        features: Iterable[Mapping[str, Any]],
        resolution: int,
        *,
        exclude_ftypes: Iterable[int] = (COASTLINE_FTYPE,),
        discharge_field: str = "qama",
        name_field: str = "gnis_name",
        spacing_degrees: float = 2e-4,
    ) -> FlowCellGraph:
        """Trace flow lines onto H3; lines must be digitized with the flow.

        A ``flowdir`` property other than 1 is rejected rather than reversed.
        ``discharge`` keeps the largest reported value of ``discharge_field``
        among lines touching a cell; missing or negative values count as zero.
        """
        excluded = set(exclude_ftypes)
        edges: set[tuple[str, str]] = set()
        discharge: dict[str, float] = {}
        names: dict[str, Counter[str]] = {}
        used = 0
        for feature in features:
            properties = feature.get("properties") or {}
            if properties.get("ftype") in excluded:
                continue
            if properties.get("flowdir", 1) != 1:
                raise ValueError("Flow lines must be digitized in the flow direction")
            raw = properties.get(discharge_field)
            flow = float(raw) if isinstance(raw, (int, float)) and raw > 0 else 0.0
            label = properties.get(name_field) or ""
            for vertices in line_parts(feature["geometry"]):
                cells = trace_cells(
                    vertices, resolution, spacing_degrees=spacing_degrees
                )
                for cell in cells:
                    discharge[cell] = max(discharge.get(cell, 0.0), flow)
                    names.setdefault(cell, Counter())[label] += 1
                edges.update(zip(cells, cells[1:]))
            used += 1
        if not used:
            raise ValueError("No flow lines remain after excluding feature types")
        return cls(
            resolution=resolution,
            edges=frozenset(edges),
            discharge=dict(discharge),
            stream_names={cell: dict(counter) for cell, counter in names.items()},
        )

    def cells(self) -> tuple[str, ...]:
        """Return every traced cell in sorted order."""
        return tuple(sorted(self.discharge))

    def predecessors(self, cell: str) -> tuple[str, ...]:
        """Return cells with a traced edge into ``cell``, sorted."""
        return tuple(sorted(source for source, target in self.edges if target == cell))

    def sinks(self) -> tuple[str, ...]:
        """Return cells without an outgoing traced edge."""
        sources = {source for source, _ in self.edges}
        return tuple(cell for cell in self.cells() if cell not in sources)

    def hop_distance(self, outlet: str) -> dict[str, int]:
        """Breadth-first hop count from each cell to ``outlet`` along flow edges."""
        incoming: dict[str, list[str]] = {}
        for source, target in self.edges:
            incoming.setdefault(target, []).append(source)
        distance = {outlet: 0}
        queue = deque([outlet])
        while queue:
            current = queue.popleft()
            for source in sorted(incoming.get(current, [])):
                if source not in distance:
                    distance[source] = distance[current] + 1
                    queue.append(source)
        return distance

    def stream_label(self, cells: Iterable[str]) -> str:
        """Most frequent nonempty GNIS name along ``cells``; empty when unnamed."""
        counter: Counter[str] = Counter()
        for cell in cells:
            for name, count in self.stream_names.get(cell, {}).items():
                if name:
                    counter[name] += count
        if not counter:
            return ""
        return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def walk_upstream(
        self,
        start: str,
        *,
        length: int,
        distance: Mapping[str, int],
        exclude: Iterable[str] = (),
        name: str | None = None,
    ) -> tuple[str, ...]:
        """Follow the largest-discharge upstream predecessor for ``length`` cells.

        Each step moves one hop farther from the outlet. With ``name`` the walk
        prefers predecessors that carry that GNIS name.
        """
        blocked = set(exclude)
        path: list[str] = []
        current = start
        for _ in range(length):
            candidates = [
                cell
                for cell in self.predecessors(current)
                if cell not in blocked
                and cell not in path
                and distance.get(cell) == distance[current] + 1
            ]
            if not candidates:
                break

            def rank(cell: str) -> tuple[int, float, str]:
                named = int(bool(name) and name in self.stream_names.get(cell, {}))
                return (named, self.discharge.get(cell, 0.0), cell)

            current = max(candidates, key=rank)
            path.append(current)
        return tuple(path)


def seaward_neighbors(
    outlet: str,
    coastline: Iterable[Sequence[tuple[float, float]]],
    *,
    seaward: str = "west",
) -> tuple[str, ...]:
    """Return outlet neighbors whose centers lie seaward of a coastline.

    The coastline longitude at a cell-center latitude is interpolated from the
    supplied polylines; ``seaward`` names the open-water side. Cells are ranked
    by their distance past the coastline, farthest first.
    """
    if seaward not in {"west", "east"}:
        raise ValueError("seaward must be 'west' or 'east'")
    segments = [
        segment for vertices in coastline for segment in zip(vertices, vertices[1:])
    ]

    def coast_longitude(latitude: float) -> float | None:
        crossings = [
            x0 + (latitude - y0) * (x1 - x0) / (y1 - y0)
            for (x0, y0), (x1, y1) in segments
            if y0 != y1 and min(y0, y1) <= latitude <= max(y0, y1)
        ]
        if not crossings:
            return None
        return min(crossings) if seaward == "west" else max(crossings)

    ranked: list[tuple[float, str]] = []
    for cell in h3.grid_ring(outlet, 1):
        latitude, longitude = h3.cell_to_latlng(cell)
        coast = coast_longitude(latitude)
        if coast is None:
            continue
        offset = coast - longitude if seaward == "west" else longitude - coast
        if offset > 0:
            ranked.append((offset, cell))
    return tuple(
        cell for _, cell in sorted(ranked, key=lambda item: (-item[0], item[1]))
    )


@dataclass(frozen=True)
class H3FlowNetwork:
    """A validated typed tree of H3 cells draining to one terminal sea cell."""

    resolution: int
    cells: tuple[str, ...]
    roles: Mapping[str, str]
    labels: Mapping[str, str]
    successor: Mapping[str, str | None]
    upstream: Mapping[str, str | None]
    side: Mapping[str, str | None]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check canonical cells, H3-neighbor relations and a single acyclic tree."""
        cells = [_check_cell(cell) for cell in self.cells]
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("Network cells must be nonempty and unique")
        if {h3.get_resolution(cell) for cell in cells} != {self.resolution}:
            raise ValueError("Network cells must share the declared resolution")
        domain = set(cells)
        for name in ("roles", "labels", *RELATIONS):
            if set(getattr(self, name)) != domain:
                raise ValueError(
                    f"{name} must be defined for exactly the network cells"
                )
        if any(role not in ROLES for role in self.roles.values()):
            raise ValueError(f"Roles must be drawn from {ROLES}")
        for relation in RELATIONS:
            for source, target in getattr(self, relation).items():
                if target is None:
                    continue
                if target not in domain or not h3.are_neighbor_cells(source, target):
                    raise ValueError(
                        f"{relation} relation {source}->{target} must join H3 neighbors"
                    )
        terminals = [cell for cell in cells if self.successor[cell] is None]
        if len(terminals) != 1 or self.roles[terminals[0]] != "sea":
            raise ValueError("Exactly one terminal cell with role 'sea' is required")
        distance = self.hop_distance()
        if set(distance) != domain:
            raise ValueError("Every cell must drain to the terminal cell")
        for cell in cells:
            for relation in ("upstream", "side"):
                origin = getattr(self, relation)[cell]
                if origin is not None and self.successor[origin] != cell:
                    raise ValueError(
                        f"{relation} cell of {cell} must drain into {cell}"
                    )
            if (
                self.upstream[cell] is not None
                and self.upstream[cell] == self.side[cell]
            ):
                raise ValueError("upstream and side relations must differ")

    def terminal(self) -> str:
        """Return the unique sea cell without a successor."""
        return next(cell for cell in self.cells if self.successor[cell] is None)

    def index(self) -> dict[str, int]:
        """Return the state index of each cell in matrix order."""
        return {cell: position for position, cell in enumerate(self.cells)}

    def hop_distance(self) -> dict[str, int]:
        """Return successor-chain hop counts to the terminal cell."""
        distance: dict[str, int] = {}
        for cell in self.cells:
            chain: list[str] = []
            current: str | None = cell
            while current is not None and current not in distance:
                if current in chain or len(chain) > len(self.cells):
                    raise ValueError("Successor relation contains a cycle")
                chain.append(current)
                current = self.successor.get(current)
            base = -1 if current is None else distance[current]
            for offset, member in enumerate(reversed(chain), start=1):
                distance[member] = base + offset
        return distance

    def relation_pairs(self) -> list[tuple[str, str, str]]:
        """Return ``(relation, source, target)`` triples in state order."""
        return [
            (relation, cell, target)
            for cell in self.cells
            for relation in RELATIONS
            if (target := getattr(self, relation)[cell]) is not None
        ]

    def h3_adjacency(self) -> dict[str, tuple[str, ...]]:
        """Return the H3 neighbors of each cell that lie inside the network."""
        domain = set(self.cells)
        return {
            cell: tuple(
                neighbor
                for neighbor in self.cells
                if neighbor in domain
                and neighbor != cell
                and h3.are_neighbor_cells(cell, neighbor)
            )
            for cell in self.cells
        }

    def reachable(self, sources: Iterable[str], steps: int) -> frozenset[str]:
        """Cells reachable in at most ``steps`` moves along identity or relations."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        moves: dict[str, set[str]] = {cell: {cell} for cell in self.cells}
        for _, source, target in self.relation_pairs():
            moves[source].add(target)
        frontier = set(sources)
        if not frontier <= set(self.cells):
            raise ValueError("Sources must be network cells")
        for _ in range(steps):
            frontier = {target for cell in frontier for target in moves[cell]}
        return frozenset(frontier)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe description in state order."""
        return {
            "resolution": self.resolution,
            "cells": [
                {
                    "cell": cell,
                    "role": self.roles[cell],
                    "label": self.labels[cell],
                    "successor": self.successor[cell],
                    "upstream": self.upstream[cell],
                    "side": self.side[cell],
                    "latlng": list(h3.cell_to_latlng(cell)),
                }
                for cell in self.cells
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> H3FlowNetwork:
        """Rebuild and revalidate a network written by :meth:`to_dict`."""
        rows = list(data["cells"])
        return cls(
            resolution=int(data["resolution"]),
            cells=tuple(row["cell"] for row in rows),
            roles={row["cell"]: row["role"] for row in rows},
            labels={row["cell"]: row["label"] for row in rows},
            successor={row["cell"]: row["successor"] for row in rows},
            upstream={row["cell"]: row["upstream"] for row in rows},
            side={row["cell"]: row["side"] for row in rows},
        )

    @property
    def digest(self) -> str:
        """SHA-256 of the sorted compact JSON description."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FlowNetworkSelection:
    """Declared rules that select a typed river network from a traced graph.

    ``natal_stream`` names the tributary walked for ``natal_cells`` cells.
    Every other side channel joining the main stem is walked for
    ``side_channel_cells`` cells, or one cell when its confluence lies within
    ``slough_hops`` hops of the outlet (an estuary slough).
    """

    resolution: int
    natal_stream: str
    natal_cells: int = 4
    side_channel_cells: int = 3
    slough_hops: int = 2
    main_stem_min_discharge: float = 1000.0
    seaward: str = "west"
    sea_label: str = "Pacific Ocean"
    main_label: str | None = None


def compile_flow_network(
    features: Sequence[Mapping[str, Any]],
    selection: FlowNetworkSelection,
    *,
    spacing_degrees: float = 2e-4,
) -> H3FlowNetwork:
    """Trace flow lines and select a typed sea/main/natal/side-channel network.

    The traced graph must have exactly one sink (the outlet). The main stem
    follows the largest-discharge predecessor upstream from the outlet while
    discharge stays at least ``main_stem_min_discharge``. One sea cell is the
    outlet neighbor farthest past the coastline (features with the coastline
    ftype). Cell order: sea, main stem from the outlet upstream, then side
    channels by confluence from the outlet upstream.
    """
    graph = FlowCellGraph.from_features(
        features, selection.resolution, spacing_degrees=spacing_degrees
    )
    sinks = graph.sinks()
    if len(sinks) != 1:
        raise ValueError(f"Expected one outlet sink, found {len(sinks)}")
    outlet = sinks[0]
    distance = graph.hop_distance(outlet)
    main = [outlet]
    while True:
        candidates = [
            cell
            for cell in graph.predecessors(main[-1])
            if distance.get(cell) == distance[main[-1]] + 1
            and graph.discharge.get(cell, 0.0) >= selection.main_stem_min_discharge
        ]
        if not candidates:
            break
        main.append(max(candidates, key=lambda cell: (graph.discharge[cell], cell)))
    coast = [
        vertices
        for feature in features
        if (feature.get("properties") or {}).get("ftype") == COASTLINE_FTYPE
        for vertices in line_parts(feature["geometry"])
    ]
    seaward = seaward_neighbors(outlet, coast, seaward=selection.seaward)
    if not seaward:
        raise ValueError("No outlet neighbor lies seaward of the coastline")
    sea = seaward[0]
    main_label = selection.main_label or graph.stream_label(main)

    roles: dict[str, str] = {sea: "sea"}
    labels: dict[str, str] = {sea: selection.sea_label}
    successor: dict[str, str | None] = {sea: None, outlet: sea}
    upstream: dict[str, str | None] = {sea: outlet}
    side: dict[str, str | None] = {sea: None}
    for lower, upper in zip(main, main[1:] + [None]):
        roles[lower] = "main"
        labels[lower] = main_label
        upstream[lower] = upper
        side[lower] = None
        if upper is not None:
            successor[upper] = lower

    taken = set(main) | {sea}
    channels: list[tuple[str, ...]] = []
    for confluence in main:
        options = [
            cell
            for cell in graph.predecessors(confluence)
            if cell not in taken and distance.get(cell) == distance[confluence] + 1
        ]
        if not options:
            continue
        first = max(
            options,
            key=lambda cell: (
                selection.natal_stream in graph.stream_names.get(cell, {}),
                graph.discharge.get(cell, 0.0),
                cell,
            ),
        )
        if selection.natal_stream in graph.stream_names.get(first, {}):
            role, length = "natal", selection.natal_cells
        elif distance[confluence] <= selection.slough_hops:
            role, length = "slough", 1
        else:
            role, length = "tributary", selection.side_channel_cells
        walk = (first,) + graph.walk_upstream(
            first,
            length=length - 1,
            distance=distance,
            exclude=taken,
            name=selection.natal_stream if role == "natal" else None,
        )
        label = graph.stream_label(walk)
        if role == "natal":
            label = selection.natal_stream
        elif not label:
            label = f"{main_label} {role} channel"
        side[confluence] = walk[0]
        successor[walk[0]] = confluence
        for lower, upper in zip(walk, walk[1:] + (None,)):
            roles[lower] = role
            labels[lower] = label
            upstream[lower] = upper
            side[lower] = None
            if upper is not None:
                successor[upper] = lower
        taken.update(walk)
        channels.append(walk)
    if not any(roles[walk[0]] == "natal" for walk in channels):
        raise ValueError(
            f"Natal stream {selection.natal_stream!r} does not join the main stem"
        )
    cells = (sea, *main, *(cell for walk in channels for cell in walk))
    return H3FlowNetwork(
        resolution=selection.resolution,
        cells=cells,
        roles=roles,
        labels=labels,
        successor=successor,
        upstream=upstream,
        side=side,
    )
