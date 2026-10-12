"""Typed H3 flow networks traced from real NHDPlus HR lower Smith River lines."""

from __future__ import annotations

import json
from pathlib import Path

import h3
import pytest

from geo_infer_act.core.h3_flow_network import (
    FlowCellGraph,
    FlowNetworkSelection,
    H3FlowNetwork,
    compile_flow_network,
    seaward_neighbors,
    trace_cells,
)

ACT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ACT_ROOT.parent
FLOWLINES = (
    REPO_ROOT
    / "GEO-INFER-PLACE/src/geo_infer_place/hydrography/data/smith_expanded/flowlines.geojson"
)
DOMAIN = ACT_ROOT / "examples/cascadia/data/smith_river_homing.domain.json"


@pytest.fixture(scope="module")
def features() -> list[dict]:
    """The retained 59-feature lower Smith River excerpt."""
    return json.loads(FLOWLINES.read_text(encoding="utf-8"))["features"]


@pytest.fixture(scope="module")
def network(features: list[dict]) -> H3FlowNetwork:
    """The typed Smith River network selected for the homing example."""
    return compile_flow_network(
        features, FlowNetworkSelection(resolution=8, natal_stream="Rowdy Creek")
    )


def test_trace_cells_fills_coarse_steps_with_h3_neighbors() -> None:
    """A coarsely sampled line still yields a chain of H3 neighbors."""
    vertices = [(-124.21, 41.88), (-124.13, 41.96)]
    cells = trace_cells(vertices, 8, spacing_degrees=0.05)
    assert len(cells) > 3
    assert all(h3.are_neighbor_cells(a, b) for a, b in zip(cells, cells[1:]))
    assert cells[0] == h3.latlng_to_cell(41.88, -124.21, 8)
    assert cells[-1] == h3.latlng_to_cell(41.96, -124.13, 8)


def test_flow_cell_graph_rejects_lines_against_the_flow() -> None:
    """A line not digitized with the flow is rejected instead of reversed."""
    feature = {
        "geometry": {
            "type": "LineString",
            "coordinates": [[-124.2, 41.9], [-124.1, 41.9]],
        },
        "properties": {"ftype": 460, "flowdir": 0, "qama": 5.0},
    }
    with pytest.raises(ValueError, match="flow direction"):
        FlowCellGraph.from_features([feature], 8)


def test_smith_river_network_is_a_typed_tree(network: H3FlowNetwork) -> None:
    """Roles, the sea terminal, H3-neighbor relations and hop distances hold."""
    roles = [network.roles[cell] for cell in network.cells]
    assert len(network.cells) == 26
    assert {role: roles.count(role) for role in set(roles)} == {
        "sea": 1,
        "main": 16,
        "slough": 2,
        "tributary": 3,
        "natal": 4,
    }
    assert network.roles[network.terminal()] == "sea"
    assert all(h3.are_neighbor_cells(a, b) for _, a, b in network.relation_pairs())
    hops = network.hop_distance()
    for cell in network.cells:
        successor = network.successor[cell]
        assert successor is None or hops[successor] == hops[cell] - 1
    assert {
        network.labels[cell] for cell in network.cells if network.roles[cell] == "natal"
    } == {"Rowdy Creek"}
    assert {
        network.labels[cell]
        for cell in network.cells
        if network.roles[cell] == "tributary"
    } == {"Tryon Creek"}


def test_network_round_trip_and_digest_match_committed_domain(
    network: H3FlowNetwork,
) -> None:
    """Serialization round-trips and the digest matches the committed sidecar."""
    assert H3FlowNetwork.from_dict(network.to_dict()) == network
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))
    assert domain["network_sha256"] == network.digest
    assert [row["cell"] for row in domain["network"]["cells"]] == list(network.cells)


def _floats(value: object, path: str = "") -> list[str]:
    """Paths of every float in a JSON-like value."""
    if isinstance(value, float):
        return [path]
    if isinstance(value, dict):
        return [
            hit for key, item in value.items() for hit in _floats(item, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            hit for k, item in enumerate(value) for hit in _floats(item, f"{path}[{k}]")
        ]
    return []


def test_serialized_network_and_domain_are_platform_independent(
    network: H3FlowNetwork,
) -> None:
    """Committed bytes and digests carry no value computed by the platform's math library.

    The network serializes only identifiers, roles, labels and relations. In the
    committed domain only declared inputs carry floats: the preference vector C
    and the two selection records read from configuration. Every other value is
    an identifier, an integer or an exact fraction, so a computed float, such as
    a cell coordinate, cannot reach the fixture.
    """
    assert _floats(network.to_dict()) == []
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))
    declared = (".C[", ".flowlines.selection.", ".network_selection.")
    paths = _floats(domain)
    assert paths and all(path.startswith(declared) for path in paths)
    assert all(float(value).is_integer() for value in domain["C"])


def test_validation_rejects_a_non_neighbor_relation(network: H3FlowNetwork) -> None:
    """Pointing a successor at a distant cell fails validation."""
    data = network.to_dict()
    data["cells"][5]["successor"] = data["cells"][16]["cell"]
    with pytest.raises(ValueError, match="H3 neighbors"):
        H3FlowNetwork.from_dict(data)


def test_reachable_cone_is_monotone_and_bounds_homing(network: H3FlowNetwork) -> None:
    """The reachable cone grows monotonically and reaches home first at 13 steps."""
    start = [network.terminal(), network.upstream[network.terminal()]]
    home = next(
        cell
        for cell in network.cells
        if network.roles[cell] == "natal" and network.upstream[cell] is None
    )
    cones = [network.reachable(start, steps) for steps in range(20)]
    assert all(earlier <= later for earlier, later in zip(cones, cones[1:]))
    assert cones[-1] == frozenset(network.cells)
    assert next(steps for steps, cone in enumerate(cones) if home in cone) == 13


def test_sea_cell_lies_seaward_of_the_coastline(
    features: list[dict], network: H3FlowNetwork
) -> None:
    """The selected sea cell is the outlet neighbor farthest past the coastline."""
    coast = [
        [tuple(point[:2]) for point in feature["geometry"]["coordinates"]]
        for feature in features
        if feature["properties"]["ftype"] == 566
    ]
    outlet = network.upstream[network.terminal()]
    ranked = seaward_neighbors(outlet, coast)
    assert ranked[0] == network.terminal()
    with pytest.raises(ValueError, match="seaward"):
        seaward_neighbors(outlet, coast, seaward="north")
