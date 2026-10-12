"""Analytical exact-optimization contracts for the portable supply-chain solver."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from geo_infer_log.core import supply_chain
from geo_infer_log.core.supply_chain import (
    FacilityLocator,
    NetworkOptimizer,
    SupplyChainModel,
)
from geo_infer_log.models.schemas import SupplyChainNetwork


def _sites():
    return [
        {"id": "west", "location": (0.0, 0.0)},
        {"id": "middle", "location": (1.0, 0.0)},
        {"id": "east", "location": (2.0, 0.0)},
    ]


def test_exact_two_facility_solution_serves_both_endpoints():
    # Independent equatorial distance oracle: choosing the two end points
    # yields zero distance; every other pair yields positive weighted distance.
    demands = [
        {"location": (0.0, 0.0), "demand": 5.0},
        {"location": (2.0, 0.0), "demand": 1.0},
    ]
    selected = FacilityLocator().locate_facilities(_sites(), demands, 2)
    assert {site["id"] for site in selected} == {"west", "east"}


def test_exact_one_facility_solution_uses_demand_weights():
    demands = [
        {"location": (0.0, 0.0), "demand": 5.0},
        {"location": (2.0, 0.0), "demand": 1.0},
    ]
    # Costs in equatorial degree-distance units are west=2, middle=6, east=10.
    selected = FacilityLocator().locate_facilities(_sites(), demands, 1)
    assert [site["id"] for site in selected] == ["west"]


def test_distance_infeasibility_preserves_previous_selection():
    locator = FacilityLocator()
    previous = [{"id": "previous"}]
    locator.selected_facilities = previous
    with pytest.raises(ValueError, match="infeasible"):
        locator.locate_facilities(
            _sites(),
            [{"location": (0.0, 0.0)}, {"location": (2.0, 0.0)}],
            1,
            max_distance=30.0,
        )
    assert locator.selected_facilities is previous


def test_network_budget_constraint_keeps_only_affordable_site():
    sites = [
        {"id": "west", "location": (0.0, 0.0), "fixed_cost": 20},
        {"id": "east", "location": (2.0, 0.0), "fixed_cost": 50},
    ]
    result = NetworkOptimizer().optimize_network(
        sites,
        [{"id": "demand", "location": (2.0, 0.0)}],
        {"budget": 25, "max_facilities": 1},
    )
    assert [site["id"] for site in result["selected_facilities"]] == ["west"]
    assert result["total_cost"] == 20


def test_flow_solution_conserves_supply_and_demand():
    model = SupplyChainModel()
    model.graph = nx.DiGraph()
    model.graph.add_edge("supply", "middle", capacity=3, cost=3)
    model.graph.add_edge("middle", "demand", capacity=3, cost=5)
    result = model.optimize_flow(
        [{"id": "demand", "quantity": 2}],
        [{"id": "supply", "quantity": 2}],
    )
    assert result["status"] == "Optimal"
    assert [edge["flow"] for edge in result["flows"]] == [2.0, 2.0]
    assert result["total_cost"] == 16.0


def _two_node_network(**link_properties):
    return SupplyChainNetwork(
        id="two-node",
        name="Analytical two-node flow",
        facilities=[
            {
                "id": identifier,
                "name": identifier,
                "location": (longitude, 0.0),
                "type": "warehouse",
                "capacity": 10.0,
                "operating_cost": 0.0,
            }
            for identifier, longitude in [("supply", 0.0), ("demand", 1.0)]
        ],
        links=[
            {
                "from": "supply",
                "to": "demand",
                "distance": 1.0,
                "time": 2.0,
                "cost": 3.0,
                **link_properties,
            }
        ],
    )


@pytest.mark.parametrize(
    "properties", [{}, {"capacity": float("inf")}, {"capacity": None}]
)
def test_public_network_unlimited_capacity_preserves_real_flow(properties):
    network = _two_node_network(**properties)
    original = network.model_dump()
    model = SupplyChainModel()
    model.load_network(network)
    result = model.optimize_flow(
        [{"id": "demand", "quantity": 2}],
        [{"id": "supply", "quantity": 2}],
    )
    # One link with unit cost 3 and conserved supply/demand 2 costs exactly 6.
    assert result["status"] == "Optimal"
    assert result["flows"] == [{"from": "supply", "to": "demand", "flow": 2.0}]
    assert result["total_cost"] == 6.0
    assert network.model_dump() == original


@pytest.mark.parametrize("capacity", [0.0, 1.0])
def test_finite_insufficient_capacity_is_actually_infeasible(capacity):
    model = SupplyChainModel()
    model.load_network(_two_node_network(capacity=capacity))
    with pytest.raises(ValueError, match="infeasible"):
        model.optimize_flow(
            [{"id": "demand", "quantity": 2}],
            [{"id": "supply", "quantity": 2}],
        )


@pytest.mark.parametrize("capacity", [float("nan"), float("-inf"), -1.0, True, "bad"])
def test_invalid_capacity_rejects_before_replacing_loaded_network(capacity):
    model = SupplyChainModel()
    original = _two_node_network()
    model.load_network(original)
    graph = model.graph
    with pytest.raises(ValueError, match="link capacity"):
        model.load_network(_two_node_network(capacity=capacity))
    assert model.network is original
    assert model.graph is graph


@pytest.mark.parametrize("capacity", [float("nan"), float("-inf"), -1.0])
def test_invalid_direct_graph_capacity_rejects_before_model_creation(
    capacity, monkeypatch
):
    model = SupplyChainModel()
    model.graph = nx.DiGraph()
    model.graph.add_edge("supply", "demand", capacity=capacity, cost=3)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("invalid bound reached optimization model construction")

    monkeypatch.setattr(supply_chain.pulp, "LpProblem", unexpected_model)
    with pytest.raises(ValueError, match="link capacity"):
        model.optimize_flow(
            [{"id": "demand", "quantity": 2}],
            [{"id": "supply", "quantity": 2}],
        )


def test_solver_failure_is_not_replaced_by_a_heuristic(monkeypatch):
    monkeypatch.setattr(
        supply_chain,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, status=4, x=None, message="forced solver error"
        ),
    )
    with pytest.raises(RuntimeError, match="forced solver error"):
        FacilityLocator().locate_facilities(_sites(), [{"location": (0.0, 0.0)}], 1)


@pytest.mark.parametrize("count", [0, 4, True, 1.5])
def test_invalid_facility_counts_fail_before_solving(count):
    with pytest.raises(ValueError, match="num_facilities"):
        FacilityLocator().locate_facilities(_sites(), [{"location": (0.0, 0.0)}], count)
