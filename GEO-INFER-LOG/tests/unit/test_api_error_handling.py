"""Regression tests for LOG API error handling (GS-223, LOG-EXC-01).

Domain ``ValueError``s raised by core logic are client faults and must map to
HTTP 400. Unexpected exceptions (TypeError, AttributeError, ...) are server
faults and must surface as HTTP 500 with a generic message instead of being
masked as 400s that leak internal exception text. GS-223 fixed the transport
router; LOG-EXC-01 extends the probe to the sibling routes / supply-chain /
delivery routers and pins the delivery coverage endpoint's fail-loud
behavior on malformed service-area geometry (no fabricated zero coverage).
"""

from typing import Any
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from geo_infer_log.api.errors import register_error_handlers
from geo_infer_log.api.transport import get_multimodal_planner, router
from geo_infer_log.api.delivery import get_last_mile_router, router as delivery_router
from geo_infer_log.api.routes import get_route_optimizer, router as routes_router
from geo_infer_log.api.supply_chain import (
    get_supply_chain_model,
    router as supply_chain_router,
)


class _FailingPlanner:
    """Stub planner whose plan_route raises the configured exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def plan_route(self, **kwargs: Any) -> dict[str, Any]:
        raise self._exc


def _client(planner: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    register_error_handlers(app)
    app.dependency_overrides[get_multimodal_planner] = lambda: planner
    return TestClient(app)


def _route_payload() -> dict[str, Any]:
    return {
        "origin": [13.404954, 52.520008],
        "destination": [13.36, 52.49],
        "allowed_modes": ["car"],
        "preferences": None,
    }


class TestTransportErrorMapping:
    """plan_route must distinguish domain errors from server faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the planner becomes a 400 with its message."""
        client = _client(_FailingPlanner(ValueError("Route could not be completed")))

        response = client.post("/transport/route", json=_route_payload())

        assert response.status_code == 400
        assert "Route could not be completed" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        """A TypeError is a server fault: 500 with a generic, non-leaking body."""
        client = _client(
            _FailingPlanner(TypeError("'int' object is not subscriptable"))
        )

        response = client.post("/transport/route", json=_route_payload())

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not subscriptable" not in response.text

    def test_attribute_error_surfaces_as_500(self) -> None:
        """AttributeError (e.g. from a malformed internal object) is a 500 too."""
        client = _client(_FailingPlanner(AttributeError("'NoneType' has no 'graph'")))

        response = client.post("/transport/route", json=_route_payload())

        assert response.status_code == 500
        assert response.json()["error"]["status_code"] == 500


def _router_client(router_obj: Any, getter: Callable[[], Any], stub: Any) -> TestClient:
    """Build a TestClient hosting one router with a stubbed dependency."""
    app = FastAPI()
    app.include_router(router_obj)
    register_error_handlers(app)
    app.dependency_overrides[getter] = lambda: stub
    return TestClient(app)


def _bare_client(router_obj: Any) -> TestClient:
    """Build a TestClient hosting one router with its real dependencies."""
    app = FastAPI()
    app.include_router(router_obj)
    register_error_handlers(app)
    return TestClient(app)


class _FailingOptimizer:
    """Stub route optimizer whose optimize_route raises the configured exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def optimize_route(self, **kwargs: Any) -> dict[str, Any]:
        raise self._exc


class _FailingSupplyChainModel:
    """Stub model whose optimize_flow raises the configured exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def optimize_flow(self, **kwargs: Any) -> dict[str, Any]:
        raise self._exc


class _FailingLastMileRouter:
    """Stub router whose optimize_deliveries raises the configured exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def optimize_deliveries(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise self._exc


_ROUTE_PAYLOAD: dict[str, Any] = {
    "origin": [13.404954, 52.520008],
    "destination": [13.36, 52.49],
    "parameters": None,
}

_FLOW_PAYLOAD: dict[str, Any] = {
    "network_id": "network-001",
    "demand_points": [
        {"id": "dp-001", "location": [8.6821, 50.1109], "demand": 200, "priority": 1},
    ],
    "supply_points": [
        {
            "id": "sp-001",
            "location": [18.0686, 59.3293],
            "supply": 500,
            "reliability": 0.95,
        },
    ],
    "objective": "cost",
}

_DELIVERY_PAYLOAD: dict[str, Any] = {
    "depot": {
        "name": "Berlin Warehouse",
        "coordinates": [13.404954, 52.520008],
        "type": "depot",
    },
    "deliveries": [
        {
            "name": "Customer A",
            "coordinates": [13.5, 52.5],
            "type": "customer",
            "service_time": 15,
            "priority": 1,
        },
    ],
    "vehicles": [
        {
            "id": "truck-001",
            "type": "truck",
            "capacity": 1000,
            "max_range": 500,
            "speed": 80,
            "cost_per_km": 1.2,
            "emissions_per_km": 0.8,
            "location": [13.404954, 52.520008],
        },
    ],
    "constraints": {},
}


class TestRoutesRouterErrorMapping:
    """/routes handlers must distinguish domain errors from server faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the optimizer becomes a 400 with its message."""
        client = _router_client(
            routes_router,
            get_route_optimizer,
            _FailingOptimizer(ValueError("No route found")),
        )

        response = client.post("/routes/optimize", json=_ROUTE_PAYLOAD)

        assert response.status_code == 400
        assert "No route found" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        """A TypeError is a server fault: 500 with a generic, non-leaking body."""
        client = _router_client(
            routes_router,
            get_route_optimizer,
            _FailingOptimizer(TypeError("'int' object is not subscriptable")),
        )

        response = client.post("/routes/optimize", json=_ROUTE_PAYLOAD)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not subscriptable" not in response.text


class TestSupplyChainRouterErrorMapping:
    """/supply-chain handlers must distinguish domain errors from server faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the model becomes a 400 with its message."""
        client = _router_client(
            supply_chain_router,
            get_supply_chain_model,
            _FailingSupplyChainModel(ValueError("Network graph must be built")),
        )

        response = client.post("/supply-chain/flow", json=_FLOW_PAYLOAD)

        assert response.status_code == 400
        assert "Network graph must be built" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        """A TypeError is a server fault: 500 with a generic, non-leaking body."""
        client = _router_client(
            supply_chain_router,
            get_supply_chain_model,
            _FailingSupplyChainModel(TypeError("unsupported operand type")),
        )

        response = client.post("/supply-chain/flow", json=_FLOW_PAYLOAD)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "unsupported operand" not in response.text


class TestDeliveryRouterErrorMapping:
    """/delivery handlers must distinguish domain errors from server faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the router becomes a 400 with its message."""
        client = _router_client(
            delivery_router,
            get_last_mile_router,
            _FailingLastMileRouter(ValueError("Vehicle capacity exceeded")),
        )

        response = client.post("/delivery/optimize", json=_DELIVERY_PAYLOAD)

        assert response.status_code == 400
        assert "Vehicle capacity exceeded" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        """A TypeError is a server fault: 500 with a generic, non-leaking body."""
        client = _router_client(
            delivery_router,
            get_last_mile_router,
            _FailingLastMileRouter(TypeError("'NoneType' object is not iterable")),
        )

        response = client.post("/delivery/optimize", json=_DELIVERY_PAYLOAD)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not iterable" not in response.text


class TestCoverageMalformedGeometry:
    """Malformed service-area geometry must fail loudly, not fabricate coverage."""

    def test_malformed_geometry_maps_to_400_without_fabricated_coverage(self) -> None:
        """A malformed GeoJSON service area yields 400 naming the depot."""
        client = _bare_client(delivery_router)

        response = client.post(
            "/delivery/coverage",
            json={
                "service_areas": {"depot-1": {"type": "Bogus", "coordinates": []}},
                "demand_points": [{"id": "d1", "location": [13.4, 52.5]}],
            },
        )

        assert response.status_code == 400
        assert "depot-1" in response.json()["detail"]
        assert "covered_points" not in response.json()

    def test_valid_geometry_preserves_coverage_contract(self) -> None:
        """The success-path coverage contract is unchanged by the error swap."""
        client = _bare_client(delivery_router)

        response = client.post(
            "/delivery/coverage",
            json={
                "service_areas": {
                    "depot-1": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [13.3, 52.4],
                                [13.5, 52.4],
                                [13.5, 52.6],
                                [13.3, 52.6],
                                [13.3, 52.4],
                            ]
                        ],
                    }
                },
                "demand_points": [
                    {"id": "d1", "location": [13.4, 52.5]},
                    {"id": "d2", "location": [13.6, 52.5]},
                ],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_points"] == 2
        assert body["covered_points"] == 1
        assert body["depot_coverage"]["depot-1"]["points"] == ["d1"]
