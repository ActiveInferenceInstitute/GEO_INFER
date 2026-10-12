"""Success- and error-path contract tests for the LOG API endpoints.

LOG-EXC-01 covered the error mapping of the four optimize endpoints and the
coverage endpoint's fail-loud behavior. This module extends the probe to the
remaining routes / supply-chain / delivery endpoints: each endpoint is pinned
for its success-path status code and response shape (with the core service
stubbed behind the same ``Depends`` getters the error tests override), and for
the domain ``ValueError`` -> HTTP 400 / unexpected-exception -> HTTP 500
contract established by GS-223 and LOG-EXC-01.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from geo_infer_log.api.errors import register_error_handlers
from geo_infer_log.api.delivery import (
    get_delivery_scheduler,
    get_service_area_analyzer,
    router as delivery_router,
)
from geo_infer_log.api.routes import (
    get_fleet_manager,
    get_vehicle_router,
    router as routes_router,
)
from geo_infer_log.api.supply_chain import (
    get_facility_locator,
    get_network_optimizer,
    get_resilience_analyzer,
    get_supply_chain_model,
    router as supply_chain_router,
)
from geo_infer_log.models.schemas import Vehicle


# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_VEHICLE_PAYLOAD: dict[str, Any] = {
    "id": "truck-001",
    "type": "truck",
    "capacity": 1000,
    "max_range": 500,
    "speed": 80,
    "cost_per_km": 1.2,
    "emissions_per_km": 0.8,
    "location": [13.404954, 52.520008],
}

_DEPOT: dict[str, Any] = {
    "name": "Berlin Warehouse",
    "coordinates": [13.404954, 52.520008],
    "type": "depot",
}

_CUSTOMER_A: dict[str, Any] = {
    "name": "Customer A",
    "coordinates": [13.5, 52.5],
    "type": "customer",
    "service_time": 15,
    "priority": 1,
}

_VRP_PAYLOAD: dict[str, Any] = {
    "depot": _DEPOT,
    "deliveries": [_CUSTOMER_A],
    "vehicles": [_VEHICLE_PAYLOAD],
    "constraints": {"max_route_duration": 480},
}

_SCHEDULE_PAYLOAD: dict[str, Any] = {
    "depot": _DEPOT,
    "deliveries": [_CUSTOMER_A],
    "vehicles": [_VEHICLE_PAYLOAD],
    "start_date": "2026-09-15T08:00:00",
    "end_date": "2026-09-16T18:00:00",
    "max_deliveries_per_day": 30,
}

_NETWORK_PAYLOAD: dict[str, Any] = {
    "network": {
        "id": "network-001",
        "name": "European Distribution Network",
        "facilities": [
            {
                "id": "dc-001",
                "name": "Berlin Distribution Center",
                "location": [13.4050, 52.5200],
                "type": "distribution_center",
                "capacity": 5000,
                "operating_cost": 10000,
            },
        ],
        "links": [
            {
                "from": "dc-001",
                "to": "wh-001",
                "distance": 504,
                "time": 300,
                "cost": 600,
                "capacity": 1000,
            }
        ],
    }
}

_FACILITY_PAYLOAD: dict[str, Any] = {
    "candidates": [
        {"id": "c1", "location": [13.4050, 52.5200], "cost": 10000},
        {"id": "c2", "location": [11.5820, 48.1351], "cost": 8000},
    ],
    "demand_points": [
        {"id": "d1", "location": [8.6821, 50.1109], "demand": 200},
    ],
    "num_facilities": 1,
    "max_distance": 500,
}

_NETWORK_OPT_PAYLOAD: dict[str, Any] = {
    "locations": [
        {"id": "loc1", "location": [13.4050, 52.5200], "cost": 10000},
        {"id": "loc2", "location": [11.5820, 48.1351], "cost": 8000},
    ],
    "demand_points": [
        {"id": "d1", "location": [8.6821, 50.1109], "demand": 200},
    ],
    "constraints": {"max_facilities": 3, "max_distance": 500, "budget": 30000},
}


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingFleet:
    """Fleet-manager double that records registrations and serves lookups."""

    def __init__(self) -> None:
        self.vehicles: dict[str, Vehicle] = {}

    def add_vehicle(self, vehicle: Vehicle) -> None:
        self.vehicles[vehicle.id] = vehicle


class _FailingAdder:
    """Fleet-manager double whose add_vehicle raises the configured error."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def add_vehicle(self, vehicle: Vehicle) -> None:
        raise self._exc


class _VehicleRouterDouble:
    """VehicleRouter double exposing the fleet manager and a canned VRP result."""

    def __init__(self, fleet: Any, result: dict[str, Any]) -> None:
        self.fleet_manager = fleet
        self._result = result

    def solve_vrp(self, **kwargs: Any) -> dict[str, Any]:
        return self._result


class _FailingVRPRouter:
    """VehicleRouter double whose solve_vrp raises the configured error."""

    def __init__(self, exc: Exception) -> None:
        self.fleet_manager = _RecordingFleet()
        self._exc = exc

    def solve_vrp(self, **kwargs: Any) -> dict[str, Any]:
        raise self._exc


class _RouteLike:
    """Stand-in for a core Route with the model_dump() the handlers rely on."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class _GeoFrameLike:
    """Stand-in for a GeoDataFrame exposing only to_json()."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def to_json(self) -> str:
        return self._payload


class _SupplyModelDouble:
    """SupplyChainModel double with a canned load_network behavior."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.loaded: Any = None
        self._exc = exc

    def load_network(self, network: Any) -> None:
        if self._exc is not None:
            raise self._exc
        self.loaded = network


class _AnalyzerDouble:
    """ResilienceAnalyzer double returning canned analysis payloads."""

    def __init__(
        self,
        disruption: dict[str, Any],
        critical_nodes: list[str],
        improvements: list[dict[str, Any]],
        exc: Exception | None = None,
    ) -> None:
        self._disruption = disruption
        self._critical_nodes = critical_nodes
        self._improvements = improvements
        self._exc = exc

    def simulate_disruption(self, **kwargs: Any) -> dict[str, Any]:
        if self._exc is not None:
            raise self._exc
        return self._disruption

    def identify_critical_nodes(self) -> list[str]:
        if self._exc is not None:
            raise self._exc
        return self._critical_nodes

    def suggest_improvements(self) -> list[dict[str, Any]]:
        if self._exc is not None:
            raise self._exc
        return self._improvements


class _LocatorDouble:
    """FacilityLocator double with a canned siting result."""

    def __init__(self, result: list[dict[str, Any]]) -> None:
        self._result = result

    def locate_facilities(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._result


class _NetworkOptimizerDouble:
    """NetworkOptimizer double with a canned design result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def optimize_network(self, **kwargs: Any) -> dict[str, Any]:
        return self._result


class _SchedulerDouble:
    """DeliveryScheduler double serving canned schedules and reschedules."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._schedule: dict[str, Any] = {
            "schedule": [
                {
                    "date": "2026-09-15",
                    "route_ids": ["route-001"],
                    "deliveries": 1,
                }
            ],
            "total_deliveries": 1,
        }
        self._daily: list[dict[str, Any]] = [
            {"route_id": "route-001", "vehicle_id": "truck-001", "stops": 1}
        ]
        self._reschedule: dict[str, Any] = {
            "status": "success",
            "route_id": "route-001",
        }
        self._exc = exc

    def create_schedule(self, **kwargs: Any) -> dict[str, Any]:
        if self._exc is not None:
            raise self._exc
        return self._schedule

    def get_daily_schedule(self, date: Any) -> list[_RouteLike]:
        if self._exc is not None:
            raise self._exc
        return [_RouteLike(self._daily[0])]

    def get_vehicle_schedule(self, vehicle_id: str) -> list[_RouteLike]:
        if self._exc is not None:
            raise self._exc
        return [_RouteLike(self._daily[0])]

    def reschedule_delivery(self, **kwargs: Any) -> dict[str, Any]:
        if self._exc is not None:
            raise self._exc
        return self._reschedule


class _ServiceAreaDouble:
    """ServiceAreaAnalyzer double returning a GeoDataFrame-like area."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._geojson = '{"type": "FeatureCollection", "features": []}'
        self._exc = exc

    def create_service_area(self, **kwargs: Any) -> _GeoFrameLike:
        if self._exc is not None:
            raise self._exc
        return _GeoFrameLike(self._geojson)


# ---------------------------------------------------------------------------
# Client builders (same convention as test_api_error_handling.py)
# ---------------------------------------------------------------------------


def _client() -> tuple[TestClient, dict[str, Any]]:
    """Build a TestClient over all three routers with swappable doubles."""
    doubles: dict[str, Any] = {}
    app = FastAPI()
    app.include_router(routes_router)
    app.include_router(supply_chain_router)
    app.include_router(delivery_router)
    register_error_handlers(app)

    fleet = _RecordingFleet()
    doubles["fleet"] = fleet
    app.dependency_overrides[get_fleet_manager] = lambda: fleet

    vrp_router = _VehicleRouterDouble(
        fleet,
        {"routes": [{"vehicle_id": "truck-001", "stops": ["Customer A"]}]},
    )
    doubles["vrp_router"] = vrp_router
    app.dependency_overrides[get_vehicle_router] = lambda: vrp_router

    model = _SupplyModelDouble()
    doubles["model"] = model
    app.dependency_overrides[get_supply_chain_model] = lambda: model

    analyzer = _AnalyzerDouble(
        disruption={
            "disrupted_nodes": ["wh-001"],
            "service_level": 0.75,
            "affected_deliveries": 12,
        },
        critical_nodes=["dc-001"],
        improvements=[{"action": "add_warehouse", "benefit": 0.2}],
    )
    doubles["analyzer"] = analyzer
    app.dependency_overrides[get_resilience_analyzer] = lambda: analyzer

    locator = _LocatorDouble(
        [{"facility": {"id": "c1", "location": [13.4050, 52.5200]}, "assigned": 1}]
    )
    app.dependency_overrides[get_facility_locator] = lambda: locator

    optimizer = _NetworkOptimizerDouble(
        {"selected_locations": ["loc1"], "total_cost": 10000}
    )
    app.dependency_overrides[get_network_optimizer] = lambda: optimizer

    scheduler = _SchedulerDouble()
    doubles["scheduler"] = scheduler
    app.dependency_overrides[get_delivery_scheduler] = lambda: scheduler

    area = _ServiceAreaDouble()
    app.dependency_overrides[get_service_area_analyzer] = lambda: area

    return TestClient(app), doubles


def _client_with(router_obj: Any, getter: Any, double: Any) -> TestClient:
    """Build a TestClient hosting one router with a single overridden double."""
    app = FastAPI()
    app.include_router(router_obj)
    register_error_handlers(app)
    app.dependency_overrides[getter] = lambda: double
    return TestClient(app)


# ---------------------------------------------------------------------------
# /routes: register + list vehicles, VRP
# ---------------------------------------------------------------------------


class TestFleetEndpoints:
    """/routes/vehicles and /routes/vrp success and error contracts."""

    def test_register_vehicle_success(self) -> None:
        """Registering a vehicle returns the success envelope and stores it."""
        client, doubles = _client()

        response = client.post("/routes/vehicles", json={"vehicle": _VEHICLE_PAYLOAD})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "truck-001" in body["message"]
        assert "truck-001" in doubles["fleet"].vehicles

    def test_register_vehicle_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the fleet manager becomes a 400."""
        client = _client_with(
            routes_router,
            get_fleet_manager,
            _FailingAdder(ValueError("Vehicle already registered")),
        )

        response = client.post("/routes/vehicles", json={"vehicle": _VEHICLE_PAYLOAD})

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_get_vehicles_returns_registered_fleet(self) -> None:
        """The fleet listing round-trips registered vehicles as Vehicle models."""
        client, doubles = _client()
        client.post("/routes/vehicles", json={"vehicle": _VEHICLE_PAYLOAD})

        response = client.get("/routes/vehicles")

        assert response.status_code == 200
        body = response.json()
        assert [v["id"] for v in body] == ["truck-001"]
        assert body[0]["capacity"] == 1000

    def test_solve_vrp_success(self) -> None:
        """The VRP endpoint registers vehicles and passes through the result."""
        client, doubles = _client()

        response = client.post("/routes/vrp", json=_VRP_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["routes"][0]["vehicle_id"] == "truck-001"
        assert list(doubles["vrp_router"].fleet_manager.vehicles) == ["truck-001"]

    def test_solve_vrp_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the VRP solver becomes a 400."""
        client = _client_with(
            routes_router,
            get_vehicle_router,
            _FailingVRPRouter(ValueError("No feasible routes")),
        )

        response = client.post("/routes/vrp", json=_VRP_PAYLOAD)

        assert response.status_code == 400
        assert "No feasible routes" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /supply-chain: networks, resilience, facility location, network design
# ---------------------------------------------------------------------------


class TestSupplyChainEndpoints:
    """Supply-chain endpoint success and error contracts."""

    def test_create_network_success(self) -> None:
        """Creating a network returns the success envelope and stores the model."""
        client, doubles = _client()

        response = client.post("/supply-chain/networks", json=_NETWORK_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "network-001" in body["message"]
        assert doubles["model"].loaded is not None
        assert doubles["model"].loaded.id == "network-001"

    def test_create_network_value_error_maps_to_400(self) -> None:
        """A domain ValueError from load_network becomes a 400."""
        client = _client_with(
            supply_chain_router,
            get_supply_chain_model,
            _SupplyModelDouble(ValueError("Link references unknown facility")),
        )

        response = client.post("/supply-chain/networks", json=_NETWORK_PAYLOAD)

        assert response.status_code == 400
        assert "unknown facility" in response.json()["detail"]

    def test_disruption_analysis_success(self) -> None:
        """The disruption endpoint passes the analyzer result through verbatim."""
        client, _ = _client()

        response = client.post(
            "/supply-chain/resilience/disruption",
            json={
                "network_id": "network-001",
                "disrupted_nodes": ["wh-001"],
                "disrupted_edges": [],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["service_level"] == 0.75
        assert body["affected_deliveries"] == 12

    def test_disruption_analysis_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the analyzer becomes a 400."""
        client = _client_with(
            supply_chain_router,
            get_resilience_analyzer,
            _AnalyzerDouble({}, [], [], ValueError("Network graph must be built")),
        )

        response = client.post(
            "/supply-chain/resilience/disruption",
            json={"network_id": "network-001"},
        )

        assert response.status_code == 400
        assert "Network graph must be built" in response.json()["detail"]

    def test_critical_nodes_success(self) -> None:
        """Critical-node identification returns a plain list of node ids."""
        client, _ = _client()

        response = client.get("/supply-chain/resilience/critical-nodes")

        assert response.status_code == 200
        assert response.json() == ["dc-001"]

    def test_critical_nodes_value_error_maps_to_400(self) -> None:
        """A domain ValueError from node identification becomes a 400."""
        client = _client_with(
            supply_chain_router,
            get_resilience_analyzer,
            _AnalyzerDouble({}, [], [], ValueError("Network graph must be built")),
        )

        response = client.get("/supply-chain/resilience/critical-nodes")

        assert response.status_code == 400
        assert "Network graph must be built" in response.json()["detail"]

    def test_improvement_suggestions_success(self) -> None:
        """Improvement suggestions pass through as a list of dicts."""
        client, _ = _client()

        response = client.post("/supply-chain/resilience/improvements")

        assert response.status_code == 200
        body = response.json()
        assert body == [{"action": "add_warehouse", "benefit": 0.2}]

    def test_improvement_suggestions_value_error_maps_to_400(self) -> None:
        """A domain ValueError from suggestion generation becomes a 400."""
        client = _client_with(
            supply_chain_router,
            get_resilience_analyzer,
            _AnalyzerDouble({}, [], [], ValueError("Network graph must be built")),
        )

        response = client.post("/supply-chain/resilience/improvements")

        assert response.status_code == 400
        assert "Network graph must be built" in response.json()["detail"]

    def test_facility_location_success(self) -> None:
        """Facility siting passes the locator result through verbatim."""
        client, _ = _client()

        response = client.post(
            "/supply-chain/facility-location", json=_FACILITY_PAYLOAD
        )

        assert response.status_code == 200
        body = response.json()
        assert body[0]["facility"]["id"] == "c1"
        assert body[0]["assigned"] == 1

    def test_facility_location_value_error_maps_to_400(self) -> None:
        """A domain ValueError from siting becomes a 400."""

        class _FailingLocator:
            def locate_facilities(self, **kwargs: Any) -> list[dict[str, Any]]:
                raise ValueError("num_facilities exceeds candidates")

        client = _client_with(
            supply_chain_router, get_facility_locator, _FailingLocator()
        )

        response = client.post(
            "/supply-chain/facility-location", json=_FACILITY_PAYLOAD
        )

        assert response.status_code == 400
        assert "exceeds candidates" in response.json()["detail"]

    def test_network_optimization_success(self) -> None:
        """Network design optimization passes the optimizer result through."""
        client, _ = _client()

        response = client.post(
            "/supply-chain/network-optimization", json=_NETWORK_OPT_PAYLOAD
        )

        assert response.status_code == 200
        body = response.json()
        assert body["selected_locations"] == ["loc1"]
        assert body["total_cost"] == 10000

    def test_network_optimization_value_error_maps_to_400(self) -> None:
        """A domain ValueError from network design becomes a 400."""

        class _FailingOptimizer:
            def optimize_network(self, **kwargs: Any) -> dict[str, Any]:
                raise ValueError("Budget constraint infeasible")

        client = _client_with(
            supply_chain_router, get_network_optimizer, _FailingOptimizer()
        )

        response = client.post(
            "/supply-chain/network-optimization", json=_NETWORK_OPT_PAYLOAD
        )

        assert response.status_code == 400
        assert "infeasible" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /delivery: schedule lifecycle, service areas
# ---------------------------------------------------------------------------


class TestDeliveryScheduleEndpoints:
    """Delivery scheduling endpoint success and error contracts."""

    def test_create_schedule_success(self) -> None:
        """Schedule creation returns the scheduler's plan verbatim."""
        client, _ = _client()

        response = client.post("/delivery/schedule", json=_SCHEDULE_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["total_deliveries"] == 1
        assert body["schedule"][0]["route_ids"] == ["route-001"]

    def test_create_schedule_value_error_maps_to_400(self) -> None:
        """A domain ValueError from scheduling becomes a 400."""
        client = _client_with(
            delivery_router,
            get_delivery_scheduler,
            _SchedulerDouble(ValueError("End date precedes start date")),
        )

        response = client.post("/delivery/schedule", json=_SCHEDULE_PAYLOAD)

        assert response.status_code == 400
        assert "precedes start date" in response.json()["detail"]

    def test_get_daily_schedule_success(self) -> None:
        """Daily schedule lookup returns the dumped route dicts."""
        client, _ = _client()

        response = client.get("/delivery/schedule/2026-09-15")

        assert response.status_code == 200
        body = response.json()
        assert body == [
            {"route_id": "route-001", "vehicle_id": "truck-001", "stops": 1}
        ]

    def test_get_daily_schedule_unparsable_date_maps_to_400(self) -> None:
        """An unparsable date is a client fault, not a crash."""
        client, _ = _client()

        response = client.get("/delivery/schedule/not-a-date")

        assert response.status_code == 400

    def test_get_daily_schedule_value_error_maps_to_400(self) -> None:
        """A domain ValueError from the scheduler becomes a 400."""
        client = _client_with(
            delivery_router,
            get_delivery_scheduler,
            _SchedulerDouble(ValueError("No schedule created")),
        )

        response = client.get("/delivery/schedule/2026-09-15")

        assert response.status_code == 400
        assert "No schedule created" in response.json()["detail"]

    def test_get_vehicle_schedule_success(self) -> None:
        """Per-vehicle schedule lookup returns the dumped route dicts."""
        client, _ = _client()

        response = client.get("/delivery/schedule/vehicle/truck-001")

        assert response.status_code == 200
        body = response.json()
        assert body[0]["vehicle_id"] == "truck-001"

    def test_get_vehicle_schedule_value_error_maps_to_400(self) -> None:
        """A domain ValueError from vehicle lookup becomes a 400."""
        client = _client_with(
            delivery_router,
            get_delivery_scheduler,
            _SchedulerDouble(ValueError("Unknown vehicle")),
        )

        response = client.get("/delivery/schedule/vehicle/truck-001")

        assert response.status_code == 400
        assert "Unknown vehicle" in response.json()["detail"]

    def test_reschedule_success(self) -> None:
        """Rescheduling returns the scheduler's confirmation verbatim."""
        client, _ = _client()

        response = client.post(
            "/delivery/reschedule",
            json={
                "route_id": "route-001",
                "delivery_idx": 0,
                "new_date": "2026-09-16T14:00:00",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["route_id"] == "route-001"

    def test_reschedule_value_error_maps_to_400(self) -> None:
        """A domain ValueError from rescheduling becomes a 400."""
        client = _client_with(
            delivery_router,
            get_delivery_scheduler,
            _SchedulerDouble(ValueError("Delivery index out of range")),
        )

        response = client.post(
            "/delivery/reschedule",
            json={
                "route_id": "route-001",
                "delivery_idx": 9,
                "new_date": "2026-09-16T14:00:00",
            },
        )

        assert response.status_code == 400
        assert "out of range" in response.json()["detail"]

    def test_service_area_success(self) -> None:
        """Service-area creation wraps the GeoJSON in the documented envelope."""
        client, _ = _client()

        response = client.post(
            "/delivery/service-area",
            json={
                "depot_id": "depot-001",
                "depot_location": [13.404954, 52.520008],
                "max_time": 60,
                "max_distance": 30,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["depot_id"] == "depot-001"
        assert body["max_time"] == 60
        assert body["max_distance"] == 30
        assert "FeatureCollection" in body["area"]

    def test_service_area_value_error_maps_to_400(self) -> None:
        """A domain ValueError from area computation becomes a 400."""
        client = _client_with(
            delivery_router,
            get_service_area_analyzer,
            _ServiceAreaDouble(ValueError("Requires max_time or max_distance")),
        )

        response = client.post(
            "/delivery/service-area",
            json={
                "depot_id": "depot-001",
                "depot_location": [13.404954, 52.520008],
            },
        )

        assert response.status_code == 400
        assert "max_time or max_distance" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Cross-cutting: unexpected exceptions stay 500 on the newly covered endpoints
# ---------------------------------------------------------------------------


class TestUnexpectedFaultsStay500:
    """TypeError-grade faults must not be masked as 400s on new endpoints."""

    def test_vrp_type_error_surfaces_as_500(self) -> None:
        """A TypeError from the VRP solver is a 500 with a generic body."""
        client = _client_with(
            routes_router,
            get_vehicle_router,
            _FailingVRPRouter(TypeError("'int' object is not subscriptable")),
        )

        response = client.post("/routes/vrp", json=_VRP_PAYLOAD)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not subscriptable" not in response.text

    def test_reschedule_type_error_surfaces_as_500(self) -> None:
        """A TypeError from rescheduling is a 500 with a generic body."""
        client = _client_with(
            delivery_router,
            get_delivery_scheduler,
            _SchedulerDouble(TypeError("'NoneType' object is not subscriptable")),
        )

        response = client.post(
            "/delivery/reschedule",
            json={
                "route_id": "route-001",
                "delivery_idx": 0,
                "new_date": "2026-09-16T14:00:00",
            },
        )

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not subscriptable" not in response.text
