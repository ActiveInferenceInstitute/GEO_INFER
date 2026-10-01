"""
Tests for the GEO-INFER-LOG routing module.
"""

import pytest
import networkx as nx
import numpy as np

from geo_infer_log.core.routing import (
    Vehicle,
    VehicleType,
    RoutingParameters,
    RouteOptimizer,
    FleetManager,
    TravelTimeEstimator,
    MultiObjectiveOptimizer,
    RealTimeTracker,
    save_gpickle,
)


class TestVehicle:
    """Test suite for Vehicle dataclass."""

    def test_vehicle_creation(self):
        """Test creating a vehicle."""
        vehicle = Vehicle(
            id="TEST_001",
            type=VehicleType.VAN,
            capacity=500,
            max_range=200,
            speed=40,
            cost_per_km=0.8,
            emissions_per_km=0.25,
            location=(-118.25, 34.05),
        )

        assert vehicle.id == "TEST_001"
        assert vehicle.type == VehicleType.VAN
        assert vehicle.capacity == 500

    def test_vehicle_types(self):
        """Test all vehicle types."""
        types = [
            VehicleType.TRUCK,
            VehicleType.VAN,
            VehicleType.CAR,
            VehicleType.BIKE,
            VehicleType.DRONE,
        ]
        assert len(types) == 5


class TestRoutingParameters:
    """Test suite for RoutingParameters."""

    def test_default_parameters(self):
        """Test default routing parameters."""
        params = RoutingParameters()

        assert params.weight_factor == "time"
        assert params.avoid_highways is False
        assert params.traffic_model == "best_guess"

    def test_custom_parameters(self):
        """Test custom routing parameters."""
        params = RoutingParameters(
            weight_factor="distance", avoid_highways=True, avoid_tolls=True
        )

        assert params.weight_factor == "distance"
        assert params.avoid_highways is True


class TestFleetManager:
    """Test suite for FleetManager."""

    @pytest.fixture
    def fleet_manager(self):
        """Create a fleet manager with vehicles."""
        fm = FleetManager()

        vehicle1 = Vehicle(
            id="VAN_001",
            type=VehicleType.VAN,
            capacity=500,
            max_range=200,
            speed=40,
            cost_per_km=0.8,
            emissions_per_km=0.25,
            location=(-118.25, 34.05),
        )
        vehicle2 = Vehicle(
            id="TRUCK_001",
            type=VehicleType.TRUCK,
            capacity=2000,
            max_range=300,
            speed=35,
            cost_per_km=1.2,
            emissions_per_km=0.45,
            location=(-118.25, 34.05),
        )

        fm.add_vehicle(vehicle1)
        fm.add_vehicle(vehicle2)

        return fm

    def test_add_vehicle(self, fleet_manager):
        """Test adding vehicles to fleet."""
        status = fleet_manager.get_fleet_status()

        assert status["total_vehicles"] == 2
        assert "VAN_001" in status["vehicles"]
        assert "TRUCK_001" in status["vehicles"]

    def test_fleet_status(self, fleet_manager):
        """Test fleet status reporting."""
        status = fleet_manager.get_fleet_status()

        assert status["available_vehicles"] == 2
        assert status["assigned_vehicles"] == 0


class TestTravelTimeEstimator:
    """Test suite for TravelTimeEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create an estimator."""
        return TravelTimeEstimator(use_historical_data=True)

    def test_estimate_travel_time(self, estimator):
        """Test travel time estimation."""
        origin = (-118.25, 34.05)
        destination = (-118.30, 34.10)

        time = estimator.estimate_travel_time(origin, destination)

        assert time > 0
        assert time < 60  # Should be less than an hour for this distance

    def test_traffic_factor_rush_hour(self, estimator):
        """Test traffic factor during rush hour."""
        factor = estimator._get_traffic_factor("2024-01-15T08:00:00")

        assert factor == 1.5  # Rush hour

    def test_traffic_factor_night(self, estimator):
        """Test traffic factor at night."""
        factor = estimator._get_traffic_factor("2024-01-15T03:00:00")

        assert factor == 0.9  # Night time (faster)

    def test_calculate_time_matrix(self, estimator):
        """Test time matrix calculation."""
        locations = [(-118.25, 34.05), (-118.30, 34.10), (-118.20, 34.00)]

        matrix = estimator.calculate_time_matrix(locations)

        assert matrix.shape == (3, 3)
        assert np.diag(matrix).sum() == 0  # Diagonal is zero
        assert matrix[0, 1] > 0
        assert matrix[1, 0] > 0

    def test_calculate_distance_matrix(self, estimator):
        """Test distance matrix calculation."""
        locations = [
            (-118.25, 34.05),
            (-118.30, 34.10),
        ]

        matrix = estimator.calculate_distance_matrix(locations)

        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0
        assert matrix[0, 1] > 0
        # Distance should be symmetric
        assert abs(matrix[0, 1] - matrix[1, 0]) < 0.01

    def test_estimate_arrival_times(self, estimator):
        """Test arrival time estimation."""
        route = [(-118.25, 34.05), (-118.28, 34.08), (-118.30, 34.10)]

        arrivals = estimator.estimate_arrival_times(
            route=route, departure_time="2024-01-15T09:00:00", service_times=[0, 5, 5]
        )

        assert len(arrivals) == 3
        assert arrivals[0] == "2024-01-15T09:00:00"
        # Each subsequent arrival should be later
        for i in range(1, len(arrivals)):
            assert arrivals[i] > arrivals[i - 1]


class TestMultiObjectiveOptimizer:
    """Test suite for MultiObjectiveOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create an optimizer."""
        return MultiObjectiveOptimizer(objectives=["distance", "time", "emissions"])

    def test_init_weights(self, optimizer):
        """Test initial weight distribution."""
        assert len(optimizer.weights) == 3
        assert sum(optimizer.weights.values()) == pytest.approx(1.0)

    def test_set_weights(self, optimizer):
        """Test setting custom weights."""
        optimizer.set_weights({"distance": 0.2, "time": 0.5, "emissions": 0.3})

        assert optimizer.weights["time"] == pytest.approx(0.5)

    def test_calculate_pareto_front(self, optimizer):
        """Test Pareto front calculation."""
        solutions = [
            {"id": "A", "distance": 10, "time": 30, "emissions": 5},
            {"id": "B", "distance": 8, "time": 35, "emissions": 4},
            {"id": "C", "distance": 12, "time": 25, "emissions": 6},
            {"id": "D", "distance": 15, "time": 40, "emissions": 8},  # Dominated
        ]

        pareto = optimizer.calculate_pareto_front(solutions)

        # D should be dominated
        assert all(s["id"] != "D" for s in pareto)
        assert len(pareto) <= 3

    def test_select_compromise(self, optimizer):
        """Test compromise solution selection."""
        pareto = [
            {"id": "A", "distance": 10, "time": 30, "emissions": 5},
            {"id": "B", "distance": 8, "time": 35, "emissions": 4},
        ]

        optimizer.set_weights({"distance": 0.5, "time": 0.3, "emissions": 0.2})

        compromise = optimizer.select_compromise(pareto)

        assert compromise is not None
        assert "id" in compromise


class TestRealTimeTracker:
    """Test suite for RealTimeTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker."""
        return RealTimeTracker()

    def test_update_position(self, tracker):
        """Test position updates."""
        result = tracker.update_position(
            "VAN_001", (-118.26, 34.06), "2024-01-15T09:30:00"
        )

        assert result["vehicle_id"] == "VAN_001"
        assert result["position"] == (-118.26, 34.06)

    def test_get_fleet_positions(self, tracker):
        """Test getting all positions."""
        tracker.update_position("VAN_001", (-118.26, 34.06), "2024-01-15T09:30:00")
        tracker.update_position("VAN_002", (-118.27, 34.07), "2024-01-15T09:30:00")

        positions = tracker.get_fleet_positions()

        assert len(positions) == 2
        assert "VAN_001" in positions
        assert "VAN_002" in positions

    def test_is_at_stop(self, tracker):
        """Test stop detection."""
        position = (-118.25, 34.05)
        stop = (-118.25, 34.05)

        assert tracker._is_at_stop(position, stop, threshold_km=0.1)

        far_stop = (-118.30, 34.10)
        assert not tracker._is_at_stop(position, far_stop, threshold_km=0.1)

    def test_calculate_eta(self, tracker):
        """Test ETA calculation."""
        estimator = TravelTimeEstimator()

        tracker.update_position("VAN_001", (-118.25, 34.05), "2024-01-15T09:00:00")

        eta = tracker.calculate_eta("VAN_001", (-118.30, 34.10), estimator)

        assert eta is not None
        assert "2024-01-15" in eta

    def test_calculate_eta_unknown_vehicle(self, tracker):
        """Test ETA for unknown vehicle."""
        estimator = TravelTimeEstimator()

        eta = tracker.calculate_eta("UNKNOWN", (-118.30, 34.10), estimator)

        assert eta is None


class TestNetworkLoading:
    """Tests for network loading and routing over loaded networks."""

    @staticmethod
    def _build_network() -> "nx.Graph":
        """Two-node connected network plus an isolated second component."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(1, x=-118.25, y=34.05)
        graph.add_node(2, x=-118.20, y=34.05)
        graph.add_node(3, x=-118.30, y=34.10)  # disconnected component
        graph.add_edge(1, 2, distance=4.5, time=9.0)
        return graph

    def test_gpickle_round_trip(self, tmp_path):
        """A graph written with save_gpickle loads via RouteOptimizer."""
        graph = self._build_network()
        network_file = tmp_path / "network.gpickle"
        save_gpickle(str(network_file), graph)

        optimizer = RouteOptimizer()
        optimizer.load_network(str(network_file))

        assert set(optimizer.network.nodes) == {1, 2, 3}
        assert optimizer.network[1][2]["distance"] == 4.5

    def test_optimize_route_no_path(self, tmp_path):
        """Disconnected components yield a no-path result, not zero distances."""
        graph = self._build_network()
        network_file = tmp_path / "network.gpickle"
        save_gpickle(str(network_file), graph)

        optimizer = RouteOptimizer()
        optimizer.load_network(str(network_file))

        result = optimizer.optimize_route((-118.30, 34.10), (-118.20, 34.05))
        assert result == {"error": "No path found"}

    def test_optimize_route_without_network(self):
        """Network-free routing works via haversine nearest-neighbor order."""
        origin = (-118.25, 34.05)
        destination = (-118.32, 34.04)
        waypoints = [(-118.28, 34.08), (-118.30, 34.02)]

        optimizer = RouteOptimizer()
        result = optimizer.optimize_route(origin, destination, waypoints=waypoints)

        assert "error" not in result
        assert result["distance"] > 0
        assert result["travel_time"] > 0
        # Ordered stops: origin, both waypoints, destination.
        assert len(result["path"]) == 4

    def test_optimize_route_waypoint_destination_unreachable(self, tmp_path):
        """An unreachable destination yields an error, not a partial route."""
        graph = self._build_network()
        network_file = tmp_path / "network.gpickle"
        save_gpickle(str(network_file), graph)

        optimizer = RouteOptimizer()
        optimizer.load_network(str(network_file))

        # Node 3 is in a disconnected component, so the final leg
        # (waypoint 2 -> destination 3) cannot be completed.
        result = optimizer.optimize_route(
            (-118.25, 34.05), (-118.30, 34.10), waypoints=[(-118.20, 34.05)]
        )

        assert "error" in result
