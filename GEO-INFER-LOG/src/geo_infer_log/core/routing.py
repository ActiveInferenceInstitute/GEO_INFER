"""
Routing optimization components for the GEO-INFER-LOG module.

This module provides classes for optimizing routes, managing fleets,
and estimating travel times with geospatial intelligence.
"""

import pickle
import numpy as np
import geopandas as gpd
import networkx as nx
from typing import Any, cast
import logging
from geo_infer_log.models.schemas import (
    Vehicle,
    RoutingParameters,
    VehicleType as VehicleType,
)
from geo_infer_log.utils.geo import haversine_distance
from scipy.spatial import KDTree
from shapely.geometry import LineString
from geo_infer_log.core.secure_serialization import (
    CONTEXT_NETWORK,
    sign_payload,
    verify_payload,
)

logger = logging.getLogger(__name__)

_DEFAULT_SPEED_KMH = 30.0  # Default urban speed for network-free routing


def save_gpickle(path: str, graph: nx.Graph) -> None:
    """Pickle a graph into an authenticated GISP1 envelope.

    Supported writer counterpart to :func:`_load_gpickle`: the serialized
    bytes are signed under the ``log.routing.network`` context (see
    :mod:`geo_infer_log.core.secure_serialization`).

    Args:
        path: Path to write the envelope to
        graph: The NetworkX graph to persist
    """
    envelope = sign_payload(pickle.dumps(graph), context=CONTEXT_NETWORK)
    with open(path, "wb") as handle:
        handle.write(envelope)


def _load_gpickle(path: str) -> nx.Graph:
    """Load a graph pickled and enveloped by :func:`save_gpickle`.

    ``nx.read_gpickle`` was removed in networkx 3.0; the authenticated
    :func:`save_gpickle`/``_load_gpickle`` pair is the supported
    replacement. The GISP1 envelope is verified before the bytes reach
    ``pickle.loads`` — unsigned, truncated, tampered, or cross-context
    network files are refused instead of deserialized.

    Args:
        path: Path to the enveloped graph file

    Returns:
        The unpickled NetworkX graph

    Raises:
        geo_infer_log.core.secure_serialization.PayloadSecurityError:
            If the file is not a valid GISP1 envelope.
    """
    with open(path, "rb") as handle:
        envelope = handle.read()
    verified = verify_payload(envelope, context=CONTEXT_NETWORK)
    return pickle.loads(verified)


class RouteOptimizer:
    """Base class for route optimization."""

    def __init__(self, parameters: RoutingParameters | None = None):
        """Initialize a route optimizer.

        Args:
            parameters: Routing parameters
        """
        self.parameters = parameters or RoutingParameters()
        self.network: nx.Graph | None = None
        self.vehicles: list[Vehicle] = []
        self._node_tree: KDTree | None = None
        self._node_indices: list[int] | None = None
        self._nodes_list: list[int] | None = None

    def load_network(self, network_file: str) -> None:
        """Load a transportation network from a file.

        Args:
            network_file: Path to network file (GraphML or signed gpickle envelope)
        """
        try:
            if network_file.endswith(".graphml"):
                self.network = nx.read_graphml(network_file)
            else:
                self.network = _load_gpickle(network_file)

            self._build_spatial_index()
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            raise

    def _build_spatial_index(self) -> None:
        """Build KDTree for fast nearest node lookup."""
        if self.network is not None:
            self._nodes_list = list(self.network.nodes)
            coords = []
            valid_nodes = []

            for node in self._nodes_list:
                # Support various coordinate attribute names
                data = self.network.nodes[node]
                lon = data.get("x", data.get("lon", data.get("longitude")))
                lat = data.get("y", data.get("lat", data.get("latitude")))

                if lon is not None and lat is not None:
                    coords.append((lon, lat))
                    valid_nodes.append(node)

            if coords:
                self._node_tree = KDTree(coords)
                self._node_indices = valid_nodes
            else:
                logger.warning(
                    "No coordinates found in network nodes. Nearest neighbor lookup will fail."
                )

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet.

        Args:
            vehicle: Vehicle to add
        """
        self.vehicles.append(vehicle)

    def optimize_route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        waypoints: list[tuple[float, float]] | None = None,
    ) -> dict:
        """Optimize a route between origin and destination.

        With a loaded network, routing runs over the network graph. Without
        a network, stops are ordered greedily (nearest neighbor) by haversine
        distance and travel time assumes the default urban speed.

        Args:
            origin: (lon, lat) of starting point
            destination: (lon, lat) of ending point
            waypoints: Optional list of (lon, lat) points to visit

        Returns:
            Dictionary with optimized route information, or {"error": ...}
            when no route can be completed
        """
        if self.network is None:
            return self._optimize_route_haversine(origin, destination, waypoints)

        # Find nearest nodes in the network
        origin_node = self._find_nearest_node(origin)
        dest_node = self._find_nearest_node(destination)

        waypoint_nodes = []
        if waypoints:
            waypoint_nodes = [self._find_nearest_node(wp) for wp in waypoints]

        # Solve the routing problem. Both branches bind path/distance/
        # travel_time or return an error result; the tuple is deliberately
        # unpacked outside each try/except so the names are unambiguously
        # bound before use below.
        if not waypoint_nodes:
            # Simple shortest path
            try:
                solved = (
                    nx.shortest_path(
                        self.network,
                        origin_node,
                        dest_node,
                        weight=self.parameters.weight_factor,
                    ),
                    nx.shortest_path_length(
                        self.network, origin_node, dest_node, weight="distance"
                    ),
                    nx.shortest_path_length(
                        self.network, origin_node, dest_node, weight="time"
                    ),
                )
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between {origin_node} and {dest_node}")
                return {"error": "No path found"}
            path, distance, travel_time = solved
        else:
            # With waypoints - solve as TSP
            try:
                solved = self._solve_with_waypoints(
                    origin_node, dest_node, waypoint_nodes
                )
            except ValueError as exc:
                logger.warning("Waypoint routing failed: %s", exc)
                return {"error": str(exc)}
            path, distance, travel_time = solved

        # Extract route geometry
        route_geometry = self._extract_route_geometry(path)

        # Compile results
        return {
            "path": path,
            "distance": distance,  # km
            "travel_time": travel_time,  # minutes
            "geometry": route_geometry,
            "origin": origin,
            "destination": destination,
            "waypoints": waypoints or [],
        }

    def _optimize_route_haversine(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        waypoints: list[tuple[float, float]] | None = None,
    ) -> dict:
        """Route over great-circle distances without a loaded network.

        Stops are visited in greedy nearest-neighbor order; distance is the
        sum of haversine legs and travel time assumes
        ``_DEFAULT_SPEED_KMH``.

        Args:
            origin: (lon, lat) of starting point
            destination: (lon, lat) of ending point
            waypoints: Optional list of (lon, lat) points to visit

        Returns:
            Dictionary with optimized route information
        """
        remaining = list(waypoints or [])
        current = origin
        ordered: list[tuple[float, float]] = []
        while remaining:
            nearest = min(remaining, key=lambda p: haversine_distance(current, p))
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest

        stops = [origin] + ordered + [destination]
        distance = sum(haversine_distance(a, b) for a, b in zip(stops, stops[1:]))
        travel_time = distance / _DEFAULT_SPEED_KMH * 60.0

        coords = [(float(p[0]), float(p[1])) for p in stops]
        return {
            "path": coords,
            "distance": distance,  # km
            "travel_time": travel_time,  # minutes
            "geometry": gpd.GeoSeries([LineString(coords)]),
            "origin": origin,
            "destination": destination,
            "waypoints": waypoints or [],
        }

    def _find_nearest_node(self, point: tuple[float, float]) -> int:
        """Find the nearest node in the network to a point.

        Args:
            point: (lon, lat) coordinate

        Returns:
            Node ID in the network
        """
        if self._node_tree is not None:
            dist, idx = self._node_tree.query(point)
            assert self._node_indices is not None
            return cast(int, self._node_indices[idx])

        # Fallback if no index (slow)
        min_dist = float("inf")
        nearest = None

        assert self.network is not None
        for node in self.network.nodes:
            data = self.network.nodes[node]
            lon = data.get("x", data.get("lon", 0))
            lat = data.get("y", data.get("lat", 0))
            dist = (lon - point[0]) ** 2 + (lat - point[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return cast(int, nearest or list(self.network.nodes)[0])

    def _solve_with_waypoints(
        self, origin_node: int, dest_node: int, waypoint_nodes: list[int]
    ) -> tuple[list, float, float]:
        """Solve routing problem with waypoints (TSP-like).

        Args:
            origin_node: Starting node
            dest_node: Ending node
            waypoint_nodes: List of nodes to visit

        Returns:
            Tuple of (path, distance, travel_time)

        Raises:
            ValueError: When the route cannot be completed (e.g. the
                destination is unreachable from the last waypoint)
        """
        try:
            # Construct a subgraph containing only relevant nodes for TSP approximation
            # Simple greedy approach: Nearest Neighbor
            current_node = origin_node
            unvisited = set(waypoint_nodes)
            full_path: list[int] = []
            total_distance = 0.0
            total_time = 0.0

            while unvisited:
                # Find nearest unvisited
                nearest_node = None
                min_dist = float("inf")

                for node in unvisited:
                    try:
                        d = nx.shortest_path_length(
                            self.network, current_node, node, weight="distance"
                        )
                        if d < min_dist:
                            min_dist = d
                            nearest_node = node
                    except nx.NetworkXNoPath:
                        continue

                if nearest_node:
                    # Get path to nearest
                    segment = nx.shortest_path(
                        self.network,
                        current_node,
                        nearest_node,
                        weight=self.parameters.weight_factor,
                    )
                    if full_path:
                        full_path.extend(segment[1:])  # Avoid duplicating node
                    else:
                        full_path.extend(segment)

                    total_distance += min_dist
                    # Add time approx
                    try:
                        t = nx.shortest_path_length(
                            self.network, current_node, nearest_node, weight="time"
                        )
                        total_time += t
                    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                        logger.debug(
                            f"Could not compute time from {current_node} to {nearest_node}"
                        )

                    current_node = nearest_node
                    unvisited.remove(nearest_node)
                else:
                    break  # Cannot reach remaining nodes

            # Finally go to destination
            try:
                final_segment = nx.shortest_path(
                    self.network,
                    current_node,
                    dest_node,
                    weight=self.parameters.weight_factor,
                )
                if full_path:
                    full_path.extend(final_segment[1:])
                else:
                    full_path.extend(final_segment)

                total_distance += nx.shortest_path_length(
                    self.network, current_node, dest_node, weight="distance"
                )
                total_time += nx.shortest_path_length(
                    self.network, current_node, dest_node, weight="time"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError) as exc:
                raise ValueError(
                    f"Route could not be completed: no path from "
                    f"{current_node} to destination {dest_node}"
                ) from exc

            return full_path, total_distance, total_time

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"TSP solving failed: {e}")
            raise ValueError(f"Waypoint routing failed: {e}") from e

    def _extract_route_geometry(self, path: list[int]) -> gpd.GeoSeries:
        """Extract the geometry of a route from the path.

        Args:
            path: List of node IDs

        Returns:
            GeoSeries with route geometry
        """
        coords = []
        assert self.network is not None
        for node in path:
            data = self.network.nodes[node]
            x = data.get("x", data.get("lon"))
            y = data.get("y", data.get("lat"))
            if x is not None and y is not None:
                coords.append((x, y))

        if len(coords) >= 2:
            return gpd.GeoSeries([LineString(coords)])
        return gpd.GeoSeries([])


class FleetManager:
    """Manages a fleet of vehicles and their assignments."""

    def __init__(self) -> None:
        """Initialize a fleet manager."""
        self.vehicles: dict[str, Vehicle] = {}  # id -> Vehicle
        self.assignments: dict[str, dict[str, Any]] = {}  # vehicle_id -> assignment
        self.route_optimizer = RouteOptimizer()

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet.

        Args:
            vehicle: Vehicle to add
        """
        self.vehicles[vehicle.id] = vehicle
        self.route_optimizer.add_vehicle(vehicle)

    def assign_delivery(
        self,
        vehicle_id: str,
        delivery_points: list[tuple[float, float]],
        depot: tuple[float, float],
    ) -> dict:
        """Assign a delivery route to a vehicle.

        Args:
            vehicle_id: ID of vehicle to assign
            delivery_points: List of (lon, lat) points for deliveries
            depot: (lon, lat) of the depot location

        Returns:
            Assignment information
        """
        if vehicle_id not in self.vehicles:
            raise ValueError(f"Vehicle {vehicle_id} not found")

        # Optimize route from depot through all delivery points and back
        route = self.route_optimizer.optimize_route(
            origin=depot, destination=depot, waypoints=delivery_points
        )

        # Create assignment
        assignment = {
            "vehicle_id": vehicle_id,
            "route": route,
            "depot": depot,
            "delivery_points": delivery_points,
            "start_time": None,  # To be set when executed
            "estimated_completion_time": None,  # To be set when executed
        }

        self.assignments[vehicle_id] = assignment
        return assignment

    def get_fleet_status(self) -> dict:
        """Get the current status of the fleet.

        Returns:
            Dictionary with fleet status information
        """
        assigned = [vid for vid in self.assignments]
        available = [vid for vid in self.vehicles if vid not in assigned]

        return {
            "total_vehicles": len(self.vehicles),
            "assigned_vehicles": len(assigned),
            "available_vehicles": len(available),
            "vehicles": self.vehicles,
            "assignments": self.assignments,
        }


class VehicleRouter:
    """Plans and executes complex vehicle routing problems."""

    def __init__(self, fleet_manager: FleetManager):
        """Initialize a vehicle router.

        Args:
            fleet_manager: Fleet manager instance
        """
        self.fleet_manager = fleet_manager

    def solve_vrp(
        self,
        deliveries: list[dict],
        depots: list[tuple[float, float]],
        constraints: dict,
    ) -> dict:
        """Solve a vehicle routing problem.

        Args:
            deliveries: List of delivery information {'id', 'location': (lon, lat), 'demand'}
            depots: List of depot locations
            constraints: Dictionary of constraints

        Returns:
            Solution to the VRP
        """
        # Greedy cluster-first implementation

        solution: dict[str, Any] = {"routes": {}, "unassigned": []}

        # Simple heuristic: Assign nearest deliveries to vehicles until capacity/range constraint

        available_vehicles = list(self.fleet_manager.vehicles.values())
        unassigned_deliveries = deliveries.copy()

        for vehicle in available_vehicles:
            if not unassigned_deliveries:
                break

            # Find a depot (use first for simplicity or nearest)
            depot = depots[0]

            route_points = []
            current_load = 0
            current_pos = depot

            # While capacity allows, add nearest delivery
            while unassigned_deliveries:
                nearest_idx = -1
                min_dist = float("inf")

                for i, d in enumerate(unassigned_deliveries):
                    loc = d["location"]
                    dist = (loc[0] - current_pos[0]) ** 2 + (
                        loc[1] - current_pos[1]
                    ) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = i

                if nearest_idx >= 0:
                    d = unassigned_deliveries[nearest_idx]
                    if current_load + d.get("demand", 0) <= vehicle.capacity:
                        route_points.append(d["location"])
                        current_load += d.get("demand", 0)
                        current_pos = d["location"]
                        unassigned_deliveries.pop(nearest_idx)
                    else:
                        break  # Vehicle full
                else:
                    break

            if route_points:
                # Optimize the route sequence
                assignment = self.fleet_manager.assign_delivery(
                    vehicle.id, route_points, depot
                )
                solution["routes"][vehicle.id] = assignment

        solution["unassigned"] = unassigned_deliveries
        return solution


class TravelTimeEstimator:
    """Estimates travel times between points considering traffic and conditions."""

    def __init__(self, use_historical_data: bool = True):
        """Initialize a travel time estimator.

        Args:
            use_historical_data: Whether to use historical traffic data
        """
        self.use_historical_data = use_historical_data
        self.historical_data: Any | None = None

    def load_historical_data(self, data_file: str) -> None:
        """Load historical traffic data.

        Args:
            data_file: Path to data file (CSV)
        """
        import pandas as pd

        try:
            # Expecting CSV with columns: origin_id, dest_id, hour, travel_time
            self.historical_data = pd.read_csv(data_file)
            logger.info(f"Loaded {len(self.historical_data)} historical records")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise

    def estimate_travel_time(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        departure_time: str | None = None,
    ) -> float:
        """Estimate travel time between points.

        Args:
            origin: (lon, lat) of starting point
            destination: (lon, lat) of ending point
            departure_time: Optional departure time (ISO format)

        Returns:
            Estimated travel time in minutes
        """
        distance = haversine_distance(origin, destination)  # km

        # Estimate time based on average speed (30 km/h urban, adjusted for traffic)
        base_speed = _DEFAULT_SPEED_KMH  # km/h urban

        # Adjust for time of day if departure_time provided
        traffic_factor = 1.0
        if departure_time and self.use_historical_data:
            traffic_factor = self._get_traffic_factor(departure_time)

        adjusted_speed = base_speed / traffic_factor
        travel_time = (distance / adjusted_speed) * 60  # minutes

        return travel_time

    def _get_traffic_factor(self, departure_time: str) -> float:
        """Get traffic adjustment factor based on time of day.

        Args:
            departure_time: ISO format datetime

        Returns:
            Traffic factor (1.0 = normal, >1.0 = slower)
        """
        from datetime import datetime

        try:
            dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
            hour = dt.hour

            # Peak hours have higher factors (slower traffic)
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                return 1.5  # Rush hour
            elif 10 <= hour <= 15:
                return 1.1  # Daytime
            elif 19 <= hour <= 22:
                return 1.2  # Evening
            else:
                return 0.9  # Night (faster)
        except Exception:
            return 1.0

    def calculate_time_matrix(
        self, locations: list[tuple[float, float]], departure_time: str | None = None
    ) -> np.ndarray:
        """Calculate a travel time matrix between all locations.

        Args:
            locations: List of (lon, lat) coordinates
            departure_time: Optional departure time

        Returns:
            NxN numpy array of travel times in minutes
        """
        n = len(locations)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self.estimate_travel_time(
                        locations[i], locations[j], departure_time
                    )

        return matrix

    def calculate_distance_matrix(
        self, locations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Calculate a distance matrix between all locations.

        Args:
            locations: List of (lon, lat) coordinates

        Returns:
            NxN numpy array of distances in km
        """
        n = len(locations)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = haversine_distance(locations[i], locations[j])

        return matrix

    def estimate_arrival_times(
        self,
        route: list[tuple[float, float]],
        departure_time: str,
        service_times: list[float] | None = None,
    ) -> list[str]:
        """Estimate arrival times at each stop along a route.

        Args:
            route: List of (lon, lat) coordinates
            departure_time: ISO format departure time
            service_times: Optional list of service times at each stop (minutes)

        Returns:
            List of estimated arrival times in ISO format
        """
        from datetime import datetime, timedelta

        current_time = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        arrivals = [departure_time]

        if service_times is None:
            service_times = [0.0] * len(route)

        for i in range(1, len(route)):
            travel = self.estimate_travel_time(route[i - 1], route[i])
            service = service_times[i - 1] if i - 1 < len(service_times) else 0

            current_time += timedelta(minutes=travel + service)
            arrivals.append(current_time.isoformat())

        return arrivals


class MultiObjectiveOptimizer:
    """Multi-objective optimization for logistics routing."""

    def __init__(self, objectives: list[str]):
        """Initialize multi-objective optimizer.

        Args:
            objectives: List of objective names (e.g., ['time', 'distance', 'emissions'])
        """
        self.objectives = objectives
        self.weights = {obj: 1.0 / len(objectives) for obj in objectives}

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set objective weights.

        Args:
            weights: Dictionary of objective -> weight
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def calculate_pareto_front(self, solutions: list[dict]) -> list[dict]:
        """Calculate the Pareto front from a set of solutions.

        Args:
            solutions: List of solution dictionaries with objective values

        Returns:
            List of non-dominated solutions
        """
        pareto_front = []

        for solution in solutions:
            is_dominated = False

            for other in solutions:
                if solution == other:
                    continue

                # Check if 'other' dominates 'solution'
                dominates = all(
                    other.get(obj, float("inf")) <= solution.get(obj, float("inf"))
                    for obj in self.objectives
                ) and any(
                    other.get(obj, float("inf")) < solution.get(obj, float("inf"))
                    for obj in self.objectives
                )

                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(solution)

        return pareto_front

    def select_compromise(self, pareto_front: list[dict]) -> dict:
        """Select a compromise solution from the Pareto front.

        Args:
            pareto_front: List of Pareto-optimal solutions

        Returns:
            Best compromise solution based on weights
        """
        if not pareto_front:
            return {}

        # Normalize objectives
        min_vals = {
            obj: min(s.get(obj, 0) for s in pareto_front) for obj in self.objectives
        }
        max_vals = {
            obj: max(s.get(obj, 0) for s in pareto_front) for obj in self.objectives
        }

        best_solution: dict[str, Any] | None = None
        best_score = float("inf")

        for solution in pareto_front:
            weighted_sum = 0
            for obj in self.objectives:
                val = solution.get(obj, 0)
                range_val = max_vals[obj] - min_vals[obj]
                if range_val > 0:
                    normalized = (val - min_vals[obj]) / range_val
                else:
                    normalized = 0
                weighted_sum += self.weights[obj] * normalized

            if weighted_sum < best_score:
                best_score = weighted_sum
                best_solution = solution

        assert best_solution is not None
        return best_solution


class RealTimeTracker:
    """Real-time tracking and dynamic re-routing."""

    def __init__(self) -> None:
        """Initialize real-time tracker."""
        # vehicle_id -> (lon, lat, timestamp)
        self.vehicle_positions: dict[str, tuple[float, float, str]] = {}
        self.active_routes: dict[str, dict[str, Any]] = {}  # vehicle_id -> route info
        self.events: list[Any] = []  # List of events (delays, completions, etc.)

    def update_position(
        self, vehicle_id: str, position: tuple[float, float], timestamp: str
    ) -> dict:
        """Update vehicle position.

        Args:
            vehicle_id: ID of the vehicle
            position: (lon, lat) current position
            timestamp: ISO format timestamp

        Returns:
            Update status and any triggered events
        """
        self.vehicle_positions[vehicle_id] = (position[0], position[1], timestamp)

        result: dict[str, Any] = {
            "vehicle_id": vehicle_id,
            "position": position,
            "timestamp": timestamp,
            "events": [],
        }

        # Check if vehicle is on route
        if vehicle_id in self.active_routes:
            route_info = self.active_routes[vehicle_id]

            # Check for arrival at next stop
            if self._is_at_stop(position, route_info.get("next_stop")):
                result["events"].append(
                    {
                        "type": "arrival",
                        "stop": route_info.get("next_stop"),
                        "timestamp": timestamp,
                    }
                )

        return result

    def _is_at_stop(
        self,
        position: tuple[float, float],
        stop: tuple[float, float] | None,
        threshold_km: float = 0.1,
    ) -> bool:
        """Check if position is at a stop.

        Args:
            position: Current position
            stop: Stop position
            threshold_km: Distance threshold in km

        Returns:
            True if at stop
        """
        if stop is None:
            return False

        return haversine_distance(position, stop) <= threshold_km

    def get_fleet_positions(self) -> dict:
        """Get current positions of all tracked vehicles.

        Returns:
            Dictionary of vehicle positions
        """
        return {
            vid: {"lon": pos[0], "lat": pos[1], "timestamp": pos[2]}
            for vid, pos in self.vehicle_positions.items()
        }

    def calculate_eta(
        self,
        vehicle_id: str,
        destination: tuple[float, float],
        estimator: TravelTimeEstimator,
    ) -> str | None:
        """Calculate ETA for a vehicle to reach destination.

        Args:
            vehicle_id: ID of the vehicle
            destination: (lon, lat) destination
            estimator: Travel time estimator

        Returns:
            Estimated arrival time in ISO format, or None if vehicle not tracked
        """
        from datetime import datetime, timedelta

        if vehicle_id not in self.vehicle_positions:
            return None

        pos = self.vehicle_positions[vehicle_id]
        current_pos = (pos[0], pos[1])
        current_time = datetime.fromisoformat(pos[2].replace("Z", "+00:00"))

        travel_time = estimator.estimate_travel_time(current_pos, destination)
        eta = current_time + timedelta(minutes=travel_time)

        return eta.isoformat()
