"""
Routing engine module.

Provides multi-modal routing, optimization, and path finding
for transportation networks.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingAlgorithm(Enum):
    """Routing algorithms."""

    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    BELLMAN_FORD = "bellman_ford"
    BIDIRECTIONAL = "bidirectional"


class OptimizationCriteria(Enum):
    """Route optimization criteria."""

    TIME = "time"
    DISTANCE = "distance"
    COST = "cost"
    EMISSIONS = "emissions"


@dataclass
class Route:
    """Represents a computed route."""

    route_id: str
    origin: str
    destination: str
    path: list[str]
    total_distance_m: float
    total_time_s: float
    geometry: list[dict[str, float]] | None = None
    instructions: list[str] = field(default_factory=list)
    alternatives: list["Route"] = field(default_factory=list)
    route_source: str = "network"


class RoutingEngine:
    """
    Multi-modal routing engine with optimization capabilities.

    Supports various algorithms, real-time traffic integration,
    and multi-criteria optimization.
    """

    def __init__(
        self,
        network: Any = None,
        algorithm: str = "dijkstra",
        modes: list[str] | None = None,
        real_time_traffic: bool = False,
    ):
        """
        Initialize routing engine.

        Args:
            network: Transport network to route on
            algorithm: Routing algorithm to use
            modes: Supported transport modes
            real_time_traffic: Enable real-time traffic
        """
        self.network = network
        self.algorithm = RoutingAlgorithm(algorithm)
        self.modes = modes or ["car"]
        self.real_time_traffic = real_time_traffic
        self._traffic_data: dict[str, float] = {}

        logger.info(f"Initialized RoutingEngine with {algorithm} algorithm")

    def set_network(self, network: Any) -> None:
        """Set the transport network."""
        self.network = network

    def route(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        optimization: str = "time",
    ) -> Route:
        """
        Calculate a route between origin and destination.

        Args:
            origin: Origin point or node
            destination: Destination point or node
            optimization: Optimization criteria

        Returns:
            Computed Route object. ``route.route_source`` records how the
            route was produced: ``"network"`` when computed on the graph,
            ``"estimated_fallback"`` when no network is available and the
            route is a haversine estimate. Callers should check
            ``route_source`` before treating metrics as exact. When no
            path exists on the network, or the origin/destination node is
            not present in the graph, the returned Route has an empty
            ``path`` and zero distance/time.
        """
        origin_id = origin.get("node_id") or origin.get("id") or "origin"
        dest_id = destination.get("node_id") or destination.get("id") or "destination"

        # Get path from network
        path: list[str] = []
        total_distance = 0.0
        total_time = 0.0
        route_source = "network"

        if self.network and hasattr(self.network, "graph"):
            import networkx as nx

            graph = self.network.graph

            # Choose weight based on optimization
            weight = "travel_time" if optimization == "time" else "length"

            try:
                # Apply traffic adjustment if enabled
                if self.real_time_traffic:
                    self._apply_traffic_weights(graph)

                # Prefer traffic-adjusted weights when they exist
                if optimization == "time" and any(
                    "travel_time_adjusted" in data
                    for _, _, data in graph.edges(data=True)
                ):
                    weight = "travel_time_adjusted"

                # Calculate path
                if self.algorithm == RoutingAlgorithm.A_STAR:
                    path = list(nx.astar_path(graph, origin_id, dest_id, weight=weight))
                elif self.algorithm == RoutingAlgorithm.BELLMAN_FORD:
                    path = list(
                        nx.bellman_ford_path(graph, origin_id, dest_id, weight=weight)
                    )
                else:  # Default to Dijkstra
                    path = list(
                        nx.dijkstra_path(graph, origin_id, dest_id, weight=weight)
                    )

                # Calculate totals (time total honours the weight actually used)
                time_key = weight if optimization == "time" else "travel_time"
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i + 1])
                    if edge_data:
                        total_distance += float(edge_data.get("length", 0))
                        total_time += float(
                            edge_data.get(time_key, edge_data.get("travel_time", 0))
                        )

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                logger.warning(f"No path found from {origin_id} to {dest_id}")
                path = []
        else:
            # Fallback for when no network is set
            route_source = "estimated_fallback"
            path = [origin_id, dest_id]
            total_distance = float(self._estimate_distance(origin, destination))
            total_time = total_distance / 13.9  # ~50 km/h in m/s

        # Generate turn-by-turn instructions
        instructions = self._generate_instructions(path)

        route = Route(
            route_id=f"route_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            origin=origin_id,
            destination=dest_id,
            path=path,
            total_distance_m=total_distance,
            total_time_s=total_time,
            instructions=instructions,
            route_source=route_source,
        )

        logger.info(
            f"Calculated route: {len(path)} nodes, {total_distance / 1000:.1f}km, {total_time / 60:.1f}min"
        )
        return route

    def _estimate_distance(
        self, origin: dict[str, Any], destination: dict[str, Any]
    ) -> float:
        """Estimate distance using Haversine formula."""
        import math

        lat1 = origin.get("lat", 0) * math.pi / 180
        lat2 = destination.get("lat", 0) * math.pi / 180
        lon1 = origin.get("lon", 0) * math.pi / 180
        lon2 = destination.get("lon", 0) * math.pi / 180

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return float(6371000 * c)  # Earth radius in meters

    def _apply_traffic_weights(self, graph: Any) -> None:
        """Apply traffic data to edge weights."""
        for u, v, data in graph.edges(data=True):
            edge_id = data.get("edge_id", f"{u}_{v}")
            traffic_factor = self._traffic_data.get(edge_id, 1.0)
            data["travel_time_adjusted"] = data.get("travel_time", 0) * traffic_factor

    def _generate_instructions(self, path: list[str]) -> list[str]:
        """Generate turn-by-turn instructions."""
        if len(path) < 2:
            return ["Route not found"]

        instructions = [f"Start at {path[0]}"]

        for i in range(1, len(path) - 1):
            instructions.append(f"Continue through {path[i]}")

        instructions.append(f"Arrive at destination: {path[-1]}")

        return instructions

    def optimize_route(
        self,
        waypoints: list[dict[str, Any]],
        constraints: dict[str, Any],
        objective: str = "minimize_time",
    ) -> dict[str, Any]:
        """
        Optimize route through multiple waypoints.

        Args:
            waypoints: List of waypoints to visit
            constraints: Routing constraints
            objective: Optimization objective

        Returns:
            Optimized route with waypoint order
        """
        if len(waypoints) < 2:
            return {"error": "Need at least 2 waypoints"}

        # Simple TSP-like optimization using nearest neighbor
        origin = waypoints[0]
        remaining = waypoints[1:]
        ordered = [origin]
        total_distance = 0.0

        current = origin
        while remaining:
            # Find nearest unvisited
            nearest = None
            nearest_dist = float("inf")

            for wp in remaining:
                dist = self._estimate_distance(current, wp)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = wp

            if nearest:
                ordered.append(nearest)
                remaining.remove(nearest)
                total_distance += nearest_dist
                current = nearest

        result = {
            "optimized_order": [wp.get("id", i) for i, wp in enumerate(ordered)],
            "waypoints": ordered,
            "estimated_distance_m": total_distance,
            "estimated_time_s": total_distance / 13.9,  # ~50 km/h
            "optimization_objective": objective,
            "constraints_applied": constraints,
        }

        logger.info(f"Optimized route through {len(waypoints)} waypoints")
        return result

    def calculate_matrix(
        self,
        origins: list[dict[str, Any]],
        destinations: list[dict[str, Any]],
        metric: str = "time",
    ) -> dict[str, Any]:
        """
        Calculate origin-destination matrix.

        Args:
            origins: Origin points
            destinations: Destination points
            metric: Metric to calculate ('time', 'distance')

        Returns:
            OD matrix with costs
        """
        matrix = []

        for i, origin in enumerate(origins):
            row = []
            for j, dest in enumerate(destinations):
                route = self.route(origin, dest, optimization=metric)

                if metric == "time":
                    value = route.total_time_s
                else:
                    value = route.total_distance_m

                row.append(round(value, 2))
            matrix.append(row)

        result = {
            "origins": [o.get("id", i) for i, o in enumerate(origins)],
            "destinations": [d.get("id", i) for i, d in enumerate(destinations)],
            "metric": metric,
            "matrix": matrix,
            "shape": [len(origins), len(destinations)],
        }

        logger.info(f"Calculated {len(origins)}x{len(destinations)} OD matrix")
        return result

    def find_alternatives(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        count: int = 3,
        variation: float = 0.2,
    ) -> list[Route]:
        """
        Find alternative routes.

        Args:
            origin: Origin point
            destination: Destination point
            count: Number of alternatives
            variation: Maximum deviation factor

        Returns:
            List of alternative routes
        """
        alternatives = []

        # Calculate primary route
        primary = self.route(origin, destination)
        alternatives.append(primary)
        seen_paths = {tuple(primary.path)}

        if self.network and hasattr(self.network, "graph") and len(primary.path) > 2:
            import networkx as nx

            graph = self.network.graph.copy()

            # Find alternatives by penalizing primary route edges
            for _ in range(count - 1):
                # Increase weights on current route
                for i in range(len(primary.path) - 1):
                    u, v = primary.path[i], primary.path[i + 1]
                    if graph.has_edge(u, v):
                        data = graph.get_edge_data(u, v)
                        if data:
                            data["travel_time"] = data.get("travel_time", 0) * (
                                1 + variation
                            )

                try:
                    alt_path = nx.dijkstra_path(
                        graph, primary.origin, primary.destination, weight="travel_time"
                    )

                    # Calculate alternative route metrics
                    total_distance = 0
                    total_time = 0
                    orig_graph = self.network.graph

                    for i in range(len(alt_path) - 1):
                        edge_data = orig_graph.get_edge_data(
                            alt_path[i], alt_path[i + 1]
                        )
                        if edge_data:
                            total_distance += edge_data.get("length", 0)
                            total_time += edge_data.get("travel_time", 0)

                    alt_route = Route(
                        route_id=f"alt_{len(alternatives)}",
                        origin=primary.origin,
                        destination=primary.destination,
                        path=alt_path,
                        total_distance_m=total_distance,
                        total_time_s=total_time,
                    )

                    if tuple(alt_path) not in seen_paths:
                        seen_paths.add(tuple(alt_path))
                        alternatives.append(alt_route)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break

        logger.info(f"Found {len(alternatives)} routes including primary")
        return alternatives

    def update_traffic(self, traffic_data: dict[str, float]) -> None:
        """
        Update traffic data for real-time routing.

        Args:
            traffic_data: Edge ID to traffic factor mapping
        """
        self._traffic_data.update(traffic_data)
        logger.debug(f"Updated traffic data for {len(traffic_data)} edges")
