"""
Advanced geospatial routing algorithms for GEO-INFER-COMMS.

This module implements sophisticated geospatial routing algorithms including
spatial indexing, proximity-based routing, load balancing, and intelligent
message distribution based on geospatial context and network topology.
"""

from __future__ import annotations
import heapq
import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, UTC

from geo_infer_comms.models.spatial import GeospatialPoint, SpatialFilter, SpatialIndex
from geo_infer_comms.models.message import MessageResponse, MessagePriority


class AdvancedSpatialRouter:
    """
    Advanced geospatial message router with intelligent routing algorithms.

    Implements sophisticated routing strategies including spatial clustering,
    network topology awareness, load balancing, and adaptive routing based
    on real-time geospatial conditions.
    """

    def __init__(
        self,
        spatial_index: SpatialIndex,
        routing_strategy: str = "proximity",
        adaptive_routing: bool = True,
        load_balancing: bool = True,
    ):
        self.spatial_index = spatial_index
        self.routing_strategy = routing_strategy
        self.adaptive_routing = adaptive_routing
        self.load_balancing = load_balancing

        # Routing state
        self.network_topology: dict[
            str, dict[str, float]
        ] = {}  # node -> node -> distance
        self.node_loads: dict[str, float] = {}
        self.node_locations: dict[str, GeospatialPoint] = {}
        self.routing_history: list[dict[str, Any]] = []

        # Performance metrics
        self.routing_metrics = SpatialRoutingMetrics()

        self.logger = logging.getLogger(__name__)

    def register_node_location(self, node_id: str, location: GeospatialPoint) -> None:
        """Register a node's geospatial location for proximity routing."""
        self.node_locations[node_id] = location
        self.logger.info(
            f"Node location registered: {node_id} at "
            f"{location.latitude}, {location.longitude}"
        )

    def route_message(
        self,
        message: MessageResponse,
        target_nodes: list[str],
        routing_context: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """
        Route message using advanced geospatial algorithms.

        Args:
            message: Message to route
            target_nodes: List of target node IDs
            routing_context: Additional routing context

        Returns:
            Dictionary mapping node IDs to optimal routes
        """
        if not message.geospatial_data:
            # Fall back to simple routing for non-geospatial messages
            return self._simple_route_message(target_nodes)

        # Get message location
        message_location = message.geospatial_data.location

        # Find optimal routes based on strategy
        if self.routing_strategy == "proximity":
            return self._proximity_based_routing(message_location, target_nodes)
        elif self.routing_strategy == "network_aware":
            return self._network_aware_routing(message_location, target_nodes)
        elif self.routing_strategy == "load_balanced":
            return self._load_balanced_routing(message_location, target_nodes)
        elif self.routing_strategy == "adaptive":
            return self._adaptive_routing(
                message_location, target_nodes, routing_context
            )
        else:
            # Default to proximity routing
            return self._proximity_based_routing(message_location, target_nodes)

    def _proximity_based_routing(
        self, message_location: GeospatialPoint, target_nodes: list[str]
    ) -> dict[str, list[str]]:
        """Route messages based on proximity to message location."""
        routes = {}

        for node_id in target_nodes:
            # Find shortest path to node (simplified - would use proper routing algorithm)
            route = self._calculate_proximity_route(message_location, node_id)
            routes[node_id] = route

        return routes

    def _simple_route_message(self, target_nodes: list[str]) -> dict[str, list[str]]:
        """Simple routing for non-geospatial messages."""
        return {node: [node] for node in target_nodes}

    def _network_aware_routing(
        self, message_location: GeospatialPoint, target_nodes: list[str]
    ) -> dict[str, list[str]]:
        """Route messages considering network topology and latency."""
        routes = {}

        for node_id in target_nodes:
            # Consider network topology in routing
            route = self._calculate_network_route(message_location, node_id)
            routes[node_id] = route

        return routes

    def _load_balanced_routing(
        self, message_location: GeospatialPoint, target_nodes: list[str]
    ) -> dict[str, list[str]]:
        """Route messages with load balancing consideration."""
        routes = {}

        # Sort nodes by current load (lower load = higher priority)
        sorted_nodes = sorted(target_nodes, key=lambda n: self.node_loads.get(n, 0))

        for node_id in sorted_nodes:
            route = self._calculate_proximity_route(message_location, node_id)
            routes[node_id] = route

        return routes

    def _adaptive_routing(
        self,
        message_location: GeospatialPoint,
        target_nodes: list[str],
        routing_context: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """Adaptive routing based on context and conditions."""
        routes = {}

        # Analyze routing context
        context = routing_context or {}
        urgency = context.get("urgency", "normal")
        network_conditions = context.get("network_conditions", {})

        for node_id in target_nodes:
            # Adapt routing based on conditions
            if urgency == "high" and network_conditions.get("congestion"):
                # Use alternative routes for high urgency
                route = self._calculate_alternative_route(message_location, node_id)
            else:
                route = self._calculate_proximity_route(message_location, node_id)

            routes[node_id] = route

        return routes

    def _calculate_proximity_route(
        self, message_location: GeospatialPoint, target_node: str
    ) -> list[str]:
        """Calculate route based on proximity using Dijkstra over network topology."""
        if not self.network_topology:
            return [target_node]

        # Identify the source node closest to message_location using
        # registered node locations; fall back to lexicographic selection
        # when no locations are registered.
        source_node: str | None = None
        best_dist = float("inf")
        for node_id in self.network_topology:
            loc = self.node_locations.get(node_id)
            if loc is not None:
                dist = message_location.distance_to(loc)
                if dist < best_dist:
                    best_dist = dist
                    source_node = node_id
            elif source_node is None or node_id < source_node:
                source_node = node_id
        if source_node is None or source_node == target_node:
            return [target_node]

        # Dijkstra shortest path
        dist: dict[str, float] = {n: float("inf") for n in self.network_topology}
        prev: dict[str, str | None] = {n: None for n in self.network_topology}
        dist[source_node] = 0.0
        visited: set[str] = set()
        heap: list[tuple[float, str]] = [(0.0, source_node)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == target_node:
                break
            for v, weight in self.network_topology.get(u, {}).items():
                alt = d + weight
                if alt < dist.get(v, float("inf")):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        # Reconstruct path
        path: list[str] = []
        node: str | None = target_node
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()

        return path if path and path[0] == source_node else [target_node]

    def _calculate_network_route(
        self, message_location: GeospatialPoint, target_node: str
    ) -> list[str]:
        """Calculate route considering network topology via Dijkstra."""
        # Delegate to the topology-aware proximity router
        return self._calculate_proximity_route(message_location, target_node)

    def _calculate_alternative_route(
        self, message_location: GeospatialPoint, target_node: str
    ) -> list[str]:
        """Calculate alternative route that avoids congested (high-load) nodes."""
        primary = self._calculate_proximity_route(message_location, target_node)

        if not self.network_topology or len(primary) <= 2:
            return primary  # no room for an alternative

        # Build a reduced topology excluding the highest-load interior nodes
        interior = set(primary[1:-1])
        if not interior:
            return primary

        # Find the single highest-load interior node to avoid
        worst_node = max(interior, key=lambda n: self.node_loads.get(n, 0))
        reduced_topo: dict[str, dict[str, float]] = {}
        for u, neighbors in self.network_topology.items():
            if u == worst_node:
                continue
            reduced_topo[u] = {v: w for v, w in neighbors.items() if v != worst_node}

        # Save & swap topology, run Dijkstra on reduced graph
        original_topo = self.network_topology
        self.network_topology = reduced_topo
        alt = self._calculate_proximity_route(message_location, target_node)
        self.network_topology = original_topo

        return alt if alt and len(alt) > 1 else primary

    def update_network_topology(self, topology: dict[str, dict[str, float]]) -> None:
        """Update network topology information."""
        self.network_topology = topology
        self.logger.info("Network topology updated")

    def update_node_loads(self, node_loads: dict[str, float]) -> None:
        """Update current node load information."""
        self.node_loads = node_loads
        self.logger.debug("Node loads updated")

    def record_routing_result(
        self,
        message_id: str,
        route: list[str],
        success: bool,
        latency: float | None = None,
    ) -> None:
        """Record routing result for performance analysis."""
        routing_record = {
            "message_id": message_id,
            "route": route,
            "success": success,
            "latency": latency,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.routing_history.append(routing_record)

        # Keep only recent history
        if len(self.routing_history) > 10000:
            self.routing_history = self.routing_history[-10000:]

        # Update metrics
        if success:
            self.routing_metrics.successful_routes += 1
        else:
            self.routing_metrics.failed_routes += 1

        if latency:
            self.routing_metrics.total_latency += latency
            self.routing_metrics.route_count += 1

    def get_routing_analytics(self) -> dict[str, Any]:
        """Get routing performance analytics."""
        return {
            "routing_strategy": self.routing_strategy,
            "total_routes": len(self.routing_history),
            "success_rate": (
                self.routing_metrics.successful_routes
                / max(
                    self.routing_metrics.successful_routes
                    + self.routing_metrics.failed_routes,
                    1,
                )
                * 100
            ),
            "average_latency": (
                self.routing_metrics.total_latency
                / max(self.routing_metrics.route_count, 1)
            ),
            "metrics": self.routing_metrics.to_dict(),
        }


@dataclass
class SpatialRoutingMetrics:
    """Metrics for spatial routing performance."""

    successful_routes: int = 0
    failed_routes: int = 0
    total_latency: float = 0.0
    route_count: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "successful_routes": self.successful_routes,
            "failed_routes": self.failed_routes,
            "success_rate": (
                self.successful_routes
                / max(self.successful_routes + self.failed_routes, 1)
                * 100
            ),
            "total_latency": self.total_latency,
            "average_latency": (self.total_latency / max(self.route_count, 1)),
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.successful_routes = 0
        self.failed_routes = 0
        self.total_latency = 0.0
        self.route_count = 0
        self.start_time = datetime.now(UTC)


class GeospatialLoadBalancer:
    """
    Geospatial load balancer for message distribution.

    Distributes messages across nodes based on geospatial proximity,
    current load, and network conditions for optimal performance.
    """

    def __init__(
        self,
        nodes: list[str],
        load_threshold: float = 0.8,
        rebalance_interval: int = 300,  # seconds
    ):
        self.nodes = nodes
        self.load_threshold = load_threshold
        self.rebalance_interval = rebalance_interval

        # Load tracking
        self.node_loads: dict[str, float] = {node: 0.0 for node in nodes}
        self.load_history: dict[str, list[float]] = {node: [] for node in nodes}

        # Geospatial mapping
        self.node_locations: dict[str, GeospatialPoint] = {}

        self.logger = logging.getLogger(__name__)

    def register_node_location(self, node_id: str, location: GeospatialPoint) -> None:
        """Register a node's geospatial location."""
        self.node_locations[node_id] = location
        self.logger.info(
            f"Node location registered: {node_id} at {location.latitude}, {location.longitude}"
        )

    def update_node_load(self, node_id: str, load: float) -> None:
        """Update load information for a node."""
        if node_id not in self.nodes:
            return

        self.node_loads[node_id] = load

        # Keep load history
        if node_id not in self.load_history:
            self.load_history[node_id] = []

        self.load_history[node_id].append(load)

        # Keep only recent history
        if len(self.load_history[node_id]) > 100:
            self.load_history[node_id] = self.load_history[node_id][-100:]

    def select_optimal_node(
        self,
        message_location: GeospatialPoint,
        exclude_nodes: list[str] | None = None,
    ) -> str | None:
        """Select the optimal node for a message based on location and load."""
        exclude_nodes = exclude_nodes or []

        # Filter out excluded nodes and overloaded nodes
        available_nodes = [
            node
            for node in self.nodes
            if node not in exclude_nodes
            and self.node_loads.get(node, 0) < self.load_threshold
        ]

        if not available_nodes:
            return None

        # Score nodes based on proximity and load
        node_scores = []
        for node_id in available_nodes:
            node_location = self.node_locations.get(node_id)
            if not node_location:
                continue

            # Calculate proximity score (inverse of distance)
            distance = message_location.distance_to(node_location)
            proximity_score = 1.0 / (1.0 + distance)  # Closer = higher score

            # Calculate load score (inverse of load)
            load = self.node_loads.get(node_id, 0)
            load_score = 1.0 - load  # Lower load = higher score

            # Combined score (weighted average)
            combined_score = (proximity_score * 0.7) + (load_score * 0.3)

            node_scores.append((node_id, combined_score))

        if not node_scores:
            return None

        # Select node with highest score
        best_node = max(node_scores, key=lambda x: x[1])[0]

        self.logger.debug(
            f"Selected node {best_node} for message at {message_location.latitude}, {message_location.longitude}"
        )
        return best_node

    def get_load_distribution(self) -> dict[str, Any]:
        """Get current load distribution across nodes."""
        return {
            "node_loads": self.node_loads,
            "average_load": sum(self.node_loads.values()) / len(self.node_loads),
            "max_load": max(self.node_loads.values()),
            "min_load": min(self.node_loads.values()),
            "overloaded_nodes": [
                node
                for node, load in self.node_loads.items()
                if load >= self.load_threshold
            ],
        }


class SpatialClusteringRouter:
    """
    Spatial clustering-based message router.

    Groups messages and nodes into spatial clusters for efficient
    routing and load distribution based on geographic proximity.
    """

    def __init__(self, cluster_radius_km: float = 50.0, max_cluster_size: int = 100):
        self.cluster_radius_km = cluster_radius_km
        self.max_cluster_size = max_cluster_size

        # Cluster management
        self.clusters: dict[str, SpatialCluster] = {}
        self.node_clusters: dict[str, str] = {}  # node_id -> cluster_id

        self.logger = logging.getLogger(__name__)

    def add_node_to_cluster(self, node_id: str, location: GeospatialPoint) -> str:
        """Add a node to an appropriate cluster."""
        # Find existing cluster or create new one
        cluster_id = self._find_or_create_cluster(location)

        # Add node to cluster
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = SpatialCluster(cluster_id, location)

        self.clusters[cluster_id].add_node(node_id, location)
        self.node_clusters[node_id] = cluster_id

        return cluster_id

    def route_to_cluster(
        self, message_location: GeospatialPoint, cluster_strategy: str = "nearest"
    ) -> str | None:
        """Route message to appropriate cluster."""
        if cluster_strategy == "nearest":
            return self._find_nearest_cluster(message_location)
        elif cluster_strategy == "load_balanced":
            return self._find_load_balanced_cluster(message_location)
        else:
            return self._find_nearest_cluster(message_location)

    def _find_or_create_cluster(self, location: GeospatialPoint) -> str:
        """Find existing cluster or create new one for location."""
        for cluster_id, cluster in self.clusters.items():
            if cluster.contains_location(location, self.cluster_radius_km):
                return cluster_id

        # Create new cluster
        cluster_id = f"cluster_{len(self.clusters)}"
        return cluster_id

    def _find_nearest_cluster(self, location: GeospatialPoint) -> str | None:
        """Find the nearest cluster to a location."""
        if not self.clusters:
            return None

        nearest_cluster = None
        min_distance = float("inf")

        for cluster in self.clusters.values():
            distance = location.distance_to(cluster.center)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster.cluster_id

        return nearest_cluster

    def _find_load_balanced_cluster(self, location: GeospatialPoint) -> str | None:
        """Find cluster with best load balance."""
        if not self.clusters:
            return None

        best_cluster = None
        best_score = float("-inf")

        for cluster in self.clusters.values():
            # Score based on proximity and cluster load
            distance = location.distance_to(cluster.center)
            proximity_score = 1.0 / (1.0 + distance)

            # Simple load score (fewer nodes = lower load)
            load_score = 1.0 / (1.0 + len(cluster.nodes))

            total_score = proximity_score + load_score

            if total_score > best_score:
                best_score = total_score
                best_cluster = cluster.cluster_id

        return best_cluster

    def get_cluster_info(self, cluster_id: str) -> dict[str, Any] | None:
        """Get information about a specific cluster."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return None

        return {
            "cluster_id": cluster.cluster_id,
            "center": {
                "longitude": cluster.center.longitude,
                "latitude": cluster.center.latitude,
            },
            "node_count": len(cluster.nodes),
            "nodes": list(cluster.nodes.keys()),
            "radius_km": self.cluster_radius_km,
        }


@dataclass
class SpatialCluster:
    """Represents a spatial cluster of nodes."""

    cluster_id: str
    center: GeospatialPoint
    nodes: dict[str, GeospatialPoint] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_node(self, node_id: str, location: GeospatialPoint) -> None:
        """Add a node to this cluster."""
        self.nodes[node_id] = location
        # Recalculate center if needed
        self._recalculate_center()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from this cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._recalculate_center()

    def contains_location(self, location: GeospatialPoint, radius_km: float) -> bool:
        """Check if location is within cluster radius."""
        distance = self.center.distance_to(location)
        return distance <= (radius_km * 1000)  # Convert km to meters

    def _recalculate_center(self) -> None:
        """Recalculate cluster center based on node locations."""
        if not self.nodes:
            return

        # Simple center calculation (average of node locations)
        total_lon = sum(node.longitude for node in self.nodes.values())
        total_lat = sum(node.latitude for node in self.nodes.values())

        avg_lon = total_lon / len(self.nodes)
        avg_lat = total_lat / len(self.nodes)

        self.center = GeospatialPoint(longitude=avg_lon, latitude=avg_lat)


class AdaptiveRoutingEngine:
    """
    Adaptive routing engine that learns and optimizes routing patterns.

    Uses machine learning techniques to analyze routing patterns,
    predict optimal routes, and adapt to changing network conditions.
    """

    def __init__(self, learning_rate: float = 0.01, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Routing knowledge base
        self.route_performance: dict[
            str, dict[str, float]
        ] = {}  # route_key -> performance_metrics
        self.message_patterns: dict[
            str, dict[str, Any]
        ] = {}  # pattern_key -> pattern_data

        # Adaptive parameters
        self.routing_weights: dict[str, float] = {
            "proximity": 0.4,
            "latency": 0.3,
            "reliability": 0.2,
            "load_balance": 0.1,
        }

        self.logger = logging.getLogger(__name__)

    def learn_from_routing_result(
        self, message: MessageResponse, route: list[str], performance: dict[str, Any]
    ) -> None:
        """Learn from a routing result to improve future routing."""
        route_key = self._generate_route_key(message, route)

        if route_key not in self.route_performance:
            self.route_performance[route_key] = {
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "sample_count": 0,
            }

        # Update performance metrics using exponential moving average
        current_perf = self.route_performance[route_key]
        alpha = self.learning_rate

        current_perf["success_rate"] = (1 - alpha) * current_perf[
            "success_rate"
        ] + alpha * (1.0 if performance.get("success", False) else 0.0)

        if "latency" in performance:
            current_perf["avg_latency"] = (1 - alpha) * current_perf[
                "avg_latency"
            ] + alpha * performance["latency"]

        current_perf["sample_count"] += 1

        # Update routing weights based on performance
        self._update_routing_weights(route_key, performance)

    def predict_optimal_route(
        self, message: MessageResponse, available_routes: dict[str, list[str]]
    ) -> str | None:
        """Predict the optimal route for a message."""
        if not available_routes:
            return None

        best_route = None
        best_score = float("-inf")

        for route_id, route in available_routes.items():
            score = self._calculate_route_score(message, route, route_id)

            if score > best_score:
                best_score = score
                best_route = route_id

        return best_route

    def _generate_route_key(self, message: MessageResponse, route: list[str]) -> str:
        """Generate a unique key for a route."""
        # MessageResponse uses ``use_enum_values``, so message_type and
        # priority are already plain strings; calling ``.value`` on them
        # raises AttributeError. Keep enum members working too, in case a
        # caller passes raw enum values.
        message_type = getattr(message.message_type, "value", message.message_type)
        priority = getattr(message.priority, "value", message.priority)
        key_parts = [
            message_type,
            priority,
            str(len(route)),
            "_".join(route[:2]),  # First two nodes
        ]

        return "_".join(key_parts)

    def _calculate_route_score(
        self, message: MessageResponse, route: list[str], route_id: str
    ) -> float:
        """Calculate score for a route."""
        route_key = self._generate_route_key(message, route)

        if route_key not in self.route_performance:
            # No historical data, use default scoring
            return self._default_route_score(message, route)

        performance = self.route_performance[route_key]

        # Calculate score based on learned performance
        success_score = (
            performance["success_rate"] * self.routing_weights["reliability"]
        )
        latency_score = (
            max(0, 1.0 - (performance["avg_latency"] / 1000))
            * self.routing_weights["latency"]
        )

        # Add proximity score if message has location
        proximity_score = 0.0
        if message.geospatial_data and len(route) > 0:
            # Simplified proximity scoring
            proximity_score = self.routing_weights["proximity"]

        total_score = success_score + latency_score + proximity_score

        return total_score

    def _default_route_score(self, message: MessageResponse, route: list[str]) -> float:
        """Calculate default score for unknown routes."""
        # Base score
        base_score = 0.5

        # Adjust for message priority
        priority_scores = {
            MessagePriority.LOW: 0.3,
            MessagePriority.NORMAL: 0.5,
            MessagePriority.HIGH: 0.7,
            MessagePriority.URGENT: 0.9,
        }
        priority_score = priority_scores.get(message.priority, 0.5)

        # Adjust for route length (shorter routes preferred)
        length_penalty = max(0, 1.0 - (len(route) - 1) * 0.1)

        return base_score * priority_score * length_penalty

    def _update_routing_weights(
        self, route_key: str, performance: dict[str, Any]
    ) -> None:
        """Update routing weights based on performance feedback."""
        # Simple adaptive weighting - in production would be more sophisticated
        if performance.get("success", False):
            # Increase weight for successful routing factors
            self.routing_weights["reliability"] = min(
                0.9, self.routing_weights["reliability"] + 0.01
            )
        else:
            # Decrease weight for unsuccessful routing factors
            self.routing_weights["reliability"] = max(
                0.1, self.routing_weights["reliability"] - 0.01
            )

    def get_routing_insights(self) -> dict[str, Any]:
        """Get insights into routing performance and patterns."""
        total_routes = len(self.route_performance)
        successful_routes = sum(
            1 for perf in self.route_performance.values() if perf["success_rate"] > 0.8
        )

        return {
            "total_routes_tracked": total_routes,
            "high_performance_routes": successful_routes,
            "routing_weights": self.routing_weights,
            "average_success_rate": (
                sum(perf["success_rate"] for perf in self.route_performance.values())
                / max(total_routes, 1)
            ),
        }


class GeospatialMessageQueue:
    """
    Geospatial message queue with priority and spatial organization.

    Provides efficient message queuing with geospatial indexing,
    priority handling, and intelligent message distribution.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue: list[
            tuple[float, str, MessageResponse]
        ] = []  # (priority_score, message_id, message)
        self.spatial_index: dict[str, list[str]] = {}  # location_key -> message_ids
        self.message_store: dict[str, MessageResponse] = {}

        self.logger = logging.getLogger(__name__)

    def enqueue_message(self, message: MessageResponse) -> bool:
        """Add message to geospatial queue."""
        if len(self.queue) >= self.max_size:
            # Remove lowest priority message if queue is full
            if self.queue:
                self.queue.pop(0)  # Remove lowest priority (highest score first)

        # Calculate priority score (lower = higher priority)
        priority_score = self._calculate_priority_score(message)

        # Add to queue
        heapq.heappush(self.queue, (priority_score, message.message_id, message))

        # Add to message store
        self.message_store[message.message_id] = message

        # Add to spatial index if geospatial data exists
        if message.geospatial_data:
            location_key = self._generate_location_key(message.geospatial_data.location)
            if location_key not in self.spatial_index:
                self.spatial_index[location_key] = []
            self.spatial_index[location_key].append(message.message_id)

        return True

    def dequeue_message(
        self, spatial_filter: SpatialFilter | None = None
    ) -> MessageResponse | None:
        """Dequeue highest priority message, optionally filtered by spatial criteria."""
        if not self.queue:
            return None

        # Find highest priority message that matches spatial filter
        candidates = []

        for i, (priority_score, message_id, message) in enumerate(self.queue):
            if spatial_filter and message.geospatial_data:
                if not spatial_filter.matches_location(
                    message.geospatial_data.location
                ):
                    continue

            candidates.append((priority_score, message_id, message, i))

        if not candidates:
            return None

        # Select highest priority (lowest score) message
        best_candidate = min(candidates, key=lambda x: x[0])
        priority_score, message_id, message, index = best_candidate

        # Remove from queue
        del self.queue[index]

        # Re-heapify (simplified - in production would maintain heap property)
        heapq.heapify(self.queue)

        # Remove from spatial index if present
        if message.geospatial_data:
            location_key = self._generate_location_key(message.geospatial_data.location)
            if location_key in self.spatial_index:
                try:
                    self.spatial_index[location_key].remove(message_id)
                    if not self.spatial_index[location_key]:
                        del self.spatial_index[location_key]
                except ValueError:
                    pass

        return message

    def get_messages_by_location(
        self, location: GeospatialPoint, radius_km: float = 1.0, limit: int = 100
    ) -> list[MessageResponse]:
        """Get messages near a specific location."""
        nearby_messages = []

        # Find location keys within radius
        for location_key, message_ids in self.spatial_index.items():
            # Parse location from key (simplified)
            try:
                lon, lat = map(float, location_key.split(","))
                key_location = GeospatialPoint(longitude=lon, latitude=lat)

                if location.distance_to(key_location) <= (radius_km * 1000):
                    for message_id in message_ids:
                        message = self.message_store.get(message_id)
                        if message:
                            nearby_messages.append(message)
            except (ValueError, IndexError):
                continue

        # Sort by priority and limit
        nearby_messages.sort(key=lambda m: self._calculate_priority_score(m))
        return nearby_messages[:limit]

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": len(self.queue),
            "spatial_index_size": len(self.spatial_index),
            "message_store_size": len(self.message_store),
            "max_size": self.max_size,
        }

    def _calculate_priority_score(self, message: MessageResponse) -> float:
        """Calculate priority score for message (lower = higher priority)."""
        base_priority = {
            MessagePriority.URGENT: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 4,
        }

        priority_score = base_priority.get(message.priority, 3)

        # Adjust for message age (older messages get slight priority boost)
        age_seconds = (datetime.now(UTC) - message.timestamp).total_seconds()
        age_adjustment = min(0.5, age_seconds / 3600)  # Max 0.5 point adjustment

        return priority_score - age_adjustment

    def _generate_location_key(self, location: GeospatialPoint) -> str:
        """Generate a location key for spatial indexing."""
        # Use rounded coordinates for grouping nearby messages
        return f"{location.longitude:.3f},{location.latitude:.3f}"


class SpatialRoutingOptimizer:
    """
    Optimizer for spatial routing algorithms.

    Analyzes routing performance and optimizes routing parameters
    based on historical data and current conditions.
    """

    def __init__(self, routing_engine: AdvancedSpatialRouter):
        self.routing_engine = routing_engine
        self.optimization_history: list[dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    def analyze_routing_performance(self) -> dict[str, Any]:
        """Analyze routing performance and suggest optimizations."""
        history = self.routing_engine.routing_history[-1000:]  # Last 1000 routes

        if not history:
            return {"message": "No routing history available"}

        return {
            "total_routes_analyzed": len(history),
            "success_rate": sum(1 for r in history if r["success"])
            / len(history)
            * 100,
            # ``latency`` is stored as None for failed routes; a plain
            # ``r.get("latency", 0)`` would then sum int + None and crash.
            "average_latency": sum(r.get("latency") or 0 for r in history)
            / len(history),
            "optimization_suggestions": self._generate_optimization_suggestions(
                history
            ),
        }

    def _generate_optimization_suggestions(
        self, history: list[dict[str, Any]]
    ) -> list[str]:
        """Generate optimization suggestions based on history."""
        suggestions = []

        success_rate = sum(1 for r in history if r["success"]) / len(history)

        if success_rate < 0.9:
            suggestions.append("Consider increasing retry attempts for failed routes")

        if any((r.get("latency") or 0) > 1000 for r in history):  # > 1 second
            suggestions.append(
                "Consider optimizing network paths for high-latency routes"
            )

        return suggestions
