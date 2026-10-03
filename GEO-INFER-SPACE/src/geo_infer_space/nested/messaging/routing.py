"""
Message Routing for H3 Nested Systems.

This module provides sophisticated routing algorithms for message delivery
across boundaries and hierarchies in nested geospatial systems.
"""

import logging
import heapq
from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for message delivery."""

    SHORTEST_PATH = "shortest_path"
    LEAST_CONGESTED = "least_congested"
    HIERARCHICAL = "hierarchical"
    BOUNDARY_AWARE = "boundary_aware"
    LOAD_BALANCED = "load_balanced"
    GEOGRAPHIC = "geographic"


class RouteMetric(Enum):
    """Metrics for route evaluation."""

    DISTANCE = "distance"
    HOP_COUNT = "hop_count"
    LATENCY = "latency"
    BANDWIDTH = "bandwidth"
    RELIABILITY = "reliability"
    COST = "cost"


@dataclass
class RouteSegment:
    """
    Represents a segment of a routing path.
    """

    from_node: str
    to_node: str

    # Segment properties
    distance: float = 0.0
    latency: float = 0.0
    bandwidth: float = float("inf")
    reliability: float = 1.0
    cost: float = 1.0

    # Boundary information
    crosses_boundary: bool = False
    boundary_id: str | None = None
    boundary_type: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Route:
    """
    Represents a complete routing path.
    """

    route_id: str
    source: str
    destination: str
    segments: list[RouteSegment] = field(default_factory=list)

    # Route metrics
    total_distance: float = 0.0
    total_latency: float = 0.0
    min_bandwidth: float = float("inf")
    reliability: float = 1.0
    total_cost: float = 0.0
    hop_count: int = 0

    # Route properties
    strategy: RoutingStrategy = RoutingStrategy.SHORTEST_PATH
    is_valid: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    use_count: int = 0

    def __post_init__(self) -> None:
        """Calculate route metrics after creation."""
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate aggregate route metrics."""
        if not self.segments:
            return

        self.total_distance = sum(seg.distance for seg in self.segments)
        self.total_latency = sum(seg.latency for seg in self.segments)
        self.min_bandwidth = min(seg.bandwidth for seg in self.segments)
        self.reliability = 1.0
        for seg in self.segments:
            self.reliability *= seg.reliability
        self.total_cost = sum(seg.cost for seg in self.segments)
        self.hop_count = len(self.segments)

    def get_path(self) -> list[str]:
        """Get the node path for this route."""
        if not self.segments:
            return []

        path = [self.segments[0].from_node]
        for segment in self.segments:
            path.append(segment.to_node)

        return path

    def crosses_boundaries(self) -> bool:
        """Check if route crosses any boundaries."""
        return any(seg.crosses_boundary for seg in self.segments)

    def get_boundary_crossings(self) -> list[str]:
        """Get list of boundary IDs crossed by this route."""
        return [
            seg.boundary_id
            for seg in self.segments
            if seg.crosses_boundary and seg.boundary_id
        ]


class MessageRouter:
    """
    Advanced message router for H3 nested systems.

    Provides multiple routing strategies and algorithms for efficient
    message delivery across complex nested geospatial systems.
    """

    def __init__(self, name: str = "MessageRouter") -> None:
        """
        Initialize message router.

        Args:
            name: Router name for identification
        """
        self.name = name

        # Network topology
        self.nodes: set[str] = set()
        self.edges: dict[tuple[str, str], RouteSegment] = {}
        self.adjacency: dict[str, set[str]] = defaultdict(set)

        # Routing tables
        self.routing_tables: dict[RoutingStrategy, dict[tuple[str, str], Route]] = {
            strategy: {} for strategy in RoutingStrategy
        }

        # Route cache
        self.route_cache: dict[tuple[str, str, RoutingStrategy], Route] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Network state
        self.node_loads: dict[str, float] = defaultdict(float)
        self.edge_loads: dict[tuple[str, str], float] = defaultdict(float)
        self.node_properties: dict[str, dict[str, Any]] = {}

        # Boundary information
        self.boundary_manager = None  # Will be set externally

        # Statistics
        self.routing_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def add_node(
        self,
        node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a node to the routing network.

        Args:
            node_id: Node identifier
            properties: Optional node properties
        """
        self.nodes.add(node_id)

        if properties:
            self.node_properties[node_id] = dict(properties)
        else:
            self.node_properties.setdefault(node_id, {})

        self.updated_at = datetime.now(UTC)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        distance: float = 1.0,
        latency: float = 0.1,
        bandwidth: float = float("inf"),
        reliability: float = 1.0,
        cost: float = 1.0,
        bidirectional: bool = True,
        crosses_boundary: bool = False,
        boundary_id: str | None = None,
    ) -> None:
        """
        Add an edge to the routing network.

        Args:
            from_node: Source node
            to_node: Target node
            distance: Edge distance
            latency: Edge latency
            bandwidth: Edge bandwidth
            reliability: Edge reliability
            cost: Edge cost
            bidirectional: Whether edge is bidirectional
            crosses_boundary: Whether edge crosses a boundary
            boundary_id: ID of boundary crossed (if any)
        """
        segment = RouteSegment(
            from_node=from_node,
            to_node=to_node,
            distance=distance,
            latency=latency,
            bandwidth=bandwidth,
            reliability=reliability,
            cost=cost,
            crosses_boundary=crosses_boundary,
            boundary_id=boundary_id,
        )

        self.edges[(from_node, to_node)] = segment
        self.adjacency[from_node].add(to_node)

        if bidirectional:
            reverse_segment = RouteSegment(
                from_node=to_node,
                to_node=from_node,
                distance=distance,
                latency=latency,
                bandwidth=bandwidth,
                reliability=reliability,
                cost=cost,
                crosses_boundary=crosses_boundary,
                boundary_id=boundary_id,
            )

            self.edges[(to_node, from_node)] = reverse_segment
            self.adjacency[to_node].add(from_node)

        # Clear route cache as topology changed
        self.route_cache.clear()
        self.updated_at = datetime.now(UTC)

    def remove_edge(
        self,
        from_node: str,
        to_node: str,
        bidirectional: bool = True,
    ) -> None:
        """Remove an edge from the routing network."""
        if (from_node, to_node) in self.edges:
            del self.edges[(from_node, to_node)]
            self.adjacency[from_node].discard(to_node)

        if bidirectional and (to_node, from_node) in self.edges:
            del self.edges[(to_node, from_node)]
            self.adjacency[to_node].discard(from_node)

        # Clear route cache
        self.route_cache.clear()
        self.updated_at = datetime.now(UTC)

    def find_route(
        self,
        source: str,
        destination: str,
        strategy: RoutingStrategy = RoutingStrategy.SHORTEST_PATH,
        metric: RouteMetric = RouteMetric.DISTANCE,
        use_cache: bool = True,
    ) -> Route | None:
        """
        Find a route between source and destination.

        Args:
            source: Source node
            destination: Destination node
            strategy: Routing strategy to use
            metric: Metric to optimize for
            use_cache: Whether to use route cache

        Returns:
            Route object or None if no route found
        """
        # Check cache first
        cache_key = (source, destination, strategy)
        if use_cache and cache_key in self.route_cache:
            self.cache_hits += 1
            cached_route = self.route_cache[cache_key]
            cached_route.last_used = datetime.now(UTC)
            cached_route.use_count += 1
            return cached_route

        self.cache_misses += 1

        # Find route based on strategy
        if strategy == RoutingStrategy.SHORTEST_PATH:
            route = self._find_shortest_path(source, destination, metric)
        elif strategy == RoutingStrategy.LEAST_CONGESTED:
            route = self._find_least_congested_path(source, destination)
        elif strategy == RoutingStrategy.HIERARCHICAL:
            route = self._find_hierarchical_path(source, destination)
        elif strategy == RoutingStrategy.BOUNDARY_AWARE:
            route = self._find_boundary_aware_path(source, destination)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            route = self._find_load_balanced_path(source, destination)
        elif strategy == RoutingStrategy.GEOGRAPHIC:
            route = self._find_geographic_path(source, destination)
        else:
            logger.warning(  # type: ignore[unreachable]
                f"Unknown routing strategy: {strategy}"
            )
            route = self._find_shortest_path(source, destination, metric)

        # Cache the route
        if route and use_cache:
            self.route_cache[cache_key] = route

        # Update statistics
        self.routing_stats[f"{strategy.value}_requests"] += 1
        if route:
            self.routing_stats[f"{strategy.value}_success"] += 1
        else:
            self.routing_stats[f"{strategy.value}_failed"] += 1

        return route

    def _find_shortest_path(
        self, source: str, destination: str, metric: RouteMetric = RouteMetric.DISTANCE
    ) -> Route | None:
        """Find shortest path using Dijkstra's algorithm."""
        if source not in self.nodes or destination not in self.nodes:
            return None

        if source == destination:
            return Route(
                route_id=f"route_{source}_{destination}",
                source=source,
                destination=destination,
                strategy=RoutingStrategy.SHORTEST_PATH,
            )

        # Dijkstra's algorithm
        distances = {node: float("inf") for node in self.nodes}
        distances[source] = 0
        previous = {}
        visited = set()

        # Priority queue: (distance, node)
        pq: list[tuple[float, str]] = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == destination:
                break

            # Check neighbors
            for neighbor in self.adjacency[current_node]:
                if neighbor in visited:
                    continue

                edge = self.edges.get((current_node, neighbor))
                if not edge:
                    continue

                # Calculate edge weight based on metric
                if metric == RouteMetric.DISTANCE:
                    weight = edge.distance
                elif metric == RouteMetric.LATENCY:
                    weight = edge.latency
                elif metric == RouteMetric.COST:
                    weight = edge.cost
                elif metric == RouteMetric.HOP_COUNT:
                    weight = 1
                elif metric == RouteMetric.RELIABILITY:
                    weight = 1 - edge.reliability  # Lower is better
                else:
                    weight = edge.distance

                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if destination not in previous and destination != source:
            return None  # No path found

        path = []
        current = destination
        while current != source:
            path.append(current)
            current = previous[current]
        path.append(source)
        path.reverse()

        # Create route segments
        segments = []
        for i in range(len(path) - 1):
            edge = self.edges.get((path[i], path[i + 1]))
            if edge:
                segments.append(edge)

        route = Route(
            route_id=f"route_{source}_{destination}_{datetime.now(UTC).timestamp()}",
            source=source,
            destination=destination,
            segments=segments,
            strategy=RoutingStrategy.SHORTEST_PATH,
        )

        return route

    def _find_least_congested_path(self, source: str, destination: str) -> Route | None:
        """Find path with least congestion."""
        # Use modified Dijkstra with congestion weights
        if source not in self.nodes or destination not in self.nodes:
            return None

        distances = {node: float("inf") for node in self.nodes}
        distances[source] = 0
        previous = {}
        visited = set()

        pq: list[tuple[float, str]] = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == destination:
                break

            for neighbor in self.adjacency[current_node]:
                if neighbor in visited:
                    continue

                edge = self.edges.get((current_node, neighbor))
                if not edge:
                    continue

                # Calculate congestion-aware weight
                node_load = self.node_loads[current_node]
                edge_load = self.edge_loads[(current_node, neighbor)]

                # Combine distance with load factors
                congestion_factor = 1 + node_load + edge_load
                weight = edge.distance * congestion_factor

                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path (same as shortest path)
        return self._reconstruct_route(
            source, destination, previous, RoutingStrategy.LEAST_CONGESTED
        )

    def _find_hierarchical_path(self, source: str, destination: str) -> Route | None:
        """Find path using hierarchical routing."""
        # This would implement hierarchical routing based on H3 resolution levels
        # For now, fall back to shortest path
        return self._find_shortest_path(source, destination)

    def _find_boundary_aware_path(self, source: str, destination: str) -> Route | None:
        """Find path that considers boundary crossings."""
        if source not in self.nodes or destination not in self.nodes:
            return None

        distances = {node: float("inf") for node in self.nodes}
        distances[source] = 0
        previous = {}
        visited = set()

        pq: list[tuple[float, str]] = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == destination:
                break

            for neighbor in self.adjacency[current_node]:
                if neighbor in visited:
                    continue

                edge = self.edges.get((current_node, neighbor))
                if not edge:
                    continue

                # Penalize boundary crossings
                weight = edge.distance
                if edge.crosses_boundary:
                    weight *= 2.0  # Double cost for boundary crossings

                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        return self._reconstruct_route(
            source, destination, previous, RoutingStrategy.BOUNDARY_AWARE
        )

    def _find_load_balanced_path(self, source: str, destination: str) -> Route | None:
        """Find path that balances load across the network."""
        # Similar to least congested but with different load calculation
        return self._find_least_congested_path(source, destination)

    def _find_geographic_path(self, source: str, destination: str) -> Route | None:
        """Find the path minimizing total geographic distance.

        Edge weights are geographic distances, so the geographic strategy
        coincides with the distance-minimizing shortest path; when H3 is
        unavailable the same distance-minimizing walk applies unchanged.
        """
        return self._find_shortest_path(source, destination)

    def _reconstruct_route(
        self,
        source: str,
        destination: str,
        previous: dict[str, str],
        strategy: RoutingStrategy,
    ) -> Route | None:
        """Reconstruct route from previous node mapping."""
        if destination not in previous and destination != source:
            return None

        path = []
        current = destination
        while current != source:
            path.append(current)
            if current not in previous:
                return None
            current = previous[current]
        path.append(source)
        path.reverse()

        # Create route segments
        segments = []
        for i in range(len(path) - 1):
            edge = self.edges.get((path[i], path[i + 1]))
            if edge:
                segments.append(edge)

        route = Route(
            route_id=f"route_{source}_{destination}_{datetime.now(UTC).timestamp()}",
            source=source,
            destination=destination,
            segments=segments,
            strategy=strategy,
        )

        return route

    def update_node_load(self, node_id: str, load: float) -> None:
        """Update load for a node."""
        self.node_loads[node_id] = load

    def update_edge_load(self, from_node: str, to_node: str, load: float) -> None:
        """Update load for an edge."""
        self.edge_loads[(from_node, to_node)] = load

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get routing statistics."""
        cache_ratio = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )

        return {
            "router_name": self.name,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "cached_routes": len(self.route_cache),
            "cache_hit_ratio": cache_ratio,
            "routing_stats": dict(self.routing_stats),
            "average_node_load": (
                sum(self.node_loads.values()) / len(self.node_loads)
                if self.node_loads
                else 0
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear the route cache."""
        self.route_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
