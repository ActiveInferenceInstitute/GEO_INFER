"""
Transport network module.

Provides network topology construction, analysis, and management
for transportation systems.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class RoadClass(Enum):
    """Road classification types."""

    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    SERVICE = "service"
    PATH = "path"


class TransportMode(Enum):
    """Transportation modes."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    RAIL = "rail"
    SUBWAY = "subway"


@dataclass
class NetworkNode:
    """Represents a node in the transport network."""

    node_id: str
    location: dict[str, float]  # lat, lon
    node_type: str = "intersection"
    elevation: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEdge:
    """Represents an edge (road segment) in the transport network."""

    edge_id: str
    from_node: str
    to_node: str
    road_class: RoadClass
    length_m: float
    speed_limit_kmh: float = 50
    lanes: int = 1
    one_way: bool = False
    geometry: list[dict[str, float]] | None = None
    properties: dict[str, Any] = field(default_factory=dict)


class TransportNetwork:
    """
    Build and analyze transportation network topology.

    Supports multi-modal networks, connectivity analysis, and
    network statistics calculation.
    """

    def __init__(
        self,
        network_type: str = "road",
        modes: list[str] | None = None,
        crs: str = "EPSG:4326",
    ):
        """
        Initialize transport network.

        Args:
            network_type: Type of network ('road', 'rail', 'multimodal')
            modes: Supported transport modes
            crs: Coordinate reference system
        """
        self.network_type = network_type
        self.modes = modes or ["car", "bicycle", "pedestrian"]
        self.crs = crs
        self._graph = nx.DiGraph()
        self._nodes: dict[str, NetworkNode] = {}
        self._edges: dict[str, NetworkEdge] = {}
        logger.info(f"Initialized TransportNetwork of type {network_type}")

    def build_from_edges(
        self,
        edges: list[dict[str, Any]],
        nodes: list[dict[str, Any]] | None = None,
        attributes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build network from edge list.

        Args:
            edges: List of edge definitions
            nodes: Optional node definitions
            attributes: Extra edge attribute keys from the edge dicts to
                carry onto the graph edges (e.g. ``["surface", "oneway"]``).
                Base attributes (edge_id, length, speed_limit, travel_time,
                road_class) are always included.

        Returns:
            Network build summary
        """
        # Add nodes if provided
        if nodes:
            for node_data in nodes:
                node = NetworkNode(
                    node_id=node_data.get("id", f"node_{len(self._nodes)}"),
                    location=node_data.get("location", {}),
                    node_type=node_data.get("type", "intersection"),
                    elevation=node_data.get("elevation"),
                )
                self._nodes[node.node_id] = node
                self._graph.add_node(node.node_id, **node_data)

        # Add edges
        for edge_data in edges:
            from_node = edge_data.get("from")
            to_node = edge_data.get("to")
            edge = NetworkEdge(
                edge_id=edge_data.get("id", f"edge_{len(self._edges)}"),
                from_node=str(from_node) if from_node is not None else "",
                to_node=str(to_node) if to_node is not None else "",
                road_class=RoadClass(edge_data.get("road_class", "secondary")),
                length_m=edge_data.get("length_m", 100),
                speed_limit_kmh=edge_data.get("speed_limit", 50),
                lanes=edge_data.get("lanes", 1),
                one_way=edge_data.get("one_way", False),
                geometry=edge_data.get("geometry"),
            )
            self._edges[edge.edge_id] = edge

            # Calculate travel time
            travel_time_s = (edge.length_m / 1000) / edge.speed_limit_kmh * 3600

            # Add to graph
            self._graph.add_edge(
                edge.from_node,
                edge.to_node,
                edge_id=edge.edge_id,
                length=edge.length_m,
                speed_limit=edge.speed_limit_kmh,
                travel_time=travel_time_s,
                road_class=edge.road_class.value,
                **{
                    attr: edge_data[attr]
                    for attr in (attributes or [])
                    if attr in edge_data
                },
            )

            # If not one-way, add reverse edge
            if not edge.one_way:
                self._graph.add_edge(
                    edge.to_node,
                    edge.from_node,
                    edge_id=f"{edge.edge_id}_rev",
                    length=edge.length_m,
                    speed_limit=edge.speed_limit_kmh,
                    travel_time=travel_time_s,
                    road_class=edge.road_class.value,
                    **{
                        attr: edge_data[attr]
                        for attr in (attributes or [])
                        if attr in edge_data
                    },
                )

            # Ensure nodes exist
            if edge.from_node not in self._nodes:
                self._nodes[edge.from_node] = NetworkNode(
                    node_id=edge.from_node, location={}
                )
                self._graph.add_node(edge.from_node)
            if edge.to_node not in self._nodes:
                self._nodes[edge.to_node] = NetworkNode(
                    node_id=edge.to_node, location={}
                )
                self._graph.add_node(edge.to_node)

        summary = {
            "nodes_created": len(self._nodes),
            "edges_created": len(self._edges),
            "network_type": self.network_type,
            "is_connected": nx.is_weakly_connected(self._graph)
            if self._graph.number_of_nodes() > 0
            else False,
        }

        logger.info(
            f"Built network with {len(self._nodes)} nodes and {len(self._edges)} edges"
        )
        return summary

    def analyze_connectivity(
        self,
        method: str = "components",
        origin: str | None = None,
        destinations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze network connectivity.

        Args:
            method: Analysis method ('components', 'reachability', 'betweenness')
            origin: Origin node for reachability analysis
            destinations: Destination nodes

        Returns:
            Connectivity analysis results
        """
        result = {
            "method": method,
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
        }

        if method == "components":
            # Strongly connected components
            sccs = list(nx.strongly_connected_components(self._graph))
            wccs = list(nx.weakly_connected_components(self._graph))

            result["strongly_connected_components"] = len(sccs)
            result["weakly_connected_components"] = len(wccs)
            result["largest_scc_size"] = max(len(c) for c in sccs) if sccs else 0
            result["is_strongly_connected"] = len(sccs) == 1

        elif method == "reachability" and origin:
            # Nodes reachable from origin
            reachable = nx.descendants(self._graph, origin)
            reachable.add(origin)

            result["origin"] = origin
            result["reachable_nodes"] = len(reachable)
            result["reachability_ratio"] = (
                len(reachable) / self._graph.number_of_nodes()
                if self._graph.number_of_nodes() > 0
                else 0
            )

            if destinations:
                result["destinations_reachable"] = [
                    d for d in destinations if d in reachable
                ]
                result["destinations_unreachable"] = [
                    d for d in destinations if d not in reachable
                ]

        elif method == "betweenness":
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(self._graph, weight="length")

            # Find critical nodes
            sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

            result["critical_nodes"] = [
                {"node_id": node, "centrality": round(cent, 4)}
                for node, cent in sorted_nodes[:10]
            ]

        elif method == "critical_links":
            # Delegate to GEO-INFER-LOG if available for edge-betweenness analysis
            try:
                from geo_infer_log.core.transport import TransportationNetworkAnalyzer

                analyzer = TransportationNetworkAnalyzer()
                analyzer.network = self._graph  # Reuse our NetworkX graph
                critical = analyzer.identify_critical_links(top_n=10)
                result["critical_links"] = [{"from": u, "to": v} for u, v in critical]
                result["source"] = "geo_infer_log"
            except ImportError:
                # Fallback: compute edge betweenness directly
                edge_bc = nx.edge_betweenness_centrality(self._graph, weight="length")
                sorted_edges = sorted(
                    edge_bc.items(), key=lambda x: x[1], reverse=True
                )[:10]
                result["critical_links"] = [
                    {"from": u, "to": v, "centrality": round(c, 4)}
                    for (u, v), c in sorted_edges
                ]
                result["source"] = "fallback"

        logger.info(f"Connectivity analysis ({method}) completed")
        return result

    def calculate_centrality(
        self,
        centrality_type: str = "betweenness",
        weight: str = "length",
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Calculate network centrality measures.

        Args:
            centrality_type: Type of centrality ('betweenness', 'closeness', 'degree')
            weight: Edge weight attribute
            top_n: Number of top nodes to return

        Returns:
            Centrality analysis results
        """
        if centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(self._graph, weight=weight)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(self._graph, distance=weight)
        elif centrality_type == "degree":
            centrality = nx.degree_centrality(self._graph)
        else:
            centrality = nx.betweenness_centrality(self._graph)

        # Sort and get top N
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        result = {
            "centrality_type": centrality_type,
            "weight": weight,
            "top_nodes": [
                {"node_id": node, "centrality": round(cent, 4)}
                for node, cent in sorted_nodes[:top_n]
            ],
            "mean_centrality": sum(centrality.values()) / len(centrality)
            if centrality
            else 0,
        }

        logger.info(f"Calculated {centrality_type} centrality")
        return result

    def get_statistics(self) -> dict[str, Any]:
        """
        Get network statistics.

        Returns:
            Network statistics summary
        """
        stats = {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "network_type": self.network_type,
            "modes": self.modes,
        }

        if self._graph.number_of_nodes() > 0:
            # Calculate additional statistics
            stats["density"] = nx.density(self._graph)
            stats["is_directed"] = self._graph.is_directed()

            # Degree statistics
            in_degrees = [d for _, d in self._graph.in_degree()]
            out_degrees = [d for _, d in self._graph.out_degree()]

            stats["avg_in_degree"] = (
                sum(in_degrees) / len(in_degrees) if in_degrees else 0
            )
            stats["avg_out_degree"] = (
                sum(out_degrees) / len(out_degrees) if out_degrees else 0
            )

            # Total network length
            total_length = sum(
                data.get("length", 0) for _, _, data in self._graph.edges(data=True)
            )
            stats["total_length_km"] = total_length / 1000

            # Road class distribution
            road_classes: dict[str, int] = {}
            for _, _, data in self._graph.edges(data=True):
                rc = data.get("road_class", "unknown")
                road_classes[rc] = road_classes.get(rc, 0) + 1
            stats["road_class_distribution"] = road_classes

        return stats

    def get_subgraph(
        self, nodes: list[str] | None = None, bbox: dict[str, float] | None = None
    ) -> "TransportNetwork":
        """
        Extract a subgraph from the network.

        Args:
            nodes: Nodes to include
            bbox: Bounding box with ``min_lat``/``max_lat``/``min_lon``/
                ``max_lon`` keys. Only nodes that carry a location are
                considered; nodes without coordinates (e.g. implicit
                nodes created by ``build_from_edges`` without explicit
                node definitions) cannot be spatially filtered and are
                excluded with a warning.

        Returns:
            New TransportNetwork with subgraph
        """
        if nodes:
            subgraph = self._graph.subgraph(nodes).copy()
        elif bbox:
            filtered_nodes = []
            unlocated = 0
            for node_id, node in self._nodes.items():
                loc = node.location
                if not loc:
                    unlocated += 1
                    continue
                if bbox.get("min_lat", -90) <= loc.get("lat", 0) <= bbox.get(
                    "max_lat", 90
                ) and bbox.get("min_lon", -180) <= loc.get("lon", 0) <= bbox.get(
                    "max_lon", 180
                ):
                    filtered_nodes.append(node_id)
            if unlocated:
                logger.warning(
                    "get_subgraph(bbox=...) excluded %d node(s) without a location; "
                    "provide node coordinates via build_from_edges(nodes=...) to include them",
                    unlocated,
                )
            subgraph = self._graph.subgraph(filtered_nodes).copy()
        else:
            subgraph = self._graph.copy()

        # Create new network
        new_network = TransportNetwork(
            network_type=self.network_type, modes=self.modes, crs=self.crs
        )
        new_network._graph = subgraph

        # Copy nodes and edges
        for node_id in subgraph.nodes():
            if node_id in self._nodes:
                new_network._nodes[node_id] = self._nodes[node_id]

        for u, v, data in subgraph.edges(data=True):
            edge_id = data.get("edge_id", f"{u}_{v}")
            if edge_id in self._edges:
                new_network._edges[edge_id] = self._edges[edge_id]

        return new_network

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph
