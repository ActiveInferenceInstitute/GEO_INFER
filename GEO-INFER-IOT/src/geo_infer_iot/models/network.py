"""
Network Topology Data Models

This module defines data models for sensor network topology, communication
patterns, and network management for IoT sensor deployments.
"""

import logging
from typing import Any, cast
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import StrEnum
import networkx as nx
import h3
import numpy as np

logger = logging.getLogger(__name__)


class NetworkTopologyType(StrEnum):
    """Network topology types."""

    MESH = "mesh"
    STAR = "star"
    HIERARCHICAL = "hierarchical"
    BUS = "bus"
    RING = "ring"
    HYBRID = "hybrid"


class CommunicationProtocol(StrEnum):
    """Communication protocol types."""

    MQTT = "MQTT"
    COAP = "CoAP"
    LORAWAN = "LoRaWAN"
    HTTP = "HTTP"
    WEBSOCKET = "WebSocket"
    BLUETOOTH = "Bluetooth"
    ZIGBEE = "Zigbee"
    CUSTOM = "custom"


class NetworkNode(BaseModel):
    """Network node representing a sensor or gateway."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(
        ..., description="Type of node (sensor, gateway, coordinator)"
    )
    sensor_id: str | None = Field(
        None, description="Associated sensor ID if this is a sensor node"
    )

    # Network position and connections
    parent_node: str | None = Field(
        None, description="Parent node in hierarchical topology"
    )
    child_nodes: list[str] = Field(
        default_factory=list, description="Child nodes in hierarchical topology"
    )
    connected_nodes: list[str] = Field(
        default_factory=list, description="Directly connected nodes"
    )

    # Location information
    latitude: float | None = Field(None, description="Node latitude")
    longitude: float | None = Field(None, description="Node longitude")
    h3_index: str | None = Field(None, description="H3 hexagonal index")

    # Network properties
    protocol: CommunicationProtocol = Field(
        CommunicationProtocol.MQTT, description="Communication protocol"
    )
    transmission_range: float | None = Field(
        None, description="Transmission range in meters"
    )
    battery_level: float | None = Field(None, description="Battery level percentage")
    signal_strength: float | None = Field(None, description="Signal strength dBm")

    # Operational status
    status: str = Field("active", description="Node status")
    last_seen: datetime | None = Field(None, description="Last communication timestamp")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")

    # Performance metrics
    packets_sent: int = Field(0, description="Total packets sent")
    packets_received: int = Field(0, description="Total packets received")
    packet_loss_rate: float = Field(0.0, description="Packet loss rate percentage")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Auto-generate H3 index if coordinates provided
        if (
            self.latitude is not None
            and self.longitude is not None
            and not self.h3_index
        ):
            self.h3_index = h3.latlng_to_cell(self.latitude, self.longitude, 8)

    def add_connection(self, node_id: str) -> None:
        """Add a connection to another node."""
        if node_id not in self.connected_nodes:
            self.connected_nodes.append(node_id)

    def remove_connection(self, node_id: str) -> None:
        """Remove a connection to another node."""
        if node_id in self.connected_nodes:
            self.connected_nodes.remove(node_id)

    def get_health_score(self) -> float:
        """Calculate node health score based on various metrics."""
        score = 1.0

        # Battery level factor
        if self.battery_level is not None:
            if self.battery_level < 20:
                score *= 0.6
            elif self.battery_level < 50:
                score *= 0.8

        # Signal strength factor
        if self.signal_strength is not None:
            if self.signal_strength < -80:
                score *= 0.7
            elif self.signal_strength < -60:
                score *= 0.85

        # Packet loss factor
        if self.packet_loss_rate > 10:
            score *= 0.7
        elif self.packet_loss_rate > 5:
            score *= 0.85

        # Status factor
        if self.status != "active":
            score *= 0.5

        return max(0.0, min(1.0, score))


class NetworkLink(BaseModel):
    """Network link between two nodes."""

    link_id: str = Field(..., description="Unique link identifier")
    source_node: str = Field(..., description="Source node ID")
    target_node: str = Field(..., description="Target node ID")

    # Link properties
    link_type: str = Field(
        "wireless", description="Link type (wireless, wired, optical)"
    )
    protocol: CommunicationProtocol = Field(
        CommunicationProtocol.MQTT, description="Link protocol"
    )
    bandwidth_mbps: float | None = Field(None, description="Link bandwidth in Mbps")
    latency_ms: float | None = Field(None, description="Link latency in milliseconds")

    # Link quality metrics
    signal_quality: float = Field(1.0, description="Signal quality score")
    reliability: float = Field(1.0, description="Link reliability score")
    utilization: float = Field(0.0, description="Link utilization percentage")
    packet_loss_rate: float = Field(0.0, description="Link packet loss rate percentage")

    # Status
    status: str = Field("active", description="Link status")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_performance_score(self) -> float:
        """Calculate link performance score."""
        # Weighted combination of quality metrics
        return (
            0.4 * self.signal_quality
            + 0.3 * self.reliability
            + 0.3 * (1.0 - min(1.0, self.utilization))
        )


class NetworkTopology(BaseModel):
    """Complete network topology model."""

    topology_id: str = Field(..., description="Unique topology identifier")
    name: str = Field(..., description="Topology name")
    description: str | None = Field(None, description="Topology description")

    # Topology configuration
    topology_type: NetworkTopologyType = Field(
        NetworkTopologyType.MESH, description="Network topology type"
    )
    protocol: CommunicationProtocol = Field(
        CommunicationProtocol.MQTT, description="Primary protocol"
    )

    # Network structure
    nodes: dict[str, NetworkNode] = Field(
        default_factory=dict, description="Network nodes"
    )
    links: dict[str, NetworkLink] = Field(
        default_factory=dict, description="Network links"
    )

    # Network properties
    total_nodes: int = Field(0, description="Total number of nodes")
    active_nodes: int = Field(0, description="Number of active nodes")
    total_links: int = Field(0, description="Total number of links")
    active_links: int = Field(0, description="Number of active links")

    # Coverage area
    spatial_bounds: dict[str, float] | None = Field(
        None, description="Network coverage bounds"
    )
    h3_resolution: int = Field(8, description="H3 resolution for spatial indexing")

    # Performance metrics
    average_latency_ms: float | None = Field(
        None, description="Average network latency"
    )
    packet_loss_rate: float = Field(0.0, description="Overall packet loss rate")
    network_efficiency: float = Field(1.0, description="Network efficiency score")

    # Status and metadata
    status: str = Field("active", description="Network status")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Update derived fields
        self._update_derived_fields()

    def _update_derived_fields(self) -> None:
        """Update derived fields based on current nodes and links."""
        self.total_nodes = len(self.nodes)
        self.active_nodes = len(
            [n for n in self.nodes.values() if n.status == "active"]
        )
        self.total_links = len(self.links)
        self.active_links = len(
            [link for link in self.links.values() if link.status == "active"]
        )

        # Calculate network metrics
        self._calculate_network_metrics()

    def _calculate_network_metrics(self) -> None:
        """Calculate network performance metrics."""
        if not self.links:
            return

        # Calculate average latency
        latencies = [
            link.latency_ms
            for link in self.links.values()
            if link.latency_ms is not None
        ]
        if latencies:
            self.average_latency_ms = sum(latencies) / len(latencies)

        # Calculate overall packet loss
        loss_rates = [link.packet_loss_rate for link in self.links.values()]
        self.packet_loss_rate = sum(loss_rates) / len(loss_rates) if loss_rates else 0.0

        # Calculate network efficiency
        efficiency_factors = []
        for link in self.links.values():
            # Efficiency based on utilization and reliability
            efficiency = link.reliability * (1.0 - min(1.0, link.utilization))
            efficiency_factors.append(efficiency)

        self.network_efficiency = (
            sum(efficiency_factors) / len(efficiency_factors)
            if efficiency_factors
            else 1.0
        )

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the network."""
        self.nodes[node.node_id] = node
        self._update_derived_fields()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        if node_id in self.nodes:
            del self.nodes[node_id]

            # Remove associated links
            links_to_remove = [
                link_id
                for link_id, link in self.links.items()
                if link.source_node == node_id or link.target_node == node_id
            ]
            for link_id in links_to_remove:
                del self.links[link_id]

            self._update_derived_fields()

    def add_link(self, link: NetworkLink) -> None:
        """Add a link to the network."""
        self.links[link.link_id] = link

        # Update node connections
        if link.source_node in self.nodes:
            self.nodes[link.source_node].add_connection(link.target_node)
        if link.target_node in self.nodes:
            self.nodes[link.target_node].add_connection(link.source_node)

        self._update_derived_fields()

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the network."""
        if link_id in self.links:
            link = self.links[link_id]

            # Update node connections
            if link.source_node in self.nodes:
                self.nodes[link.source_node].remove_connection(link.target_node)
            if link.target_node in self.nodes:
                self.nodes[link.target_node].remove_connection(link.source_node)

            del self.links[link_id]
            self._update_derived_fields()

    def get_connected_components(self) -> list[list[str]]:
        """Get connected components in the network."""
        if not self.nodes:
            return []

        # Build adjacency list
        adjacency = {
            node_id: node.connected_nodes for node_id, node in self.nodes.items()
        }

        # Find connected components
        visited: set[str] = set()
        components: list[list[str]] = []

        def dfs(node_id: str, component: list[str]) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            component.append(node_id)
            for neighbor in adjacency.get(node_id, []):
                dfs(neighbor, component)

        for node_id in self.nodes:
            if node_id not in visited:
                component: list[str] = []
                dfs(node_id, component)
                if component:
                    components.append(component)

        return components

    def get_network_diameter(self) -> int | None:
        """Calculate network diameter (longest shortest path)."""
        if not self.nodes:
            return None

        try:
            # Create NetworkX graph
            G = nx.Graph()

            # Add nodes
            for node_id, node in self.nodes.items():
                G.add_node(node_id, **node.model_dump())

            # Add edges
            for link in self.links.values():
                if link.status == "active":
                    G.add_edge(
                        link.source_node,
                        link.target_node,
                        weight=1,
                        **link.model_dump(),
                    )

            if not nx.is_connected(G):
                return None  # Network is not fully connected

            # Calculate diameter
            return cast(int, nx.diameter(G))

        except Exception as e:
            logger.warning(f"Error calculating network diameter: {e}")
            return None

    def get_sensor_coverage(self) -> dict[str, Any]:
        """Get sensor coverage analysis."""
        sensor_nodes = [
            node for node in self.nodes.values() if node.node_type == "sensor"
        ]

        if not sensor_nodes:
            return {"total_sensors": 0, "coverage_area": 0.0}

        # Calculate coverage bounds
        latitudes = [n.latitude for n in sensor_nodes if n.latitude is not None]
        longitudes = [n.longitude for n in sensor_nodes if n.longitude is not None]

        if not latitudes or not longitudes:
            return {"total_sensors": len(sensor_nodes), "coverage_area": 0.0}

        lat_range = max(latitudes) - min(latitudes)
        lon_range = max(longitudes) - min(longitudes)

        # Rough coverage area calculation
        coverage_area = lat_range * lon_range * 111 * 111  # km²

        return {
            "total_sensors": len(sensor_nodes),
            "active_sensors": len([n for n in sensor_nodes if n.status == "active"]),
            "coverage_bounds": {
                "lat_min": min(latitudes),
                "lat_max": max(latitudes),
                "lon_min": min(longitudes),
                "lon_max": max(longitudes),
            },
            "coverage_area_km2": coverage_area,
            "average_signal_strength": np.mean(
                [
                    n.signal_strength
                    for n in sensor_nodes
                    if n.signal_strength is not None
                ]
            ),
            "network_connectivity": len(self.get_connected_components())
            == 1,  # Fully connected
        }


class NetworkEvent(BaseModel):
    """Network event for monitoring and debugging."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of network event")

    # Event details
    node_id: str | None = Field(None, description="Affected node ID")
    link_id: str | None = Field(None, description="Affected link ID")
    severity: str = Field("info", description="Event severity")

    # Event data
    message: str = Field(..., description="Event message")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional event details"
    )

    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Event timestamp"
    )
    duration_ms: float | None = Field(
        None, description="Event duration in milliseconds"
    )

    # Context
    network_id: str | None = Field(None, description="Associated network ID")
    session_id: str | None = Field(None, description="Associated session ID")

    @field_validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate event severity."""
        valid_severities = ["debug", "info", "warning", "error", "critical"]
        if v not in valid_severities:
            raise ValueError(
                f"Invalid severity '{v}'. Must be one of: {valid_severities}"
            )
        return v


class NetworkConfiguration(BaseModel):
    """Network configuration and deployment settings."""

    config_id: str = Field(..., description="Unique configuration identifier")
    network_id: str = Field(..., description="Associated network ID")

    # Configuration settings
    topology_config: dict[str, Any] = Field(
        default_factory=dict, description="Topology-specific configuration"
    )
    protocol_config: dict[str, Any] = Field(
        default_factory=dict, description="Protocol-specific configuration"
    )
    security_config: dict[str, Any] = Field(
        default_factory=dict, description="Security configuration"
    )

    # Deployment settings
    deployment_mode: str = Field("production", description="Deployment mode")
    auto_scaling: bool = Field(False, description="Enable automatic scaling")
    redundancy_level: int = Field(1, description="Network redundancy level")

    # Performance settings
    max_latency_ms: float | None = Field(None, description="Maximum acceptable latency")
    target_reliability: float = Field(0.99, description="Target network reliability")
    bandwidth_limits: dict[str, float] = Field(
        default_factory=dict, description="Bandwidth limits per node type"
    )

    # Monitoring settings
    monitoring_enabled: bool = Field(True, description="Enable network monitoring")
    alert_thresholds: dict[str, float] = Field(
        default_factory=dict, description="Alert threshold configuration"
    )

    # Version and metadata
    version: str = Field("1.0.0", description="Configuration version")
    description: str = Field("", description="Configuration description")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("deployment_mode")
    def validate_deployment_mode(cls, v: str) -> str:
        """Validate deployment mode."""
        valid_modes = ["development", "staging", "production", "testing"]
        if v not in valid_modes:
            raise ValueError(
                f"Invalid deployment mode '{v}'. Must be one of: {valid_modes}"
            )
        return v


class NetworkPerformance(BaseModel):
    """Network performance metrics and analysis."""

    performance_id: str = Field(..., description="Unique performance record ID")
    network_id: str = Field(..., description="Associated network ID")

    # Performance metrics
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Measurement timestamp"
    )

    # Detailed breakdown
    node_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Per-node metrics"
    )
    link_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Per-link metrics"
    )

    # Analysis results
    performance_score: float = Field(1.0, description="Overall performance score")
    bottlenecks: list[dict[str, Any]] = Field(
        default_factory=list, description="Identified bottlenecks"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Performance recommendations"
    )

    # Comparison data
    baseline_metrics: dict[str, float] | None = Field(
        None, description="Baseline metrics for comparison"
    )
    trend_analysis: dict[str, Any] | None = Field(
        None, description="Trend analysis results"
    )

    def add_metric(
        self,
        metric_name: str,
        value: float,
        node_id: str | None = None,
        link_id: str | None = None,
    ) -> None:
        """Add a performance metric."""
        if node_id:
            if node_id not in self.node_metrics:
                self.node_metrics[node_id] = {}
            self.node_metrics[node_id][metric_name] = value
        elif link_id:
            if link_id not in self.link_metrics:
                self.link_metrics[link_id] = {}
            self.link_metrics[link_id][metric_name] = value
        else:
            self.metrics[metric_name] = value

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of key metrics
        weights = {
            "reliability": 0.3,
            "latency": 0.25,
            "throughput": 0.25,
            "efficiency": 0.2,
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in self.metrics:
                # Normalize metrics (assuming higher is better except for latency)
                if metric == "latency":
                    # Lower latency is better
                    normalized = max(
                        0, 1.0 - (self.metrics[metric] / 1000.0)
                    )  # Assume 1000ms max
                else:
                    # Higher values are better
                    normalized = min(1.0, self.metrics[metric])

                score += normalized * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify network bottlenecks."""
        bottlenecks = []

        # Check for high latency links
        for link_id, metrics in self.link_metrics.items():
            latency = metrics.get("latency_ms", 0)
            if latency > 100:  # High latency threshold
                bottlenecks.append(
                    {
                        "type": "high_latency",
                        "component": "link",
                        "component_id": link_id,
                        "metric": "latency_ms",
                        "value": latency,
                        "threshold": 100,
                    }
                )

        # Check for overloaded nodes
        for node_id, metrics in self.node_metrics.items():
            utilization = metrics.get("cpu_utilization", 0)
            if utilization > 80:  # High utilization threshold
                bottlenecks.append(
                    {
                        "type": "high_utilization",
                        "component": "node",
                        "component_id": node_id,
                        "metric": "cpu_utilization",
                        "value": utilization,
                        "threshold": 80,
                    }
                )

        return bottlenecks
