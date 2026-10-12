"""
Transportation planning components for the GEO-INFER-LOG module.

This module provides classes for multimodal transportation planning,
transportation network analysis, and emissions calculation.
"""

import logging
import pandas as pd
import networkx as nx
from typing import Any
import matplotlib.pyplot as plt

from geo_infer_log.models.schemas import VehicleType, FuelType, Vehicle, Route
from geo_infer_log.core.routing import _load_gpickle
from geo_infer_log.utils.geo import haversine_distance

logger = logging.getLogger(__name__)


class MultiModalPlanner:
    """Plans and optimizes multimodal transportation."""

    def __init__(self) -> None:
        """Initialize a multimodal transportation planner."""
        self.networks: dict[str, nx.Graph] = {}  # mode -> network graph
        # List of transfer points between modes
        self.transfer_points: list[dict[str, Any]] = []

    def load_network(self, mode: str, network_file: str) -> None:
        """Load a transportation network for a specific mode.

        Args:
            mode: Transportation mode
            network_file: Path to network file
        """
        # Load network from file
        network = _load_gpickle(network_file)
        self.networks[mode] = network

    def add_transfer_point(
        self,
        location: tuple[float, float],
        name: str,
        modes: list[str],
        transfer_time: dict[tuple[str, str], int],
    ) -> None:
        """Add a transfer point between transportation modes.

        Args:
            location: (lon, lat) of transfer point
            name: Name of transfer point
            modes: List of modes available at this point
            transfer_time: Dict of (from_mode, to_mode) -> transfer time in minutes
        """
        transfer_id = len(self.transfer_points)
        transfer_point = {
            "id": transfer_id,
            "location": location,
            "name": name,
            "modes": modes,
            "transfer_time": transfer_time,
        }
        self.transfer_points.append(transfer_point)

        # Connect transfer point to mode networks
        for mode in modes:
            if mode in self.networks:
                network = self.networks[mode]
                if network.nodes:
                    nearest_node = min(
                        network.nodes,
                        key=lambda n: (
                            (network.nodes[n].get("x", 0) - location[0]) ** 2
                            + (network.nodes[n].get("y", 0) - location[1]) ** 2
                        ),
                    )
                    network.add_node(
                        f"transfer_{transfer_id}_{mode}",
                        x=location[0],
                        y=location[1],
                        transfer=True,
                    )
                    network.add_edge(
                        nearest_node,
                        f"transfer_{transfer_id}_{mode}",
                        weight=min(transfer_time.values(), default=0),
                    )

    def plan_route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        allowed_modes: list[str],
        preferences: dict[str, Any] | None = None,
    ) -> dict:
        """Plan a multimodal route between origin and destination.

        Args:
            origin: (lon, lat) of origin
            destination: (lon, lat) of destination
            allowed_modes: List of allowed transportation modes
            preferences: Dict of routing preferences

        Returns:
            Dictionary with route information
        """
        if not preferences:
            preferences = {
                "cost_weight": 1.0,
                "time_weight": 1.0,
                "emissions_weight": 1.0,
                "transfers_weight": 1.0,
            }

        # Build multimodal graph
        multimodal_graph = self._build_multimodal_graph(allowed_modes)

        # Find nearest nodes to origin and destination
        origin_nodes = {}
        destination_nodes = {}

        for mode in allowed_modes:
            if mode in self.networks:
                net = self.networks[mode]

                # Nearest-node lookup via Haversine distance
                def _nearest(net: Any, lon: float, lat: float) -> Any | None:
                    best, best_d = None, float("inf")
                    for n, d in net.nodes(data=True):
                        nx_val = d.get("x", d.get("lon", 0))
                        ny_val = d.get("y", d.get("lat", 0))
                        dist = haversine_distance((lon, lat), (nx_val, ny_val))
                        if dist < best_d:
                            best, best_d = n, dist
                    return best

                origin_nodes[mode] = _nearest(net, origin[0], origin[1])
                destination_nodes[mode] = _nearest(net, destination[0], destination[1])

        # Build multimodal graph and find shortest path
        multimodal_graph = self._build_multimodal_graph(allowed_modes)

        # Map origin/destination into the prefixed multimodal graph
        best_route = None
        best_cost = float("inf")
        cost_attr = "weight"

        for o_mode, o_node in origin_nodes.items():
            for d_mode, d_node in destination_nodes.items():
                src = f"{o_mode}_{o_node}"
                dst = f"{d_mode}_{d_node}"
                if src not in multimodal_graph or dst not in multimodal_graph:
                    continue
                try:
                    path = nx.dijkstra_path(
                        multimodal_graph, src, dst, weight=cost_attr
                    )
                    cost = nx.dijkstra_path_length(
                        multimodal_graph, src, dst, weight=cost_attr
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_route = path
                except nx.NetworkXNoPath:
                    continue

        # Build route segments from the discovered path
        segments: list[dict[str, Any]] = []
        if best_route:
            current_mode = None
            seg_start = None
            seg_dist = 0.0
            for i, node in enumerate(best_route):
                node_data = multimodal_graph.nodes[node]
                mode = node_data.get("mode", "unknown")
                if mode != current_mode:
                    if current_mode is not None:
                        segments.append(
                            {
                                "mode": current_mode,
                                "origin": seg_start,
                                "destination": (
                                    node_data.get("x", 0),
                                    node_data.get("y", 0),
                                ),
                                "distance": seg_dist,
                                "time": seg_dist / 50.0 * 60,  # est. 50 km/h average
                                "cost": seg_dist * 0.3,
                                "emissions": seg_dist * 0.15,
                            }
                        )
                    current_mode = mode
                    seg_start = (node_data.get("x", 0), node_data.get("y", 0))
                    seg_dist = 0.0
                if i > 0:
                    edge_data = (
                        multimodal_graph.get_edge_data(best_route[i - 1], node) or {}
                    )
                    seg_dist += edge_data.get("distance", edge_data.get("weight", 1))
            # Final segment
            if current_mode and seg_start:
                last_data = multimodal_graph.nodes[best_route[-1]]
                segments.append(
                    {
                        "mode": current_mode,
                        "origin": seg_start,
                        "destination": (last_data.get("x", 0), last_data.get("y", 0)),
                        "distance": seg_dist,
                        "time": seg_dist / 50.0 * 60,
                        "cost": seg_dist * 0.3,
                        "emissions": seg_dist * 0.15,
                    }
                )
        else:
            # Direct great-circle fallback
            direct_dist = haversine_distance(origin, destination)
            segments.append(
                {
                    "mode": allowed_modes[0] if allowed_modes else "unknown",
                    "origin": origin,
                    "destination": destination,
                    "distance": direct_dist,
                    "time": direct_dist / 50.0 * 60,
                    "cost": direct_dist * 0.3,
                    "emissions": direct_dist * 0.15,
                }
            )

        total_distance = sum(s["distance"] for s in segments)
        total_time = sum(s["time"] for s in segments)
        total_cost = sum(s["cost"] for s in segments)
        total_emissions = sum(s["emissions"] for s in segments)

        return {
            "segments": segments,
            "total_distance": total_distance,
            "total_time": total_time,
            "total_cost": total_cost,
            "total_emissions": total_emissions,
            "num_transfers": len(segments) - 1,
        }

    def _build_multimodal_graph(self, modes: list[str]) -> nx.DiGraph:
        """Build a multimodal graph combining specified mode networks.

        Args:
            modes: List of transportation modes to include

        Returns:
            Multimodal directed graph
        """
        # Create a new graph for multimodal network
        multimodal_graph = nx.DiGraph()

        # Add all mode networks
        for mode in modes:
            if mode in self.networks:
                for node, data in self.networks[mode].nodes(data=True):
                    multimodal_graph.add_node(f"{mode}_{node}", **data, mode=mode)

                for u, v, data in self.networks[mode].edges(data=True):
                    multimodal_graph.add_edge(
                        f"{mode}_{u}", f"{mode}_{v}", **data, mode=mode
                    )

        # Add transfer edges between networks
        for transfer in self.transfer_points:
            modes = transfer["modes"]
            for from_mode in modes:
                for to_mode in modes:
                    if from_mode != to_mode:
                        transfer_time = transfer["transfer_time"].get(
                            (from_mode, to_mode), 15
                        )
                        transfer_node_from = f"transfer_{transfer['id']}_{from_mode}"
                        transfer_node_to = f"transfer_{transfer['id']}_{to_mode}"
                        if (
                            transfer_node_from in multimodal_graph
                            and transfer_node_to in multimodal_graph
                        ):
                            multimodal_graph.add_edge(
                                transfer_node_from,
                                transfer_node_to,
                                weight=transfer_time,
                                transfer=True,
                            )

        return multimodal_graph

    def compare_routes(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        mode_combinations: list[list[str]],
    ) -> pd.DataFrame:
        """Compare different multimodal routes between origin and destination.

        Args:
            origin: (lon, lat) of origin
            destination: (lon, lat) of destination
            mode_combinations: List of mode combinations to compare

        Returns:
            DataFrame with route comparisons
        """
        comparisons = []

        for modes in mode_combinations:
            route = self.plan_route(origin, destination, modes)

            comparisons.append(
                {
                    "modes": "-".join(modes),
                    "total_distance": route["total_distance"],
                    "total_time": route["total_time"],
                    "total_cost": route["total_cost"],
                    "total_emissions": route["total_emissions"],
                    "num_transfers": route["num_transfers"],
                }
            )

        return pd.DataFrame(comparisons)


class TransportationNetworkAnalyzer:
    """Analyzes transportation networks and flows."""

    def __init__(self) -> None:
        """Initialize a transportation network analyzer."""
        self.network: nx.Graph | None = None
        self.flow_data: Any | None = None

    def load_network(self, network_file: str) -> None:
        """Load a transportation network from a file.

        Args:
            network_file: Path to network file
        """
        self.network = _load_gpickle(network_file)

    def load_flow_data(self, flow_file: str) -> None:
        """Load transportation flow data from a file.

        Args:
            flow_file: Path to flow data file
        """
        # Auto-detect format from extension
        import os

        ext = os.path.splitext(flow_file)[1].lower()
        if ext == ".csv":
            self.flow_data = pd.read_csv(flow_file)
        elif ext in (".json", ".geojson"):
            self.flow_data = pd.read_json(flow_file)
        elif ext in (".parquet",):
            self.flow_data = pd.read_parquet(flow_file)
        else:
            # Attempt CSV as default
            self.flow_data = pd.read_csv(flow_file)
        logger.info(
            "Loaded flow data: %d records from %s", len(self.flow_data), flow_file
        )

    def calculate_network_metrics(self) -> dict:
        """Calculate metrics for the transportation network.

        Returns:
            Dictionary with network metrics
        """
        if not self.network:
            raise ValueError("Network must be loaded before calculating metrics")

        # Calculate basic network metrics
        metrics = {
            "num_nodes": self.network.number_of_nodes(),
            "num_edges": self.network.number_of_edges(),
            "density": nx.density(self.network),
            "diameter": (
                nx.diameter(self.network)
                if nx.is_strongly_connected(self.network)
                else float("inf")
            ),
            "average_shortest_path_length": (
                nx.average_shortest_path_length(self.network)
                if nx.is_strongly_connected(self.network)
                else float("inf")
            ),
            "average_degree": (
                sum(dict(self.network.degree()).values())
                / self.network.number_of_nodes()
                if self.network.number_of_nodes() > 0
                else 0
            ),
        }

        # Calculate centrality measures
        centrality = nx.betweenness_centrality(self.network)
        metrics["max_betweenness_centrality"] = (
            max(centrality.values()) if centrality else 0
        )
        metrics["average_betweenness_centrality"] = (
            sum(centrality.values()) / len(centrality) if centrality else 0
        )

        return metrics

    def identify_critical_links(self, top_n: int = 10) -> list[tuple[str, str]]:
        """Identify critical links in the transportation network.

        Args:
            top_n: Number of critical links to identify

        Returns:
            List of critical link tuples (u, v)
        """
        if not self.network:
            raise ValueError("Network must be loaded before identifying critical links")

        # Calculate edge betweenness centrality
        edge_centrality = nx.edge_betweenness_centrality(self.network)

        # Sort by centrality and return top links
        critical_links = sorted(
            edge_centrality.items(), key=lambda x: x[1], reverse=True
        )
        return [link for link, _ in critical_links[:top_n]]

    def _edge_flows(self) -> dict[tuple[Any, Any], float]:
        """Aggregate flow records into per-edge totals.

        Expects a DataFrame with columns like 'origin'/'source'/'from',
        'destination'/'target'/'to', and 'flow'/'volume'/'count'.

        Returns:
            Mapping of (origin, destination) to total flow
        """
        edge_flows: dict[tuple[Any, Any], float] = {}
        if isinstance(self.flow_data, pd.DataFrame) and not self.flow_data.empty:
            origin_col = next(
                (
                    c
                    for c in self.flow_data.columns
                    if c in ("origin", "source", "from")
                ),
                None,
            )
            dest_col = next(
                (
                    c
                    for c in self.flow_data.columns
                    if c in ("destination", "target", "to")
                ),
                None,
            )
            flow_col = next(
                (c for c in self.flow_data.columns if c in ("flow", "volume", "count")),
                None,
            )
            if origin_col and dest_col and flow_col:
                for _, row in self.flow_data.iterrows():
                    key = (row[origin_col], row[dest_col])
                    edge_flows[key] = edge_flows.get(key, 0) + row[flow_col]
        return edge_flows

    def analyze_flow(self) -> dict:
        """Analyze transportation flow in the network.

        Returns:
            Dictionary with flow analysis results
        """
        if self.network is None or self.flow_data is None:
            raise ValueError("Network and flow data must be loaded before analysis")

        edge_flows = self._edge_flows()
        if edge_flows:
            total_flow = sum(edge_flows.values())
            max_flow_edge = (
                max(edge_flows, key=lambda k: edge_flows[k]) if edge_flows else None
            )
            max_flow_val = edge_flows[max_flow_edge] if max_flow_edge else 0

            # Identify edges where flow > 80% of capacity
            congestion_points = []
            for (u, v), flow in edge_flows.items():
                if self.network.has_edge(u, v):
                    capacity = self.network[u][v].get("capacity", float("inf"))
                    if capacity > 0 and flow / capacity > 0.8:
                        congestion_points.append(
                            {
                                "edge": (u, v),
                                "flow": flow,
                                "capacity": capacity,
                                "utilization": flow / capacity,
                            }
                        )

            return {
                "total_flow": total_flow,
                "max_flow": max_flow_val,
                "max_flow_edge": max_flow_edge,
                "num_edges_with_flow": len(edge_flows),
                "congestion_points": congestion_points,
            }

        # Fallback: compute max-flow between first/last node if we have capacity data
        nodes = list(self.network.nodes())
        if len(nodes) >= 2:
            try:
                max_flow_val, flow_dict = nx.maximum_flow(
                    self.network, nodes[0], nodes[-1], capacity="capacity"
                )
                return {
                    "total_flow": max_flow_val,
                    "max_flow": max_flow_val,
                    "congestion_points": [],
                }
            except (nx.NetworkXError, nx.NetworkXUnbounded):
                pass

        return {
            "total_flow": 0,
            "max_flow": 0,
            "congestion_points": [],
        }

    def visualize_network(
        self, with_flow: bool = False, highlight_critical: bool = False
    ) -> None:
        """Visualize the transportation network.

        Args:
            with_flow: Whether to visualize flow data
            highlight_critical: Whether to highlight critical links
        """
        if not self.network:
            raise ValueError("Network must be loaded before visualization")

        # Create a plot
        plt.figure(figsize=(12, 10))

        # Draw the network
        pos = nx.spring_layout(self.network)
        nx.draw_networkx_nodes(self.network, pos, node_size=50)
        nx.draw_networkx_edges(self.network, pos, width=1, alpha=0.5)

        if highlight_critical:
            # Highlight critical links
            critical_links = self.identify_critical_links()
            nx.draw_networkx_edges(
                self.network, pos, edgelist=critical_links, width=3, edge_color="red"
            )

        if with_flow:
            edge_flows = self._edge_flows()
            max_flow = max(edge_flows.values(), default=0.0)
            edge_widths = []
            for u, v in self.network.edges():
                flow = edge_flows.get((u, v), 0.0)
                edge_widths.append(
                    max(0.5, flow / max_flow * 5) if max_flow > 0 else 0.5
                )
            nx.draw_networkx_edges(
                self.network, pos, width=edge_widths, alpha=0.6, edge_color="blue"
            )

        plt.title("Transportation Network")
        plt.axis("off")
        plt.tight_layout()


class TrafficSimulator:
    """Simulates traffic patterns and congestion."""

    def __init__(self, network: nx.DiGraph | None = None):
        """Initialize a traffic simulator.

        Args:
            network: Transportation network graph
        """
        self.network = network
        self.time_periods: list[str] = []
        # (u, v) -> speed by time period
        self.edge_speeds: dict[tuple[str, str], dict[str, float]] = {}

    def load_network(self, network_file: str) -> None:
        """Load a transportation network from a file.

        Args:
            network_file: Path to network file
        """
        self.network = _load_gpickle(network_file)

    def set_time_periods(self, periods: list[str]) -> None:
        """Set time periods for traffic simulation.

        Args:
            periods: List of time period labels
        """
        self.time_periods = periods

        # Initialize speeds for all edges and time periods
        if self.network:
            for u, v in self.network.edges():
                self.edge_speeds[(u, v)] = {period: 0 for period in periods}

    def set_edge_speeds(self, edge: tuple[str, str], speeds: dict[str, float]) -> None:
        """Set speeds for an edge by time period.

        Args:
            edge: Edge tuple (u, v)
            speeds: Dict of time period -> speed in km/h
        """
        assert self.network is not None
        if edge not in self.network.edges():
            raise ValueError(f"Edge {edge} not in network")

        for period, speed in speeds.items():
            if period not in self.time_periods:
                raise ValueError(f"Time period {period} not defined")

            self.edge_speeds[edge][period] = speed

    def simulate_traffic(
        self, origin: str, destination: str, departure_time: str
    ) -> dict:
        """Simulate traffic for a route from origin to destination.

        Args:
            origin: Origin node
            destination: Destination node
            departure_time: Departure time period

        Returns:
            Dictionary with simulation results
        """
        if not self.network:
            raise ValueError("Network must be loaded before simulation")

        if departure_time not in self.time_periods:
            raise ValueError(f"Time period {departure_time} not defined")

        # Create a copy of the network with speeds for the departure time
        temp_network = self.network.copy()

        for u, v, data in temp_network.edges(data=True):
            speed = self.edge_speeds.get((u, v), {}).get(
                departure_time, data.get("free_flow_speed", 50)
            )

            # Calculate travel time based on speed and distance
            distance = data.get("distance", 1)
            travel_time = (
                (distance / speed) * 60 if speed > 0 else float("inf")
            )  # minutes

            temp_network[u][v]["travel_time"] = travel_time

        # Find shortest path based on travel time
        try:
            path = nx.shortest_path(
                temp_network, origin, destination, weight="travel_time"
            )
            travel_time = nx.shortest_path_length(
                temp_network, origin, destination, weight="travel_time"
            )
            distance = sum(
                temp_network[u][v].get("distance", 0)
                for u, v in zip(path[:-1], path[1:])
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
            travel_time = float("inf")
            distance = 0

        return {
            "path": path,
            "travel_time": travel_time,
            "distance": distance,
            "departure_time": departure_time,
            "estimated_arrival_time": None,  # Would calculate based on departure time
        }

    def analyze_congestion(
        self, time_period: str | None = None, congestion_threshold: float = 0.7
    ) -> dict:
        """Analyze network congestion.

        Args:
            time_period: Time period to analyze (None for all periods)
            congestion_threshold: Threshold ratio of flow/capacity for congestion

        Returns:
            Dictionary with congestion analysis results
        """
        if not self.network:
            raise ValueError("Network must be loaded before analysis")

        periods = [time_period] if time_period else self.time_periods

        congestion_results = {}
        for period in periods:
            congested_edges = []
            total_edges = 0
            for u, v, data in self.network.edges(data=True):
                total_edges += 1
                free_flow = data.get("free_flow_speed", 60)
                current_speed = self.edge_speeds.get((u, v), {}).get(period, free_flow)
                if free_flow > 0:
                    speed_ratio = current_speed / free_flow
                    if speed_ratio <= congestion_threshold:
                        congested_edges.append(
                            {
                                "edge": (u, v),
                                "free_flow_speed": free_flow,
                                "current_speed": current_speed,
                                "speed_ratio": speed_ratio,
                            }
                        )
            congestion_ratio = (
                len(congested_edges) / total_edges if total_edges > 0 else 0.0
            )
            congestion_results[period] = {
                "congested_edges": congested_edges,
                "congestion_ratio": congestion_ratio,
                "total_edges": total_edges,
            }

        return congestion_results


class EmissionsCalculator:
    """Calculates transportation emissions."""

    def __init__(self) -> None:
        """Initialize an emissions calculator."""
        # Default emissions factors by vehicle type and fuel type (kg CO2e per km)
        self.emissions_factors: dict[tuple[VehicleType, FuelType | None], float] = {
            (VehicleType.TRUCK, FuelType.DIESEL): 0.9,
            (VehicleType.TRUCK, FuelType.ELECTRIC): 0.2,
            (VehicleType.VAN, FuelType.DIESEL): 0.5,
            (VehicleType.VAN, FuelType.ELECTRIC): 0.15,
            (VehicleType.CAR, FuelType.GASOLINE): 0.2,
            (VehicleType.CAR, FuelType.ELECTRIC): 0.1,
            (VehicleType.BIKE, FuelType.ELECTRIC): 0.01,
            (VehicleType.BIKE, None): 0.0,
            (VehicleType.TRAIN, FuelType.ELECTRIC): 0.05,
            (VehicleType.SHIP, FuelType.DIESEL): 0.4,
            (VehicleType.AIRPLANE, FuelType.JET_FUEL): 2.0,
        }

    def set_emissions_factor(
        self, vehicle_type: VehicleType, fuel_type: FuelType | None, factor: float
    ) -> None:
        """Set an emissions factor for a vehicle and fuel type.

        Args:
            vehicle_type: Type of vehicle
            fuel_type: Type of fuel (None for non-motorized)
            factor: Emissions factor in kg CO2e per km
        """
        self.emissions_factors[(vehicle_type, fuel_type)] = factor

    def calculate_route_emissions(
        self,
        vehicle: Vehicle,
        distance: float,
        load_factor: float = 1.0,
        terrain_factor: float = 1.0,
    ) -> float:
        """Calculate emissions for a route with a specific vehicle.

        Args:
            vehicle: Vehicle to use for calculation
            distance: Route distance in km
            load_factor: Factor for vehicle load (1.0 = full load)
            terrain_factor: Factor for terrain (1.0 = flat terrain)

        Returns:
            Emissions in kg CO2e
        """
        # Get base emissions factor
        base_factor = self.emissions_factors.get(
            (vehicle.type, vehicle.fuel_type),
            vehicle.emissions_per_km,  # Fallback to vehicle's own factor
        )

        # Apply adjustment factors
        adjusted_factor = base_factor * load_factor * terrain_factor

        # Calculate total emissions
        emissions = distance * adjusted_factor

        return emissions

    def compare_emissions(
        self, route: dict, vehicle_options: list[Vehicle]
    ) -> pd.DataFrame:
        """Compare emissions for different vehicle options on a route.

        Args:
            route: Route information with distance
            vehicle_options: List of vehicle options to compare

        Returns:
            DataFrame with emissions comparison
        """
        distance = route.get("distance", 0)

        comparisons = []
        for vehicle in vehicle_options:
            emissions = self.calculate_route_emissions(vehicle, distance)

            comparisons.append(
                {
                    "vehicle_id": vehicle.id,
                    "vehicle_type": vehicle.type,
                    "fuel_type": vehicle.fuel_type,
                    "emissions": emissions,
                    "emissions_per_km": emissions / distance if distance > 0 else 0,
                }
            )

        return pd.DataFrame(comparisons)

    def calculate_fleet_emissions(
        self, fleet: list[Vehicle], routes: list[Route]
    ) -> dict:
        """Calculate total emissions for a fleet of vehicles.

        Args:
            fleet: List of vehicles in the fleet
            routes: List of routes assigned to vehicles

        Returns:
            Dictionary with emissions statistics
        """
        vehicle_map = {vehicle.id: vehicle for vehicle in fleet}

        emissions = []
        for route in routes:
            if route.vehicle_id in vehicle_map:
                vehicle = vehicle_map[route.vehicle_id]
                route_emissions = self.calculate_route_emissions(
                    vehicle, route.total_distance
                )

                emissions.append(
                    {
                        "route_id": route.id,
                        "vehicle_id": vehicle.id,
                        "distance": route.total_distance,
                        "emissions": route_emissions,
                    }
                )

        emissions_df = pd.DataFrame(emissions) if emissions else pd.DataFrame()

        return {
            "total_emissions": (
                emissions_df["emissions"].sum() if not emissions_df.empty else 0
            ),
            "total_distance": (
                emissions_df["distance"].sum() if not emissions_df.empty else 0
            ),
            "average_emissions_per_km": (
                emissions_df["emissions"].sum() / emissions_df["distance"].sum()
                if not emissions_df.empty and emissions_df["distance"].sum() > 0
                else 0
            ),
            "emissions_by_vehicle": (
                emissions_df.groupby("vehicle_id")["emissions"].sum().to_dict()
                if not emissions_df.empty
                else {}
            ),
        }
