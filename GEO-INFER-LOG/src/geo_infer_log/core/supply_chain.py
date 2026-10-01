"""
Supply chain modeling components for the GEO-INFER-LOG module.

This module provides classes for supply chain network design,
resilience analysis, facility location, and inventory management.
"""

import logging
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pulp
from shapely.geometry import LineString

from geo_infer_log.models.schemas import SupplyChainNetwork
from geo_infer_log.utils.geo import haversine_distance

logger = logging.getLogger(__name__)


class SupplyChainModel:
    """Base class for supply chain network modeling."""

    def __init__(self, network: SupplyChainNetwork | None = None):
        """Initialize a supply chain model.

        Args:
            network: Supply chain network to model
        """
        self.network = network
        self.graph: nx.DiGraph | None = None

    def load_network(self, network: SupplyChainNetwork) -> None:
        """Load a supply chain network.

        Args:
            network: Supply chain network to load
        """
        self.network = network
        self._build_graph()

    def _build_graph(self) -> None:
        """Build a graph representation of the supply chain network."""
        if not self.network:
            return

        self.graph = nx.DiGraph()

        # Add facilities as nodes
        for facility in self.network.facilities:
            self.graph.add_node(
                facility.id,
                name=facility.name,
                location=facility.location,
                type=facility.type,
                capacity=facility.capacity,
                operating_cost=facility.operating_cost,
            )

        # Add links as edges
        for link in self.network.links:
            self.graph.add_edge(
                link["from"],
                link["to"],
                distance=link["distance"],
                time=link["time"],
                cost=link["cost"],
                capacity=link.get("capacity", float("inf")),
            )

    def optimize_flow(
        self,
        demand_points: list[dict],
        supply_points: list[dict],
        objective: str = "cost",
    ) -> dict:
        """Optimize flow in the supply chain network.

        Args:
            demand_points: List of demand points with quantities
            supply_points: List of supply points with quantities
            objective: Objective function ('cost', 'time', 'distance')

        Returns:
            Dictionary with optimized flow information
        """
        if not self.graph:
            raise ValueError("Network graph must be built before optimization")

        # Create optimization model
        model = pulp.LpProblem("SupplyChainFlow", pulp.LpMinimize)

        # Create decision variables for flow on each edge
        edges = list(self.graph.edges(data=True))
        flow_vars = {}
        for u, v, data in edges:
            flow_vars[(u, v)] = pulp.LpVariable(
                f"flow_{u}_{v}", lowBound=0, upBound=data.get("capacity", None)
            )

        # Objective: minimize total cost/time/distance
        obj_attr = objective if objective in ("cost", "time", "distance") else "cost"
        model += pulp.lpSum(
            flow_vars[(u, v)] * data.get(obj_attr, 1) for u, v, data in edges
        )

        # Supply constraints
        supply_map = {sp["id"]: sp.get("quantity", 0) for sp in supply_points}
        demand_map = {dp["id"]: dp.get("quantity", 0) for dp in demand_points}

        for node in self.graph.nodes():
            inflow = pulp.lpSum(
                flow_vars.get((u, node), 0) for u in self.graph.predecessors(node)
            )
            outflow = pulp.lpSum(
                flow_vars.get((node, v), 0) for v in self.graph.successors(node)
            )
            supply = supply_map.get(node, 0)
            demand = demand_map.get(node, 0)
            model += (inflow + supply - outflow - demand == 0, f"balance_{node}")

        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract results
        flows = []
        total_cost = 0.0
        total_time = 0.0
        total_distance = 0.0
        for u, v, data in edges:
            val = flow_vars[(u, v)].varValue or 0.0
            if val > 0:
                flows.append({"from": u, "to": v, "flow": val})
                total_cost += val * data.get("cost", 0)
                total_time += val * data.get("time", 0)
                total_distance += val * data.get("distance", 0)

        logger.info(
            "Optimized supply chain flow: %d active links, status=%s",
            len(flows),
            pulp.LpStatus[model.status],
        )
        return {
            "status": pulp.LpStatus[model.status],
            "total_cost": total_cost,
            "total_time": total_time,
            "total_distance": total_distance,
            "flows": flows,
        }

    def visualize_network(self) -> gpd.GeoDataFrame:
        """Visualize the supply chain network.

        Returns:
            GeoDataFrame with network visualization
        """
        if not self.network:
            raise ValueError("No network loaded")

        # Create nodes GeoDataFrame
        nodes = []
        for facility in self.network.facilities:
            nodes.append(
                {
                    "id": facility.id,
                    "name": facility.name,
                    "type": facility.type,
                    "capacity": facility.capacity,
                    "geometry": gpd.points_from_xy(
                        [facility.location[0]], [facility.location[1]]
                    )[0],
                }
            )

        nodes_gdf = gpd.GeoDataFrame(nodes)

        # Create edges GeoDataFrame with LineString geometries
        edge_rows = []
        for link in self.network.links:
            from_fac = next(
                (f for f in self.network.facilities if f.id == link["from"]), None
            )
            to_fac = next(
                (f for f in self.network.facilities if f.id == link["to"]), None
            )
            if from_fac and to_fac:
                line = LineString(
                    [
                        (from_fac.location[0], from_fac.location[1]),
                        (to_fac.location[0], to_fac.location[1]),
                    ]
                )
                edge_rows.append(
                    {
                        "from": link["from"],
                        "to": link["to"],
                        "distance": link.get("distance", 0),
                        "cost": link.get("cost", 0),
                        "geometry": line,
                    }
                )

        if edge_rows:
            edges_gdf = gpd.GeoDataFrame(edge_rows)
            return pd.concat([nodes_gdf, edges_gdf], ignore_index=True)

        return nodes_gdf


class ResilienceAnalyzer:
    """Analyzes and improves supply chain resilience."""

    def __init__(self, supply_chain_model: SupplyChainModel):
        """Initialize a resilience analyzer.

        Args:
            supply_chain_model: Supply chain model to analyze
        """
        self.supply_chain_model = supply_chain_model

    def identify_critical_nodes(self) -> list[str]:
        """Identify critical nodes in the supply chain.

        Returns:
            List of critical node IDs
        """
        if not self.supply_chain_model.graph:
            raise ValueError("Supply chain model must have a graph")

        # Use centrality measures to identify critical nodes
        centrality = nx.betweenness_centrality(self.supply_chain_model.graph)

        # Sort by centrality and return top nodes
        critical_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in critical_nodes[:5]]

    def simulate_disruption(
        self, disrupted_nodes: list[str], disrupted_edges: list[tuple[str, str]]
    ) -> dict:
        """Simulate a disruption in the supply chain.

        Args:
            disrupted_nodes: List of node IDs that are disrupted
            disrupted_edges: List of edge tuples that are disrupted

        Returns:
            Dictionary with disruption impact metrics
        """
        if not self.supply_chain_model.graph:
            raise ValueError("Supply chain model must have a graph")

        # Create a copy of the graph
        g = self.supply_chain_model.graph.copy()

        # Remove disrupted nodes and edges
        g.remove_nodes_from(disrupted_nodes)
        g.remove_edges_from(disrupted_edges)

        # Analyze connectivity
        connected = nx.is_strongly_connected(g)
        components = list(nx.strongly_connected_components(g))

        # Calculate impact metrics
        impact = {
            "network_connected": connected,
            "components": len(components),
            "largest_component_size": (
                len(max(components, key=len)) if components else 0
            ),
            "connectivity_ratio": (
                len(max(components, key=len)) / g.number_of_nodes()
                if g.number_of_nodes() > 0
                else 0
            ),
        }

        return impact

    def suggest_improvements(self) -> list[dict]:
        """Suggest improvements to increase supply chain resilience.

        Returns:
            List of improvement suggestions
        """
        if not self.supply_chain_model.graph:
            raise ValueError("Supply chain model must have a graph")

        g = self.supply_chain_model.graph
        suggestions: list[dict] = []

        # 1. Identify articulation points (single-point-of-failure nodes)
        undirected = g.to_undirected()
        try:
            aps = list(nx.articulation_points(undirected))
        except nx.NetworkXError:
            aps = []
        for node in aps:
            suggestions.append(
                {
                    "type": "redundancy",
                    "action": f"Add backup link/supplier bypassing node '{node}' — it is an articulation point",
                    "location": node,
                    "impact": "high",
                }
            )

        # 2. Identify bridge edges
        try:
            bridges = list(nx.bridges(undirected))
        except nx.NetworkXError:
            bridges = []
        for u, v in bridges:
            suggestions.append(
                {
                    "type": "redundancy",
                    "action": f"Add alternate route for bridge edge ({u} → {v})",
                    "location": f"{u}-{v}",
                    "impact": "high",
                }
            )

        # 3. Low-degree nodes (single source/sink)
        for node in g.nodes():
            in_deg = g.in_degree(node)
            out_deg = g.out_degree(node)
            if in_deg == 1 and out_deg >= 1:
                suggestions.append(
                    {
                        "type": "inventory",
                        "action": f"Increase safety stock at '{node}' — single supplier",
                        "location": node,
                        "impact": "medium",
                    }
                )

        logger.info("Generated %d resilience improvement suggestions", len(suggestions))
        return suggestions


class NetworkOptimizer:
    """Optimizes supply chain network design."""

    def __init__(self, supply_chain_model: SupplyChainModel | None = None):
        """Initialize a network optimizer.

        Args:
            supply_chain_model: Supply chain model to optimize
        """
        self.supply_chain_model = supply_chain_model

    def optimize_network(
        self, locations: list[dict], demand_points: list[dict], constraints: dict
    ) -> dict:
        """Optimize the supply chain network design.

        Args:
            locations: Potential facility locations
            demand_points: Customer demand points
            constraints: Optimization constraints

        Returns:
            Dictionary with optimized network design
        """
        # Facility location via PuLP MILP (p-median)
        n_locs = len(locations)
        n_demand = len(demand_points)
        max_facilities = constraints.get("max_facilities", n_locs)
        budget = constraints.get("budget", float("inf"))

        model = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)

        # Binary variables: open facility j
        y = [pulp.LpVariable(f"open_{j}", cat="Binary") for j in range(n_locs)]
        # Assignment variables: demand i served by facility j
        x = [
            [
                pulp.LpVariable(f"assign_{i}_{j}", lowBound=0, upBound=1)
                for j in range(n_locs)
            ]
            for i in range(n_demand)
        ]

        # Distance matrix
        dist = np.zeros((n_demand, n_locs))
        for i, dp in enumerate(demand_points):
            for j, loc in enumerate(locations):
                dp_coord = dp.get("location", (0, 0))
                loc_coord = loc.get("location", (0, 0))
                dist[i][j] = haversine_distance(dp_coord, loc_coord)

        # Objective: minimize weighted distance
        model += pulp.lpSum(
            dp.get("demand", 1) * dist[i][j] * x[i][j]
            for i, dp in enumerate(demand_points)
            for j in range(n_locs)
        )

        # Constraints
        for i in range(n_demand):
            model += (
                pulp.lpSum(x[i][j] for j in range(n_locs)) == 1
            )  # each demand assigned
        for i in range(n_demand):
            for j in range(n_locs):
                model += x[i][j] <= y[j]  # only assign to open facility
        model += pulp.lpSum(y) <= max_facilities
        model += (
            pulp.lpSum(y[j] * locations[j].get("fixed_cost", 0) for j in range(n_locs))
            <= budget
        )

        model.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [
            locations[j] for j in range(n_locs) if y[j].varValue and y[j].varValue > 0.5
        ]
        links = []
        for i, dp in enumerate(demand_points):
            for j in range(n_locs):
                if x[i][j].varValue and x[i][j].varValue > 0.5:
                    links.append(
                        {
                            "from": locations[j].get("id", j),
                            "to": dp.get("id", i),
                            "distance": dist[i][j],
                        }
                    )

        total_cost = sum(loc.get("fixed_cost", 0) for loc in selected)
        service_level = len(
            [
                link
                for link in links
                if link["distance"] <= constraints.get("max_distance", float("inf"))
            ]
        ) / max(n_demand, 1)

        logger.info("Optimized network: %d facilities selected", len(selected))
        return {
            "selected_facilities": selected,
            "links": links,
            "total_cost": total_cost,
            "service_level": service_level,
        }

    def evaluate_design(self, network: SupplyChainNetwork) -> dict:
        """Evaluate a supply chain network design.

        Args:
            network: Supply chain network to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate using real graph metrics
        temp_model = SupplyChainModel(network)
        temp_model._build_graph()
        g = temp_model.graph

        if not g or g.number_of_nodes() == 0:
            return {
                "total_cost": 0,
                "service_level": 0,
                "average_distance": 0,
                "resilience_score": 0,
            }

        total_cost = sum(d.get("operating_cost", 0) for _, d in g.nodes(data=True))
        total_cost += sum(d.get("cost", 0) for _, _, d in g.edges(data=True))

        distances = [d.get("distance", 0) for _, _, d in g.edges(data=True)]
        avg_dist = np.mean(distances) if distances else 0.0

        # Resilience: ratio of edge connectivity to nodes
        try:
            edge_conn = nx.edge_connectivity(g)
        except nx.NetworkXError:
            edge_conn = 0
        resilience = edge_conn / max(g.number_of_nodes(), 1)

        return {
            "total_cost": total_cost,
            "service_level": 1.0 if nx.is_weakly_connected(g) else 0.5,
            "average_distance": float(avg_dist),
            "resilience_score": float(resilience),
        }


class FacilityLocator:
    """Optimizes facility locations in supply chains."""

    def __init__(self) -> None:
        """Initialize a facility locator."""
        self.selected_facilities: list[dict] = []
        self.coverage_results: dict = {}

    def locate_facilities(
        self,
        candidates: list[dict],
        demand_points: list[dict],
        num_facilities: int,
        max_distance: float | None = None,
    ) -> list[dict]:
        """Determine optimal facility locations.

        Args:
            candidates: Candidate facility locations
            demand_points: Customer demand points
            num_facilities: Number of facilities to locate
            max_distance: Maximum distance constraint

        Returns:
            List of selected facility locations
        """
        # P-median via PuLP MILP
        n_cand = len(candidates)
        n_dem = len(demand_points)

        model = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)
        y = [pulp.LpVariable(f"open_{j}", cat="Binary") for j in range(n_cand)]
        x = [
            [
                pulp.LpVariable(f"assign_{i}_{j}", lowBound=0, upBound=1)
                for j in range(n_cand)
            ]
            for i in range(n_dem)
        ]

        dist = np.zeros((n_dem, n_cand))
        for i, dp in enumerate(demand_points):
            for j, cand in enumerate(candidates):
                dp_loc = dp.get("location", (0, 0))
                c_loc = cand.get("location", (0, 0))
                dist[i][j] = haversine_distance(dp_loc, c_loc)

        # Objective: minimize demand-weighted distance
        model += pulp.lpSum(
            dp.get("demand", 1) * dist[i][j] * x[i][j]
            for i, dp in enumerate(demand_points)
            for j in range(n_cand)
        )

        for i in range(n_dem):
            model += pulp.lpSum(x[i][j] for j in range(n_cand)) == 1
        for i in range(n_dem):
            for j in range(n_cand):
                model += x[i][j] <= y[j]
                if max_distance is not None:
                    if dist[i][j] > max_distance:
                        model += x[i][j] == 0
        model += pulp.lpSum(y) == num_facilities

        solver = pulp.PULP_CBC_CMD(msg=0)
        try:
            model.solve(solver)
            self.selected_facilities = [
                candidates[j]
                for j in range(n_cand)
                if y[j].varValue and y[j].varValue > 0.5
            ]
        except (OSError, pulp.PulpSolverError):
            # The bundled CBC executable is platform-specific. Keep this
            # library path deterministic with a local weighted-distance
            # selector when that optional binary cannot execute.
            scores = [
                sum(
                    dp.get("demand", 1) * dist[i][j]
                    for i, dp in enumerate(demand_points)
                )
                for j in range(n_cand)
            ]
            selected = np.argsort(scores)[:num_facilities]
            self.selected_facilities = [candidates[int(j)] for j in selected]
        logger.info("Located %d facilities via p-median", len(self.selected_facilities))
        return self.selected_facilities

    def analyze_coverage(
        self, facilities: list[dict], demand_points: list[dict], max_distance: float
    ) -> dict:
        """Analyze coverage of demand points by facilities.

        Args:
            facilities: Facility locations
            demand_points: Customer demand points
            max_distance: Maximum service distance

        Returns:
            Dictionary with coverage analysis
        """
        # Real Haversine-based coverage analysis
        covered = 0
        total_demand = 0
        distances = []

        for dp in demand_points:
            dp_loc = dp.get("location", (0, 0))
            min_dist = float("inf")
            for fac in facilities:
                fac_loc = fac.get("location", (0, 0))
                d = haversine_distance(dp_loc, fac_loc)
                min_dist = min(min_dist, d)
            distances.append(min_dist)
            demand_qty = dp.get("demand", 1)
            total_demand += demand_qty
            if min_dist <= max_distance:
                covered += demand_qty

        self.coverage_results = {
            "covered_points": covered,
            "coverage_ratio": covered / total_demand if total_demand > 0 else 0.0,
            "average_distance": float(np.mean(distances)) if distances else 0.0,
            "max_distance_observed": float(np.max(distances)) if distances else 0.0,
        }
        return self.coverage_results


class InventoryManager:
    """Manages inventory in supply chain networks."""

    def __init__(self) -> None:
        """Initialize an inventory manager."""
        self.inventory_levels: dict = {}
        self.reorder_points: dict = {}

    def optimize_inventory(
        self,
        facilities: list[dict],
        demand_data: dict,
        lead_times: dict,
        service_level: float = 0.95,
    ) -> dict:
        """Optimize inventory levels across facilities.

        Args:
            facilities: Facility information
            demand_data: Historical demand data
            lead_times: Supplier lead times
            service_level: Target service level

        Returns:
            Dictionary with optimized inventory levels
        """
        # EOQ (Economic Order Quantity) and safety stock optimization
        from scipy import stats

        safety_stocks = {}
        reorder_points = {}
        order_quantities = {}
        total_cost = 0.0

        z_score = stats.norm.ppf(service_level)  # e.g., 1.645 for 95%

        for fac in facilities:
            fac_id = fac.get("id", str(fac))
            demand_series = demand_data.get(fac_id, [])
            if not demand_series:
                continue

            mean_demand = float(np.mean(demand_series))
            std_demand = float(np.std(demand_series))
            lead_time = lead_times.get(fac_id, 1)  # periods

            # Safety stock = z * σ_demand * √(lead_time)
            ss = z_score * std_demand * math.sqrt(lead_time)
            safety_stocks[fac_id] = ss

            # Reorder point = mean_demand * lead_time + safety_stock
            rop = mean_demand * lead_time + ss
            reorder_points[fac_id] = rop
            self.reorder_points[fac_id] = rop

            # EOQ = √(2 * D * S / H) where S=ordering cost, H=holding cost
            annual_demand = mean_demand * 365
            ordering_cost = fac.get("ordering_cost", 100)
            holding_cost = fac.get("holding_cost", 10)
            eoq = math.sqrt(2 * annual_demand * ordering_cost / max(holding_cost, 0.01))
            order_quantities[fac_id] = eoq

            # Annual cost = ordering + holding + safety stock holding
            annual_ordering = (annual_demand / max(eoq, 1)) * ordering_cost
            annual_holding = (eoq / 2 + ss) * holding_cost
            total_cost += annual_ordering + annual_holding

            self.inventory_levels[fac_id] = eoq / 2 + ss

        logger.info(
            "Optimized inventory for %d facilities, total cost=%.0f",
            len(safety_stocks),
            total_cost,
        )
        return {
            "safety_stocks": safety_stocks,
            "reorder_points": reorder_points,
            "order_quantities": order_quantities,
            "total_inventory_cost": total_cost,
        }

    def simulate_inventory_policy(
        self, policy: dict, demand_data: dict, lead_times: dict, simulation_period: int
    ) -> dict:
        """Simulate an inventory policy.

        Args:
            policy: Inventory policy parameters
            demand_data: Historical demand data
            lead_times: Supplier lead times
            simulation_period: Number of periods to simulate

        Returns:
            Dictionary with simulation results
        """
        # Monte Carlo inventory simulation
        rng = np.random.default_rng(42)

        reorder_point = policy.get("reorder_point", 100)
        order_quantity = policy.get("order_quantity", 200)
        initial_inventory = policy.get("initial_inventory", order_quantity)

        inventory = initial_inventory
        on_order = 0
        pending_orders: list = []  # (arrival_period, quantity)
        stockouts = 0
        inventory_history = []

        # Use first facility's demand as baseline
        demand_key = next(iter(demand_data), None)
        demand_series = demand_data.get(demand_key, [100]) if demand_key else [100]
        mean_demand = float(np.mean(demand_series))
        std_demand = (
            float(np.std(demand_series))
            if len(demand_series) > 1
            else mean_demand * 0.2
        )

        lead_key = next(iter(lead_times), None)
        lead_time = lead_times.get(lead_key, 3) if lead_key else 3

        for period in range(simulation_period):
            # Receive pending orders
            arrived = [q for (t, q) in pending_orders if t <= period]
            pending_orders = [(t, q) for (t, q) in pending_orders if t > period]
            inventory += sum(arrived)
            on_order -= sum(arrived)

            # Generate stochastic demand
            demand = max(0, rng.normal(mean_demand, std_demand))
            inventory -= demand

            if inventory < 0:
                stockouts += 1
                inventory = 0  # lost sales model

            # Check reorder point
            if inventory <= reorder_point and on_order == 0:
                pending_orders.append((period + lead_time, order_quantity))
                on_order += order_quantity

            inventory_history.append(inventory)

        inv_arr = np.array(inventory_history)
        total_demand = mean_demand * simulation_period
        turns = (
            total_demand / max(np.mean(inv_arr), 0.01) if np.mean(inv_arr) > 0 else 0
        )

        return {
            "stockouts": stockouts,
            "average_inventory": float(np.mean(inv_arr)),
            "max_inventory": float(np.max(inv_arr)),
            "inventory_turns": float(turns),
            "service_level": 1.0 - stockouts / max(simulation_period, 1),
        }
