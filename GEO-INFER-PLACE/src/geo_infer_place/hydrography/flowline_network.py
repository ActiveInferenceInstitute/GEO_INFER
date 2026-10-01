"""
Flowline Network and Topology Validation for Pacific Northwest Rivers.

This module provides graph-based representation, topology validation,
traversal (upstream/downstream), Strahler stream order checks, and H3
spatial integration for NHDPlus High Resolution (NHDPlus HR) flowline networks.
"""

from __future__ import annotations

import logging
from itertools import islice
import math

from typing import Any, cast

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import MultiLineString

from geo_infer_space.utils.h3_utils import latlng_to_cell

logger = logging.getLogger(__name__)


def normalize_flowlines(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preserve USGS attributes and add canonical aliases used by the network API."""
    frame = frame.copy()
    if frame.crs is None:
        raise ValueError("Flowlines require an explicit CRS")
    frame = frame.to_crs("EPSG:4326")
    if not frame.empty:
        if (
            not frame.geometry.is_valid.all()
            or frame.geometry.is_empty.any()
            or not frame.geometry.geom_type.isin(
                ["LineString", "MultiLineString"]
            ).all()
        ):
            raise ValueError("Flowlines require valid nonempty linear geometries")
        west, south, east, north = frame.total_bounds
        if not all(
            math.isfinite(value) for value in (west, south, east, north)
        ) or not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("Flowline coordinates must lie within WGS84 bounds")
    aliases = {
        "comid": ("nhdplusid", "NHDPlusID", "COMID"),
        "from_node": ("fromnode", "FromNode"),
        "to_node": ("tonode", "ToNode"),
        "stream_order": ("streamorde", "StreamOrder", "StreamOrde"),
        "length_km": ("lengthkm", "LengthKM"),
        "drainage_area_sqkm": ("totdasqkm", "TotDASqKM"),
    }
    for target, sources in aliases.items():
        if target not in frame.columns:
            for source in sources:
                if source in frame.columns:
                    frame[target] = frame[source]
                    if target == "stream_order":
                        # USGS uses negative sentinels for unassigned order.
                        values = pd.to_numeric(frame[target], errors="raise")
                        frame[target] = values.mask(values < 0).astype("Int64")
                    break
    return frame


class FlowlineTopologyValidator:
    """Validates hydrological flowline network topology and consistency.

    Performs graph-level consistency checks:
    - Directed acyclic graph (DAG) verification (flow cannot cycle)
    - Node connectivity and dangling reach detection
    - Strahler stream order monotonicity (downstream order must be >= upstream order)
    - Outlet and headwater identification
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        """Initialize validator with a directed flowline graph.

        Args:
            graph: Directed NetworkX graph where edges represent flowlines
                   directed from upstream node to downstream node.
        """
        self.graph = graph

    def validate_is_dag(self) -> dict[str, Any]:
        """Validate that the flowline network is a directed acyclic graph (no loops)."""
        is_dag = nx.is_directed_acyclic_graph(self.graph)
        cycles: list[list[Any]] = []
        if not is_dag:
            try:
                cycles = list(islice(nx.simple_cycles(self.graph), 11))
            except Exception as e:
                logger.warning(f"Error computing simple cycles: {e}")
        return {
            "is_dag": is_dag,
            "has_cycles": not is_dag,
            "cycle_count": len(cycles),
            "cycle_count_is_lower_bound": len(cycles) > 10,
            "cycles": cycles[:10],
        }

    def validate_strahler_monotonicity(self) -> dict[str, Any]:
        """Verify that stream order is non-decreasing in the downstream direction.

        Along any valid flowline path (u -> v), stream_order(v) >= stream_order(u).
        """
        violations: list[dict[str, Any]] = []
        for u, v, data in self.graph.edges(data=True):
            u_order = data.get("stream_order")
            # Look at downstream edges from v
            downstream_edges = self.graph.out_edges(v, data=True)
            for _, w, ds_data in downstream_edges:
                w_order = ds_data.get("stream_order")
                if (
                    u_order is not None
                    and w_order is not None
                    and int(w_order) < int(u_order)
                ):
                    violations.append(
                        {
                            "upstream_comid": data.get("comid"),
                            "upstream_order": int(u_order),
                            "downstream_comid": ds_data.get("comid"),
                            "downstream_order": int(w_order),
                            "from_node": u,
                            "junction_node": v,
                            "to_node": w,
                        }
                    )
        return {
            "monotonic": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations,
        }

    def find_headwaters(self) -> list[Any]:
        """Find headwater nodes (nodes with in-degree == 0 and out-degree > 0)."""
        return [
            node
            for node in self.graph.nodes()
            if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) > 0
        ]

    def find_outlets(self) -> list[Any]:
        """Find terminal outlet nodes (nodes with out-degree == 0 and in-degree > 0)."""
        return [
            node
            for node in self.graph.nodes()
            if self.graph.out_degree(node) == 0 and self.graph.in_degree(node) > 0
        ]

    def validate_all(self) -> dict[str, Any]:
        """Run full battery of hydrological network validation checks."""
        dag_res = self.validate_is_dag()
        monotonic_res = self.validate_strahler_monotonicity()
        headwaters = self.find_headwaters()
        outlets = self.find_outlets()
        connected_components = list(nx.weakly_connected_components(self.graph))

        unknown_orders = sum(
            data.get("stream_order") is None
            for _, _, data in self.graph.edges(data=True)
        )
        return {
            "unknown_stream_order_count": unknown_orders,
            "valid": bool(self.graph.number_of_edges())
            and dag_res["is_dag"]
            and monotonic_res["monotonic"],
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "headwater_count": len(headwaters),
            "outlet_count": len(outlets),
            "component_count": len(connected_components),
            "dag_validation": dag_res,
            "strahler_monotonicity": monotonic_res,
            "headwaters": headwaters,
            "outlets": outlets,
        }


class CascadiaFlowlineNetwork:
    """Hydrological river flowline network for Cascadia and Pacific Northwest basins.

    Provides graph creation from NHDPlus HR GeoDataFrames or GeoJSON collections,
    topology traversal, drainage path finding, and spatial integration with H3 grids.
    """

    def __init__(self, flowlines_gdf: gpd.GeoDataFrame | None = None) -> None:
        """Initialize flowline network.

        Args:
            flowlines_gdf: Optional GeoDataFrame of NHDPlus HR flowlines.
        """
        self.flowlines_gdf = (
            normalize_flowlines(flowlines_gdf)
            if flowlines_gdf is not None
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
        self.graph = nx.MultiDiGraph()
        self._comid_to_edge: dict[int, tuple[Any, Any]] = {}
        if not self.flowlines_gdf.empty:
            self._build_graph()

    @classmethod
    def from_geojson(cls, path_or_dict: Any) -> CascadiaFlowlineNetwork:
        """Construct CascadiaFlowlineNetwork from GeoJSON file path or dict."""
        if isinstance(path_or_dict, str):
            gdf = gpd.read_file(path_or_dict)
        elif isinstance(path_or_dict, bytes):
            import io

            gdf = gpd.read_file(io.BytesIO(path_or_dict))
        elif hasattr(path_or_dict, "exists"):  # Path object
            gdf = gpd.read_file(str(path_or_dict))
        elif isinstance(path_or_dict, dict):
            gdf = gpd.GeoDataFrame.from_features(
                path_or_dict.get("features", []), crs="EPSG:4326"
            )
        else:
            raise ValueError(f"Unsupported GeoJSON source type: {type(path_or_dict)}")
        return cls(gdf)

    def _build_graph(self) -> None:
        """Build directed network graph from flowlines GeoDataFrame."""
        self.graph.clear()
        self._comid_to_edge.clear()

        required = {"comid", "from_node", "to_node", "stream_order", "length_km"}
        missing = required - set(self.flowlines_gdf.columns)
        if missing:
            raise ValueError(f"Flowline topology attributes missing: {sorted(missing)}")
        for _, row in self.flowlines_gdf.iterrows():
            if any(pd.isna(row[key]) for key in required - {"stream_order"}):
                raise ValueError("Null flowline ID, node, order or length")
            raw_id = row["comid"]
            comid = int(raw_id)
            if float(raw_id) != comid or comid in self._comid_to_edge:
                raise ValueError(f"Invalid or duplicate flowline ID: {raw_id}")
            from_node, to_node = row["from_node"], row["to_node"]
            length_km = float(row["length_km"])
            order = None if pd.isna(row["stream_order"]) else float(row["stream_order"])
            if not math.isfinite(length_km) or length_km < 0:
                raise ValueError("Flowline length must be finite and nonnegative")
            if order is not None and (
                not math.isfinite(order) or order < 0 or int(order) != order
            ):
                raise ValueError("Stream order must be a nonnegative integer")
            if (
                row.geometry is None
                or row.geometry.is_empty
                or row.geometry.geom_type not in {"LineString", "MultiLineString"}
            ):
                raise ValueError("Flowline geometry must be a nonempty line")
            edge_attrs = row.to_dict()
            edge_attrs.update(
                comid=comid,
                length_km=length_km,
                stream_order=int(order) if order is not None else None,
            )
            self.graph.add_edge(from_node, to_node, key=comid, **edge_attrs)
            self._comid_to_edge[comid] = (from_node, to_node)

        logger.info(
            "Built Cascadia flowline graph with %d nodes and %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def validate(self) -> dict[str, Any]:
        """Validate network topology and return diagnostic report."""
        validator = FlowlineTopologyValidator(self.graph)
        return validator.validate_all()

    def get_flowline_by_comid(self, comid: int) -> dict[str, Any] | None:
        """Retrieve edge attributes for a given flowline COMID."""
        edge = self._comid_to_edge.get(comid)
        if edge is None:
            return None
        data = self.graph.get_edge_data(edge[0], edge[1], key=comid)
        return cast(dict[str, Any] | None, data)

    def trace_downstream(self, comid: int) -> list[dict[str, Any]]:
        """Trace downstream from a flowline reach to the terminal outlet.

        Returns:
            List of edge attribute dictionaries ordered from upstream to downstream.
        """
        edge = self._comid_to_edge.get(comid)
        if edge is None:
            return []

        path_edges: list[dict[str, Any]] = []
        current_data = self.graph.get_edge_data(edge[0], edge[1], key=comid)
        if current_data:
            path_edges.append(dict(current_data))

        current_node = edge[1]
        visited_nodes: set[Any] = {edge[0], current_node}

        while self.graph.out_degree(current_node) > 0:
            out_edges = list(self.graph.out_edges(current_node, data=True))
            if not out_edges:
                break
            # In dendritic river networks, choose highest stream order edge if multiple
            next_edge = max(
                out_edges,
                key=lambda e: (
                    -1
                    if e[2].get("stream_order") is None
                    else int(e[2]["stream_order"])
                ),
            )
            next_node = next_edge[1]
            if next_node in visited_nodes:
                logger.warning(
                    "Cycle detected at node %s during downstream trace", current_node
                )
                break
            visited_nodes.add(next_node)
            path_edges.append(dict(next_edge[2]))
            current_node = next_node

        return path_edges

    def trace_upstream(self, comid: int) -> list[dict[str, Any]]:
        """Trace all upstream tributary flowlines contributing to a given reach.

        Returns:
            List of upstream edge attribute dictionaries.
        """
        edge = self._comid_to_edge.get(comid)
        if edge is None:
            return []

        upstream_comids: set[int] = {comid}
        start_node = edge[0]

        # Reverse BFS/DFS upstream
        nodes_to_visit = [start_node]
        visited_nodes: set[Any] = {edge[1], start_node}

        while nodes_to_visit:
            curr = nodes_to_visit.pop()
            in_edges = self.graph.in_edges(curr, data=True)
            for u, _, data in in_edges:
                c = data.get("comid")
                if c is not None and c not in upstream_comids:
                    upstream_comids.add(c)
                if u not in visited_nodes:
                    visited_nodes.add(u)
                    nodes_to_visit.append(u)

        results: list[dict[str, Any]] = []
        for c in sorted(upstream_comids):
            e = self._comid_to_edge.get(c)
            if e:
                d = self.graph.get_edge_data(e[0], e[1], key=c)
                if d:
                    results.append(dict(d))
        return results

    def calculate_downstream_distance(self, comid: int) -> float:
        """Calculate total downstream flow distance (km) from reach to outlet."""
        path = self.trace_downstream(comid)
        return sum(float(e.get("length_km", 0.0)) for e in path)

    def calculate_upstream_network_length(self, comid: int) -> float:
        """Calculate total tributary network length (km) upstream of reach."""
        upstream = self.trace_upstream(comid)
        return sum(float(e.get("length_km", 0.0)) for e in upstream)

    def get_pnw_high_order_flowlines(
        self, min_stream_order: int = 5
    ) -> gpd.GeoDataFrame:
        """Filter and return high-order flowlines (Strahler order >= min_stream_order)."""
        if self.flowlines_gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        col = (
            "stream_order"
            if "stream_order" in self.flowlines_gdf.columns
            else "StreamOrder"
            if "StreamOrder" in self.flowlines_gdf.columns
            else None
        )
        if type(min_stream_order) is not int or min_stream_order < 0:
            raise ValueError("min_stream_order must be a nonnegative integer")
        if col:
            return self.flowlines_gdf[
                self.flowlines_gdf[col] >= min_stream_order
            ].copy()
        return self.flowlines_gdf.copy()

    def index_to_h3(self, resolution: int = 8) -> dict[str, dict[str, Any]]:
        """Build an approximate H3 index from vertices and segment midpoints.

        This samples coverage rather than enumerating every intersected cell.
        Native source lengths are apportioned equally among sampled cells; use
        GeoInferSurfaceWater.run_analysis for geometric line-cell measurements.
        Method metadata accompanies each cell so estimates remain identifiable.
        """
        if type(resolution) is not int or not 0 <= resolution <= 15:
            raise ValueError("H3 resolution must be an integer from 0 to 15")
        if self.flowlines_gdf.empty:
            return {}

        hex_data: dict[str, dict[str, Any]] = {}

        for _, row in self.flowlines_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            raw_order = row.get("stream_order", row.get("StreamOrder"))
            order = None if raw_order is None or pd.isna(raw_order) else int(raw_order)
            name = str(
                row.get("gnis_name", row.get("GNIS_NAME", "Unnamed River"))
                or "Unnamed River"
            )
            length = float(row.get("length_km", row.get("LengthKM", 1.0)))
            comid = int(row.get("comid", row.get("COMID", 0)))

            # Sample points along flowline geometry to map into H3 cells
            sampled_cells: set[str] = set()
            lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
            for line in lines:
                coords = list(line.coords)
                for pt in coords:
                    c = latlng_to_cell(pt[1], pt[0], resolution)
                    if c:
                        sampled_cells.add(c)
                # Midpoint sampling for long segments
                for i in range(len(coords) - 1):
                    mid_x = (coords[i][0] + coords[i + 1][0]) / 2.0
                    mid_y = (coords[i][1] + coords[i + 1][1]) / 2.0
                    c = latlng_to_cell(mid_y, mid_x, resolution)
                    if c:
                        sampled_cells.add(c)

            length_per_cell = length / max(1, len(sampled_cells))

            for cell in sampled_cells:
                if cell not in hex_data:
                    hex_data[cell] = {
                        "flowline_length_km": 0.0,
                        "max_stream_order": None,
                        "flowline_count": 0,
                        "coverage_method": "vertex_midpoint_sampling",
                        "length_method": "equal_share_source_length",
                        "river_names": set(),
                        "comids": [],
                    }
                hex_data[cell]["flowline_length_km"] = round(
                    hex_data[cell]["flowline_length_km"] + length_per_cell, 4
                )
                if order is not None:
                    previous = hex_data[cell]["max_stream_order"]
                    hex_data[cell]["max_stream_order"] = (
                        order if previous is None else max(previous, order)
                    )
                hex_data[cell]["flowline_count"] += 1
                hex_data[cell]["river_names"].add(name)
                hex_data[cell]["comids"].append(comid)

        # Convert sets to lists for JSON serializability
        for cell, metrics in hex_data.items():
            metrics["river_names"] = sorted(list(metrics["river_names"]))

        return hex_data
