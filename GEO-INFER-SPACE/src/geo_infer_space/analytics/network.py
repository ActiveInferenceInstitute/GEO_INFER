"""
Network analysis module for advanced spatial analysis.

This module provides comprehensive network-based spatial operations including
shortest path analysis, service area calculations, routing, and accessibility
analysis using NetworkX and OSMnx.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import Any
from shapely.geometry import Point, LineString

logger = logging.getLogger(__name__)

try:
    import osmnx as osmnx

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    logger.warning(
        "OSMnx not available. Some network analysis features will be limited."
    )


def shortest_path(
    network_gdf: gpd.GeoDataFrame,
    start_point: Point,
    end_point: Point,
    weight_column: str = "length",
    impedance_factor: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate shortest path between two points on a network.

    Args:
        network_gdf: GeoDataFrame representing network edges
        start_point: Starting point geometry
        end_point: Ending point geometry
        weight_column: Column name for edge weights
        impedance_factor: Factor to multiply weights by

    Returns:
        Dictionary with path geometry, distance, and route information
    """
    try:
        # Create NetworkX graph from GeoDataFrame
        G = _create_graph_from_gdf(network_gdf, weight_column, impedance_factor)

        # Find nearest nodes to start and end points
        start_node = _find_nearest_node(G, start_point, network_gdf)
        end_node = _find_nearest_node(G, end_point, network_gdf)

        if start_node is None or end_node is None:
            raise ValueError("Could not find network nodes near start/end points")

        # Calculate shortest path
        try:
            path_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_column)
            path_length = nx.shortest_path_length(
                G, start_node, end_node, weight=weight_column
            )
        except nx.NetworkXNoPath:
            return {
                "path_geometry": None,
                "total_distance": float("inf"),
                "path_nodes": [],
                "success": False,
                "message": "No path found between points",
            }

        # Create path geometry
        path_coords = []
        for i in range(len(path_nodes) - 1):
            edge_data = G[path_nodes[i]][path_nodes[i + 1]]
            if "geometry" in edge_data:
                geom = edge_data["geometry"]
                if hasattr(geom, "coords"):
                    path_coords.extend(list(geom.coords))

        path_geometry = LineString(path_coords) if path_coords else None

        return {
            "path_geometry": path_geometry,
            "total_distance": path_length,
            "path_nodes": path_nodes,
            "success": True,
            "message": f"Path found with {len(path_nodes)} nodes",
        }

    except Exception as e:
        logger.error(f"Shortest path calculation failed: {e}")
        return {
            "path_geometry": None,
            "total_distance": float("inf"),
            "path_nodes": [],
            "success": False,
            "message": str(e),
        }


def service_area(
    network_gdf: gpd.GeoDataFrame,
    center_point: Point,
    max_distance: float,
    weight_column: str = "length",
) -> gpd.GeoDataFrame:
    """
    Calculate service area (isochrone) from a center point.

    Args:
        network_gdf: GeoDataFrame representing network edges
        center_point: Center point for service area
        max_distance: Maximum distance/time for service area
        weight_column: Column name for edge weights

    Returns:
        GeoDataFrame with service area polygon
    """
    try:
        # Create NetworkX graph
        G = _create_graph_from_gdf(network_gdf, weight_column)

        # Find nearest node to center point
        center_node = _find_nearest_node(G, center_point, network_gdf)
        if center_node is None:
            raise ValueError("Could not find network node near center point")

        # Calculate shortest paths to all reachable nodes within distance
        distances = nx.single_source_dijkstra_path_length(
            G, center_node, cutoff=max_distance, weight=weight_column
        )

        # Get coordinates of reachable nodes
        reachable_coords = []
        for node, distance in distances.items():
            if distance <= max_distance:
                node_data = G.nodes[node]
                if "x" in node_data and "y" in node_data:
                    reachable_coords.append((node_data["x"], node_data["y"]))

        if len(reachable_coords) < 3:
            # Not enough points for polygon
            return gpd.GeoDataFrame(columns=["geometry"], crs=network_gdf.crs)

        # Create convex hull as service area approximation
        from shapely.geometry import MultiPoint

        service_area_geom = MultiPoint(reachable_coords).convex_hull

        result_gdf = gpd.GeoDataFrame(
            [{"geometry": service_area_geom, "max_distance": max_distance}],
            crs=network_gdf.crs,
        )

        logger.info(
            f"Service area calculated with {len(reachable_coords)} reachable points"
        )
        return result_gdf

    except Exception as e:
        logger.error(f"Service area calculation failed: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=network_gdf.crs)


def network_connectivity(
    network_gdf: gpd.GeoDataFrame, weight_column: str = "length"
) -> dict[str, Any]:
    """
    Analyze network connectivity metrics.

    Args:
        network_gdf: GeoDataFrame representing network edges
        weight_column: Column name for edge weights

    Returns:
        Dictionary with connectivity metrics
    """
    try:
        G = _create_graph_from_gdf(network_gdf, weight_column)

        # Basic network metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Connectivity metrics
        is_connected = (
            nx.is_connected(G) if not G.is_directed() else nx.is_strongly_connected(G)
        )
        num_components = (
            nx.number_connected_components(G)
            if not G.is_directed()
            else nx.number_strongly_connected_components(G)
        )

        # Degree statistics
        degrees = dict(G.degree())
        avg_degree = np.mean(list(degrees.values())) if degrees else 0

        # Centrality measures (sample for large networks)
        sample_size = min(100, num_nodes)
        if sample_size > 0:
            sample_nodes = list(G.nodes())[:sample_size]
            betweenness = nx.betweenness_centrality(G.subgraph(sample_nodes))
            closeness = nx.closeness_centrality(G.subgraph(sample_nodes))
            avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0
            avg_closeness = np.mean(list(closeness.values())) if closeness else 0
        else:
            avg_betweenness = 0
            avg_closeness = 0

        # Network density
        density = nx.density(G)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "is_connected": is_connected,
            "num_components": num_components,
            "average_degree": avg_degree,
            "network_density": density,
            "average_betweenness_centrality": avg_betweenness,
            "average_closeness_centrality": avg_closeness,
        }

    except Exception as e:
        logger.error(f"Network connectivity analysis failed: {e}")
        return {"error": str(e)}


def routing_analysis(
    network_gdf: gpd.GeoDataFrame,
    origins: list[Point],
    destinations: list[Point],
    weight_column: str = "length",
) -> pd.DataFrame:
    """
    Perform origin-destination routing analysis.

    Args:
        network_gdf: GeoDataFrame representing network edges
        origins: List of origin points
        destinations: List of destination points
        weight_column: Column name for edge weights

    Returns:
        DataFrame with O-D matrix results
    """
    try:
        G = _create_graph_from_gdf(network_gdf, weight_column)

        # Find nearest nodes for all origins and destinations
        origin_nodes = [_find_nearest_node(G, point, network_gdf) for point in origins]
        dest_nodes = [
            _find_nearest_node(G, point, network_gdf) for point in destinations
        ]

        results = []

        for i, origin_node in enumerate(origin_nodes):
            if origin_node is None:
                continue

            for j, dest_node in enumerate(dest_nodes):
                if dest_node is None:
                    continue

                try:
                    distance = nx.shortest_path_length(
                        G, origin_node, dest_node, weight=weight_column
                    )
                    path = nx.shortest_path(
                        G, origin_node, dest_node, weight=weight_column
                    )

                    results.append(
                        {
                            "origin_id": i,
                            "destination_id": j,
                            "distance": distance,
                            "path_length": len(path),
                            "success": True,
                        }
                    )

                except nx.NetworkXNoPath:
                    results.append(
                        {
                            "origin_id": i,
                            "destination_id": j,
                            "distance": float("inf"),
                            "path_length": 0,
                            "success": False,
                        }
                    )

        result_df = pd.DataFrame(results)
        logger.info(f"Routing analysis completed: {len(results)} O-D pairs")
        return result_df

    except Exception as e:
        logger.error(f"Routing analysis failed: {e}")
        return pd.DataFrame()


def accessibility_analysis(
    network_gdf: gpd.GeoDataFrame,
    origins: list[Point],
    destinations: list[Point],
    max_distance: float,
    weight_column: str = "length",
) -> pd.DataFrame:
    """
    Calculate accessibility metrics from origins to destinations.

    Args:
        network_gdf: GeoDataFrame representing network edges
        origins: List of origin points
        destinations: List of destination points
        max_distance: Maximum distance to consider accessible
        weight_column: Column name for edge weights

    Returns:
        DataFrame with accessibility metrics per origin
    """
    try:
        G = _create_graph_from_gdf(network_gdf, weight_column)

        origin_nodes = [_find_nearest_node(G, point, network_gdf) for point in origins]
        dest_nodes = [
            _find_nearest_node(G, point, network_gdf) for point in destinations
        ]

        results = []

        for i, origin_node in enumerate(origin_nodes):
            if origin_node is None:
                continue

            accessible_count = 0
            total_distance = 0
            min_distance: float | None = float("inf")
            distances = []

            for dest_node in dest_nodes:
                if dest_node is None:
                    continue

                try:
                    distance = nx.shortest_path_length(
                        G, origin_node, dest_node, weight=weight_column
                    )

                    if distance <= max_distance:
                        accessible_count += 1
                        total_distance += distance
                        min_distance = min(min_distance, distance)
                        distances.append(distance)

                except nx.NetworkXNoPath:
                    continue

            # Calculate accessibility metrics
            accessibility_ratio = (
                accessible_count / len(destinations) if destinations else 0
            )
            avg_distance = (
                total_distance / accessible_count
                if accessible_count > 0
                else float("inf")
            )
            min_distance = min_distance if min_distance != float("inf") else None

            results.append(
                {
                    "origin_id": i,
                    "accessible_destinations": accessible_count,
                    "accessibility_ratio": accessibility_ratio,
                    "average_distance": avg_distance,
                    "minimum_distance": min_distance,
                    "total_destinations": len(destinations),
                }
            )

        result_df = pd.DataFrame(results)
        logger.info(f"Accessibility analysis completed for {len(results)} origins")
        return result_df

    except Exception as e:
        logger.error(f"Accessibility analysis failed: {e}")
        return pd.DataFrame()


def _create_graph_from_gdf(
    gdf: gpd.GeoDataFrame, weight_column: str = "length", impedance_factor: float = 1.0
) -> nx.Graph:
    """Create NetworkX graph from GeoDataFrame of network edges."""
    G = nx.Graph()

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if isinstance(geom, LineString):
            coords = list(geom.coords)
            start_node = coords[0]
            end_node = coords[-1]

            # Add nodes with coordinates
            G.add_node(start_node, x=start_node[0], y=start_node[1])
            G.add_node(end_node, x=end_node[0], y=end_node[1])

            # Add edge with weight
            weight = row.get(weight_column, geom.length) * impedance_factor
            G.add_edge(
                start_node,
                end_node,
                **{weight_column: weight, "geometry": geom, "edge_id": idx},
            )

    return G


def _find_nearest_node(
    G: nx.Graph, point: Point, network_gdf: gpd.GeoDataFrame
) -> tuple[float, float] | None:
    """Find nearest network node to a point."""
    if not G.nodes():
        return None

    min_distance = float("inf")
    nearest_node = None

    for node in G.nodes():
        node_point = Point(G.nodes[node]["x"], G.nodes[node]["y"])
        distance = point.distance(node_point)

        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node
