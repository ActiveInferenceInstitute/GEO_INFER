"""
Visualization utility functions for GEO-INFER-LOG.

This module provides utility functions for visualizing routes,
networks, service areas, and other logistics data.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import math
import networkx as nx
import geopandas as gpd
import folium
from shapely.geometry import LineString
from typing import Any

try:
    import contextily as ctx
except ImportError:
    ctx = None


def _coordinate(point: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate and normalize a ``(longitude, latitude)`` coordinate."""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ValueError(f"{name} must be a (longitude, latitude) pair")
    try:
        longitude, latitude = float(point[0]), float(point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError(f"{name} coordinates must be finite")
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError(f"{name} coordinates are outside geographic bounds")
    return longitude, latitude


def _route(
    route: list[tuple[float, float]], name: str = "route"
) -> list[tuple[float, float]]:
    """Validate a route and require enough points to form a line."""
    if not route or len(route) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    return [_coordinate(point, f"{name}[{index}]") for index, point in enumerate(route)]


def plot_route(
    route: list[tuple[float, float]],
    points_of_interest: list[tuple[float, float]] | None = None,
    labels: list[str] | None = None,
    title: str = "Route Map",
    figsize: tuple[int, int] = (12, 8),
    basemap: bool = True,
) -> plt.Figure:
    """
    Plot a route on a map.

    Args:
        route: List of (longitude, latitude) coordinates
        points_of_interest: Optional list of additional points to display
        labels: Optional list of labels for points of interest
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        basemap: Whether to include a basemap

    Returns:
        Matplotlib figure
    """
    route = _route(route)

    # Convert route to GeoDataFrame
    route_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([p[0] for p in route], [p[1] for p in route]),
        crs="EPSG:4326",
    )

    # Convert to Web Mercator for basemap compatibility
    route_gdf = route_gdf.to_crs(epsg=3857)

    # Extract route line
    route_line = gpd.GeoDataFrame(
        geometry=[LineString(route)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Add basemap if requested
    if basemap and ctx is not None:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot route line
    route_line.plot(ax=ax, color="blue", linewidth=2, alpha=0.7)

    # Plot route points
    route_gdf.plot(ax=ax, color="blue", markersize=30, marker="o")

    # Add origin and destination markers
    origin = route_gdf.iloc[[0]]
    destination = route_gdf.iloc[[-1]]

    origin.plot(ax=ax, color="green", markersize=100, marker="^")
    destination.plot(ax=ax, color="red", markersize=100, marker="v")

    # Add points of interest if provided
    if points_of_interest:
        points_of_interest = [
            _coordinate(point, f"points_of_interest[{index}]")
            for index, point in enumerate(points_of_interest)
        ]
        if labels is not None and len(labels) != len(points_of_interest):
            raise ValueError("labels must have one entry per point of interest")
        poi_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                [p[0] for p in points_of_interest], [p[1] for p in points_of_interest]
            ),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        poi_gdf.plot(ax=ax, color="purple", markersize=80, marker="s")

        # Add labels if provided
        if labels:
            for i, (_, point) in enumerate(poi_gdf.iterrows()):
                if i < len(labels):
                    ax.annotate(
                        labels[i],
                        (point.geometry.x, point.geometry.y),
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7
                        ),
                    )

    # Add legend
    handles = [
        Patch(color="green", label="Origin"),
        Patch(color="red", label="Destination"),
        Patch(color="blue", label="Route"),
    ]

    if points_of_interest:
        handles.append(Patch(color="purple", label="Points of Interest"))

    ax.legend(handles=handles, loc="upper right")

    # Set title and axis labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    return fig


def plot_network(
    graph: nx.Graph,
    node_positions: dict[Any, tuple[float, float]] | None = None,
    node_colors: dict[Any, str] | None = None,
    edge_weights: dict[tuple[Any, Any], float] | None = None,
    highlight_path: list[Any] | None = None,
    title: str = "Network Graph",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot a network graph.

    Args:
        graph: NetworkX graph
        node_positions: Dictionary mapping node IDs to (x, y) positions
        node_colors: Dictionary mapping node IDs to colors
        edge_weights: Dictionary mapping edge tuples to weights
        highlight_path: List of node IDs forming a path to highlight
        title: Title for the plot
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib figure
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx graph")
    if highlight_path:
        missing = set(highlight_path) - set(graph.nodes)
        if missing:
            raise ValueError(
                f"highlight_path contains unknown nodes: {sorted(missing)}"
            )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get or calculate node positions
    if node_positions is None:
        pos = nx.spring_layout(graph)
    else:
        pos = node_positions

    # Set default node colors if not provided
    if node_colors is None:
        node_colors = {node: "lightblue" for node in graph.nodes()}

    # Extract node colors in the order of graph.nodes()
    node_color_list = [node_colors.get(node, "lightblue") for node in graph.nodes()]

    # Set default edge weights if not provided
    if edge_weights is None:
        edge_weights = {edge: 1.0 for edge in graph.edges()}

    # Extract edge weights in the order of graph.edges()
    edge_width_list = [edge_weights.get(edge, 1.0) for edge in graph.edges()]

    # Draw the network
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_color_list, node_size=300, alpha=0.8, ax=ax
    )

    nx.draw_networkx_edges(
        graph, pos, width=edge_width_list, alpha=0.6, ax=ax, edge_color="gray"
    )

    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")

    # Highlight a path if provided
    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(
            graph, pos, edgelist=path_edges, width=3, alpha=1.0, edge_color="red"
        )

        # Highlight path nodes
        path_nodes = {node: "red" for node in highlight_path}
        path_node_color_list = [
            path_nodes.get(node, node_colors.get(node, "lightblue"))
            for node in highlight_path
        ]

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=highlight_path,
            node_color=path_node_color_list,
            node_size=400,
            alpha=1.0,
            ax=ax,
        )

    # Set title
    ax.set_title(title, fontsize=16)

    # Remove axis
    ax.axis("off")

    fig.tight_layout()

    return fig


def plot_service_area(
    service_areas: dict[str, gpd.GeoDataFrame],
    facilities: gpd.GeoDataFrame | None = None,
    demand_points: gpd.GeoDataFrame | None = None,
    title: str = "Service Area Map",
    figsize: tuple[int, int] = (12, 8),
    basemap: bool = True,
) -> plt.Figure:
    """
    Plot service areas on a map.

    Args:
        service_areas: Dictionary mapping service area IDs to GeoDataFrames with Polygon geometry
        facilities: Optional GeoDataFrame of facility points
        demand_points: Optional GeoDataFrame of demand points
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        basemap: Whether to include a basemap

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get color cycle
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Plot service areas
    for i, (area_id, area_gdf) in enumerate(service_areas.items()):
        color = colors[i % len(colors)]

        # Ensure the GeoDataFrame is in Web Mercator
        if area_gdf.crs != "EPSG:3857":
            area_gdf = area_gdf.to_crs(epsg=3857)

        area_gdf.plot(ax=ax, color=color, alpha=0.3, edgecolor=color, linewidth=2)

    # Plot facilities if provided
    if facilities is not None:
        # Ensure the GeoDataFrame is in Web Mercator
        if facilities.crs != "EPSG:3857":
            facilities = facilities.to_crs(epsg=3857)

        facilities.plot(ax=ax, color="black", markersize=100, marker="^")

        # Add facility labels if 'name' column exists
        if "name" in facilities.columns:
            for idx, row in facilities.iterrows():
                ax.annotate(
                    row["name"],
                    (row.geometry.x, row.geometry.y),
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

    # Plot demand points if provided
    if demand_points is not None:
        # Ensure the GeoDataFrame is in Web Mercator
        if demand_points.crs != "EPSG:3857":
            demand_points = demand_points.to_crs(epsg=3857)

        demand_points.plot(ax=ax, color="red", markersize=30, marker="o", alpha=0.7)

    # Add basemap if requested
    if basemap and ctx is not None:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Create legend
    handles = []

    for i, area_id in enumerate(service_areas.keys()):
        color = colors[i % len(colors)]
        handles.append(Patch(color=color, alpha=0.3, label=f"Service Area: {area_id}"))

    if facilities is not None:
        handles.append(Patch(color="black", label="Facilities"))

    if demand_points is not None:
        handles.append(Patch(color="red", alpha=0.7, label="Demand Points"))

    ax.legend(handles=handles, loc="upper right")

    # Set title
    ax.set_title(title, fontsize=16)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    return fig


def create_interactive_map(
    routes: list[list[tuple[float, float]]] | None = None,
    service_areas: dict[str, gpd.GeoDataFrame] | None = None,
    facilities: gpd.GeoDataFrame | None = None,
    demand_points: gpd.GeoDataFrame | None = None,
    center: tuple[float, float] | None = None,
    zoom: int = 10,
) -> folium.Map:
    """
    Create an interactive map with Folium.

    Args:
        routes: Optional list of routes, each a list of (longitude, latitude) coordinates
        service_areas: Optional dictionary mapping service area IDs to GeoDataFrames with Polygon geometry
        facilities: Optional GeoDataFrame of facility points
        demand_points: Optional GeoDataFrame of demand points
        center: Optional (latitude, longitude) for the center of the map
        zoom: Initial zoom level

    Returns:
        Folium map
    """
    if not isinstance(zoom, int) or not 0 <= zoom <= 18:
        raise ValueError("zoom must be an integer between 0 and 18")
    if routes:
        routes = [
            _route(route, f"routes[{index}]") for index, route in enumerate(routes)
        ]
    if center is not None:
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            raise ValueError("center must be a (latitude, longitude) pair")
        latitude, longitude = _coordinate((center[1], center[0]), "center")
        center = (latitude, longitude)

    # Determine center of map
    if center is None:
        # Try to infer from other inputs
        if routes and routes[0]:
            first_point = routes[0][0]
            center = (first_point[1], first_point[0])  # (lat, lon) for Folium
        elif facilities is not None and not facilities.empty:
            first_facility = facilities.iloc[0]
            center = (first_facility.geometry.y, first_facility.geometry.x)
        elif demand_points is not None and not demand_points.empty:
            first_point = demand_points.iloc[0]
            center = (first_point.geometry.y, first_point.geometry.x)
        else:
            # Default to Berlin
            center = (52.5200, 13.4050)

    # Create map
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    # Add routes if provided
    if routes:
        route_colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "darkred",
            "cadetblue",
            "darkgreen",
            "darkpurple",
            "black",
        ]

        for i, route in enumerate(routes):
            color = route_colors[i % len(route_colors)]

            # Create line for route
            route_line = folium.PolyLine(
                locations=[(lat, lon) for lon, lat in route],
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"Route {i + 1}",
            )
            route_line.add_to(m)

            # Add markers for start and end
            if route:
                folium.Marker(
                    location=(route[0][1], route[0][0]),
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                    tooltip=f"Start of Route {i + 1}",
                ).add_to(m)

                folium.Marker(
                    location=(route[-1][1], route[-1][0]),
                    icon=folium.Icon(color="red", icon="stop", prefix="fa"),
                    tooltip=f"End of Route {i + 1}",
                ).add_to(m)

    # Add service areas if provided
    if service_areas:
        area_colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "cadetblue",
            "darkred",
            "darkgreen",
            "darkpurple",
            "black",
        ]

        for i, (area_id, area_gdf) in enumerate(service_areas.items()):
            color = area_colors[i % len(area_colors)]

            # Convert to GeoJSON
            area_json = area_gdf.__geo_interface__

            # Add to map
            folium.GeoJson(
                area_json,
                style_function=lambda x, color=color: {
                    "fillColor": color,
                    "color": color,
                    "weight": 2,
                    "fillOpacity": 0.3,
                },
                tooltip=f"Service Area: {area_id}",
            ).add_to(m)

    # Add facilities if provided
    if facilities is not None:
        for _, facility in facilities.iterrows():
            name = facility.get("name", "Facility")

            folium.Marker(
                location=(facility.geometry.y, facility.geometry.x),
                icon=folium.Icon(color="black", icon="building", prefix="fa"),
                tooltip=name,
            ).add_to(m)

    # Add demand points if provided
    if demand_points is not None:
        # Create cluster for demand points
        marker_cluster = folium.plugins.MarkerCluster()  # type: ignore[attr-defined]
        marker_cluster.add_to(m)

        for _, point in demand_points.iterrows():
            name = point.get("name", "Demand Point")

            folium.Marker(
                location=(point.geometry.y, point.geometry.x),
                icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
                tooltip=name,
            ).add_to(marker_cluster)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m
