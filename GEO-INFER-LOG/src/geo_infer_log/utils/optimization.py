"""
Optimization utility functions for GEO-INFER-LOG.

This module provides optimization utility functions for solving
problems like TSP and VRP.
"""

from typing import Any, cast

import numpy as np

from geo_infer_log.utils.geo import haversine_distance

_ORTOOLS_MODULES: tuple[object | None, object | None] | None = None
_HAS_ORTOOLS: bool | None = None


def _load_ortools() -> tuple[object | None, object | None]:
    """Import ortools on demand; return (routing_enums_pb2, pywrapcp) or Nones.

    The import is lazy because a module-level ``from ortools...`` segfaults
    under some C-extension load orders (pyarrow/pandas/shapely-loaded pytest
    processes, both locally and in CI). The availability probe therefore
    happens on first use instead of at import time.
    """
    global _ORTOOLS_MODULES, _HAS_ORTOOLS
    if _HAS_ORTOOLS is not None:
        return _ORTOOLS_MODULES
    try:
        from ortools.constraint_solver import (
            pywrapcp,
            routing_enums_pb2,
        )

        _ORTOOLS_MODULES = (routing_enums_pb2, pywrapcp)
        _HAS_ORTOOLS = True
    except ImportError:
        _ORTOOLS_MODULES = (None, None)
        _HAS_ORTOOLS = False
    return _ORTOOLS_MODULES


def _require_ortools() -> tuple[Any, Any]:
    """Return the ortools modules or raise ImportError if not installed."""
    _load_ortools()
    if not _HAS_ORTOOLS:
        raise ImportError(
            "ortools is required for optimization functions. "
            "Install with: pip install ortools"
        )


def solve_tsp(
    points: list[tuple[float, float]],
    start_index: int = 0,
    end_index: int | None = None,
    time_windows: list[tuple[int, int]] | None = None,
    time_matrix: list[list[int]] | None = None,
) -> dict:
    """
    Solve a Traveling Salesman Problem (TSP).

    Args:
        points: List of (longitude, latitude) coordinates
        start_index: Index of the starting point
        end_index: Index of the ending point (if None, uses start_index)
        time_windows: Optional list of (start_time, end_time) for each point
        time_matrix: Optional time matrix between points

    Returns:
        Dictionary with solution information
    """
    routing_enums_pb2, pywrapcp = _require_ortools()

    if not points:
        raise ValueError("Points list cannot be empty")

    if start_index < 0 or start_index >= len(points):
        raise ValueError(f"Start index {start_index} out of range")

    if end_index is not None and (end_index < 0 or end_index >= len(points)):
        raise ValueError(f"End index {end_index} out of range")

    # Set default end index
    if end_index is None:
        end_index = start_index

    # Create the distance matrix
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i, j] = haversine_distance(points[i], points[j])

    # Create the time matrix if not provided
    if time_matrix is None:
        # Assume average speed of 60 km/h -> 1 km / minute
        time_matrix = [
            [int(distance_matrix[i, j]) for j in range(num_points)]
            for i in range(num_points)
        ]

    # Create a routing index manager
    manager = pywrapcp.RoutingIndexManager(num_points, 1, [start_index], [end_index])

    # Create a routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create distance and time callbacks
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node] * 1000)  # Convert to meters

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cast(int, time_matrix[from_node][to_node])

    # Register callbacks
    distance_transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Set the cost function for distance
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_callback_index)

    # Add time dimension if time windows are provided
    if time_windows:
        routing.AddDimension(
            time_transit_callback_index,
            30,  # Allow waiting time
            1440,  # Maximum time: 24 hours in minutes
            False,  # Don't force start cumul to zero
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # Add time window constraints
        for location_idx, time_window in enumerate(time_windows):
            if location_idx == start_index:
                continue  # Skip depot
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Process the solution
    if not solution:
        return {"status": "No solution found", "route": [], "distance": 0, "time": 0}

    # Extract the route
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        index = solution.Value(routing.NextVar(index))

    node_index = manager.IndexToNode(index)
    route.append(node_index)

    # Calculate total distance and time
    total_distance = 0
    total_time = 0

    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        total_distance += distance_matrix[from_node, to_node]
        total_time += time_matrix[from_node][to_node]

    return {
        "status": "Solution found",
        "route": route,
        "distance": total_distance,
        "time": total_time,
    }


def solve_vrp(
    depots: list[tuple[float, float]],
    deliveries: list[tuple[float, float]],
    num_vehicles: int,
    vehicle_capacities: list[float] | None = None,
    delivery_demands: list[float] | None = None,
    time_windows: list[tuple[int, int]] | None = None,
    max_distance: float | None = None,
    max_time: int | None = None,
) -> dict:
    """
    Solve a Vehicle Routing Problem (VRP).

    Args:
        depots: List of (longitude, latitude) depot coordinates
        deliveries: List of (longitude, latitude) delivery coordinates
        num_vehicles: Number of vehicles to use
        vehicle_capacities: Optional list of vehicle capacities
        delivery_demands: Optional list of delivery demands
        time_windows: Optional list of (start_time, end_time) for each delivery
        max_distance: Optional maximum distance per vehicle
        max_time: Optional maximum time per vehicle

    Returns:
        Dictionary with solution information
    """
    routing_enums_pb2, pywrapcp = _require_ortools()

    if not depots:
        raise ValueError("Depots list cannot be empty")

    if not deliveries:
        raise ValueError("Deliveries list cannot be empty")

    if num_vehicles <= 0:
        raise ValueError("Number of vehicles must be positive")

    # Use the first depot as starting and ending point
    depot = depots[0]

    # Create a combined list of all locations
    locations = [depot] + deliveries
    num_locations = len(locations)

    # Create the distance matrix
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i, j] = haversine_distance(locations[i], locations[j])

    # Create the time matrix
    # Assume average speed of 60 km/h -> 1 km / minute
    time_matrix = [
        [int(distance_matrix[i, j]) for j in range(num_locations)]
        for i in range(num_locations)
    ]

    # Create a routing index manager
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)

    # Create a routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create distance and time callbacks
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node] * 1000)  # Convert to meters

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cast(int, time_matrix[from_node][to_node])

    # Register callbacks
    distance_transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Set the cost function for distance
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_callback_index)

    # Add distance dimension
    routing.AddDimension(
        distance_transit_callback_index,
        0,  # No slack
        int(max_distance * 1000)
        if max_distance
        else 3000000,  # Maximum distance in meters (default: 3000 km)
        True,  # Start cumul to zero
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")

    # Add time dimension
    routing.AddDimension(
        time_transit_callback_index,
        30,  # Allow 30 minutes of waiting time
        max_time or 1440,  # Maximum time (default: 24 hours in minutes)
        False,  # Don't force start cumul to zero
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Add capacity constraints if specified
    if vehicle_capacities and delivery_demands:

        def demand_callback(from_index: int) -> float:
            """Return the demand of the node."""
            from_node = manager.IndexToNode(from_index)
            # Depot has no demand
            if from_node == 0:
                return 0
            return cast(float, delivery_demands[from_node - 1])

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

    # Add time window constraints if specified
    if time_windows:
        for location_idx, time_window in enumerate(time_windows):
            if location_idx == 0:
                continue  # Skip depot
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add time window for depot
        depot_index = manager.NodeToIndex(0)
        time_dimension.CumulVar(depot_index).SetRange(0, 1440)  # 24 hours in minutes

    # Set vehicle costs for each vehicle
    for vehicle_id in range(num_vehicles):
        # Set a limit on maximum distance per vehicle
        if max_distance:
            distance_dimension.SetSpanCostCoefficientForVehicle(1, vehicle_id)

        # Set a limit on maximum time per vehicle
        if max_time:
            time_dimension.SetSpanCostCoefficientForVehicle(1, vehicle_id)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30  # 30 seconds time limit

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Process the solution
    if not solution:
        return {
            "status": "No solution found",
            "routes": [],
            "total_distance": 0,
            "total_time": 0,
            "num_vehicles_used": 0,
        }

    # Extract the routes
    routes = []
    total_distance = 0
    total_time = 0
    num_vehicles_used = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)

        # Skip empty routes
        if solution.Value(routing.NextVar(index)) == routing.End(vehicle_id):
            continue

        num_vehicles_used += 1
        route = []
        route_distance = 0
        route_time = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)

            previous_index = index
            index = solution.Value(routing.NextVar(index))

            from_node = manager.IndexToNode(previous_index)
            to_node = manager.IndexToNode(index)

            route_distance += distance_matrix[from_node, to_node]
            route_time += time_matrix[from_node][to_node]

        # Add depot as the end of the route
        route.append(0)

        # Build route with actual locations
        route_locations = [locations[i] for i in route]

        routes.append(
            {
                "vehicle_id": vehicle_id,
                "route": route,
                "locations": route_locations,
                "distance": route_distance,
                "time": route_time,
            }
        )

        total_distance += route_distance
        total_time += route_time

    return {
        "status": "Solution found",
        "routes": routes,
        "total_distance": total_distance,
        "total_time": total_time,
        "num_vehicles_used": num_vehicles_used,
    }
