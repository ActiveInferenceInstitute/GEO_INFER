#!/usr/bin/env python3
"""
GEO-INFER-LOG Example: Last-Mile Delivery Optimization

This example demonstrates comprehensive last-mile delivery optimization
including fleet management, route optimization, and real-time tracking.

Runs entirely network-free: RouteOptimizer falls back to haversine-based
nearest-neighbor routing when no road network is loaded.
"""

from geo_infer_log import (
    FleetManager,
    MultiObjectiveOptimizer,
    RealTimeTracker,
    TravelTimeEstimator,
    Vehicle,
    VehicleRouter,
    VehicleType,
)


def main():
    print("=" * 60)
    print("GEO-INFER-LOG: Last-Mile Delivery Optimization")
    print("=" * 60)

    # 1. Set Up Fleet
    print("\n1. Setting Up Delivery Fleet...")

    fleet = FleetManager()

    vehicles = [
        Vehicle(
            id="VAN_001",
            type=VehicleType.VAN,
            capacity=500,
            max_range=200,
            speed=40,
            cost_per_km=0.8,
            emissions_per_km=0.25,
            location=(-118.25, 34.05),
        ),
        Vehicle(
            id="VAN_002",
            type=VehicleType.VAN,
            capacity=500,
            max_range=200,
            speed=40,
            cost_per_km=0.8,
            emissions_per_km=0.25,
            location=(-118.25, 34.05),
        ),
        Vehicle(
            id="TRUCK_001",
            type=VehicleType.TRUCK,
            capacity=2000,
            max_range=300,
            speed=35,
            cost_per_km=1.2,
            emissions_per_km=0.45,
            location=(-118.25, 34.05),
        ),
        Vehicle(
            id="BIKE_001",
            type=VehicleType.BIKE,
            capacity=20,
            max_range=30,
            speed=15,
            cost_per_km=0.05,
            emissions_per_km=0.0,
            location=(-118.25, 34.05),
        ),
    ]

    for vehicle in vehicles:
        fleet.add_vehicle(vehicle)

    status = fleet.get_fleet_status()
    print(f"   Total vehicles: {status['total_vehicles']}")
    print(f"   Available: {status['available_vehicles']}")

    # 2. Define Delivery Orders
    print("\n2. Processing Delivery Orders...")

    depot = (-118.25, 34.05)  # Depot location

    orders = [
        {"id": "ORD_001", "location": (-118.28, 34.08), "weight": 15},
        {"id": "ORD_002", "location": (-118.22, 34.07), "weight": 25},
        {"id": "ORD_003", "location": (-118.30, 34.02), "weight": 10},
        {"id": "ORD_004", "location": (-118.18, 34.06), "weight": 45},
        {"id": "ORD_005", "location": (-118.26, 34.10), "weight": 8},
        {"id": "ORD_006", "location": (-118.32, 34.04), "weight": 30},
        {"id": "ORD_007", "location": (-118.20, 34.03), "weight": 20},
        {"id": "ORD_008", "location": (-118.27, 34.01), "weight": 35},
    ]

    total_weight = sum(o["weight"] for o in orders)
    print(f"   Orders: {len(orders)}")
    print(f"   Total weight: {total_weight} kg")

    # 3. Calculate Distance/Time Matrices
    print("\n3. Computing Travel Time Matrix...")

    estimator = TravelTimeEstimator()

    locations = [depot] + [o["location"] for o in orders]

    time_matrix = estimator.calculate_time_matrix(locations)
    distance_matrix = estimator.calculate_distance_matrix(locations)

    print(f"   Matrix size: {len(locations)}x{len(locations)}")
    print(f"   Avg travel time: {time_matrix[time_matrix > 0].mean():.1f} min")
    print(f"   Max travel time: {time_matrix.max():.1f} min")

    # 4. Multi-Objective Optimization
    print("\n4. Running Multi-Objective Optimization...")

    optimizer = MultiObjectiveOptimizer(objectives=["distance", "time", "emissions"])

    # Set weights (prioritize time, then emissions, then distance)
    optimizer.set_weights({"distance": 0.2, "time": 0.5, "emissions": 0.3})

    print(f"   Objectives: {', '.join(optimizer.objectives)}")
    print(
        f"   Weights: time={optimizer.weights['time']:.1%}, "
        f"emissions={optimizer.weights['emissions']:.1%}, "
        f"distance={optimizer.weights['distance']:.1%}"
    )

    # 5. Solve Vehicle Routing Problem
    print("\n5. Solving Vehicle Routing Problem...")

    router = VehicleRouter(fleet)

    deliveries = [
        {"id": o["id"], "location": o["location"], "demand": o["weight"]}
        for o in orders
    ]

    solution = router.solve_vrp(
        deliveries=deliveries,
        depots=[depot],
        constraints={"max_distance": 100, "max_stops": 5},
    )

    print(f"   Vehicles routed: {len(solution['routes'])}")
    print(f"   Unassigned orders: {len(solution['unassigned'])}")

    # 6. Generate Vehicle Routes
    print("\n6. Generating Vehicle Routes...")

    # Assign orders to vehicles (simplified)
    van1_orders = orders[:3]
    van2_orders = orders[3:6]
    truck_orders = orders[6:]

    van1_coords = [o["location"] for o in van1_orders]
    van2_coords = [o["location"] for o in van2_orders]
    truck_coords = [o["location"] for o in truck_orders]

    # Create assignments (network-free haversine routing)
    assignment1 = fleet.assign_delivery("VAN_001", van1_coords, depot)
    assignment2 = fleet.assign_delivery("VAN_002", van2_coords, depot)
    assignment3 = fleet.assign_delivery("TRUCK_001", truck_coords, depot)

    print(
        f"   VAN_001: {len(van1_orders)} stops, "
        f"{assignment1['route']['distance']:.1f} km"
    )
    print(
        f"   VAN_002: {len(van2_orders)} stops, "
        f"{assignment2['route']['distance']:.1f} km"
    )
    print(
        f"   TRUCK_001: {len(truck_orders)} stops, "
        f"{assignment3['route']['distance']:.1f} km"
    )

    # 7. Estimate Arrival Times
    print("\n7. Estimating Arrival Times...")

    departure = "2024-01-15T09:00:00"

    route1 = [depot] + van1_coords + [depot]
    arrivals1 = estimator.estimate_arrival_times(
        route=route1,
        departure_time=departure,
        service_times=[0, 5, 5, 5, 0],  # 5 min service per stop
    )

    print(f"   VAN_001 departure: {departure[:16]}")
    print(f"   VAN_001 return ETA: {arrivals1[-1][:16]}")

    # 8. Set Up Real-Time Tracking
    print("\n8. Initializing Real-Time Tracking...")

    tracker = RealTimeTracker()

    # Simulate position updates
    tracker.update_position("VAN_001", (-118.26, 34.06), "2024-01-15T09:30:00")
    tracker.update_position("VAN_002", (-118.27, 34.05), "2024-01-15T09:30:00")
    tracker.update_position("TRUCK_001", (-118.24, 34.04), "2024-01-15T09:30:00")

    positions = tracker.get_fleet_positions()
    print(f"   Tracking {len(positions)} vehicles")

    for vid, pos in positions.items():
        print(f"   {vid}: ({pos['lon']:.4f}, {pos['lat']:.4f})")

    # Calculate ETA to next stop
    eta = tracker.calculate_eta("VAN_001", van1_coords[0], estimator)
    print(f"   VAN_001 ETA to next stop: {eta[:16] if eta else 'N/A'}")

    # 9. Performance Summary
    print("\n9. Route Performance Summary...")

    total_distance = (
        assignment1["route"]["distance"]
        + assignment2["route"]["distance"]
        + assignment3["route"]["distance"]
    )

    total_emissions = sum(
        v.emissions_per_km * assignment["route"]["distance"]
        for v, assignment in zip(vehicles[:3], [assignment1, assignment2, assignment3])
    )

    print(f"   Total planned distance: {total_distance:.1f} km")
    print(f"   Estimated emissions: {total_emissions:.2f} kg CO2")
    print(f"   Orders delivered: {len(orders)}")
    print(f"   Fleet utilization: {3 / len(vehicles) * 100:.0f}%")

    print("\n" + "=" * 60)
    print("Last-Mile Delivery Optimization Complete!")
    print("=" * 60)

    # Summary
    print("\nOptimization Results:")
    print(f"  - Orders: {len(orders)} deliveries")
    print(f"  - Vehicles utilized: 3/{len(vehicles)}")
    print(f"  - CO2 emissions: {total_emissions:.2f} kg")


if __name__ == "__main__":
    main()
