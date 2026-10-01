"""
Basic logistics and routing example using GEO-INFER-LOG.

This example demonstrates:
- Route optimization (network-free haversine mode)
- Vehicle fleet management
- Delivery planning
"""

from geo_infer_log import (
    FleetManager,
    RouteOptimizer,
    RoutingParameters,
    Vehicle,
    VehicleType,
)


def create_sample_vehicles() -> list[Vehicle]:
    """Create sample vehicle fleet."""
    return [
        Vehicle(
            id="truck_001",
            type=VehicleType.TRUCK,
            capacity=5000.0,  # kg
            max_range=500.0,  # km
            speed=80.0,  # km/h
            cost_per_km=1.5,
            emissions_per_km=0.2,
            location=(-122.4, 37.7),  # San Francisco
        ),
        Vehicle(
            id="van_001",
            type=VehicleType.VAN,
            capacity=1500.0,
            max_range=400.0,
            speed=60.0,
            cost_per_km=0.8,
            emissions_per_km=0.15,
            location=(-122.4, 37.7),
        ),
    ]


def create_sample_destinations() -> list[tuple[float, float]]:
    """Create sample delivery destinations."""
    # San Francisco area locations
    return [
        (-122.4194, 37.7749),  # Downtown SF
        (-122.4094, 37.7849),  # North SF
        (-122.4294, 37.7649),  # South SF
        (-122.3894, 37.7949),  # East SF
    ]


def main():
    """Run basic logistics routing example."""
    print("=" * 60)
    print("GEO-INFER-LOG: Basic Logistics Routing Example")
    print("=" * 60)

    # Step 1: Vehicle fleet setup
    print("\nStep 1: Setting up vehicle fleet...")
    vehicles = create_sample_vehicles()
    for vehicle in vehicles:
        print(f"   {vehicle.id}: {vehicle.type.value}, capacity: {vehicle.capacity}kg")

    # Step 2: Destination planning
    print("\nStep 2: Planning delivery destinations...")
    destinations = create_sample_destinations()
    for i, dest in enumerate(destinations, 1):
        print(f"   Destination {i}: ({dest[0]:.4f}, {dest[1]:.4f})")

    # Step 3: Fleet registration
    print("\nStep 3: Registering fleet...")
    fleet = FleetManager()
    for vehicle in vehicles:
        fleet.add_vehicle(vehicle)
    status = fleet.get_fleet_status()
    print(f"   Fleet ready: {status['total_vehicles']} vehicles")

    # Step 4: Route optimization (network-free haversine routing)
    print("\nStep 4: Route optimization...")
    params = RoutingParameters(
        weight_factor="time",
        avoid_tolls=False,
        traffic_model="best_guess",
    )
    optimizer = RouteOptimizer(params)
    print(f"   Weight factor: {optimizer.parameters.weight_factor}")
    print(f"   Traffic model: {optimizer.parameters.traffic_model}")

    origin = (-122.4, 37.7)
    route = optimizer.optimize_route(
        origin=origin,
        destination=destinations[-1],
        waypoints=destinations[:-1],
    )
    print(f"   Route distance: {route['distance']:.2f} km")
    print(f"   Estimated travel time: {route['travel_time']:.1f} min")
    print(f"   Stops: {len(route['path'])}")

    # Summary
    print("\n" + "=" * 60)
    print("Logistics routing example complete!")
    print("=" * 60)
    print("\nKey capabilities demonstrated:")
    print("  - Vehicle fleet management")
    print("  - Network-free route optimization")
    print("  - Delivery planning with waypoints")


if __name__ == "__main__":
    main()
