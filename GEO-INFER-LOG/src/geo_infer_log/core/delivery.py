"""
Last-mile delivery components for the GEO-INFER-LOG module.

This module provides classes for optimizing last-mile delivery,
service area analysis, and delivery scheduling.
"""

import logging
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, voronoi_diagram
from datetime import datetime, timedelta

from geo_infer_log.models.schemas import Vehicle, Location, Route, RoutingParameters
from geo_infer_log.core.routing import RouteOptimizer

try:
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# WGS84 constants
_EARTH_RADIUS_KM = 6371.0
_DEG_PER_KM_LAT = 1.0 / 111.32


def _deg_per_km_lon(lat_deg: float) -> float:
    """Return the number of degrees of longitude per km at a given latitude."""
    return 1.0 / (111.32 * math.cos(math.radians(lat_deg)))


class LastMileRouter:
    """Specialized routing for last-mile delivery."""

    def __init__(self, parameters: RoutingParameters | None = None):
        """Initialize a last-mile router.

        Args:
            parameters: Routing parameters
        """
        self.parameters = parameters or RoutingParameters()
        self.route_optimizer = RouteOptimizer(parameters)
        self.service_areas: dict[str, Polygon] = {}  # depot_id -> service area polygon

    def load_network(self, network_file: str) -> None:
        """Load a transportation network from a file.

        Args:
            network_file: Path to network file
        """
        self.route_optimizer.load_network(network_file)

    def define_service_area(
        self, depot_id: str, depot_location: tuple[float, float], max_distance: float
    ) -> Polygon:
        """Define a service area around a depot.

        Args:
            depot_id: ID of the depot
            depot_location: (lon, lat) of the depot
            max_distance: Maximum service distance in km

        Returns:
            Polygon representing the service area
        """
        lon, lat = depot_location
        # Build an elliptical buffer that accounts for latitude-dependent
        # distortion of longitude, giving a more accurate circular coverage
        # area on the WGS84 ellipsoid.
        deg_lat = max_distance * _DEG_PER_KM_LAT
        deg_lon = max_distance * _deg_per_km_lon(lat)

        # Approximate a circle with an elliptical Shapely buffer using
        # an affine-scaled unit circle (64-segment resolution).
        point = Point(lon, lat)
        # Scale to unit circle, buffer, scale back
        from shapely import affinity

        scaled = affinity.scale(point, xfact=1.0 / deg_lon, yfact=1.0 / deg_lat)
        circle = scaled.buffer(1.0, quad_segs=64)
        service_area = affinity.scale(circle, xfact=deg_lon, yfact=deg_lat)

        self.service_areas[depot_id] = service_area
        logger.info(
            "Service area defined for depot '%s': %.2f km radius",
            depot_id,
            max_distance,
        )
        return service_area

    def optimize_deliveries(
        self,
        depot: Location,
        deliveries: list[Location],
        vehicles: list[Vehicle],
        constraints: dict,
    ) -> list[Route]:
        """Optimize deliveries from a depot.

        Args:
            depot: Depot location
            deliveries: Delivery locations
            vehicles: Available vehicles
            constraints: Delivery constraints

        Returns:
            List of optimized routes
        """
        # Check if all deliveries are within service area
        if depot.name in self.service_areas:
            service_area = self.service_areas[depot.name]
            for delivery in deliveries:
                point = Point(delivery.coordinates)
                if not service_area.contains(point):
                    logger.warning(
                        "Delivery to %s is outside the service area", delivery.name
                    )

        # Group deliveries into clusters
        clusters = self._cluster_deliveries(deliveries, len(vehicles))

        # Optimize routes for each cluster
        routes: list[Route] = []
        for i, cluster in enumerate(clusters):
            if i >= len(vehicles):
                break

            vehicle = vehicles[i]

            # Create route using the route optimizer
            waypoints = [d.coordinates for d in cluster]
            route_data = self.route_optimizer.optimize_route(
                origin=depot.coordinates,
                destination=depot.coordinates,
                waypoints=waypoints,
            )

            # Create Route object
            route = Route(
                id=f"route-{len(routes) + 1}",
                vehicle_id=vehicle.id,
                stops=[depot] + cluster + [depot],
                departure_time=datetime.now(),
                estimated_arrival_time=datetime.now()
                + timedelta(minutes=route_data["travel_time"]),
                total_distance=route_data["distance"],
                total_time=route_data["travel_time"],
                total_cost=route_data["distance"] * vehicle.cost_per_km,
                total_emissions=route_data["distance"] * vehicle.emissions_per_km,
                geometry=route_data["geometry"].__geo_interface__
                if hasattr(route_data["geometry"], "__geo_interface__")
                else None,
            )

            routes.append(route)

        return routes

    def _cluster_deliveries(
        self, deliveries: list[Location], num_clusters: int
    ) -> list[list[Location]]:
        """Cluster delivery locations.

        Args:
            deliveries: Delivery locations
            num_clusters: Number of clusters to create

        Returns:
            List of delivery clusters
        """
        if num_clusters >= len(deliveries):
            return [[d] for d in deliveries]

        if num_clusters <= 0:
            return [deliveries]

        coords = np.array([d.coordinates for d in deliveries])

        if HAS_SKLEARN and len(deliveries) >= num_clusters:
            # Real KMeans geographic clustering
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(coords)
            clusters: list[list[Location]] = [[] for _ in range(num_clusters)]
            for delivery, label in zip(deliveries, labels):
                clusters[label].append(delivery)
            # Remove empty clusters
            clusters = [c for c in clusters if c]
            logger.info(
                "KMeans clustered %d deliveries into %d groups",
                len(deliveries),
                len(clusters),
            )
        else:
            # Fallback: spatial-median split based on alternating axes
            clusters = [[] for _ in range(num_clusters)]
            sorted_deliveries = sorted(deliveries, key=lambda d: d.coordinates[0])
            for i, delivery in enumerate(sorted_deliveries):
                clusters[i % num_clusters].append(delivery)

        return clusters


class DeliveryScheduler:
    """Schedules and manages delivery operations."""

    def __init__(self, router: LastMileRouter):
        """Initialize a delivery scheduler.

        Args:
            router: Last-mile router for optimizing deliveries
        """
        self.router = router
        self.schedule: dict[str, list[Route]] = {}  # date -> list of routes
        # vehicle_id -> list of routes
        self.vehicle_assignments: dict[str, list[Route]] = {}

    def create_schedule(
        self,
        depot: Location,
        deliveries: list[Location],
        vehicles: list[Vehicle],
        start_date: datetime,
        end_date: datetime,
        max_deliveries_per_day: int,
    ) -> dict:
        """Create a delivery schedule for a date range.

        Args:
            depot: Depot location
            deliveries: All delivery locations
            vehicles: Available vehicles
            start_date: Start date for scheduling
            end_date: End date for scheduling
            max_deliveries_per_day: Maximum deliveries per day

        Returns:
            Dictionary with schedule information
        """
        # Distribute deliveries across days
        current_date = start_date
        remaining_deliveries = deliveries.copy()

        while current_date <= end_date and remaining_deliveries:
            # Select deliveries for this day
            day_deliveries = remaining_deliveries[:max_deliveries_per_day]
            remaining_deliveries = remaining_deliveries[max_deliveries_per_day:]

            # Optimize routes for this day
            day_routes = self.router.optimize_deliveries(
                depot=depot,
                deliveries=day_deliveries,
                vehicles=vehicles,
                constraints={},
            )

            # Add to schedule
            date_str = current_date.strftime("%Y-%m-%d")
            self.schedule[date_str] = day_routes

            # Update vehicle assignments
            for route in day_routes:
                if route.vehicle_id not in self.vehicle_assignments:
                    self.vehicle_assignments[route.vehicle_id] = []
                self.vehicle_assignments[route.vehicle_id].append(route)

            # Move to next day
            current_date += timedelta(days=1)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_deliveries": len(deliveries),
            "scheduled_deliveries": len(deliveries) - len(remaining_deliveries),
            "unscheduled_deliveries": len(remaining_deliveries),
            "total_routes": sum(len(routes) for routes in self.schedule.values()),
        }

    def get_daily_schedule(self, date: datetime) -> list[Route]:
        """Get the delivery schedule for a specific day.

        Args:
            date: Date to get schedule for

        Returns:
            List of routes scheduled for the day
        """
        date_str = date.strftime("%Y-%m-%d")
        return self.schedule.get(date_str, [])

    def get_vehicle_schedule(self, vehicle_id: str) -> list[Route]:
        """Get the schedule for a specific vehicle.

        Args:
            vehicle_id: ID of the vehicle

        Returns:
            List of routes assigned to the vehicle
        """
        return self.vehicle_assignments.get(vehicle_id, [])

    def reschedule_delivery(
        self, route_id: str, delivery_idx: int, new_date: datetime
    ) -> dict:
        """Reschedule a delivery to a different date.

        Args:
            route_id: ID of the route containing the delivery
            delivery_idx: Index of the delivery in the route
            new_date: New date for the delivery

        Returns:
            Dictionary with rescheduling information
        """
        new_date_str = new_date.strftime("%Y-%m-%d")

        # Find and remove the delivery from its current route
        original_route = None
        removed_delivery = None
        for routes in self.schedule.values():
            for route in routes:
                if route.id == route_id:
                    original_route = route
                    if 0 <= delivery_idx < len(route.stops):
                        removed_delivery = route.stops[delivery_idx]
                        route.stops = (
                            route.stops[:delivery_idx] + route.stops[delivery_idx + 1 :]
                        )
                    break
            if original_route:
                break

        if removed_delivery is None:
            logger.warning(
                "Could not find route '%s' delivery idx %d", route_id, delivery_idx
            )
            return {"success": False, "reason": "delivery_not_found"}

        # Insert delivery into the target date's schedule
        if new_date_str not in self.schedule:
            self.schedule[new_date_str] = []

        # Append to the first existing route on that day, or create a new one
        target_routes = self.schedule[new_date_str]
        if target_routes:
            target_routes[0].stops.insert(
                -1, removed_delivery
            )  # before return-to-depot
            new_route = target_routes[0]
        else:
            new_route = Route(
                id=f"route-resched-{new_date_str}",
                vehicle_id=original_route.vehicle_id
                if original_route
                else "unassigned",
                stops=[removed_delivery],
                departure_time=new_date,
                estimated_arrival_time=new_date + timedelta(hours=1),
                total_distance=0.0,
                total_time=0.0,
                total_cost=0.0,
                total_emissions=0.0,
                geometry=None,
            )
            target_routes.append(new_route)

        logger.info(
            "Rescheduled delivery from route '%s' idx %d to %s",
            route_id,
            delivery_idx,
            new_date_str,
        )
        return {
            "success": True,
            "original_route": original_route.id if original_route else None,
            "new_route": new_route.id,
            "new_date": new_date_str,
        }


class ServiceAreaAnalyzer:
    """Analyzes and optimizes delivery service areas."""

    def __init__(self) -> None:
        """Initialize a service area analyzer."""
        self.service_areas: dict[str, Polygon] = {}  # depot_id -> service area polygon

    def create_service_area(
        self,
        depot_id: str,
        depot_location: tuple[float, float],
        max_time: int | None = None,
        max_distance: float | None = None,
    ) -> gpd.GeoDataFrame:
        """Create a service area around a depot.

        Args:
            depot_id: ID of the depot
            depot_location: (lon, lat) of the depot
            max_time: Maximum travel time in minutes
            max_distance: Maximum travel distance in km

        Returns:
            GeoDataFrame with service area
        """
        lon, lat = depot_location

        # Convert time-based specification to distance
        if max_distance is None and max_time is not None:
            # Conservative urban average: 25 km/h accounting for stops
            max_distance = max_time / 60.0 * 25.0
        elif max_distance is None:
            max_distance = 10.0  # Default 10 km

        # Build multi-ring isochrone approximation using latitude-aware
        # elliptical buffers for WGS84 accuracy
        from shapely import affinity

        point = Point(lon, lat)
        deg_lat = _DEG_PER_KM_LAT
        deg_lon = _deg_per_km_lon(lat)

        rings = []
        ring_distances = [max_distance * f for f in [0.33, 0.67, 1.0]]
        for ring_km in ring_distances:
            scaled = affinity.scale(
                point, xfact=1.0 / (ring_km * deg_lon), yfact=1.0 / (ring_km * deg_lat)
            )
            circle = scaled.buffer(1.0, quad_segs=64)
            ring = affinity.scale(
                circle, xfact=ring_km * deg_lon, yfact=ring_km * deg_lat
            )
            rings.append(ring)

        service_area = rings[-1]  # outermost ring

        # Compute real ellipsoidal area approximation
        area_km2 = service_area.area / (deg_lat * deg_lon)

        gdf = gpd.GeoDataFrame(
            {
                "depot_id": [depot_id] * len(rings),
                "ring_km": ring_distances,
                "max_time": [max_time] * len(rings),
                "area_km2": [r.area / (deg_lat * deg_lon) for r in rings],
            },
            geometry=rings,
        )

        self.service_areas[depot_id] = service_area
        logger.info(
            "Service area created for depot '%s': %.1f km, %.1f km²",
            depot_id,
            max_distance,
            area_km2,
        )
        return gdf

    def analyze_coverage(
        self, service_areas: dict[str, Polygon], demand_points: gpd.GeoDataFrame
    ) -> dict:
        """Analyze coverage of demand points by service areas.

        Args:
            service_areas: Dictionary of depot_id -> service area polygon
            demand_points: GeoDataFrame with demand points

        Returns:
            Dictionary with coverage metrics
        """
        # Calculate coverage
        covered_points = 0
        depot_coverage = {}

        for depot_id, area in service_areas.items():
            points_in_area = demand_points[demand_points.geometry.within(area)]
            depot_coverage[depot_id] = len(points_in_area)
            covered_points += len(points_in_area)

        # Calculate metrics
        total_points = len(demand_points)
        coverage_ratio = covered_points / total_points if total_points > 0 else 0

        return {
            "total_points": total_points,
            "covered_points": covered_points,
            "uncovered_points": total_points - covered_points,
            "coverage_ratio": coverage_ratio,
            "depot_coverage": depot_coverage,
        }

    def optimize_service_areas(
        self,
        depot_locations: list[tuple[str, tuple[float, float]]],
        demand_points: gpd.GeoDataFrame,
        max_distance: float,
    ) -> dict[str, Polygon]:
        """Optimize service areas for multiple depots.

        Args:
            depot_locations: List of (depot_id, (lon, lat)) tuples
            demand_points: GeoDataFrame with demand points
            max_distance: Maximum service distance in km

        Returns:
            Dictionary of depot_id -> optimized service area
        """
        if not depot_locations:
            return {}

        # Build Voronoi tessellation from depot points to partition the
        # demand space, then intersect each Voronoi cell with the
        # max-distance buffer to produce bounded service areas.
        from shapely import affinity

        depot_points = {did: Point(loc) for did, loc in depot_locations}
        all_points = unary_union(list(depot_points.values()))

        if len(depot_locations) == 1:
            # Single depot: just buffer
            did, loc = depot_locations[0]
            lat = loc[1]
            deg_lat = max_distance * _DEG_PER_KM_LAT
            deg_lon = max_distance * _deg_per_km_lon(lat)
            scaled = affinity.scale(
                Point(loc), xfact=1.0 / deg_lon, yfact=1.0 / deg_lat
            )
            circle = scaled.buffer(1.0, quad_segs=64)
            area = affinity.scale(circle, xfact=deg_lon, yfact=deg_lat)
            self.service_areas = {did: area}
            return self.service_areas

        # Compute Voronoi diagram
        voronoi_geom = voronoi_diagram(all_points)

        optimized_areas = {}
        for depot_id, loc in depot_locations:
            depot_point = depot_points[depot_id]
            lat = loc[1]
            deg_lat = max_distance * _DEG_PER_KM_LAT
            deg_lon = max_distance * _deg_per_km_lon(lat)

            # Find the Voronoi cell containing this depot
            voronoi_cell = None
            for geom in voronoi_geom.geoms:
                if geom.contains(depot_point):
                    voronoi_cell = geom
                    break
            if voronoi_cell is None:
                # Fallback: nearest cell
                voronoi_cell = min(
                    voronoi_geom.geoms, key=lambda g: g.distance(depot_point)
                )

            # Intersect Voronoi cell with distance buffer
            scaled = affinity.scale(
                depot_point, xfact=1.0 / deg_lon, yfact=1.0 / deg_lat
            )
            circle = scaled.buffer(1.0, quad_segs=64)
            buffer_area = affinity.scale(circle, xfact=deg_lon, yfact=deg_lat)
            optimized_areas[depot_id] = voronoi_cell.intersection(buffer_area)

        self.service_areas = optimized_areas
        logger.info(
            "Optimized %d service areas using Voronoi tessellation (max %.1f km)",
            len(optimized_areas),
            max_distance,
        )
        return optimized_areas
