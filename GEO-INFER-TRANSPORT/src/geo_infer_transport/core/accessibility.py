"""
Accessibility analysis module.

Provides isochrone analysis, service area calculation,
and accessibility metrics for transportation networks.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
import math

logger = logging.getLogger(__name__)


@dataclass
class Isochrone:
    """Represents an isochrone (travel time contour)."""

    center: dict[str, float]
    time_minutes: float
    mode: str
    geometry: dict[str, Any]  # GeoJSON polygon
    area_sq_km: float
    reachable_nodes: list[str] = field(default_factory=list)


@dataclass
class ServiceArea:
    """Represents a service area analysis result."""

    facility_id: str
    location: dict[str, float]
    break_values: list[float]
    polygons: list[dict[str, Any]]
    population_covered: int = 0
    coverage_statistics: dict[str, Any] = field(default_factory=dict)


class AccessibilityAnalyzer:
    """
    Analyze accessibility and generate service areas.

    Supports isochrone calculation, equity analysis,
    and multi-modal accessibility metrics.
    """

    def __init__(
        self,
        network: Any = None,
        default_mode: str = "car",
        population_data: dict[str, Any] | None = None,
    ):
        """
        Initialize accessibility analyzer.

        Args:
            network: Transport network
            default_mode: Default transport mode
            population_data: Population distribution data
        """
        self.network = network
        self.default_mode = default_mode
        self.population_data = population_data or {}
        logger.info("Initialized AccessibilityAnalyzer")

    def set_network(self, network: Any) -> None:
        """Set the transport network."""
        self.network = network

    def calculate_isochrone(
        self,
        origin: dict[str, Any],
        travel_times: list[float],
        mode: str = "car",
        departure_time: datetime | None = None,
    ) -> list[Isochrone]:
        """
        Calculate isochrones from an origin point.

        Args:
            origin: Origin point or node
            travel_times: Travel time thresholds in minutes
            mode: Transport mode
            departure_time: Departure time for time-dependent routing

        Returns:
            List of Isochrone objects for each time threshold
        """
        isochrones: list[Isochrone] = []
        origin_id = origin.get("node_id") or origin.get("id")
        origin_loc = origin.get("location", {"lat": 0, "lon": 0})

        for time_limit in sorted(travel_times):
            reachable = []

            if self.network and hasattr(self.network, "graph"):
                import networkx as nx

                graph = self.network.graph

                # Calculate shortest paths from origin
                try:
                    lengths = nx.single_source_dijkstra_path_length(
                        graph, origin_id, cutoff=time_limit * 60, weight="travel_time"
                    )
                    reachable = list(lengths.keys())
                except (nx.NetworkXError, nx.NodeNotFound, KeyError):
                    logger.warning(
                        "Origin %r not routable in network; isochrone degenerates "
                        "to the origin node only",
                        origin_id,
                    )
                    reachable = [origin_id]
            else:
                # Estimate reachable area without network
                reachable = [origin_id]

            # Estimate service area via the bounding-square heuristic
            # At 50 km/h, in time_limit minutes, you can travel time_limit * 50/60 km
            avg_speed = 50 if mode == "car" else 15 if mode == "bicycle" else 5
            radius_km = time_limit * avg_speed / 60
            area_sq_km = math.pi * radius_km**2

            # Generate the bounding-square service polygon
            polygon = self._generate_isochrone_polygon(origin_loc, radius_km)

            isochrone = Isochrone(
                center=origin_loc,
                time_minutes=time_limit,
                mode=mode,
                geometry=polygon,
                area_sq_km=round(area_sq_km, 2),
                reachable_nodes=reachable,
            )
            isochrones.append(isochrone)

        logger.info(f"Calculated {len(isochrones)} isochrones from {origin_id}")
        return isochrones

    def _generate_isochrone_polygon(
        self, center: dict[str, float], radius_km: float, num_points: int = 32
    ) -> dict[str, Any]:
        """Generate a circular polygon approximation for isochrone."""
        lat = center.get("lat", 0)
        lon = center.get("lon", 0)

        # Generate points around the circle
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Approximate degree offset (1 degree ≈ 111 km)
            dlat = radius_km / 111 * math.cos(angle)
            dlon = radius_km / (111 * math.cos(math.radians(lat))) * math.sin(angle)

            points.append([lon + dlon, lat + dlat])

        # Close the polygon
        points.append(points[0])

        return {"type": "Polygon", "coordinates": [points]}

    def generate_service_area(
        self,
        facilities: list[dict[str, Any]],
        breaks: list[float],
        mode: str = "car",
        dissolve: bool = True,
    ) -> list[ServiceArea]:
        """
        Generate service areas for facilities.

        Args:
            facilities: Facility locations
            breaks: Distance or time break values
            mode: Transport mode
            dissolve: Whether to dissolve overlapping areas

        Returns:
            List of ServiceArea objects
        """
        service_areas: list[ServiceArea] = []

        for facility in facilities:
            facility_id = facility.get("id", f"facility_{len(service_areas)}")
            location = facility.get("location", {})

            # Generate polygons for each break
            polygons: list[dict[str, Any]] = []
            for break_value in sorted(breaks):
                polygon = self._generate_isochrone_polygon(location, break_value)
                polygons.append({"break_value": break_value, "geometry": polygon})

            # Calculate population covered
            pop_covered = self._estimate_population_coverage(location, max(breaks))

            service_area = ServiceArea(
                facility_id=facility_id,
                location=location,
                break_values=breaks,
                polygons=polygons,
                population_covered=pop_covered,
                coverage_statistics={
                    "mode": mode,
                    "max_service_distance": max(breaks),
                    "area_sq_km": math.pi * max(breaks) ** 2,
                },
            )
            service_areas.append(service_area)

        logger.info(f"Generated service areas for {len(facilities)} facilities")
        return service_areas

    def _estimate_population_coverage(
        self, center: dict[str, float], radius_km: float
    ) -> int:
        """Estimate population within a radius."""
        # Default population density estimate (people per sq km)
        density = self.population_data.get("average_density", 1000)
        area = math.pi * radius_km**2
        return int(density * area)

    def analyze_equity(
        self,
        population_groups: dict[str, Any],
        accessibility_scores: dict[str, float],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze accessibility equity across population groups.

        Args:
            population_groups: Population by group
            accessibility_scores: Accessibility scores by area
            metrics: Equity metrics to calculate

        Returns:
            Equity analysis results
        """
        metrics = metrics or ["gini", "mean_difference"]

        # Calculate group averages
        group_scores: dict[str, Any] = {}
        for group_id, group_data in population_groups.items():
            areas = group_data.get("areas", [])
            scores = [accessibility_scores.get(a, 0) for a in areas]
            group_scores[group_id] = {
                "population": group_data.get("population", 0),
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            }

        # Calculate overall statistics
        all_scores = list(accessibility_scores.values())
        overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0

        # Calculate the Gini coefficient via the Lorenz-area method
        sorted_scores = sorted(all_scores)
        n = len(sorted_scores)
        if n > 0 and sum(sorted_scores) > 0:
            numerator = sum((i + 1) * s for i, s in enumerate(sorted_scores))
            gini = (2 * numerator) / (n * sum(sorted_scores)) - (n + 1) / n
        else:
            gini = 0

        disparities_out: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "group_analysis": group_scores,
            "equity_metrics": {
                "gini_coefficient": round(gini, 4),
                "overall_mean_accessibility": round(overall_mean, 4),
                "score_range": max(all_scores) - min(all_scores) if all_scores else 0,
            },
            "disparities": disparities_out,
        }

        # Identify disparities
        for group_id, data in group_scores.items():
            if data["mean_score"] < overall_mean * 0.8:
                disparities_out.append(
                    {
                        "group": group_id,
                        "mean_score": data["mean_score"],
                        "gap": round(overall_mean - data["mean_score"], 4),
                    }
                )

        logger.info(f"Equity analysis completed for {len(population_groups)} groups")
        return result

    def calculate_accessibility_index(
        self,
        origin: dict[str, Any],
        destinations: list[dict[str, Any]],
        decay_function: str = "exponential",
        beta: float = 0.1,
    ) -> dict[str, Any]:
        """
        Calculate accessibility index using gravity-based approach.

        Args:
            origin: Origin location
            destinations: Destination opportunities
            decay_function: Distance decay function
            beta: Decay parameter

        Returns:
            Accessibility index and components
        """
        origin_loc = origin.get("location", {})

        total_accessibility = 0
        components = []

        for dest in destinations:
            dest_loc = dest.get("location", {})
            opportunity = dest.get("weight", 1.0)

            # Calculate distance
            distance = self._haversine_distance(origin_loc, dest_loc)

            # Apply decay function
            if decay_function == "exponential":
                decay = math.exp(-beta * distance)
            elif decay_function == "power":
                decay = 1 / (distance**beta) if distance > 0 else 1
            else:  # Linear
                decay = max(0, 1 - beta * distance)

            contribution = opportunity * decay
            total_accessibility += contribution

            components.append(
                {
                    "destination_id": dest.get("id"),
                    "distance_km": round(distance, 2),
                    "opportunity": opportunity,
                    "decay": round(decay, 4),
                    "contribution": round(contribution, 4),
                }
            )

        result = {
            "origin": origin.get("id"),
            "accessibility_index": round(total_accessibility, 4),
            "decay_function": decay_function,
            "beta": beta,
            "destination_count": len(destinations),
            "components": sorted(
                components, key=lambda x: x["contribution"], reverse=True
            )[:10],
        }

        logger.info(f"Accessibility index: {total_accessibility:.4f}")
        return result

    def _haversine_distance(
        self, loc1: dict[str, float], loc2: dict[str, float]
    ) -> float:
        """Calculate haversine distance in km."""
        lat1 = math.radians(loc1.get("lat", 0))
        lat2 = math.radians(loc2.get("lat", 0))
        lon1 = math.radians(loc1.get("lon", 0))
        lon2 = math.radians(loc2.get("lon", 0))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return 6371 * c  # Earth radius in km
