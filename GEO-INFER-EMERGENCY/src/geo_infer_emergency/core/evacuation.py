"""
Evacuation planning module.

Provides evacuation zone delineation, route optimization,
shelter management, and clearance time estimation.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import networkx as nx

logger = logging.getLogger(__name__)


class EvacuationLevel(Enum):
    """Evacuation alert levels."""

    WARNING = "warning"  # Be prepared to evacuate
    ORDER = "order"  # Evacuate immediately
    LIFT = "lift"  # Safe to return


@dataclass
class EvacuationZone:
    """Represents an evacuation zone."""

    zone_id: str
    name: str
    geometry: dict[str, Any]  # GeoJSON geometry
    population: int
    level: EvacuationLevel = EvacuationLevel.WARNING
    special_populations: list[str] = field(default_factory=list)


@dataclass
class Shelter:
    """Represents an evacuation shelter."""

    shelter_id: str
    name: str
    location: dict[str, Any]
    capacity: int
    current_occupancy: int = 0
    services: list[str] = field(default_factory=list)
    accessible: bool = True


@dataclass
class EvacuationRoute:
    """Represents an evacuation route."""

    route_id: str
    origin_zone: str
    destination_shelter: str
    path: list[dict[str, float]]  # List of coordinates
    distance_km: float
    estimated_time_minutes: float
    capacity_vehicles_per_hour: int


class EvacuationPlanner:
    """
    Plan and execute evacuations with route optimization,
    shelter management, and special population support.
    """

    def __init__(
        self,
        road_network: nx.Graph | None = None,
        population_data: dict[str, Any] | None = None,
        shelters: list[dict[str, Any]] | None = None,
        special_needs: list[str] | None = None,
    ):
        """
        Initialize evacuation planner.

        Args:
            road_network: NetworkX graph whose nodes are routable locations
                (zones, shelters, intersections). Edges may carry ``travel_time``
                (minutes), ``distance`` (km), and ``capacity``
                (vehicles/hour) attributes.
            population_data: Population demographics
            shelters: Available shelter locations
            special_needs: Special needs facilities to consider
        """
        self.road_network = road_network
        self.population_data = population_data
        self.special_needs = special_needs or ["hospitals", "nursing_homes", "schools"]
        self._zones: dict[str, EvacuationZone] = {}
        self._shelters: dict[str, Shelter] = {}
        self._routes: dict[str, EvacuationRoute] = {}

        # Register shelters
        if shelters:
            for shelter_data in shelters:
                self.register_shelter(shelter_data)

        logger.info("Initialized EvacuationPlanner")

    def register_shelter(self, shelter_data: dict[str, Any]) -> Shelter:
        """Register a shelter in the system."""
        shelter = Shelter(
            shelter_id=shelter_data.get("id", f"shelter_{len(self._shelters)}"),
            name=shelter_data.get("name", ""),
            location=shelter_data.get("location", {}),
            capacity=shelter_data.get("capacity", 100),
            services=shelter_data.get("services", []),
            accessible=shelter_data.get("accessible", True),
        )
        self._shelters[shelter.shelter_id] = shelter
        return shelter

    def plan(
        self,
        affected_zone: dict[str, Any],
        population: dict[str, Any],
        destinations: list[dict[str, Any]],
        phasing: str = "staged",
        contraflow: bool = False,
    ) -> dict[str, Any]:
        """
        Create an evacuation plan.

        Args:
            affected_zone: Zone requiring evacuation
            population: Population in affected area
            destinations: Shelter destinations
            phasing: Evacuation phasing strategy
            contraflow: Enable contraflow lanes

        Returns:
            Complete evacuation plan
        """
        plan_id = f"evac_plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create evacuation zone
        zone = EvacuationZone(
            zone_id=affected_zone.get("id", "zone_1"),
            name=affected_zone.get("name", "Evacuation Zone"),
            geometry=affected_zone.get("geometry", {}),
            population=population.get("total", 0),
            level=EvacuationLevel(affected_zone.get("level", "order")),
            special_populations=population.get("special_populations", []),
        )
        self._zones[zone.zone_id] = zone

        # Register destinations as shelters
        for dest in destinations:
            if dest.get("id") not in self._shelters:
                self.register_shelter(dest)

        # Generate routes
        routes = self.optimize_routes(
            origins=[zone.zone_id],
            destinations=[
                str(d.get("id")) for d in destinations if d.get("id") is not None
            ],
            objectives=["clearance_time", "safety"],
            constraints={"road_capacity": True, "contraflow": contraflow},
        )

        # Calculate phasing
        phases = self._calculate_phases(zone, phasing)

        # Estimate clearance time
        clearance = self.estimate_clearance_time(
            evacuation_plan={"zone": zone, "routes": routes},
            traffic_model="dynamic_assignment",
            scenarios=["expected"],
        )

        plan = {
            "plan_id": plan_id,
            "created_at": datetime.now().isoformat(),
            "affected_zone": {
                "id": zone.zone_id,
                "name": zone.name,
                "population": zone.population,
                "level": zone.level.value,
            },
            "destinations": [
                {"id": s.shelter_id, "name": s.name, "capacity": s.capacity}
                for s in self._shelters.values()
            ],
            "routes": routes.get("routes", []),
            "phasing": {"strategy": phasing, "phases": phases},
            "contraflow": {
                "enabled": contraflow,
                "segments": self._identify_contraflow_segments() if contraflow else [],
            },
            "estimated_clearance_time_hours": clearance.get("expected", {}).get(
                "clearance_hours", 0
            ),
            "special_populations": self._plan_special_populations(zone),
            "status": "planned",
        }

        logger.info(f"Created evacuation plan {plan_id} for {zone.population} people")
        return plan

    def _calculate_phases(
        self, zone: EvacuationZone, strategy: str
    ) -> list[dict[str, Any]]:
        """Calculate evacuation phases."""
        phases = []

        if strategy == "staged":
            # Stage by distance from hazard
            phases = [
                {
                    "phase": 1,
                    "description": "Immediate danger zone",
                    "population_pct": 30,
                    "delay_hours": 0,
                },
                {
                    "phase": 2,
                    "description": "High risk zone",
                    "population_pct": 40,
                    "delay_hours": 2,
                },
                {
                    "phase": 3,
                    "description": "Precautionary zone",
                    "population_pct": 30,
                    "delay_hours": 4,
                },
            ]
        elif strategy == "simultaneous":
            phases = [
                {
                    "phase": 1,
                    "description": "All zones",
                    "population_pct": 100,
                    "delay_hours": 0,
                }
            ]
        elif strategy == "time_phased":
            # Phase by time of day
            phases = [
                {
                    "phase": 1,
                    "description": "Morning departure",
                    "population_pct": 40,
                    "delay_hours": 0,
                },
                {
                    "phase": 2,
                    "description": "Afternoon departure",
                    "population_pct": 35,
                    "delay_hours": 4,
                },
                {
                    "phase": 3,
                    "description": "Evening departure",
                    "population_pct": 25,
                    "delay_hours": 8,
                },
            ]

        return phases

    def _identify_contraflow_segments(self) -> list[dict[str, Any]]:
        """Identify road segments flagged as contraflow-capable in the network.

        A segment qualifies when its edge carries a truthy
        ``contraflow_capable`` attribute. Networks without such
        attributes yield no contraflow candidates.
        """
        if self.road_network is None:
            return []
        segments: list[dict[str, Any]] = []
        for u, v, data in self.road_network.edges(data=True):
            if data.get("contraflow_capable"):
                segments.append(
                    {
                        "road": data.get("name", f"{u}->{v}"),
                        "from_node": u,
                        "to_node": v,
                        "lanes_reversed": int(data.get("lanes_reversed", 1)),
                    }
                )
        return segments

    def _plan_special_populations(self, zone: EvacuationZone) -> dict[str, Any]:
        """Plan for special populations in the zone."""
        return {
            "facilities": zone.special_populations,
            "transportation_needed": True,
            "medical_support_needed": "hospitals" in zone.special_populations,
            "accessible_vehicles_needed": True,
        }

    def optimize_routes(
        self,
        origins: list[str],
        destinations: list[str],
        objectives: list[str],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Optimize evacuation routes with Dijkstra shortest paths.

        Routes are computed over ``self.road_network``. Every origin and
        destination must be a node of that network. Edge weights follow
        this precedence:

        * ``travel_time`` (minutes) when present on edges - drives
          ``estimated_time_minutes``;
        * otherwise ``distance`` (km) - drives both the route weight and
          ``distance_km``;
        * otherwise the graph is traversed unweighted and
          ``distance_km``/``estimated_time_minutes`` are the hop counts.

        ``distance_km`` prefers the ``distance`` edge attribute, falling
        back to ``length_m``/``length`` (metres, converted). Route
        capacity is the minimum ``capacity`` (vehicles/hour) along the
        path, or 0 when the network carries no capacity attributes.

        Args:
            origins: Origin zone IDs (nodes of the road network)
            destinations: Destination shelter IDs (nodes of the road network)
            objectives: Optimization objectives (recorded in the result)
            constraints: Routing constraints (recorded in the result)

        Returns:
            Optimized routes keyed by origin/destination pairs

        Raises:
            ValueError: If no road network is configured or an origin or
                destination is not a node of the road network.
        """
        if self.road_network is None:
            raise ValueError("road_network required for route optimization")

        graph = self.road_network
        unknown = [n for n in list(origins) + list(destinations) if n not in graph]
        if unknown:
            raise ValueError(
                f"endpoints not found in road_network: {sorted(set(unknown))}"
            )

        has_travel_time = self._edges_have(graph, "travel_time")
        has_distance = self._edges_have(graph, "distance")
        weight = (
            "travel_time" if has_travel_time else ("distance" if has_distance else None)
        )

        routes = []
        for origin in origins:
            for destination in destinations:
                route_id = f"route_{origin}_{destination}"
                try:
                    path = nx.dijkstra_path(graph, origin, destination, weight=weight)
                except nx.NetworkXNoPath:
                    logger.warning(f"No path between {origin} and {destination}")
                    continue

                distance_km = self._path_distance_km(graph, path)
                if has_travel_time:
                    estimated_time_minutes = float(
                        nx.shortest_path_length(
                            graph, origin, destination, weight="travel_time"
                        )
                    )
                elif has_distance:
                    estimated_time_minutes = float(
                        nx.shortest_path_length(
                            graph, origin, destination, weight="distance"
                        )
                    )
                else:
                    estimated_time_minutes = float(len(path) - 1)

                capacities = [
                    data["capacity"]
                    for _, _, data in _path_edges(graph, path)
                    if "capacity" in data
                ]
                capacity_vehicles_per_hour = int(min(capacities)) if capacities else 0

                routes.append(
                    {
                        "route_id": route_id,
                        "origin": origin,
                        "destination": destination,
                        "path": list(path),
                        "distance_km": round(distance_km, 3),
                        "estimated_time_minutes": round(estimated_time_minutes, 2),
                        "capacity_vehicles_per_hour": capacity_vehicles_per_hour,
                        "hazards": [],
                        "accessibility_score": 0.9,
                    }
                )

        result = {
            "optimization_objectives": objectives,
            "constraints": constraints,
            "routes": routes,
            "total_routes": len(routes),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Optimized {len(routes)} evacuation routes")
        return result

    @staticmethod
    def _edges_have(graph: nx.Graph, attribute: str) -> bool:
        """Return True when at least one edge carries the attribute."""
        return any(attribute in data for _, _, data in graph.edges(data=True))

    @staticmethod
    def _path_distance_km(graph: nx.Graph, path: list[str]) -> float:
        """Sum path length in kilometres.

        Per edge, uses ``distance`` (km) when present, otherwise
        ``length_m``/``length`` (metres, converted); edges carrying
        neither contribute 0.
        """
        total = 0.0
        for _u, _v, data in _path_edges(graph, path):
            if "distance" in data:
                total += float(data["distance"])
            elif "length_m" in data:
                total += float(data["length_m"]) * 0.001
            elif "length" in data:
                total += float(data["length"]) * 0.001
        return total

    def plan_shelters(
        self,
        shelter_locations: list[dict[str, Any]],
        population_estimate: int,
        duration_days: int,
        services: list[str],
    ) -> dict[str, Any]:
        """
        Plan shelter operations.

        Args:
            shelter_locations: Available shelter locations
            population_estimate: Estimated evacuee count
            duration_days: Expected shelter duration
            services: Required services

        Returns:
            Shelter operations plan
        """
        # Register shelters
        for loc in shelter_locations:
            if loc.get("id") not in self._shelters:
                self.register_shelter(loc)

        total_capacity = sum(s.capacity for s in self._shelters.values())
        capacity_sufficient = total_capacity >= population_estimate

        shelter_assignments = []
        remaining_population = population_estimate

        for shelter in self._shelters.values():
            assigned = min(remaining_population, shelter.capacity)
            shelter_assignments.append(
                {
                    "shelter_id": shelter.shelter_id,
                    "name": shelter.name,
                    "capacity": shelter.capacity,
                    "assigned_population": assigned,
                    "utilization": assigned / shelter.capacity
                    if shelter.capacity > 0
                    else 0,
                }
            )
            remaining_population -= assigned
            if remaining_population <= 0:
                break

        plan = {
            "population_estimate": population_estimate,
            "duration_days": duration_days,
            "total_shelter_capacity": total_capacity,
            "capacity_sufficient": capacity_sufficient,
            "overflow_population": max(0, remaining_population),
            "shelters": shelter_assignments,
            "services": {service: "planned" for service in services},
            "resource_requirements": {
                "cots": population_estimate,
                "blankets": population_estimate * 2,
                "meals_per_day": population_estimate * 3,
                "water_gallons_per_day": population_estimate * 3,
            },
            "staffing_needs": {
                "shelter_managers": len(self._shelters),
                "volunteers_per_shift": population_estimate // 50,
                "medical_staff": "medical" in services,
            },
        }

        logger.info(f"Planned shelters for {population_estimate} evacuees")
        return plan

    def plan_special_populations(
        self,
        facilities: list[dict[str, Any]],
        transportation: list[dict[str, Any]],
        receiving_facilities: list[dict[str, Any]],
        medical_support: bool = True,
    ) -> dict[str, Any]:
        """
        Plan evacuation of special populations.

        Args:
            facilities: Special needs facilities to evacuate
            transportation: Available specialized transport
            receiving_facilities: Destination facilities
            medical_support: Whether medical support is needed

        Returns:
            Special populations evacuation plan
        """
        plans = []

        for facility in facilities:
            facility_type = facility.get("type", "unknown")
            population = facility.get("population", 0)

            # Determine transport requirements
            transport_type = (
                "ambulance" if facility_type == "hospital" else "accessible_bus"
            )
            transport_capacity = 4 if transport_type == "ambulance" else 30
            trips_needed = (population + transport_capacity - 1) // transport_capacity

            # Find appropriate receiving facility
            receiving = None
            for recv in receiving_facilities:
                if (
                    recv.get("type") == facility_type
                    and recv.get("available_capacity", 0) >= population
                ):
                    receiving = recv
                    break

            plan = {
                "facility_id": facility.get("id"),
                "facility_name": facility.get("name"),
                "facility_type": facility_type,
                "population": population,
                "transport_type": transport_type,
                "trips_required": trips_needed,
                "receiving_facility": receiving.get("name")
                if receiving
                else "To be determined",
                "medical_support": medical_support and facility_type == "hospital",
                "priority": "high"
                if facility_type in ["hospital", "dialysis_center"]
                else "medium",
                "estimated_evacuation_hours": trips_needed * 1.5,
            }
            plans.append(plan)

        result = {
            "total_facilities": len(facilities),
            "total_population": sum(f.get("population", 0) for f in facilities),
            "facility_plans": plans,
            "transport_requirements": {
                "ambulances": sum(
                    1 for p in plans if p["transport_type"] == "ambulance"
                ),
                "accessible_buses": sum(
                    1 for p in plans if p["transport_type"] == "accessible_bus"
                ),
            },
            "medical_support_needed": medical_support,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Planned evacuation for {len(facilities)} special needs facilities"
        )
        return result

    def estimate_clearance_time(
        self,
        evacuation_plan: dict[str, Any],
        traffic_model: str = "dynamic_assignment",
        scenarios: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate evacuation clearance time.

        Args:
            evacuation_plan: The evacuation plan
            traffic_model: Traffic model to use
            scenarios: Scenarios to model

        Returns:
            Clearance time estimates
        """
        scenarios = scenarios or ["best_case", "expected", "worst_case"]

        zone = evacuation_plan.get("zone")
        if isinstance(zone, dict):
            population = zone.get("population", 10000)
        else:
            population = getattr(zone, "population", 10000)

        # Calculate based on vehicle loading and network capacity
        persons_per_vehicle = 2.5
        vehicles = population / persons_per_vehicle

        routes = evacuation_plan.get("routes", [])
        # Handle case where routes is a dict with "routes" key (from optimize_routes result)
        if isinstance(routes, dict):
            routes = routes.get("routes", [])
        total_capacity = (
            sum(r.get("capacity_vehicles_per_hour", 1000) for r in routes)
            if routes
            else 2000
        )

        # Base clearance time
        base_clearance = vehicles / total_capacity if total_capacity > 0 else 10

        estimates = {}
        scenario_multipliers = {"best_case": 0.7, "expected": 1.0, "worst_case": 1.5}
        for scenario in scenarios:
            multiplier = scenario_multipliers.get(scenario, 1.0)
            clearance_hours = base_clearance * multiplier

            estimates[scenario] = {
                "clearance_hours": round(clearance_hours, 1),
                "vehicles": int(vehicles),
                "network_capacity_per_hour": total_capacity,
                "loading_time_hours": 1.0 * multiplier,
                "travel_time_hours": (clearance_hours - 1.0 * multiplier)
                if clearance_hours > 1
                else 0.5,
            }

        logger.info(
            f"Estimated clearance time: {estimates.get('expected', {}).get('clearance_hours', 0)} hours"
        )
        return estimates


def _path_edges(
    graph: nx.Graph, path: list[str]
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """Return (u, v, data) triples for consecutive path node pairs."""
    return [
        (u, v, graph.get_edge_data(u, v) or {}) for u, v in zip(path[:-1], path[1:])
    ]
