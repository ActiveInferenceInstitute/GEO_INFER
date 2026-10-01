"""
Transit optimization module.

Provides transit network design, frequency optimization,
and service planning capabilities.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TransitMode(Enum):
    """Transit mode types."""

    BUS = "bus"
    RAIL = "rail"
    SUBWAY = "subway"
    TRAM = "tram"
    FERRY = "ferry"
    BRT = "brt"


@dataclass
class TransitStop:
    """Represents a transit stop."""

    stop_id: str
    name: str
    location: dict[str, float]
    routes: list[str] = field(default_factory=list)
    boarding_daily: int = 0
    amenities: list[str] = field(default_factory=list)


@dataclass
class TransitRoute:
    """Represents a transit route."""

    route_id: str
    name: str
    mode: TransitMode
    stops: list[str]
    headway_minutes: int = 30
    operating_hours: dict[str, str] = field(default_factory=dict)
    ridership_daily: int = 0


class TransitOptimizer:
    """
    Optimize transit network design and service.

    Supports frequency optimization, coverage analysis,
    and network design.
    """

    def __init__(
        self,
        transit_network: dict[str, Any] | None = None,
        demand_data: dict[str, Any] | None = None,
        optimization_objectives: list[str] | None = None,
    ):
        """
        Initialize transit optimizer.

        Args:
            transit_network: Existing transit network
            demand_data: Travel demand data
            optimization_objectives: Objectives for optimization
        """
        self.transit_network = transit_network or {}
        self.demand_data = demand_data or {}
        self.optimization_objectives = optimization_objectives or [
            "coverage",
            "ridership",
        ]
        self._stops: dict[str, TransitStop] = {}
        self._routes: dict[str, TransitRoute] = {}
        logger.info("Initialized TransitOptimizer")

    def optimize_frequencies(
        self,
        routes: list[dict[str, Any]],
        demand_patterns: dict[str, Any],
        fleet_constraints: dict[str, int],
        optimization_period: str = "peak",
    ) -> dict[str, Any]:
        """
        Optimize route frequencies.

        Args:
            routes: Transit routes
            demand_patterns: Demand by route and time
            fleet_constraints: Fleet size constraints
            optimization_period: Period to optimize for

        Returns:
            Optimized frequency plan
        """
        routes_out: list[dict[str, Any]] = []
        summary_out: dict[str, Any] = {
            "total_routes": len(routes),
            "total_vehicles_required": 0,
            "estimated_ridership_increase": 0,
        }
        optimization_result: dict[str, Any] = {
            "optimization_period": optimization_period,
            "timestamp": datetime.now().isoformat(),
            "routes": routes_out,
            "summary": summary_out,
        }

        total_vehicles = 0

        for route in routes:
            route_id = route.get("id")
            current_headway = route.get("headway_minutes", 30)

            # Get demand for this route
            route_demand = (
                demand_patterns.get(str(route_id), {}) if route_id is not None else {}
            )
            peak_demand = route_demand.get("peak_hourly", 100)

            # Calculate optimal headway based on demand
            # Target: load factor of 0.8, capacity of 50 passengers
            vehicle_capacity = route.get("vehicle_capacity", 50)
            target_load = 0.8

            trips_per_hour_needed = peak_demand / (vehicle_capacity * target_load)
            optimal_headway = (
                60 / trips_per_hour_needed if trips_per_hour_needed > 0 else 60
            )

            # Constrain headway to reasonable bounds
            optimal_headway = max(5, min(60, int(optimal_headway)))

            # Calculate vehicles needed
            route_length_hours = route.get("cycle_time_hours", 2)
            vehicles_needed = int(route_length_hours * 60 / optimal_headway) + 1
            total_vehicles += vehicles_needed

            routes_out.append(
                {
                    "route_id": route_id,
                    "current_headway": current_headway,
                    "optimal_headway": optimal_headway,
                    "headway_change": current_headway - optimal_headway,
                    "vehicles_required": vehicles_needed,
                    "expected_ridership_change_pct": round(
                        ((current_headway / optimal_headway) - 1) * 20, 1
                    ),
                }
            )

        # Check fleet constraints
        max_fleet = sum(fleet_constraints.values())
        if total_vehicles > max_fleet:
            optimization_result["constraint_violation"] = {
                "message": "Insufficient fleet for optimal frequencies",
                "vehicles_required": total_vehicles,
                "vehicles_available": max_fleet,
            }

        summary_out["total_vehicles_required"] = total_vehicles

        logger.info(f"Optimized frequencies for {len(routes)} routes")
        return optimization_result

    def analyze_coverage(
        self,
        stops: list[dict[str, Any]],
        population_zones: list[dict[str, Any]],
        walk_radius_m: float = 400,
        equity_focus: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze transit coverage.

        Args:
            stops: Transit stop locations
            population_zones: Population distribution
            walk_radius_m: Walking distance to stops
            equity_focus: Include equity metrics

        Returns:
            Coverage analysis results
        """
        import math

        total_population = sum(z.get("population", 0) for z in population_zones)
        covered_population = 0
        zone_covered_flags: list[bool] = []
        covered_zones = []
        uncovered_zones = []

        for zone in population_zones:
            zone_loc = zone.get("centroid", {})
            zone_pop = zone.get("population", 0)

            # Check if any stop is within walk radius
            is_covered = False
            for stop in stops:
                stop_loc = stop.get("location", {})

                # Calculate distance
                lat1 = math.radians(zone_loc.get("lat", 0))
                lat2 = math.radians(stop_loc.get("lat", 0))
                lon1 = math.radians(zone_loc.get("lon", 0))
                lon2 = math.radians(stop_loc.get("lon", 0))

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))
                distance_m = 6371000 * c

                if distance_m <= walk_radius_m:
                    is_covered = True
                    break
            zone_covered_flags.append(is_covered)

            if is_covered:
                covered_population += zone_pop
                covered_zones.append(zone.get("id"))
            else:
                uncovered_zones.append(
                    {"zone_id": zone.get("id"), "population": zone_pop}
                )

        coverage = {
            "walk_radius_m": walk_radius_m,
            "total_stops": len(stops),
            "total_population": total_population,
            "covered_population": covered_population,
            "coverage_rate": round(covered_population / total_population, 4)
            if total_population > 0
            else 0,
            "uncovered_zones": uncovered_zones[:10],  # Top 10
            "uncovered_population": total_population - covered_population,
        }

        if equity_focus:
            # Per-group coverage computed from zone demographics. Zones carry
            # either a "demographics" mapping (group -> population share) or a
            # single "demographic_group" name. Without demographic data no
            # equity_analysis is reported.
            group_total: dict[str, float] = {}
            group_covered: dict[str, float] = {}
            for zone, is_covered in zip(population_zones, zone_covered_flags):
                zone_pop = zone.get("population", 0)
                demographics = zone.get("demographics")
                if isinstance(demographics, dict):
                    shares: dict[str, float] = demographics
                elif zone.get("demographic_group"):
                    shares = {zone["demographic_group"]: 1.0}
                else:
                    continue
                for group, share in shares.items():
                    pop = zone_pop * float(share)
                    group_total[group] = group_total.get(group, 0.0) + pop
                    if is_covered:
                        group_covered[group] = group_covered.get(group, 0.0) + pop
            if group_total:
                coverage["equity_analysis"] = {
                    "group_coverage": {
                        group: round(group_covered.get(group, 0.0) / total, 4)
                        for group, total in group_total.items()
                    },
                    "group_population": {
                        group: round(total, 1) for group, total in group_total.items()
                    },
                }

        logger.info(
            f"Coverage analysis: {coverage['coverage_rate']:.1%} population covered"
        )
        return coverage

    def design_network(
        self,
        demand_zones: list[dict[str, Any]],
        constraints: dict[str, Any],
        mode: str = "bus",
        objective: str = "maximize_coverage",
    ) -> dict[str, Any]:
        """
        Design a transit network.

        Args:
            demand_zones: Demand by zone
            constraints: Budget and operational constraints
            mode: Transit mode
            objective: Optimization objective

        Returns:
            Proposed network design
        """
        proposed_routes_out: list[dict[str, Any]] = []
        proposed_stops_out: list[dict[str, Any]] = []
        design: dict[str, Any] = {
            "mode": mode,
            "objective": objective,
            "timestamp": datetime.now().isoformat(),
            "proposed_routes": proposed_routes_out,
            "proposed_stops": proposed_stops_out,
            "metrics": {},
        }

        # Sort zones by demand
        sorted_zones = sorted(
            demand_zones, key=lambda z: z.get("demand", 0), reverse=True
        )

        # Identify high-demand corridors
        max_routes = constraints.get("max_routes", 5)

        for i in range(min(max_routes, len(sorted_zones) - 1)):
            origin = sorted_zones[i]
            destination = sorted_zones[(i + 1) % len(sorted_zones)]

            route: dict[str, Any] = {
                "route_id": f"route_{i + 1}",
                "name": f"Route {i + 1}",
                "mode": mode,
                "origin": origin.get("id"),
                "destination": destination.get("id"),
                "estimated_ridership": (
                    origin.get("demand", 0) + destination.get("demand", 0)
                )
                / 2,
                "recommended_headway": 15 if i < 2 else 30,
                "length_km": 10,  # Baseline
            }
            proposed_routes_out.append(route)

            # Add stops for this route
            proposed_stops_out.extend(
                [
                    {
                        "stop_id": f"stop_{i}_1",
                        "location": origin.get("centroid", {}),
                        "route": route["route_id"],
                    },
                    {
                        "stop_id": f"stop_{i}_2",
                        "location": destination.get("centroid", {}),
                        "route": route["route_id"],
                    },
                ]
            )

        # Calculate metrics
        design["metrics"] = {
            "total_routes": len(proposed_routes_out),
            "total_stops": len(proposed_stops_out),
            "estimated_daily_ridership": sum(
                r.get("estimated_ridership", 0) for r in proposed_routes_out
            ),
            "total_route_km": sum(r.get("length_km", 0) for r in proposed_routes_out),
        }

        logger.info(f"Designed network with {len(proposed_routes_out)} routes")
        return design

    def evaluate_scenario(
        self,
        base_network: dict[str, Any],
        proposed_changes: list[dict[str, Any]],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a network change scenario.

        Args:
            base_network: Current network
            proposed_changes: Proposed changes
            metrics: Metrics to evaluate

        Returns:
            Scenario evaluation results
        """
        metrics = metrics or ["ridership", "coverage", "cost"]

        impacts: dict[str, Any] = {}
        evaluation: dict[str, Any] = {
            "scenario_id": f"scenario_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "proposed_changes": proposed_changes,
            "impacts": impacts,
            "recommendation": "",
        }

        per_change_impacts: list[dict[str, Any]] = []
        for change in proposed_changes:
            change_type = change.get("type")
            change_impact: dict[str, Any] = {"type": change_type}

            if change_type == "add_route":
                change_impact["ridership_change"] = change.get(
                    "expected_ridership", 1000
                )
                change_impact["cost_annual"] = 500000
                change_impact["coverage_change_pct"] = 2.5
            elif change_type == "increase_frequency":
                change_impact["ridership_change"] = 500
                change_impact["cost_annual"] = 100000
                change_impact["coverage_change_pct"] = 0
            elif change_type == "extend_route":
                change_impact["ridership_change"] = 300
                change_impact["cost_annual"] = 200000
                change_impact["coverage_change_pct"] = 1.5
            else:
                logger.warning("Unknown scenario change type: %r", change_type)
                continue

            per_change_impacts.append(change_impact)

        # Aggregate impacts across ALL changes (not just the last one)
        impacts["changes"] = per_change_impacts
        impacts["ridership_change"] = sum(
            c["ridership_change"] for c in per_change_impacts
        )
        impacts["cost_annual"] = sum(c["cost_annual"] for c in per_change_impacts)
        impacts["coverage_change_pct"] = round(
            sum(c["coverage_change_pct"] for c in per_change_impacts), 4
        )
        # Calculate benefit-cost ratio
        if impacts.get("cost_annual", 0) > 0:
            ridership_value = impacts.get("ridership_change", 0) * 3 * 365  # $3/ride
            bcr = ridership_value / impacts["cost_annual"]
            evaluation["benefit_cost_ratio"] = round(bcr, 2)

            if bcr > 1.5:
                evaluation["recommendation"] = "Strongly recommended"
            elif bcr > 1.0:
                evaluation["recommendation"] = "Recommended"
            else:
                evaluation["recommendation"] = "Not recommended"

        logger.info(f"Evaluated scenario with {len(proposed_changes)} changes")
        return evaluation
