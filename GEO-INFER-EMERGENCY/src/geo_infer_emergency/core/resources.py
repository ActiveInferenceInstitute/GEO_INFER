"""
Resource deployment module.

Provides resource allocation, deployment optimization, and
real-time tracking for emergency resources.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .geo import haversine_distance_km

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Status of emergency resources."""

    AVAILABLE = "available"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    OUT_OF_SERVICE = "out_of_service"
    RETURNING = "returning"


class ResourceType(Enum):
    """Types of emergency resources."""

    ENGINE = "engine"
    TRUCK = "truck"
    AMBULANCE = "ambulance"
    RESCUE_UNIT = "rescue_unit"
    HAZMAT = "hazmat"
    HELICOPTER = "helicopter"
    DOZER = "dozer"
    WATER_TENDER = "water_tender"
    PERSONNEL = "personnel"


# Plural -> singular alias map so configured vocabularies such as the module
# default ("engines", "ambulances", "rescue_units") resolve to the singular
# ResourceType values ("engine", "ambulance", "rescue_unit").
RESOURCE_TYPE_ALIASES: dict[str, str] = {
    **{f"{rt.value}s": rt.value for rt in ResourceType},
    "personnel": "personnel",
}


@dataclass
class Resource:
    """Represents an emergency resource unit."""

    resource_id: str
    resource_type: ResourceType
    name: str
    status: ResourceStatus = ResourceStatus.AVAILABLE
    location: dict[str, float] | None = None  # lat, lon
    capacity: int = 1
    agency: str = ""
    capabilities: list[str] = field(default_factory=list)
    assigned_incident: str | None = None


class ResourceDeployer:
    """
    Optimize deployment and allocation of emergency resources.

    Allocation uses a greedy nearest-available-resource heuristic to minimize
    response time and maximize coverage (see the optimization_algorithm
    contract in __init__).
    """

    def __init__(
        self,
        resource_types: list[str] | None = None,
        optimization_algorithm: str = "greedy_nearest_resource",
        real_time_updates: bool = True,
    ):
        """
        Initialize resource deployer.

        Args:
            resource_types: Types of resources to manage. Entries may use the
                plural configured vocabulary ("engines", "ambulances",
                "rescue_units", ...) or the singular ResourceType values
                ("engine", "ambulance", "rescue_unit", ...); plurals are
                resolved through RESOURCE_TYPE_ALIASES. The set is used as an
                allowed-type filter in optimize_allocation.
            optimization_algorithm: Allocation-strategy label. The implemented
                strategy is "greedy_nearest_resource": greedy assignment of the
                nearest available resource within the response-time constraint.
                The value is echoed into optimize_allocation results; no solver
                backend is attached to it.
            real_time_updates: Enable real-time tracking
        """
        self.resource_types = list(
            resource_types
            if resource_types is not None
            else ["engines", "ambulances", "rescue_units"]
        )
        self._allowed_type_values = {
            self._resolve_resource_type(entry).value for entry in self.resource_types
        }
        self.optimization_algorithm = optimization_algorithm
        self.real_time_updates = real_time_updates
        self._resources: dict[str, Resource] = {}
        logger.info(
            f"Initialized ResourceDeployer with {optimization_algorithm} optimization"
        )

    def _resolve_resource_type(self, value: str) -> ResourceType:
        """
        Resolve a configured/serialized type name to a ResourceType.

        Accepts singular ResourceType values and plural aliases via
        RESOURCE_TYPE_ALIASES. Raises ValueError naming the valid types.
        """
        key = value.strip().lower()
        singular = RESOURCE_TYPE_ALIASES.get(key, key)
        try:
            return ResourceType(singular)
        except ValueError:
            valid = sorted(rt.value for rt in ResourceType)
            raise ValueError(
                f"geo_infer_emergency.core.resources: unknown resource type "
                f"{value!r}. Valid types: {valid} "
                "(plural forms like 'engines' are accepted as aliases)."
            ) from None

    def register_resource(self, resource: Resource) -> None:
        """Register a resource in the deployment system."""
        self._resources[resource.resource_id] = resource
        logger.debug(f"Registered resource: {resource.resource_id}")

    def optimize_allocation(
        self,
        resources: list[dict[str, Any]],
        demand_points: list[dict[str, Any]],
        constraints: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, Any]:
        """
        Optimize resource allocation to demand points.

        Args:
            resources: Available resources with locations. Each "type" entry
                is resolved via RESOURCE_TYPE_ALIASES (singular ResourceType
                values and plural aliases both accepted) and must be among
                the deployer's configured resource_types, else ValueError.

        Returns:
            Optimized allocation plan
        """
        # Register resources
        for res_data in resources:
            raw_type = res_data.get("type", "engine")
            resolved_type = self._resolve_resource_type(raw_type)
            if resolved_type.value not in self._allowed_type_values:
                raise ValueError(
                    f"geo_infer_emergency.core.resources: resource "
                    f"{res_data.get('id', '<unnamed>')!r} has type "
                    f"{resolved_type.value!r} which is not among the "
                    f"allowed types configured for this deployer: "
                    f"{sorted(self._allowed_type_values)}."
                )
            resource = Resource(
                resource_id=res_data.get("id", f"res_{len(self._resources)}"),
                resource_type=resolved_type,
                name=res_data.get("name", ""),
                location=res_data.get("location"),
                status=ResourceStatus(res_data.get("status", "available")),
                agency=res_data.get("agency", ""),
            )
            self.register_resource(resource)

        # Calculate distances and response times
        allocations = []
        unallocated_demands = []
        available_resources = [
            r for r in self._resources.values() if r.status == ResourceStatus.AVAILABLE
        ]

        for demand in demand_points:
            demand_loc = demand.get("location", {})

            # Find nearest available resource
            best_resource = None
            best_time = float("inf")

            for resource in available_resources:
                if resource.location:
                    travel_time = self._estimate_travel_time(
                        resource.location, demand_loc
                    )
                    if travel_time < best_time:
                        best_time = travel_time
                        best_resource = resource

            if best_resource and best_time <= constraints.get("response_time", 15):
                allocations.append(
                    {
                        "demand_id": demand.get("id", ""),
                        "demand_location": demand_loc,
                        "resource_id": best_resource.resource_id,
                        "resource_type": best_resource.resource_type.value,
                        "estimated_response_time": best_time,
                        "status": "allocated",
                    }
                )
                # Mark resource as assigned
                best_resource.status = ResourceStatus.ASSIGNED
                available_resources.remove(best_resource)
            else:
                unallocated_demands.append(demand.get("id", ""))

        # Calculate coverage
        total_demands = len(demand_points)
        covered_demands = len(allocations)
        coverage = covered_demands / total_demands if total_demands > 0 else 1.0

        result = {
            "optimization_algorithm": self.optimization_algorithm,
            "objectives": objectives,
            "constraints": constraints,
            "allocations": allocations,
            "unallocated_demands": unallocated_demands,
            "metrics": {
                "total_resources": len(resources),
                "resources_allocated": len(allocations),
                "total_demands": total_demands,
                "demands_covered": covered_demands,
                "coverage_rate": coverage,
                "average_response_time": (
                    sum(a["estimated_response_time"] for a in allocations)
                    / len(allocations)
                    if allocations
                    else 0
                ),
            },
            "feasible": coverage >= constraints.get("coverage", 0.8),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Optimized allocation: {covered_demands}/{total_demands} demands covered"
        )
        return result

    def _estimate_travel_time(
        self, from_loc: dict[str, float], to_loc: dict[str, float]
    ) -> float:
        """Estimate travel time between two locations."""
        distance_km = haversine_distance_km(from_loc, to_loc)

        # Assume 40 km/h average speed for emergency response
        speed = 40
        travel_time_hours = distance_km / speed
        travel_time_minutes = travel_time_hours * 60

        return travel_time_minutes

    def dynamic_redeploy(
        self,
        current_positions: list[dict[str, Any]],
        pending_incidents: list[dict[str, Any]],
        predicted_demand: dict[str, Any],
        strategy: str = "move_up",
    ) -> dict[str, Any]:
        """
        Dynamically redeploy resources based on current conditions.

        Args:
            current_positions: Current resource positions
            pending_incidents: Pending incidents without resources
            predicted_demand: Predicted future demand
            strategy: Redeployment strategy

        Returns:
            Redeployment plan
        """
        redeployments = []

        # Update resource positions
        for pos in current_positions:
            res_id = pos.get("resource_id")
            if res_id in self._resources:
                self._resources[res_id].location = pos.get("location")
        # Find available resources
        available = [
            r for r in self._resources.values() if r.status == ResourceStatus.AVAILABLE
        ]

        if strategy == "move_up":
            # Move units to cover gaps left by responding units
            gap_locations = predicted_demand.get("high_risk_areas", [])

            for gap in gap_locations:
                if available:
                    # Find available unit farthest from any pending incident,
                    # so the closest units remain held for immediate response.
                    if pending_incidents:
                        incident_locs = [
                            inc["location"]
                            for inc in pending_incidents
                            if isinstance(inc, dict)
                            and isinstance(inc.get("location"), dict)
                            and "lat" in inc["location"]
                            and "lon" in inc["location"]
                        ]
                        located = [u for u in available if u.location]
                        if incident_locs and located:
                            best_unit = max(
                                located,
                                key=lambda u: min(
                                    self._estimate_travel_time(
                                        u.location,
                                        loc,  # type: ignore[arg-type]
                                    )
                                    for loc in incident_locs
                                ),
                            )
                        else:
                            best_unit = available[0]
                    else:
                        # No pending incidents: keep first-available order.
                        best_unit = available[0]
                    gap_valid = isinstance(gap, dict) and "lat" in gap and "lon" in gap
                    if not gap_valid:
                        raise ValueError(
                            "geo_infer_emergency.core.resources: "
                            "dynamic_redeploy 'move_up' strategy requires each "
                            "high_risk_areas entry to be a dict with complete "
                            f"'lat' and 'lon' keys; got {gap!r}."
                        )
                    redeployments.append(
                        {
                            "resource_id": best_unit.resource_id,
                            "from_location": best_unit.location,
                            "to_location": gap,
                            "reason": "coverage_gap",
                            "estimated_travel_minutes": (
                                self._estimate_travel_time(
                                    best_unit.location,
                                    gap,  # type: ignore[arg-type]
                                )
                                if best_unit.location
                                else 0
                            ),
                        }
                    )
                    available.remove(best_unit)
        result = {
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "redeployments": redeployments,
            "pending_incidents_covered": len(pending_incidents),
            "units_redeployed": len(redeployments),
        }

        logger.info(f"Dynamic redeployment: {len(redeployments)} units moved")
        return result

    def manage_staging(
        self,
        staging_areas: list[dict[str, Any]],
        incoming_resources: list[dict[str, Any]],
        assignment_queue: list[dict[str, Any]],
        prioritization: str = "incident_severity",
    ) -> dict[str, Any]:
        """
        Manage staging area operations.

        Args:
            staging_areas: Available staging locations
            incoming_resources: Resources arriving
            assignment_queue: Pending assignments
            prioritization: How to prioritize assignments

        Returns:
            Staging management plan
        """
        staging_areas_out: list[dict[str, Any]] = []
        pending_queue_out: list[dict[str, Any]] = []
        staging_plan: dict[str, Any] = {
            "staging_areas": staging_areas_out,
            "incoming_assignments": [],
            "pending_queue": pending_queue_out,
            "prioritization": prioritization,
            "timestamp": datetime.now().isoformat(),
        }

        # Assign staging areas
        for i, staging in enumerate(staging_areas):
            area_plan: dict[str, Any] = {
                "staging_id": staging.get("id", f"staging_{i}"),
                "location": staging.get("location"),
                "capacity": staging.get("capacity", 50),
                "current_count": 0,
                "assigned_resources": [],
            }

            # Assign incoming resources to staging
            for resource in incoming_resources:
                if area_plan["current_count"] < area_plan["capacity"]:
                    area_plan["assigned_resources"].append(resource.get("id"))
                    area_plan["current_count"] += 1

            staging_areas_out.append(area_plan)

        # Process assignment queue by priority
        if prioritization == "incident_severity":
            sorted_queue = sorted(
                assignment_queue, key=lambda x: x.get("severity", 0), reverse=True
            )
        else:
            sorted_queue = assignment_queue

        for assignment in sorted_queue:
            pending_queue_out.append(
                {
                    "assignment_id": assignment.get("id"),
                    "incident": assignment.get("incident"),
                    "resources_needed": assignment.get("resources_needed"),
                    "priority_rank": sorted_queue.index(assignment) + 1,
                }
            )

        logger.info(
            f"Managed {len(staging_areas)} staging areas with {len(incoming_resources)} incoming resources"
        )
        return staging_plan

    def track_resources(
        self,
        resources: list[dict[str, Any]],
        update_frequency: str = "real_time",
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Track resource status and locations.

        Args:
            resources: Resources to track
            update_frequency: How often to update
            metrics: Metrics to track

        Returns:
            Resource tracking data
        """
        metrics = metrics or ["location", "status", "availability", "eta"]

        resources_out: list[dict[str, Any]] = []
        summary_out: dict[str, int] = {
            "total": len(resources),
            "available": 0,
            "assigned": 0,
            "en_route": 0,
            "on_scene": 0,
            "out_of_service": 0,
        }
        tracking: dict[str, Any] = {
            "update_frequency": update_frequency,
            "timestamp": datetime.now().isoformat(),
            "resources": resources_out,
            "summary": summary_out,
        }

        for res_data in resources:
            res_id = res_data.get("id")
            status = res_data.get("status", "available")

            resource_track: dict[str, Any] = {
                "resource_id": res_id,
                "type": res_data.get("type", "unknown"),
            }

            if "location" in metrics:
                resource_track["location"] = res_data.get("location")
            if "status" in metrics:
                resource_track["status"] = status
            if "availability" in metrics:
                resource_track["available"] = status == "available"
            if "eta" in metrics:
                resource_track["eta_minutes"] = res_data.get("eta", None)

            resources_out.append(resource_track)

            # Update summary
            if status == "available":
                summary_out["available"] += 1
            elif status == "assigned":
                summary_out["assigned"] += 1
            elif status == "en_route":
                summary_out["en_route"] += 1
            elif status == "on_scene":
                summary_out["on_scene"] += 1
            else:
                summary_out["out_of_service"] += 1

        logger.debug(f"Tracking {len(resources)} resources")
        return tracking

    def get_resource_status(self, resource_id: str) -> dict[str, Any] | None:
        """Get status of a specific resource."""
        resource = self._resources.get(resource_id)
        if resource:
            return {
                "resource_id": resource.resource_id,
                "type": resource.resource_type.value,
                "name": resource.name,
                "status": resource.status.value,
                "location": resource.location,
                "assigned_incident": resource.assigned_incident,
            }
        return None
