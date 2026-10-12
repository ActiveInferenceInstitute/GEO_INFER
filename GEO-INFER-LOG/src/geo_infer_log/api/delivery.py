"""
API endpoints for last-mile delivery in GEO-INFER-LOG.

This module provides FastAPI endpoints for last-mile delivery functionality,
service area analysis, and delivery scheduling.
"""

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Depends
from pydantic import ConfigDict, Field
from geo_infer_log.models.base import BaseModel
from datetime import datetime

from geo_infer_log.models.schemas import Vehicle, Location
from geo_infer_log.core.delivery import (
    LastMileRouter,
    DeliveryScheduler,
    ServiceAreaAnalyzer,
)


router = APIRouter(
    prefix="/delivery",
    tags=["delivery"],
    responses={404: {"description": "Not found"}},
)


class DeliveryOptimizationRequest(BaseModel):
    """Request model for delivery optimization."""

    depot: Location
    deliveries: list[Location]
    vehicles: list[Vehicle]
    constraints: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "depot": {
                    "name": "Berlin Warehouse",
                    "coordinates": [13.404954, 52.520008],
                    "type": "depot",
                },
                "deliveries": [
                    {
                        "name": "Customer A",
                        "coordinates": [13.5, 52.5],
                        "type": "customer",
                        "service_time": 15,
                        "priority": 1,
                    },
                    {
                        "name": "Customer B",
                        "coordinates": [13.4, 52.4],
                        "type": "customer",
                        "service_time": 10,
                        "priority": 2,
                    },
                ],
                "vehicles": [
                    {
                        "id": "truck-001",
                        "type": "truck",
                        "capacity": 1000,
                        "max_range": 500,
                        "speed": 80,
                        "cost_per_km": 1.2,
                        "emissions_per_km": 0.8,
                        "location": [13.404954, 52.520008],
                    }
                ],
                "constraints": {"max_route_duration": 480, "max_stops_per_route": 20},
            }
        }
    )


class ScheduleRequest(BaseModel):
    """Request model for delivery scheduling."""

    depot: Location
    deliveries: list[Location]
    vehicles: list[Vehicle]
    start_date: datetime
    end_date: datetime
    max_deliveries_per_day: int = 50

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "depot": {
                    "name": "Berlin Warehouse",
                    "coordinates": [13.404954, 52.520008],
                    "type": "depot",
                },
                "deliveries": [
                    {
                        "name": "Customer A",
                        "coordinates": [13.5, 52.5],
                        "type": "customer",
                        "service_time": 15,
                    }
                ],
                "vehicles": [
                    {
                        "id": "truck-001",
                        "type": "truck",
                        "capacity": 1000,
                        "max_range": 500,
                        "speed": 80,
                        "cost_per_km": 1.2,
                        "emissions_per_km": 0.8,
                        "location": [13.404954, 52.520008],
                    }
                ],
                "start_date": "2023-01-01T08:00:00",
                "end_date": "2023-01-07T18:00:00",
                "max_deliveries_per_day": 30,
            }
        }
    )


class ServiceAreaRequest(BaseModel):
    """Request model for service area definition."""

    depot_id: str
    depot_location: tuple[float, float]
    max_time: int | None = None
    max_distance: float | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "depot_id": "depot-001",
                "depot_location": [13.404954, 52.520008],
                "max_time": 60,  # minutes
                "max_distance": 30,  # km
            }
        }
    )


class CoverageAnalysisRequest(BaseModel):
    """Request model for service area coverage analysis."""

    service_areas: dict[str, dict]
    demand_points: list[dict]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service_areas": {
                    "depot-001": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [13.3, 52.4],
                                [13.5, 52.4],
                                [13.5, 52.6],
                                [13.3, 52.6],
                                [13.3, 52.4],
                            ]
                        ],
                    }
                },
                "demand_points": [
                    {"id": "d1", "location": [13.4, 52.5]},
                    {"id": "d2", "location": [13.6, 52.5]},
                ],
            }
        }
    )


class RescheduleRequest(BaseModel):
    """Request model for delivery rescheduling."""

    route_id: str
    delivery_idx: int
    new_date: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route_id": "route-001",
                "delivery_idx": 2,
                "new_date": "2023-01-02T14:00:00",
            }
        }
    )


# Cached module-level singletons: schedules and service areas must survive
# across requests for the API to behave correctly.
@lru_cache(maxsize=1)
def get_last_mile_router() -> LastMileRouter:
    """Dependency for last-mile router."""
    return LastMileRouter()


def get_delivery_scheduler() -> DeliveryScheduler:
    """Dependency for delivery scheduler."""
    return DeliveryScheduler(get_last_mile_router())


@lru_cache(maxsize=1)
def get_service_area_analyzer() -> ServiceAreaAnalyzer:
    """Dependency for service area analyzer."""
    return ServiceAreaAnalyzer()


@router.post("/optimize", response_model=list[dict])
async def optimize_deliveries(
    request: DeliveryOptimizationRequest,
    router: LastMileRouter = Depends(get_last_mile_router),
) -> list[dict]:
    """Optimize deliveries from a depot."""
    try:
        routes = router.optimize_deliveries(
            depot=request.depot,
            deliveries=request.deliveries,
            vehicles=request.vehicles,
            constraints=request.constraints,
        )

        # Convert route objects to dictionaries
        return [route.model_dump() for route in routes]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/schedule", response_model=dict)
async def create_schedule(
    request: ScheduleRequest,
    scheduler: DeliveryScheduler = Depends(get_delivery_scheduler),
) -> dict:
    """Create a delivery schedule for a date range."""
    try:
        result = scheduler.create_schedule(
            depot=request.depot,
            deliveries=request.deliveries,
            vehicles=request.vehicles,
            start_date=request.start_date,
            end_date=request.end_date,
            max_deliveries_per_day=request.max_deliveries_per_day,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/schedule/{date}", response_model=list[dict])
async def get_daily_schedule(
    date: str, scheduler: DeliveryScheduler = Depends(get_delivery_scheduler)
) -> list[dict]:
    """Get the delivery schedule for a specific day."""
    try:
        # Parse date string to datetime
        date_obj = datetime.fromisoformat(date)
        routes = scheduler.get_daily_schedule(date_obj)

        # Convert route objects to dictionaries
        return [route.model_dump() for route in routes]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/schedule/vehicle/{vehicle_id}", response_model=list[dict])
async def get_vehicle_schedule(
    vehicle_id: str, scheduler: DeliveryScheduler = Depends(get_delivery_scheduler)
) -> list[dict]:
    """Get the schedule for a specific vehicle."""
    try:
        routes = scheduler.get_vehicle_schedule(vehicle_id)

        # Convert route objects to dictionaries
        return [route.model_dump() for route in routes]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/reschedule", response_model=dict)
async def reschedule_delivery(
    request: RescheduleRequest,
    scheduler: DeliveryScheduler = Depends(get_delivery_scheduler),
) -> dict:
    """Reschedule a delivery to a different date."""
    try:
        result = scheduler.reschedule_delivery(
            route_id=request.route_id,
            delivery_idx=request.delivery_idx,
            new_date=request.new_date,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/service-area", response_model=dict)
async def create_service_area(
    request: ServiceAreaRequest,
    analyzer: ServiceAreaAnalyzer = Depends(get_service_area_analyzer),
) -> dict:
    """Create a service area around a depot."""
    try:
        gdf = analyzer.create_service_area(
            depot_id=request.depot_id,
            depot_location=request.depot_location,
            max_time=request.max_time,
            max_distance=request.max_distance,
        )

        # Convert GeoDataFrame to GeoJSON
        geo_json = gdf.to_json()

        return {
            "depot_id": request.depot_id,
            "max_time": request.max_time,
            "max_distance": request.max_distance,
            "area": geo_json,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/coverage", response_model=dict)
async def analyze_coverage(
    request: CoverageAnalysisRequest,
    analyzer: ServiceAreaAnalyzer = Depends(get_service_area_analyzer),
) -> dict:
    """Analyze coverage of demand points by service areas."""
    try:
        # Convert service area GeoJSON to Shapely polygons and test
        # which demand points fall inside each area
        from shapely.errors import ShapelyError
        from shapely.geometry import shape, Point

        depot_coverage: dict = {}
        covered_ids: set = set()

        for depot_id, geojson in request.service_areas.items():
            try:
                poly = shape(geojson)
            except (ValueError, TypeError, AttributeError, ShapelyError) as e:
                raise ValueError(
                    f"Invalid service-area geometry for depot {depot_id}: {e}"
                ) from e

            dep_covered: list[str] = []
            for dp in request.demand_points:
                loc = dp.get("location")
                if loc is None:
                    continue
                pt = Point(loc[0], loc[1])
                if poly.contains(pt):
                    dep_covered.append(dp.get("id", ""))
                    covered_ids.add(dp.get("id", ""))
            depot_coverage[depot_id] = {
                "covered": len(dep_covered),
                "points": dep_covered,
            }

        total = len(request.demand_points)
        covered = len(covered_ids)
        return {
            "total_points": total,
            "covered_points": covered,
            "uncovered_points": total - covered,
            "coverage_ratio": covered / total if total > 0 else 0.0,
            "depot_coverage": depot_coverage,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
