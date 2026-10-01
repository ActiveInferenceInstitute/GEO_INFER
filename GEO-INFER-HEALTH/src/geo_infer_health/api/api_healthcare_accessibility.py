from fastapi import APIRouter, HTTPException, Body, Query
from typing import Optional, Any

from geo_infer_health.models import HealthFacility, Location, PopulationData
from geo_infer_health.core.healthcare_accessibility import (
    HealthcareAccessibilityAnalyzer,
)

router = APIRouter(
    prefix="/accessibility",
    tags=["Healthcare Accessibility"],
)

# In-memory store backing the demo API
_FACILITIES_DB: list[HealthFacility] = []
_POPULATION_DATA_DB_ACC: list[
    PopulationData
] = []  # Using a different name to avoid conflict if run in same context as surveillance


def reset_stores() -> None:
    """Clear the in-memory stores (for tests and demo resets)."""
    _FACILITIES_DB.clear()
    _POPULATION_DATA_DB_ACC.clear()


@router.post("/facilities/", response_model=HealthFacility, status_code=201)
async def add_health_facility(facility: HealthFacility = Body(...)) -> HealthFacility:
    """Add a new health facility to the system."""
    _FACILITIES_DB.append(facility)
    return facility


@router.get("/facilities/", response_model=list[HealthFacility])
async def get_all_health_facilities(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> list[HealthFacility]:
    """Retrieve all health facilities with pagination."""
    return _FACILITIES_DB[offset : offset + limit]


@router.post("/facilities/nearby", response_model=list[HealthFacility])
async def find_nearby_facilities(
    latitude: float = Query(..., description="Latitude of the search center."),
    longitude: float = Query(..., description="Longitude of the search center."),
    radius_km: float = Query(..., gt=0, description="Search radius in kilometers."),
    facility_type: str | None = Query(
        None,
        description="Optional filter by facility type (e.g., 'hospital', 'clinic').",
    ),
    required_services: list[str] | None = Query(
        None, description="Optional list of required services."
    ),
) -> list[HealthFacility]:
    """Finds health facilities within a radius, with optional filters."""
    if not _FACILITIES_DB:
        raise HTTPException(status_code=404, detail="No health facilities available.")

    center_loc = Location(latitude=latitude, longitude=longitude)
    analyzer = HealthcareAccessibilityAnalyzer(
        facilities=_FACILITIES_DB, population_data=_POPULATION_DATA_DB_ACC
    )

    facilities = analyzer.find_facilities_in_radius(
        center_loc=center_loc,
        radius_km=radius_km,
        facility_type=facility_type,
        required_services=required_services,
    )
    return facilities


@router.post(
    "/facilities/nearest", response_model=Optional[dict[str, Any]]
)  # HealthFacility and distance
async def get_nearest_facility_endpoint(
    latitude: float = Query(..., description="Latitude of the origin point."),
    longitude: float = Query(..., description="Longitude of the origin point."),
    facility_type: str | None = Query(
        None, description="Optional filter by facility type."
    ),
    required_services: list[str] | None = Query(
        None, description="Optional list of required services."
    ),
) -> dict[str, Any] | None:
    """Finds the nearest health facility to a given location."""
    if not _FACILITIES_DB:
        raise HTTPException(status_code=404, detail="No health facilities available.")

    loc = Location(latitude=latitude, longitude=longitude)
    analyzer = HealthcareAccessibilityAnalyzer(
        facilities=_FACILITIES_DB, population_data=_POPULATION_DATA_DB_ACC
    )
    result: tuple[HealthFacility, float] | None = analyzer.get_nearest_facility(
        loc=loc, facility_type=facility_type, required_services=required_services
    )
    if not result:
        return None
    facility, distance = result
    return {"facility": facility, "distance_km": distance}


@router.get(
    "/facility_population_ratio/{area_id}", response_model=Optional[dict[str, Any]]
)
async def get_facility_population_ratio(
    area_id: str,
    facility_type: str | None = Query(
        None, description="Optional filter by facility type for ratio calculation."
    ),
) -> dict[str, Any]:
    """Calculates the facility-to-population ratio for a given area ID."""
    if not _FACILITIES_DB and not _POPULATION_DATA_DB_ACC:  # Basic check
        raise HTTPException(
            status_code=404,
            detail="Facility or population data not available for ratio calculation.",
        )

    analyzer = HealthcareAccessibilityAnalyzer(
        facilities=_FACILITIES_DB, population_data=_POPULATION_DATA_DB_ACC
    )
    ratio_info = analyzer.calculate_facility_to_population_ratio(
        area_id=area_id, facility_type=facility_type
    )

    if ratio_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Population data for area_id '{area_id}' not found.",
        )
    return ratio_info


# Endpoint to add population data for accessibility context (can be shared or specific)
@router.post("/population_data/", response_model=PopulationData, status_code=201)
async def add_accessibility_population_data(
    data: PopulationData = Body(...),
) -> PopulationData:
    """Add population data for accessibility analysis context."""
    _POPULATION_DATA_DB_ACC.append(data)
    return data


@router.get("/population_data/", response_model=list[PopulationData])
async def get_accessibility_population_data(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> list[PopulationData]:
    """Retrieve population data used in accessibility context."""
    return _POPULATION_DATA_DB_ACC[offset : offset + limit]
