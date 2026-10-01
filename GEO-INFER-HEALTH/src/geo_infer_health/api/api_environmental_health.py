from fastapi import APIRouter, HTTPException, Body, Query
from datetime import datetime

from geo_infer_health.models import EnvironmentalData, Location
from geo_infer_health.core.environmental_health import EnvironmentalHealthAnalyzer

router = APIRouter(
    prefix="/environment",
    tags=["Environmental Health"],
)

# In-memory store backing the demo API
_ENV_READINGS_DB: list[EnvironmentalData] = []


def reset_stores() -> None:
    """Clear the in-memory store (for tests and demo resets)."""
    _ENV_READINGS_DB.clear()


@router.post("/readings/", response_model=EnvironmentalData, status_code=201)
async def submit_environmental_reading(
    reading: EnvironmentalData = Body(...),
) -> EnvironmentalData:
    """Submit a new environmental data reading."""
    _ENV_READINGS_DB.append(reading)
    # Sort by timestamp after adding for consistent latest reading retrieval
    _ENV_READINGS_DB.sort(key=lambda r: r.timestamp)
    return reading


@router.get("/readings/", response_model=list[EnvironmentalData])
async def get_all_environmental_readings(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    parameter_name: str | None = Query(
        None, description="Filter by parameter name (e.g., 'PM2.5')"
    ),
) -> list[EnvironmentalData]:
    """Retrieve all environmental readings with pagination and optional parameter filter."""
    results = _ENV_READINGS_DB
    if parameter_name:
        results = [
            r for r in results if r.parameter_name.lower() == parameter_name.lower()
        ]
    return results[offset : offset + limit]


@router.post("/readings/near_location", response_model=list[EnvironmentalData])
async def get_readings_near_location_api(
    latitude: float = Query(..., description="Latitude of the search center."),
    longitude: float = Query(..., description="Longitude of the search center."),
    radius_km: float = Query(..., gt=0, description="Search radius in kilometers."),
    parameter_name: str | None = Query(
        None, description="Optional filter by parameter name."
    ),
    start_time_iso: str | None = Query(
        None, description="Optional start time in ISO format (YYYY-MM-DDTHH:MM:SS)."
    ),
    end_time_iso: str | None = Query(
        None, description="Optional end time in ISO format (YYYY-MM-DDTHH:MM:SS)."
    ),
) -> list[EnvironmentalData]:
    """Get environmental readings near a specific location and time window."""
    if not _ENV_READINGS_DB:
        raise HTTPException(
            status_code=404, detail="No environmental readings available."
        )

    center_loc = Location(latitude=latitude, longitude=longitude)
    analyzer = EnvironmentalHealthAnalyzer(environmental_readings=_ENV_READINGS_DB)

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    try:
        if start_time_iso:
            start_dt = datetime.fromisoformat(start_time_iso)
        if end_time_iso:
            end_dt = datetime.fromisoformat(end_time_iso)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO date format: {e}")

    readings = analyzer.get_environmental_readings_near_location(
        center_loc=center_loc,
        radius_km=radius_km,
        parameter_name=parameter_name,
        start_time=start_dt,
        end_time=end_dt,
    )
    return readings


@router.post("/exposure/average", response_model=dict[str, float | None])
async def get_average_exposure_api(
    target_locations_query: list[dict[str, float]] = Body(
        ...,
        description="List of target locations, e.g., [{'latitude': lat, 'longitude': lon}].",
    ),
    radius_km: float = Query(
        ...,
        gt=0,
        description="Radius to search for environmental data around each target location.",
    ),
    parameter_name: str = Query(
        ...,
        description="The specific environmental parameter to analyze (e.g., 'PM2.5').",
    ),
    time_window_days: int = Query(
        ...,
        ge=1,
        description="How many days back from the most recent reading to consider.",
    ),
) -> dict[str, float | None]:
    """Calculates the average exposure to an environmental parameter for a list of locations."""
    if not _ENV_READINGS_DB:
        raise HTTPException(
            status_code=404,
            detail="No environmental readings available for exposure analysis.",
        )

    try:
        target_locations = [
            Location(latitude=loc["latitude"], longitude=loc["longitude"])
            for loc in target_locations_query
        ]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Each location in target_locations_query must have 'latitude' and 'longitude' keys.",
        )
    except Exception as e:  # Broad exception for other Pydantic validation errors if Location model changes
        raise HTTPException(status_code=400, detail=f"Invalid location format: {e}")

    analyzer = EnvironmentalHealthAnalyzer(environmental_readings=_ENV_READINGS_DB)
    exposure_results = analyzer.calculate_average_exposure(
        target_locations=target_locations,
        radius_km=radius_km,
        parameter_name=parameter_name,
        time_window_days=time_window_days,
    )
    return exposure_results
