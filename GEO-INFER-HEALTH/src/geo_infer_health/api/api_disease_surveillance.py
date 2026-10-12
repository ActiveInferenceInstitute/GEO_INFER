from fastapi import APIRouter, HTTPException, Body, Query
from typing import Any

from geo_infer_health.models import DiseaseReport, Location, PopulationData
from geo_infer_health.core.disease_surveillance import DiseaseHotspotAnalyzer

router = APIRouter(
    prefix="/surveillance",
    tags=["Disease Surveillance"],
)

# In a real application, you'd have a way to load/manage this data persistently.
# The store is module-level and shared across requests.
# This is the in-memory store backing the demo API.
# Global in-memory stores are NOT suitable for production.
_DISEASE_REPORTS_DB: list[DiseaseReport] = []
_POPULATION_DATA_DB: list[PopulationData] = []


def reset_stores() -> None:
    """Clear the in-memory stores (for tests and demo resets)."""
    _DISEASE_REPORTS_DB.clear()
    _POPULATION_DATA_DB.clear()


@router.post("/reports/", response_model=DiseaseReport, status_code=201)
async def submit_disease_report(report: DiseaseReport = Body(...)) -> DiseaseReport:
    """Submit a new disease report."""
    # Basic validation or processing could happen here
    _DISEASE_REPORTS_DB.append(report)
    return report


@router.get("/reports/", response_model=list[DiseaseReport])
async def get_all_disease_reports(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> list[DiseaseReport]:
    """Retrieve all submitted disease reports with pagination."""
    return _DISEASE_REPORTS_DB[offset : offset + limit]


@router.post("/hotspots/identify", response_model=list[dict])
async def identify_disease_hotspots(
    threshold_case_count: int = Query(
        5,
        ge=1,
        description="Minimum case count to be considered a potential hotspot center.",
    ),
    scan_radius_km: float = Query(
        1.0,
        gt=0,
        description="Scan radius in kilometers around each report to count cases.",
    ),
    min_density_cases_per_sq_km: float | None = Query(
        None,
        gt=0,
        description="Optional minimum case density (cases/km^2) to define a hotspot.",
    ),
) -> list[dict]:
    """Identifies disease hotspots based on current reports."""
    if not _DISEASE_REPORTS_DB:
        raise HTTPException(
            status_code=404, detail="No disease reports available to analyze."
        )

    analyzer = DiseaseHotspotAnalyzer(
        reports=_DISEASE_REPORTS_DB, population_data=_POPULATION_DATA_DB
    )
    hotspots = analyzer.identify_simple_hotspots(
        threshold_case_count=threshold_case_count,
        scan_radius_km=scan_radius_km,
        min_density_cases_per_sq_km=min_density_cases_per_sq_km,
    )
    return hotspots


@router.post("/incidence_rate/local", response_model=dict)
async def get_local_incidence_rate(
    latitude: float = Query(..., description="Latitude of the center point."),
    longitude: float = Query(..., description="Longitude of the center point."),
    radius_km: float = Query(
        ..., gt=0, description="Radius in kilometers to calculate incidence rate."
    ),
    time_window_days: int | None = Query(
        None,
        ge=1,
        description="Optional time window in days to consider recent reports.",
    ),
) -> dict[str, Any]:
    """Calculates the local incidence rate for a given area and time window."""
    if not _DISEASE_REPORTS_DB:
        raise HTTPException(
            status_code=404, detail="No disease reports available to analyze."
        )

    center_loc = Location(latitude=latitude, longitude=longitude)
    analyzer = DiseaseHotspotAnalyzer(
        reports=_DISEASE_REPORTS_DB, population_data=_POPULATION_DATA_DB
    )

    rate, cases, population, population_estimated = (
        analyzer.calculate_local_incidence_rate(
            center_loc=center_loc,
            radius_km=radius_km,
            time_window_days=time_window_days,
        )
    )
    return {
        "center_location": center_loc,
        "radius_km": radius_km,
        "time_window_days": time_window_days,
        "incidence_rate_per_100k": rate,
        "total_cases_in_area": cases,
        "estimated_population_in_area": population,
        "population_estimated": population_estimated,
    }


@router.post("/population_data/", response_model=PopulationData, status_code=201)
async def add_population_data_area(data: PopulationData = Body(...)) -> PopulationData:
    """Add population data for a specific area."""
    _POPULATION_DATA_DB.append(data)
    return data


@router.get("/population_data/", response_model=list[PopulationData])
async def get_all_population_data(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> list[PopulationData]:
    """Retrieve all population data entries."""
    return _POPULATION_DATA_DB[offset : offset + limit]
