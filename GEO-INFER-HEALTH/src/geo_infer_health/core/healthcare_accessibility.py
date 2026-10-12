from typing import Any
from geo_infer_health.models import Location, HealthFacility, PopulationData
from geo_infer_health.utils.geospatial_utils import haversine_distance


class HealthcareAccessibilityAnalyzer:
    """Analyzes accessibility to healthcare facilities."""

    def __init__(
        self,
        facilities: list[HealthFacility],
        population_data: list[PopulationData] | None = None,
    ):
        self.facilities = facilities
        self.population_data = population_data if population_data else []

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance in kilometers between two locations."""
        return haversine_distance(loc1, loc2)

    def find_facilities_in_radius(
        self,
        center_loc: Location,
        radius_km: float,
        facility_type: str | None = None,
        required_services: list[str] | None = None,
    ) -> list[HealthFacility]:
        """Finds health facilities within a given radius, optionally filtering by type and services."""
        nearby_facilities: list[tuple[float, HealthFacility]] = []
        normalized_type = facility_type.lower() if facility_type else None
        for facility in self.facilities:
            if normalized_type and facility.facility_type.lower() != normalized_type:
                continue
            if required_services and not all(
                service in facility.services_offered for service in required_services
            ):
                continue
            distance = haversine_distance(facility.location, center_loc)
            if distance <= radius_km:
                nearby_facilities.append((distance, facility))
        return [
            facility
            for _, facility in sorted(nearby_facilities, key=lambda item: item[0])
        ]

    def get_nearest_facility(
        self,
        loc: Location,
        facility_type: str | None = None,
        required_services: list[str] | None = None,
    ) -> tuple[HealthFacility, float] | None:  # Returns (Facility, distance_km)
        """Finds the nearest health facility to a given location, with optional filters."""
        closest_facility = None
        min_distance = float("inf")

        candidate_facilities = self.facilities
        if facility_type:
            candidate_facilities = [
                f
                for f in candidate_facilities
                if f.facility_type.lower() == facility_type.lower()
            ]
        if required_services:
            candidate_facilities = [
                f
                for f in candidate_facilities
                if all(service in f.services_offered for service in required_services)
            ]

        if not candidate_facilities:
            return None

        for facility in candidate_facilities:
            distance = haversine_distance(loc, facility.location)
            if distance < min_distance:
                min_distance = distance
                closest_facility = facility

        return (closest_facility, min_distance) if closest_facility else None

    def calculate_facility_to_population_ratio(
        self,
        area_id: str,
        facility_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Count caller-supplied facilities relative to a population area's count.

        The caller supplies facilities for the region of interest; this method
        applies the type filter but does not clip facilities to area geometry.
        A zero population retains an infinite ratio and the filtered count.
        """
        target_pop_data = next(
            (p for p in self.population_data if p.area_id == area_id), None
        )
        if not target_pop_data:
            return None

        population = target_pop_data.population_count
        relevant_facilities = self.facilities
        if facility_type:
            normalized_type = facility_type.lower()
            relevant_facilities = [
                f
                for f in relevant_facilities
                if f.facility_type.lower() == normalized_type
            ]
        facility_count = len(relevant_facilities)

        if population == 0:
            return {
                "area_id": area_id,
                "facility_type_filter": facility_type,
                "ratio_per_1000_pop": float("inf"),
                "facility_count": facility_count,
                "population": 0,
                "message": "Population is zero.",
            }

        if facility_count == 0:
            return {
                "area_id": area_id,
                "facility_type_filter": facility_type,
                "ratio_per_1000_pop": 0.0,
                "facility_count": 0,
                "population": population,
                "message": "No facilities found for ratio calculation.",
            }

        # Ratio: facilities per 1,000 people for example
        ratio = (facility_count / population) * 1000

        return {
            "area_id": area_id,
            "facility_type_filter": facility_type,
            "ratio_per_1000_pop": ratio,
            "facility_count": facility_count,
            "population": population,
        }
