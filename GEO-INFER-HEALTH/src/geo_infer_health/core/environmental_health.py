from datetime import datetime, timedelta

from geo_infer_health.models import Location, EnvironmentalData
from geo_infer_health.utils.geospatial_utils import haversine_distance


class EnvironmentalHealthAnalyzer:
    """Analyzes environmental data in relation to health."""

    def __init__(self, environmental_readings: list[EnvironmentalData]):
        self.readings = sorted(environmental_readings, key=lambda r: r.timestamp)

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance in kilometers between two locations."""
        return haversine_distance(loc1, loc2)

    def get_environmental_readings_near_location(
        self,
        center_loc: Location,
        radius_km: float,
        parameter_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[EnvironmentalData]:
        """Retrieves environmental readings near a location within a given time window and for a specific parameter."""
        nearby_readings = []
        for reading in self.readings:
            if haversine_distance(reading.location, center_loc) <= radius_km:
                if (
                    parameter_name
                    and reading.parameter_name.lower() != parameter_name.lower()
                ):
                    continue
                if start_time and reading.timestamp <= start_time:
                    continue
                if end_time and reading.timestamp > end_time:
                    continue
                nearby_readings.append(reading)
        return nearby_readings

    def calculate_average_exposure(
        self,
        target_locations: list[Location],
        radius_km: float,
        parameter_name: str,
        time_window_days: int,
    ) -> dict[str, float | None]:  # Returns dict mapping location str to avg value
        """Calculates the average exposure to an environmental parameter for a list of locations.

        Args:
            target_locations: A list of Location objects.
            radius_km: Radius to search for environmental data around each target location.
            parameter_name: The specific environmental parameter to analyze (e.g., 'PM2.5').
            time_window_days: How many days back from the most recent reading to consider.

        Returns:
            A dictionary where keys are string representations of target locations
            and values are the average exposure, or None if no data.
        """

        def key_for(loc: Location) -> str:
            return f"{loc.latitude},{loc.longitude}"

        if not self.readings:
            return {key_for(loc): None for loc in target_locations}

        # Anchor the trailing window to the most recent reading in the
        # dataset (mirroring disease_surveillance.py) so historical datasets
        # older than ``time_window_days`` from *today* still yield results.
        latest_reading_time = self.readings[-1].timestamp
        start_time = latest_reading_time - timedelta(days=time_window_days)

        avg_exposure_results: dict[str, float | None] = {}
        for loc in target_locations:
            relevant_readings = self.get_environmental_readings_near_location(
                center_loc=loc,
                radius_km=radius_km,
                parameter_name=parameter_name,
                start_time=start_time,
                end_time=latest_reading_time,
            )
            if not relevant_readings:
                avg_exposure_results[key_for(loc)] = None
            else:
                total_value = sum(r.value for r in relevant_readings)
                avg_value = total_value / len(relevant_readings)
                avg_exposure_results[key_for(loc)] = avg_value
        return avg_exposure_results
