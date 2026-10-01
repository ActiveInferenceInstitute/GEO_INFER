from typing import Any
from collections import defaultdict
import math
from datetime import timedelta

from geo_infer_health.models import DiseaseReport, Location, PopulationData
from geo_infer_health.utils.geospatial_utils import haversine_distance


class DiseaseHotspotAnalyzer:
    """Analyzes disease reports to identify hotspots."""

    def __init__(
        self,
        reports: list[DiseaseReport],
        population_data: list[PopulationData] | None = None,
    ):
        self.reports = sorted(reports, key=lambda r: r.report_date)
        self.population_data = population_data if population_data else []
        # Potential pre-processing: create a spatial index for reports or population_data

    def get_cases_in_radius(
        self, center_loc: Location, radius_km: float
    ) -> list[DiseaseReport]:
        """Returns all disease reports within a given radius of a center location."""
        return [
            report
            for report in self.reports
            if haversine_distance(report.location, center_loc) <= radius_km
        ]

    def calculate_local_incidence_rate(
        self,
        center_loc: Location,
        radius_km: float,
        time_window_days: int | None = None,
    ) -> tuple[float, int, int, bool]:
        """Calculates the incidence rate within a given radius and time window.

        Incidence rate is per 100,000 population when population data is
        available. If no time window, uses all reports.

        Population estimate: each population area whose location falls
        within the radius contributes its ``population_count``; a single
        unlocated population area is used as a coarse regional estimate.

        Fallback: when no population estimate exists but cases are
        present, the raw case count is returned in the rate slot. The
        ``population_estimated`` flag distinguishes this: it is ``True``
        only when a real population figure was used, so callers never
        mistake a raw case count for a per-100k rate.

        Args:
            center_loc: Center of the analysis radius.
            radius_km: Radius in kilometres.
            time_window_days: Optional trailing window in days.

        Returns:
            Tuple of (incidence_rate, total_cases, estimated_population,
            population_estimated).
        """
        relevant_reports = self.reports
        if time_window_days and self.reports:
            latest_report_date = self.reports[-1].report_date
            start_date = latest_report_date - timedelta(days=time_window_days)
            relevant_reports = [r for r in self.reports if r.report_date >= start_date]

        cases_in_radius = [
            report
            for report in relevant_reports
            if haversine_distance(report.location, center_loc) <= radius_km
        ]
        total_cases = sum(report.case_count for report in cases_in_radius)

        estimated_population = 0
        if self.population_data:
            for pop_area in self.population_data:
                if (
                    hasattr(pop_area, "location")
                    and haversine_distance(pop_area.location, center_loc) <= radius_km
                ):
                    estimated_population += pop_area.population_count
                elif (
                    not hasattr(pop_area, "location") and len(self.population_data) == 1
                ):
                    estimated_population = pop_area.population_count
                    break

        if estimated_population > 0:
            incidence_rate = (total_cases / estimated_population) * 100000
            return incidence_rate, total_cases, estimated_population, True

        # No usable population figure: report the raw case count in the
        # rate slot with population_estimated=False.
        return float(total_cases), total_cases, 0, False

    def identify_simple_hotspots(
        self,
        threshold_case_count: int = 5,
        scan_radius_km: float = 1.0,
        min_density_cases_per_sq_km: float | None = None,
    ) -> list[dict]:
        """Identifies simple hotspots based on case counts in a radius or density.

        Returns:
            A list of dictionaries, each representing a hotspot with 'location', 'case_count', 'radius_km'.
        """
        hotspots: list[dict[str, Any]] = []
        # This is a naive approach: iterate through each report as a potential center.
        # More sophisticated methods (e.g., DBSCAN, Getis-Ord Gi*) should be used for real applications.
        for report in self.reports:
            cases_in_scan = self.get_cases_in_radius(report.location, scan_radius_km)
            current_case_count = sum(r.case_count for r in cases_in_scan)

            is_hotspot = False
            if current_case_count >= threshold_case_count:
                is_hotspot = True

            if min_density_cases_per_sq_km is not None:
                area_sq_km = math.pi * (scan_radius_km**2)
                density = current_case_count / area_sq_km if area_sq_km > 0 else 0
                if density >= min_density_cases_per_sq_km:
                    is_hotspot = True
                else:
                    if (
                        current_case_count < threshold_case_count
                    ):  # if density was the only criteria initially failing
                        is_hotspot = False

            if is_hotspot:
                # Avoid adding nearly identical hotspots by checking distance to existing ones
                is_new_hotspot = True
                for hs in hotspots:
                    if (
                        haversine_distance(Location(**hs["location"]), report.location)
                        < scan_radius_km / 2
                    ):
                        # If a new potential hotspot is too close to an existing one,
                        # update the existing one if the new one has more cases.
                        if current_case_count > hs["case_count"]:
                            hs["case_count"] = current_case_count
                            hs["location"] = report.location.model_dump()  # Pydantic v2
                        is_new_hotspot = False
                        break
                if is_new_hotspot:
                    hotspots.append(
                        {
                            "location": report.location.model_dump(),
                            "case_count": current_case_count,
                            "radius_km": scan_radius_km,
                            "comment": "Simple threshold-based hotspot",
                        }
                    )
        return hotspots

    def simulate_sir_model(
        self,
        initial_infected: int,
        population: int,
        beta: float = 0.3,
        gamma: float = 0.1,
        days: int = 100,
    ) -> dict[str, Any]:
        """
        Simulate SIR (Susceptible-Infected-Recovered) disease spread model.

        Args:
            initial_infected: Initial number of infected individuals
            population: Total population
            beta: Transmission rate (probability of infection per contact)
            gamma: Recovery rate (1/infectious period in days)
            days: Number of days to simulate

        Returns:
            Dictionary with time series for S, I, R compartments

        Raises:
            ValueError: If gamma is not strictly positive; a non-positive
                recovery rate is an impossible epidemic state and would
                make the basic reproduction number undefined.
        """
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        susceptible: list[float] = [float(population - initial_infected)]
        infected: list[float] = [float(initial_infected)]
        recovered: list[float] = [0.0]

        for _ in range(days - 1):
            s, i, r = susceptible[-1], infected[-1], recovered[-1]
            n = s + i + r

            # SIR differential equations (discrete approximation)
            new_infected = (beta * s * i) / n if n > 0 else 0
            new_recovered = gamma * i

            susceptible.append(max(0, s - new_infected))
            infected.append(max(0, i + new_infected - new_recovered))
            recovered.append(r + new_recovered)

        return {
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered,
            "days": list(range(days)),
            "basic_reproduction_number": beta / gamma,
            "peak_infected": max(infected),
            "peak_day": infected.index(max(infected)),
        }

    def find_potential_contacts(
        self,
        case_report: "DiseaseReport",
        search_radius_km: float = 0.1,
        time_window_hours: int = 48,
    ) -> list[dict]:
        """
        Find potential contacts for a given case based on proximity and time.

        Args:
            case_report: The index case to find contacts for
            search_radius_km: Search radius in kilometers
            time_window_hours: Time window in hours before/after the case

        Returns:
            List of potential contact records
        """
        contacts = []

        for report in self.reports:
            if report == case_report:
                continue

            # Check temporal proximity
            time_diff = abs(
                (report.report_date - case_report.report_date).total_seconds() / 3600
            )
            if time_diff > time_window_hours:
                continue

            # Check spatial proximity
            distance = haversine_distance(report.location, case_report.location)
            if distance <= search_radius_km:
                contact = {
                    "contact_case_id": getattr(report, "id", str(id(report))),
                    "distance_km": round(distance, 4),
                    "time_difference_hours": round(time_diff, 2),
                    "location": report.location.model_dump()
                    if hasattr(report.location, "model_dump")
                    else {
                        "latitude": report.location.latitude,
                        "longitude": report.location.longitude,
                    },
                    "report_date": str(report.report_date),
                    "risk_score": self._calculate_contact_risk(
                        distance, time_diff, search_radius_km, time_window_hours
                    ),
                }
                contacts.append(contact)

        # Sort by risk score descending
        return sorted(contacts, key=lambda x: x["risk_score"], reverse=True)

    def _calculate_contact_risk(
        self,
        distance_km: float,
        time_diff_hours: float,
        max_distance: float,
        max_time: float,
    ) -> float:
        """Calculate contact risk score based on distance and time."""
        # Risk decreases with distance and time
        distance_factor = 1 - (distance_km / max_distance) if max_distance > 0 else 1
        time_factor = 1 - (time_diff_hours / max_time) if max_time > 0 else 1

        return round((distance_factor * 0.6 + time_factor * 0.4) * 100, 1)

    def analyze_temporal_trends(
        self, time_resolution: str = "daily", metric: str = "case_count"
    ) -> dict[str, Any]:
        """
        Analyze temporal trends in disease reports.

        Args:
            time_resolution: Time resolution ('hourly', 'daily', 'weekly')
            metric: Metric to analyze ('case_count', 'incidence_rate')

        Returns:
            Trend analysis results with time series and statistics
        """
        if not self.reports:
            return {"error": "No reports to analyze"}

        # Group reports by time period
        time_series: dict[str, int] = defaultdict(int)

        for report in self.reports:
            if time_resolution == "hourly":
                key = report.report_date.strftime("%Y-%m-%d %H:00")
            elif time_resolution == "weekly":
                key = report.report_date.strftime("%Y-W%W")
            else:  # daily
                key = report.report_date.strftime("%Y-%m-%d")

            time_series[key] += report.case_count

        # Sort by time
        sorted_times = sorted(time_series.keys())
        counts = [time_series[t] for t in sorted_times]

        # Calculate statistics
        if counts:
            avg = sum(counts) / len(counts)
            variance = (
                sum((x - avg) ** 2 for x in counts) / len(counts)
                if len(counts) > 1
                else 0
            )
            std_dev = math.sqrt(variance)

            # Detect trend direction (simple linear regression slope)
            n = len(counts)
            if n > 1:
                x_mean = (n - 1) / 2
                y_mean = avg
                numerator = sum((i - x_mean) * (counts[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                slope = numerator / denominator if denominator != 0 else 0
                trend = (
                    "increasing"
                    if slope > 0.1
                    else "decreasing"
                    if slope < -0.1
                    else "stable"
                )
            else:
                slope = 0
                trend = "insufficient_data"
        else:
            avg = std_dev = slope = 0
            trend = "no_data"

        return {
            "time_resolution": time_resolution,
            "time_periods": sorted_times,
            "case_counts": counts,
            "statistics": {
                "total_cases": sum(counts),
                "mean": round(avg, 2),
                "std_dev": round(std_dev, 2),
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
                "trend_slope": round(slope, 4),
                "trend_direction": trend,
            },
        }

    def calculate_reproduction_number(
        self, serial_interval_days: float = 5.0, window_days: int = 7
    ) -> dict[str, Any]:
        """
        Estimate the effective reproduction number (Rt) over time.

        Args:
            serial_interval_days: Average time between successive cases
            window_days: Rolling window for calculation

        Returns:
            Time series of Rt estimates
        """
        if len(self.reports) < window_days:
            return {"error": "Insufficient data for Rt calculation"}

        # Group by day
        daily_cases: dict[str, int] = defaultdict(int)
        for report in self.reports:
            day = report.report_date.strftime("%Y-%m-%d")
            daily_cases[day] += report.case_count

        sorted_days = sorted(daily_cases.keys())
        case_counts = [daily_cases[d] for d in sorted_days]

        # Calculate Rt via the case-ratio method
        rt_values: list[dict[str, Any]] = []
        for i in range(window_days, len(case_counts)):
            current_window = sum(case_counts[i - window_days + 1 : i + 1])
            previous_window = (
                sum(case_counts[i - 2 * window_days + 1 : i - window_days + 1])
                if i >= 2 * window_days
                else sum(case_counts[: i - window_days + 1])
            )

            if previous_window > 0:
                rt = current_window / previous_window
            else:
                rt = 0 if current_window == 0 else float("inf")

            rt_values.append(
                {
                    "date": sorted_days[i],
                    "rt": round(rt, 2) if rt != float("inf") else None,
                    "cases_current": current_window,
                    "cases_previous": previous_window,
                }
            )

        valid_rt = [v["rt"] for v in rt_values if v["rt"] is not None]

        return {
            "serial_interval_days": serial_interval_days,
            "window_days": window_days,
            "rt_time_series": rt_values,
            "summary": {
                "latest_rt": rt_values[-1]["rt"] if rt_values else None,
                "mean_rt": round(sum(valid_rt) / len(valid_rt), 2)
                if valid_rt
                else None,
                "trend": "epidemic"
                if valid_rt and valid_rt[-1] > 1
                else "declining"
                if valid_rt
                else "unknown",
            },
        }

    def generate_risk_map_data(
        self, grid_resolution_km: float = 1.0, bbox: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Generate risk map data by gridding the study area.

        Args:
            grid_resolution_km: Grid cell size in kilometers
            bbox: Bounding box (min_lat, max_lat, min_lon, max_lon)

        Returns:
            Grid-based risk assessment data
        """
        if not self.reports:
            return {"error": "No reports to map"}

        # Determine bounding box if not provided
        if bbox is None:
            lats = [
                r.location.latitude
                for r in self.reports
                if r.location.latitude is not None and r.location.longitude is not None
            ]
            lons = [
                r.location.longitude
                for r in self.reports
                if r.location.latitude is not None and r.location.longitude is not None
            ]
            margin = 0.1  # Add margin
            bbox = {
                "min_lat": min(lats) - margin,
                "max_lat": max(lats) + margin,
                "min_lon": min(lons) - margin,
                "max_lon": max(lons) + margin,
            }

        # Create grid
        # Approximate 1 degree = 111 km
        lat_step = grid_resolution_km / 111
        lon_step = grid_resolution_km / (
            111 * math.cos(math.radians((bbox["min_lat"] + bbox["max_lat"]) / 2))
        )

        grid_cells = []
        lat = bbox["min_lat"]
        while lat < bbox["max_lat"]:
            lon = bbox["min_lon"]
            while lon < bbox["max_lon"]:
                # Count cases in this cell
                center = Location(
                    latitude=lat + lat_step / 2, longitude=lon + lon_step / 2
                )
                cases_in_cell = self.get_cases_in_radius(center, grid_resolution_km / 2)
                case_count = sum(r.case_count for r in cases_in_cell)

                if case_count > 0:
                    grid_cells.append(
                        {
                            "lat": round(lat + lat_step / 2, 4),
                            "lon": round(lon + lon_step / 2, 4),
                            "case_count": case_count,
                            "risk_level": "high"
                            if case_count >= 10
                            else "medium"
                            if case_count >= 3
                            else "low",
                        }
                    )

                lon += lon_step
            lat += lat_step

        return {
            "grid_resolution_km": grid_resolution_km,
            "bounding_box": bbox,
            "total_cells_with_cases": len(grid_cells),
            "cells": grid_cells,
            "summary": {
                "high_risk_cells": sum(
                    1 for c in grid_cells if c["risk_level"] == "high"
                ),
                "medium_risk_cells": sum(
                    1 for c in grid_cells if c["risk_level"] == "medium"
                ),
                "low_risk_cells": sum(
                    1 for c in grid_cells if c["risk_level"] == "low"
                ),
            },
        }
