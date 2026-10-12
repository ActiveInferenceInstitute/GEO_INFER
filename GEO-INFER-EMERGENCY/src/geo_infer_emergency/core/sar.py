"""
Search and rescue module.

Provides SAR mission planning, probability of containment,
search pattern generation, and team coordination.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .geo import haversine_distance_km
import math

logger = logging.getLogger(__name__)


class SearchPattern(Enum):
    """Standard SAR search patterns."""

    PARALLEL = "parallel"
    CREEPING_LINE = "creeping_line"
    EXPANDING_SQUARE = "expanding_square"
    SECTOR = "sector"
    CONTOUR = "contour"
    GRID = "grid"


class SubjectType(Enum):
    """Types of search subjects."""

    HIKER = "hiker"
    CHILD = "child"
    ELDERLY = "elderly"
    DEMENTIA = "dementia"
    HUNTER = "hunter"
    CLIMBER = "climber"
    WATER_VICTIM = "water_victim"


@dataclass
class SearchSubject:
    """Information about the search subject."""

    subject_id: str
    subject_type: SubjectType
    name: str
    age: int | None = None
    last_known_location: dict[str, float] | None = None
    last_seen_time: datetime | None = None
    medical_conditions: list[str] = field(default_factory=list)
    clothing: str = ""
    experience_level: str = "unknown"


@dataclass
class SearchTeam:
    """Represents a search team."""

    team_id: str
    name: str
    size: int
    capabilities: list[str]
    location: dict[str, float] | None = None
    assigned_sector: str | None = None
    status: str = "available"


@dataclass
class SearchArea:
    """Defines a search area with probability."""

    area_id: str
    geometry: dict[str, Any]  # GeoJSON
    probability: float = 0.5
    terrain: str = "mixed"
    search_pattern: SearchPattern | None = None
    searched: bool = False


class SearchAndRescue:
    """
    Plan and coordinate search and rescue operations with
    probabilistic search planning and team management.
    """

    # Statistical search radius multipliers by subject type (in km)
    SEARCH_RADIUS = {
        SubjectType.HIKER: 10.0,
        SubjectType.CHILD: 2.0,
        SubjectType.ELDERLY: 3.0,
        SubjectType.DEMENTIA: 3.5,
        SubjectType.HUNTER: 8.0,
        SubjectType.CLIMBER: 5.0,
        SubjectType.WATER_VICTIM: 2.0,
    }

    def __init__(
        self,
        terrain_data: dict[str, Any] | None = None,
        statistical_data: dict[str, Any] | None = None,
        team_capabilities: list[str] | None = None,
    ):
        """
        Initialize search and rescue module.

        Args:
            terrain_data: Terrain information
            statistical_data: Lost person statistics
            team_capabilities: Available team capabilities
        """
        self.terrain_data = terrain_data
        self.statistical_data = statistical_data
        self.team_capabilities = team_capabilities or ["ground", "k9", "aerial"]
        self._subjects: dict[str, SearchSubject] = {}
        self._teams: dict[str, SearchTeam] = {}
        self._search_areas: dict[str, SearchArea] = {}
        logger.info("Initialized SearchAndRescue module")

    def register_subject(self, subject_data: dict[str, Any]) -> SearchSubject:
        """Register a search subject."""
        subject = SearchSubject(
            subject_id=subject_data.get("id", f"subject_{len(self._subjects)}"),
            subject_type=SubjectType(subject_data.get("type", "hiker").lower()),
            name=subject_data.get("name", "Unknown"),
            age=subject_data.get("age"),
            last_known_location=subject_data.get("last_known_location"),
            last_seen_time=subject_data.get("last_seen_time"),
            medical_conditions=subject_data.get("medical_conditions", []),
            clothing=subject_data.get("clothing", ""),
            experience_level=subject_data.get("experience", "unknown"),
        )
        self._subjects[subject.subject_id] = subject
        return subject

    def register_team(self, team_data: dict[str, Any]) -> SearchTeam:
        """Register a search team."""
        team = SearchTeam(
            team_id=team_data.get("id", f"team_{len(self._teams)}"),
            name=team_data.get("name", "Search Team"),
            size=team_data.get("size", 4),
            capabilities=team_data.get("capabilities", ["ground"]),
            location=team_data.get("location"),
        )
        self._teams[team.team_id] = team
        return team

    def plan_mission(
        self,
        subject: dict[str, Any],
        last_known_point: dict[str, float],
        search_radius: float | None = None,
        terrain_type: str = "mixed",
        weather: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Plan a SAR mission.

        Args:
            subject: Subject information
            last_known_point: Last known location
            search_radius: Search radius in km
            terrain_type: Type of terrain
            weather: Weather conditions

        Returns:
            SAR mission plan
        """
        # Register subject
        registered = self.register_subject(subject)

        # Determine search radius from statistics if not provided
        if search_radius is None:
            search_radius = self.SEARCH_RADIUS.get(registered.subject_type, 5.0)

        # Calculate probability areas
        prob_areas = self.calculate_pod(
            subject=subject,
            search_area={"center": last_known_point, "radius_km": search_radius},
            search_effort=1.0,
            terrain_coverable="moderate",
        )

        # Generate search pattern
        pattern = self.generate_pattern(
            area={"center": last_known_point, "radius_km": search_radius},
            pattern_type="expanding_square",
            team_size=4,
            visibility_distance=50,
        )

        # Estimate resources
        mission = {
            "mission_id": f"sar_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "subject": {
                "id": registered.subject_id,
                "type": registered.subject_type.value,
                "name": registered.name,
                "age": registered.age,
            },
            "last_known_point": last_known_point,
            "search_radius_km": search_radius,
            "terrain_type": terrain_type,
            "weather": weather or {"conditions": "unknown"},
            "probability_areas": prob_areas.get("areas", []),
            "recommended_pattern": pattern,
            "resource_estimate": {
                "ground_teams": max(1, int(search_radius / 2)),
                "k9_teams": 1 if search_radius > 3 else 0,
                "aerial_assets": 1 if search_radius > 5 else 0,
                "estimated_search_hours": search_radius * 2,
            },
            "priority_areas": [
                {"area": "near_lkp", "priority": 1, "probability": 0.5},
                {"area": "trails", "priority": 2, "probability": 0.25},
                {"area": "water_sources", "priority": 3, "probability": 0.15},
            ],
            "status": "planned",
        }

        logger.info(f"Created SAR mission for {registered.name}")
        return mission

    def calculate_pod(
        self,
        subject: dict[str, Any],
        search_area: dict[str, Any],
        search_effort: float,
        terrain_coverable: str,
    ) -> dict[str, Any]:
        """
        Calculate probability of detection (POD).

        Args:
            subject: Subject characteristics
            search_area: Area being searched
            search_effort: Search effort level
            terrain_coverable: Terrain coverage factor

        Returns:
            POD calculations
        """
        # Terrain coverage factors
        terrain_factors = {
            "easy": 0.9,
            "moderate": 0.7,
            "difficult": 0.5,
            "extreme": 0.3,
        }

        coverage_factor = terrain_factors.get(terrain_coverable, 0.6)

        # Calculate POD using simple model
        # POD = 1 - e^(-coverage * effort)
        import math

        pod = 1 - math.exp(-coverage_factor * search_effort)

        # Generate probability areas (concentric rings)
        center = search_area.get("center", {"lat": 0, "lon": 0})
        radius = search_area.get("radius_km", 5)

        areas = [
            {
                "ring": 1,
                "distance_km": radius * 0.25,
                "probability": 0.50,
                "cumulative_prob": 0.50,
            },
            {
                "ring": 2,
                "distance_km": radius * 0.50,
                "probability": 0.25,
                "cumulative_prob": 0.75,
            },
            {
                "ring": 3,
                "distance_km": radius * 0.75,
                "probability": 0.15,
                "cumulative_prob": 0.90,
            },
            {
                "ring": 4,
                "distance_km": radius * 1.00,
                "probability": 0.10,
                "cumulative_prob": 1.00,
            },
        ]

        result = {
            "pod": round(pod, 3),
            "coverage_factor": coverage_factor,
            "search_effort": search_effort,
            "areas": areas,
            "center": center,
            "total_area_sq_km": math.pi * radius**2,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Calculated POD: {pod:.2%}")
        return result

    def generate_pattern(
        self,
        area: dict[str, Any],
        pattern_type: str = "expanding_square",
        team_size: int = 4,
        visibility_distance: float = 50,
    ) -> dict[str, Any]:
        """
        Generate search pattern.

        Args:
            area: Area to search
            pattern_type: Type of search pattern
            team_size: Size of search team
            visibility_distance: Visibility in meters

        Returns:
            Search pattern with waypoints
        """
        center = area.get("center", {"lat": 0, "lon": 0})
        radius_km = area.get("radius_km", 1)

        pattern = {
            "pattern_type": pattern_type,
            "center": center,
            "area_radius_km": radius_km,
            "team_size": team_size,
            "visibility_distance_m": visibility_distance,
            "track_spacing_m": visibility_distance * 2,  # For overlap
            "waypoints": [],
            "estimated_search_time_hours": 0,
            "estimated_distance_km": 0,
        }

        # Generate waypoints based on pattern type
        if pattern_type == "expanding_square":
            pattern["waypoints"] = self._generate_expanding_square(center, radius_km)
        elif pattern_type == "parallel":
            pattern["waypoints"] = self._generate_parallel_lines(
                center, radius_km, visibility_distance * 2 / 1000
            )
        elif pattern_type == "sector":
            pattern["waypoints"] = self._generate_sector(center, radius_km)
        else:
            pattern["waypoints"] = self._generate_grid(
                center, radius_km, visibility_distance * 2 / 1000
            )

        # Calculate distance and time
        pattern["estimated_distance_km"] = self._calculate_pattern_distance(
            pattern["waypoints"]
        )
        search_speed_kmh = 2.0  # Average walking speed in rough terrain
        pattern["estimated_search_time_hours"] = (
            pattern["estimated_distance_km"] / search_speed_kmh
        )

        logger.info(
            f"Generated {pattern_type} pattern with {len(pattern['waypoints'])} waypoints"
        )
        return pattern

    def _generate_expanding_square(
        self, center: dict[str, float], radius_km: float
    ) -> list[dict[str, float]]:
        """Generate expanding square pattern waypoints."""
        waypoints = [center.copy()]

        leg_length = 0.1  # km
        direction = 0  # 0=E, 1=N, 2=W, 3=S
        x, y = center.get("lon", 0), center.get("lat", 0)

        while leg_length < radius_km:
            # Each pair of legs increases length
            for _ in range(2):
                # Move in current direction
                if direction == 0:  # East
                    x += leg_length / 111  # Approximate degrees
                elif direction == 1:  # North
                    y += leg_length / 111
                elif direction == 2:  # West
                    x -= leg_length / 111
                else:  # South
                    y -= leg_length / 111

                waypoints.append({"lat": y, "lon": x})
                direction = (direction + 1) % 4

            leg_length += 0.1

        return waypoints

    def _generate_parallel_lines(
        self, center: dict[str, float], radius_km: float, spacing_km: float
    ) -> list[dict[str, float]]:
        """Generate parallel line pattern waypoints."""
        waypoints = []

        num_lines = int(2 * radius_km / spacing_km) + 1
        lat_start = center.get("lat", 0) - radius_km / 111
        lon_start = center.get("lon", 0) - radius_km / 111
        lon_end = center.get("lon", 0) + radius_km / 111

        for i in range(num_lines):
            lat = lat_start + (i * spacing_km / 111)

            if i % 2 == 0:
                waypoints.append({"lat": lat, "lon": lon_start})
                waypoints.append({"lat": lat, "lon": lon_end})
            else:
                waypoints.append({"lat": lat, "lon": lon_end})
                waypoints.append({"lat": lat, "lon": lon_start})

        return waypoints

    def _generate_sector(
        self, center: dict[str, float], radius_km: float
    ) -> list[dict[str, float]]:
        """Generate sector search pattern waypoints."""
        waypoints = [center.copy()]

        for angle in range(0, 360, 45):  # 8 sectors
            rad = math.radians(angle)

            # Go to edge
            lat = center.get("lat", 0) + (radius_km / 111) * math.cos(rad)
            lon = center.get("lon", 0) + (radius_km / 111) * math.sin(rad)
            waypoints.append({"lat": lat, "lon": lon})

            # Return to center
            waypoints.append(center.copy())

        return waypoints

    def _generate_grid(
        self, center: dict[str, float], radius_km: float, spacing_km: float
    ) -> list[dict[str, float]]:
        """Generate grid pattern waypoints."""
        return self._generate_parallel_lines(center, radius_km, spacing_km)

    def _calculate_pattern_distance(self, waypoints: list[dict[str, float]]) -> float:
        """Calculate total distance of pattern."""
        if len(waypoints) < 2:
            return 0

        total = sum(
            haversine_distance_km(waypoints[i - 1], waypoints[i])
            for i in range(1, len(waypoints))
        )
        return round(total, 2)

    def coordinate_teams(
        self,
        teams: list[dict[str, Any]],
        search_areas: list[dict[str, Any]],
        assignments: dict[str, str],
        briefing_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate search teams.

        Args:
            teams: Available search teams
            search_areas: Areas to search
            assignments: Team to area assignments
            briefing_time: Scheduled briefing time

        Returns:
            Team coordination plan
        """
        # Register teams
        for team_data in teams:
            self.register_team(team_data)

        teams_out: list[dict[str, Any]] = []
        search_areas_out: list[dict[str, Any]] = []
        assignments_out: list[dict[str, Any]] = []
        coordination: dict[str, Any] = {
            "coordination_id": f"coord_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "briefing_time": briefing_time.isoformat() if briefing_time else None,
            "teams": teams_out,
            "search_areas": search_areas_out,
            "assignments": assignments_out,
            "communication_plan": {
                "command_frequency": "SAR_CMD_1",
                "tactical_frequencies": ["SAR_TAC_1", "SAR_TAC_2"],
                "check_in_interval_minutes": 30,
            },
            "safety": {
                "weather_check_time": None,
                "recall_signal": "3 long blasts",
                "emergency_frequency": "SAR_EMER",
            },
        }

        # Process teams
        for team_id, team in self._teams.items():
            team_info: dict[str, Any] = {
                "team_id": team_id,
                "name": team.name,
                "size": team.size,
                "capabilities": team.capabilities,
                "assigned_area": assignments.get(team_id),
            }
            teams_out.append(team_info)

        # Process search areas
        for area in search_areas:
            area_info: dict[str, Any] = {
                "area_id": area.get("id"),
                "priority": area.get("priority", 1),
                "probability": area.get("probability", 0.5),
                "terrain": area.get("terrain", "mixed"),
                "assigned_team": None,
            }

            # Find assigned team
            for team_id, assigned_area in assignments.items():
                if assigned_area == area.get("id"):
                    area_info["assigned_team"] = team_id
                    break

            search_areas_out.append(area_info)

        # Create assignment list
        for team_id, area_id in assignments.items():
            assignments_out.append(
                {"team_id": team_id, "area_id": area_id, "status": "assigned"}
            )

        logger.info(f"Coordinated {len(teams)} teams across {len(search_areas)} areas")
        return coordination

    def update_probability(
        self,
        area_id: str,
        search_result: str,
        new_information: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update search probability based on results.

        Args:
            area_id: Area that was searched
            search_result: Result ('negative', 'clue_found', 'subject_found')
            new_information: Any new information

        Returns:
            Updated probability map
        """
        if area_id not in self._search_areas:
            # Create baseline
            self._search_areas[area_id] = SearchArea(
                area_id=area_id, geometry={}, probability=0.5
            )

        area = self._search_areas[area_id]
        old_prob = area.probability

        # Bayesian update based on result
        if search_result == "negative":
            # Reduce probability if not found
            pod = 0.7  # Assumed probability of detection
            area.probability = old_prob * (1 - pod) / (1 - old_prob * pod)
            area.searched = True
        elif search_result == "clue_found":
            # Increase probability
            area.probability = min(0.95, old_prob * 1.5)
        elif search_result == "subject_found":
            area.probability = 1.0
            area.searched = True

        # Normalize other areas
        if search_result == "negative":
            other_areas = [
                a for a in self._search_areas.values() if a.area_id != area_id
            ]
            if other_areas:
                prob_increase = (old_prob - area.probability) / len(other_areas)
                for other in other_areas:
                    other.probability = min(1.0, other.probability + prob_increase)

        result = {
            "area_id": area_id,
            "search_result": search_result,
            "previous_probability": round(old_prob, 3),
            "updated_probability": round(area.probability, 3),
            "searched": area.searched,
            "timestamp": datetime.now().isoformat(),
            "all_areas": [
                {"area_id": a.area_id, "probability": round(a.probability, 3)}
                for a in self._search_areas.values()
            ],
        }

        logger.info(
            f"Updated probability for {area_id}: {old_prob:.2%} -> {area.probability:.2%}"
        )
        return result
