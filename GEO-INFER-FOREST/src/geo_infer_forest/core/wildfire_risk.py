"""Wildfire risk assessment module."""

import logging
from typing import Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import xarray as xr


def _normalized(data: xr.DataArray) -> xr.DataArray:
    """Scale a DataArray to [0, 1] by its finite maximum.

    Returns ``data / max``; uniform 1.0 when the maximum is zero,
    negative, or not finite, so an all-zero input cannot propagate
    NaN into downstream risk calculations.
    """
    max_value = float(data.max())
    if max_value <= 0.0 or not np.isfinite(max_value):
        return xr.ones_like(data)
    return data / max_value


logger = logging.getLogger(__name__)


class FireDangerRating(Enum):
    """Fire danger rating categories."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class FuelType(Enum):
    """Vegetation/fuel types for fire behavior."""

    GRASS = "grass"
    SHRUB = "shrub"
    TIMBER_UNDERSTORY = "timber_understory"
    TIMBER_LITTER = "timber_litter"
    SLASH = "slash"
    DORMANT_BRUSH = "dormant_brush"


@dataclass
class FireWeatherObservation:
    """Fire weather observation data."""

    observation_id: str
    location: tuple[float, float]
    timestamp: str
    temperature_c: float
    relative_humidity: float  # Percentage
    wind_speed_kmh: float
    wind_direction_deg: float
    precipitation_mm: float
    fuel_moisture_1hr: float | None = None
    fuel_moisture_10hr: float | None = None
    fuel_moisture_100hr: float | None = None


@dataclass
class FireIncident:
    """Fire incident data."""

    incident_id: str
    name: str
    location: tuple[float, float]
    start_time: str
    area_hectares: float
    containment_pct: float = 0.0
    cause: str | None = None
    fuel_type: FuelType = FuelType.TIMBER_UNDERSTORY


class WildfireRiskAnalyzer:
    """
    Comprehensive wildfire risk assessment system.

    Provides wildfire analysis including:
    - Fire weather index calculation
    - Risk assessment and mapping
    - Fire spread modeling
    - Suppression resource planning
    - Post-fire damage assessment
    """

    def __init__(self, config: dict | None = None):
        """Initialize wildfire risk analyzer."""
        self.config = config or {}

        # Fire weather thresholds
        self.thresholds = {
            "temperature_high": 35.0,  # °C
            "humidity_low": 20.0,  # %
            "wind_speed_high": 40.0,  # km/h
            "fuel_moisture_critical": 10.0,  # %
        }

        # Spread rate by fuel type (m/min under moderate conditions)
        self.base_spread_rates = {
            FuelType.GRASS: 12.0,
            FuelType.SHRUB: 8.0,
            FuelType.TIMBER_UNDERSTORY: 5.0,
            FuelType.TIMBER_LITTER: 2.5,
            FuelType.SLASH: 6.0,
            FuelType.DORMANT_BRUSH: 4.0,
        }

        # Incident registry
        self.active_incidents: dict[str, FireIncident] = {}

    def assess_wildfire_risk(
        self,
        temperature: xr.DataArray,
        precipitation: xr.DataArray,
        fuel_load: xr.DataArray | None = None,
        wind_speed: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess wildfire risk.

        Args:
            temperature: Temperature data
            precipitation: Precipitation data
            fuel_load: Optional fuel load data
            wind_speed: Optional wind speed data

        Returns:
            Wildfire risk assessment
        """
        # Calculate the drought index from precipitation and temperature
        mean_precip = precipitation.mean(dim="time")
        current_precip = precipitation.isel(time=-1)
        drought_index = 1 - (current_precip / (mean_precip + 1e-10))
        drought_index = xr.where(drought_index < 0, 0, drought_index)
        drought_index = xr.where(drought_index > 1, 1, drought_index)

        # Temperature factor
        temp_factor = (temperature - temperature.min()) / (
            temperature.max() - temperature.min() + 1e-10
        )

        # Combined risk
        risk = (drought_index + temp_factor) / 2

        if fuel_load is not None:
            fuel_factor = _normalized(fuel_load)
            risk = (risk + fuel_factor) / 2

        if wind_speed is not None:
            wind_factor = _normalized(wind_speed)
            risk = risk * (1 + wind_factor * 0.5)  # Wind increases risk

        risk = xr.where(risk > 1, 1, risk)

        return xr.Dataset(
            {
                "wildfire_risk": risk,
                "drought_index": drought_index,
                "temperature_factor": temp_factor,
            }
        )

    def calculate_fire_weather_index(
        self, observation: FireWeatherObservation
    ) -> dict[str, Any]:
        """
        Calculate Fire Weather Index (FWI) components.

        Args:
            observation: Fire weather observation

        Returns:
            FWI components and overall index
        """
        temp = observation.temperature_c
        rh = observation.relative_humidity
        wind = observation.wind_speed_kmh
        precip = observation.precipitation_mm

        # Fine Fuel Moisture Code (FFMC), first-order formulation
        # Higher temp and lower humidity = higher FFMC = drier fine fuels
        ffmc = 100 * (1 - rh / 100) * (1 + temp / 40)
        ffmc = min(101, max(0, ffmc))

        # Duff Moisture Code (DMC), first-order formulation
        # Measures moisture in moderate duff layer
        if precip < 1.5:
            dmc = max(0, 50 - rh + temp)
        else:
            dmc = max(0, 30 - rh / 2 + temp / 2)

        # Drought Code (DC), first-order formulation
        # Measures deep duff layer moisture
        dc = max(0, 200 - precip * 10 + temp * 2)

        # Initial Spread Index (ISI)
        # Based on FFMC and wind
        isi = 0.208 * (ffmc / 10) * np.exp(0.05039 * wind)

        # Buildup Index (BUI)
        # Combination of DMC and DC
        bui = (0.8 * dmc * dc) / (dmc + 0.4 * dc + 1)

        # Fire Weather Index
        fwi = 0.1 * isi * bui**0.46

        # Classify danger rating
        if fwi < 5:
            rating = FireDangerRating.LOW
        elif fwi < 10:
            rating = FireDangerRating.MODERATE
        elif fwi < 20:
            rating = FireDangerRating.HIGH
        elif fwi < 40:
            rating = FireDangerRating.VERY_HIGH
        else:
            rating = FireDangerRating.EXTREME

        return {
            "observation_id": observation.observation_id,
            "location": observation.location,
            "timestamp": observation.timestamp,
            "components": {
                "ffmc": float(ffmc),
                "dmc": float(dmc),
                "dc": float(dc),
                "isi": float(isi),
                "bui": float(bui),
            },
            "fwi": float(fwi),
            "danger_rating": rating.value,
            "weather_conditions": {
                "temperature_c": temp,
                "humidity_pct": rh,
                "wind_speed_kmh": wind,
                "precipitation_mm": precip,
            },
        }

    def predict_fire_spread(
        self,
        ignition_points: xr.DataArray,
        fuel_load: xr.DataArray,
        wind_direction: xr.DataArray | None = None,
        spread_boost: float = 0.5,
    ) -> xr.Dataset:
        """
        Predict potential fire spread with anisotropic wind forcing.

        Spread is evaluated in the eight principal compass directions.
        Each direction d receives a multiplier

            m_d = max(1.0, 1.0 + spread_boost * cos(d - wind_direction))

        so spread is strongest downwind (d aligned with the wind vector),
        weakest upwind (multiplier floors at 1.0), and neutral in
        perpendicular directions. ``wind_direction`` is the direction the
        wind blows toward, in degrees clockwise from north; when absent
        the multipliers are all 1.0 and spread is isotropic.

        Args:
            ignition_points: Fire ignition locations
            fuel_load: Fuel load data
            wind_direction: Optional wind direction (degrees from north,
                direction the wind blows toward)
            spread_boost: Wind-driven anisotropy strength

        Returns:
            Fire spread prediction with ``spread_probability`` and
            ``potential_spread`` (isotropic base) plus
            ``directional_spread`` carrying a ``direction`` dimension at
            the eight principal compass directions (0, 45, ..., 315
            degrees).
        """
        spread_probability = _normalized(fuel_load)
        potential = ignition_points * spread_probability
        directions = np.arange(0.0, 360.0, 45.0)

        if wind_direction is None:
            multipliers = xr.DataArray(
                np.ones(directions.shape),
                coords={"direction": directions},
                dims=("direction",),
            )
        else:
            wind_values = np.asarray(wind_direction.values, dtype=float)
            # cos of the angle between each spread direction and the wind
            angle_rad = np.radians(
                directions.reshape(-1, 1) - wind_values.reshape(1, -1)
            )
            factors = np.maximum(1.0, 1.0 + spread_boost * np.cos(angle_rad))
            if wind_values.ndim == 0 or wind_values.size == 1:
                multipliers = xr.DataArray(
                    factors[:, 0],
                    coords={"direction": directions},
                    dims=("direction",),
                )
            else:
                wind_dims = tuple(wind_direction.dims)
                wind_coords = {
                    d: wind_direction.coords[d] for d in wind_direction.coords
                }
                multipliers = xr.DataArray(
                    factors.reshape((directions.size,) + wind_values.shape),
                    coords={"direction": directions, **wind_coords},
                    dims=("direction",) + wind_dims,
                )

        directional_spread = potential * multipliers

        return xr.Dataset(
            {
                "spread_probability": spread_probability,
                "potential_spread": potential,
                "directional_spread": directional_spread,
            }
        )

    def model_fire_perimeter(
        self,
        ignition_point: tuple[float, float],
        fuel_type: FuelType,
        wind_speed_kmh: float,
        wind_direction_deg: float,
        slope_pct: float,
        time_hours: float,
    ) -> dict[str, Any]:
        """
        Model fire perimeter growth using elliptical model.

        Args:
            ignition_point: (lon, lat) of ignition
            fuel_type: Type of fuel/vegetation
            wind_speed_kmh: Wind speed
            wind_direction_deg: Wind direction (degrees from north)
            slope_pct: Slope percentage
            time_hours: Time since ignition

        Returns:
            Fire perimeter model results
        """
        # Base spread rate (m/min)
        base_rate = self.base_spread_rates.get(fuel_type, 5.0)

        # Wind factor (increases head fire rate)
        wind_factor = 1 + wind_speed_kmh / 50

        # Slope factor (fire spreads faster uphill)
        slope_factor = 1 + slope_pct / 100

        # Head fire rate of spread (m/min)
        head_rate = base_rate * wind_factor * slope_factor

        # Backing fire (opposite direction)
        back_rate = base_rate * 0.3 / wind_factor

        # Flanking fire (perpendicular)
        flank_rate = (head_rate + back_rate) / 3

        # Distance traveled in given time (m)
        time_min = time_hours * 60
        head_distance = head_rate * time_min
        back_distance = back_rate * time_min
        flank_distance = flank_rate * time_min

        # Ellipse parameters
        length = head_distance + back_distance  # Major axis
        width = 2 * flank_distance  # Minor axis

        # Area (hectares)
        area_m2 = np.pi * (length / 2) * (width / 2)
        area_ha = area_m2 / 10000

        # Generate perimeter points from the ellipse model
        n_points = 36
        angles = np.linspace(0, 2 * np.pi, n_points)

        # Offset ellipse center based on head/back ratio
        offset = (head_distance - back_distance) / 2

        perimeter_x = (length / 2) * np.cos(angles) + offset
        perimeter_y = (width / 2) * np.sin(angles)

        # Rotate by wind direction
        wind_rad = np.radians(wind_direction_deg)
        rotated_x = perimeter_x * np.cos(wind_rad) - perimeter_y * np.sin(wind_rad)
        rotated_y = perimeter_x * np.sin(wind_rad) + perimeter_y * np.cos(wind_rad)

        return {
            "ignition_point": ignition_point,
            "fuel_type": fuel_type.value,
            "time_hours": time_hours,
            "spread_rates": {
                "head_m_per_min": float(head_rate),
                "back_m_per_min": float(back_rate),
                "flank_m_per_min": float(flank_rate),
            },
            "distances": {
                "head_m": float(head_distance),
                "back_m": float(back_distance),
                "flank_m": float(flank_distance),
            },
            "perimeter_length_km": float(
                np.pi * np.sqrt(2 * (length**2 + width**2) / 4) / 1000
            ),
            "area_hectares": float(area_ha),
            "perimeter_x": rotated_x.tolist(),
            "perimeter_y": rotated_y.tolist(),
        }

    def plan_suppression_resources(
        self,
        fire_size_ha: float,
        danger_rating: FireDangerRating,
        terrain_difficulty: str = "moderate",
        resources_available: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        Plan suppression resources based on fire characteristics.

        Args:
            fire_size_ha: Current fire size in hectares
            danger_rating: Current fire danger rating
            terrain_difficulty: 'easy', 'moderate', 'difficult'
            resources_available: Available resources

        Returns:
            Resource allocation plan
        """
        # Base staffing per hectare
        base_staff_per_ha = {
            FireDangerRating.LOW: 0.5,
            FireDangerRating.MODERATE: 1.0,
            FireDangerRating.HIGH: 2.0,
            FireDangerRating.VERY_HIGH: 4.0,
            FireDangerRating.EXTREME: 8.0,
        }

        terrain_multiplier = {"easy": 0.8, "moderate": 1.0, "difficult": 1.5}

        # Calculate personnel needs
        multiplier = terrain_multiplier.get(terrain_difficulty, 1.0)
        base_rate = base_staff_per_ha.get(danger_rating, 2.0)
        personnel_needed = int(np.ceil(fire_size_ha * base_rate * multiplier))

        # Equipment needs
        engines_needed = max(1, int(fire_size_ha / 50))
        dozers_needed = (
            max(0, int(fire_size_ha / 200)) if terrain_difficulty != "difficult" else 0
        )
        helicopters_needed = max(1, int(fire_size_ha / 100))
        airtankers_needed = (
            1
            if fire_size_ha > 50
            or danger_rating in [FireDangerRating.VERY_HIGH, FireDangerRating.EXTREME]
            else 0
        )

        # Calculate containment timeline
        if danger_rating == FireDangerRating.EXTREME:
            hours_to_contain = fire_size_ha * 0.5
        elif danger_rating == FireDangerRating.VERY_HIGH:
            hours_to_contain = fire_size_ha * 0.3
        else:
            hours_to_contain = fire_size_ha * 0.2

        days_to_contain = np.ceil(hours_to_contain / 12)  # 12-hour operational periods

        return {
            "fire_size_ha": fire_size_ha,
            "danger_rating": danger_rating.value,
            "terrain": terrain_difficulty,
            "personnel": {
                "firefighters_needed": personnel_needed,
                "crews_20_person": int(np.ceil(personnel_needed / 20)),
                "incident_commanders": max(1, int(personnel_needed / 50)),
            },
            "equipment": {
                "engines": engines_needed,
                "dozers": dozers_needed,
                "helicopters": helicopters_needed,
                "airtankers": airtankers_needed,
            },
            "timeline": {
                "estimated_containment_days": int(days_to_contain),
                "full_suppression_days": int(days_to_contain * 2),
            },
            "estimated_cost_usd": int(fire_size_ha * 5000 * multiplier * base_rate),
        }

    def assess_post_fire_damage(
        self,
        pre_fire_ndvi: xr.DataArray,
        post_fire_ndvi: xr.DataArray,
        land_cover: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess post-fire damage using vegetation indices.

        Args:
            pre_fire_ndvi: Pre-fire NDVI
            post_fire_ndvi: Post-fire NDVI
            land_cover: Optional land cover classification

        Returns:
            Burn severity assessment
        """
        # Differenced NDVI (dNDVI)
        dndvi = pre_fire_ndvi - post_fire_ndvi

        # Relativized dNDVI (RdNDVI)
        rdndvi = dndvi / (np.sqrt(np.abs(pre_fire_ndvi)) + 1e-10)

        # Classify burn severity
        # 0 = unburned, 1 = low, 2 = moderate, 3 = high
        severity = xr.zeros_like(dndvi)
        severity = xr.where(dndvi > 0.1, 1, severity)
        severity = xr.where(dndvi > 0.25, 2, severity)
        severity = xr.where(dndvi > 0.4, 3, severity)

        # Calculate burned area by severity class
        total_cells = float(dndvi.size)
        unburned_pct = float((severity == 0).sum()) / total_cells * 100
        low_pct = float((severity == 1).sum()) / total_cells * 100
        moderate_pct = float((severity == 2).sum()) / total_cells * 100
        high_pct = float((severity == 3).sum()) / total_cells * 100

        return xr.Dataset(
            {"dndvi": dndvi, "rdndvi": rdndvi, "burn_severity": severity},
            attrs={
                "severity_classes": {0: "Unburned", 1: "Low", 2: "Moderate", 3: "High"},
                "unburned_pct": unburned_pct,
                "low_severity_pct": low_pct,
                "moderate_severity_pct": moderate_pct,
                "high_severity_pct": high_pct,
                "total_burned_pct": low_pct + moderate_pct + high_pct,
            },
        )

    def calculate_evacuation_zones(
        self,
        fire_location: tuple[float, float],
        predicted_spread_km: float,
        wind_direction_deg: float,
        population_density: xr.DataArray | None = None,
    ) -> dict[str, Any]:
        """
        Calculate evacuation zones based on fire threat.

        Args:
            fire_location: Current fire center (lon, lat)
            predicted_spread_km: Predicted spread distance
            wind_direction_deg: Predominant wind direction
            population_density: Optional population density grid

        Returns:
            Evacuation zone recommendations
        """
        # Zone radii (km)
        immediate_radius = predicted_spread_km * 1.5
        warning_radius = predicted_spread_km * 3.0
        advisory_radius = predicted_spread_km * 5.0

        # Calculate downwind hazard zone (elliptical)
        # Fire spreads primarily in wind direction

        # Downwind extension
        downwind_factor = 2.0

        zones = {
            "fire_location": fire_location,
            "wind_direction_deg": wind_direction_deg,
            "zones": {
                "immediate_evacuation": {
                    "radius_km": immediate_radius,
                    "downwind_extension_km": immediate_radius * downwind_factor,
                    "priority": "CRITICAL",
                    "recommended_action": "Evacuate immediately",
                },
                "evacuation_warning": {
                    "radius_km": warning_radius,
                    "downwind_extension_km": warning_radius * downwind_factor,
                    "priority": "HIGH",
                    "recommended_action": "Prepare to evacuate",
                },
                "evacuation_advisory": {
                    "radius_km": advisory_radius,
                    "downwind_extension_km": advisory_radius * downwind_factor,
                    "priority": "MODERATE",
                    "recommended_action": "Be ready to leave",
                },
            },
        }

        return zones

    def register_incident(self, incident: FireIncident) -> str:
        """Register a fire incident."""
        self.active_incidents[incident.incident_id] = incident
        logger.info(f"Registered fire incident: {incident.name}")
        return incident.incident_id

    def get_active_incidents(self) -> list[FireIncident]:
        """Get list of active incidents."""
        return list(self.active_incidents.values())
