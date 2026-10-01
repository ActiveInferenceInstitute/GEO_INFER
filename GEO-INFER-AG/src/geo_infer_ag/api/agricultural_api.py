"""
Agricultural API Implementation

This module provides the AgriculturalAPI class for agricultural applications
in the GEO-INFER framework.
"""

import logging
from typing import Any
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class AgriculturalConfig:
    """Configuration for agricultural API."""

    # API settings
    api_version: str = "1.0.0"
    base_url: str = "https://api.geo-infer.ag"
    timeout: int = 30

    # Agricultural parameters
    crop_types: list[str] | None = None
    soil_types: list[str] | None = None
    climate_zones: list[str] | None = None

    def __post_init__(self) -> None:
        if self.crop_types is None:
            self.crop_types = ["corn", "soybeans", "wheat", "rice", "cotton"]
        if self.soil_types is None:
            self.soil_types = ["clay", "silt", "loam", "sandy"]
        if self.climate_zones is None:
            self.climate_zones = ["tropical", "temperate", "arid", "mediterranean"]


class AgriculturalAPI:
    """
    API client for agricultural data and analysis.

    Provides access to agricultural datasets, crop modeling,
    and precision agriculture services.
    """

    def __init__(self, config: AgriculturalConfig | None = None) -> None:
        """
        Initialize agricultural API client.

        Args:
            config: API configuration
        """
        self.config = config or AgriculturalConfig()
        self.session: Any | None = None
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize API session."""
        try:
            import requests

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": f"GEO-INFER-AG/{self.config.api_version}",
                    "Accept": "application/json",
                }
            )
            self.session = session
            logger.info("Agricultural API session initialized")
        except ImportError:
            logger.warning("Requests library not available, API functionality limited")
            self.session = None

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Fetch a JSON payload from the configured agricultural service."""
        if self.session is None:
            raise RuntimeError("requests is required for agricultural data access")

        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Agricultural service request failed for {url}: {exc}"
            ) from exc

    def get_crop_data(
        self, crop_type: str, region: str | None = None, year: int | None = None
    ) -> dict[str, Any]:
        """
        Get agricultural data for a specific crop.

        Args:
            crop_type: Type of crop
            region: Geographic region (optional)
            year: Year for data (optional)

        Returns:
            Crop data dictionary
        """
        crop_types = self.config.crop_types or []
        if crop_type not in crop_types:
            raise ValueError(f"Unsupported crop type: {crop_type}")

        data = self._request(
            f"/crops/{quote(crop_type, safe='')}",
            {"region": region, "year": year},
        )
        if not isinstance(data, dict):
            raise ValueError("Agricultural crop endpoint must return a JSON object")

        logger.info(f"Retrieved crop data for {crop_type}")
        return data

    def get_soil_data(
        self, location: dict[str, float], depth: float | None = None
    ) -> dict[str, Any]:
        """
        Get soil data for a specific location.

        Args:
            location: Dictionary with 'lat' and 'lon' keys
            depth: Soil depth in cm (optional)

        Returns:
            Soil data dictionary
        """
        if not {"lat", "lon"}.issubset(location):
            raise ValueError("location must include 'lat' and 'lon'")
        soil_data = self._request(
            "/soil",
            {"latitude": location["lat"], "longitude": location["lon"], "depth": depth},
        )
        if not isinstance(soil_data, dict):
            raise ValueError("Agricultural soil endpoint must return a JSON object")

        logger.info(f"Retrieved soil data for location {location}")
        return soil_data

    def get_weather_forecast(
        self, location: dict[str, float], days: int = 7
    ) -> list[dict[str, Any]]:
        """
        Get weather forecast for agricultural planning.

        Args:
            location: Dictionary with 'lat' and 'lon' keys
            days: Number of days to forecast

        Returns:
            List of weather forecasts
        """
        if days <= 0:
            raise ValueError("days must be positive")
        if not {"lat", "lon"}.issubset(location):
            raise ValueError("location must include 'lat' and 'lon'")
        payload = self._request(
            "/weather/forecast",
            {"latitude": location["lat"], "longitude": location["lon"], "days": days},
        )
        forecasts = payload.get("forecasts") if isinstance(payload, dict) else payload
        if not isinstance(forecasts, list):
            raise ValueError(
                "Agricultural weather endpoint must return a forecast list"
            )

        logger.info(f"Retrieved {days}-day weather forecast for location {location}")
        return forecasts

    def analyze_crop_yield(
        self,
        crop_type: str,
        location: dict[str, float],
        soil_data: dict[str, Any],
        weather_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze potential crop yield based on conditions.

        Args:
            crop_type: Type of crop
            location: Geographic location
            soil_data: Soil information
            weather_data: Weather forecast data

        Returns:
            Yield analysis results
        """
        # Simple yield prediction model
        base_yield = 8.5  # tons per hectare

        # Adjust for soil quality
        soil_factor = 1.0
        if soil_data.get("ph_level", 7.0) < 6.0 or soil_data.get("ph_level", 7.0) > 7.5:
            soil_factor *= 0.8

        if soil_data.get("organic_matter", 0) < 2.0:
            soil_factor *= 0.9

        # Adjust for weather conditions
        weather_factor = 1.0
        avg_temp, total_precip = self._weather_totals(weather_data)

        if avg_temp < 15 or avg_temp > 35:
            weather_factor *= 0.7
        elif 20 <= avg_temp <= 30:
            weather_factor *= 1.1

        if total_precip < 50:
            weather_factor *= 0.8
        elif total_precip > 200:
            weather_factor *= 0.9

        predicted_yield = base_yield * soil_factor * weather_factor

        analysis = {
            "crop_type": crop_type,
            "location": location,
            "predicted_yield": round(predicted_yield, 2),
            "confidence": 0.75,
            "factors": {
                "soil_quality": soil_factor,
                "weather_conditions": weather_factor,
                "base_yield": base_yield,
            },
            "recommendations": self._generate_recommendations(soil_data, weather_data),
        }

        logger.info(
            f"Yield analysis completed for {crop_type}: {predicted_yield:.2f} t/ha"
        )
        return analysis

    @staticmethod
    def _weather_totals(weather_data: list[dict[str, Any]]) -> tuple[float, float]:
        """Compute (mean daily high temperature, total precipitation).

        Args:
            weather_data: List of daily weather records

        Returns:
            Tuple of (average temperature_high, total precipitation)

        Raises:
            ValueError: If weather_data is empty or a record is missing a
                required key.
        """
        if not weather_data:
            raise ValueError("weather_data must contain at least one daily record")

        for index, record in enumerate(weather_data):
            if "temperature_high" not in record:
                raise ValueError(
                    f"weather_data record {index} is missing 'temperature_high'"
                )
            if "precipitation" not in record:
                raise ValueError(
                    f"weather_data record {index} is missing 'precipitation'"
                )

        avg_temp = sum(d["temperature_high"] for d in weather_data) / len(weather_data)
        total_precip = sum(d["precipitation"] for d in weather_data)
        return avg_temp, total_precip

    def _generate_recommendations(
        self, soil_data: dict[str, Any], weather_data: list[dict[str, Any]]
    ) -> list[str]:
        """Generate agricultural recommendations."""
        recommendations = []

        # Soil-based recommendations
        if soil_data.get("ph_level", 7.0) < 6.0:
            recommendations.append("Consider lime application to raise soil pH")

        if soil_data.get("organic_matter", 0) < 2.0:
            recommendations.append("Add organic matter to improve soil structure")

        if soil_data.get("nitrogen", 0) < 30:
            recommendations.append("Apply nitrogen fertilizer")

        # Weather-based recommendations
        avg_temp, total_precip = self._weather_totals(weather_data)

        if total_precip < 50:
            recommendations.append("Consider irrigation due to low precipitation")

        if avg_temp > 30:
            recommendations.append("Monitor for heat stress in crops")

        if not recommendations:
            recommendations.append("Conditions appear favorable for crop growth")

        return recommendations

    def get_precision_agriculture_data(
        self, field_id: str, sensor_type: str | None = None
    ) -> dict[str, Any]:
        """
        Get precision agriculture sensor data.

        Args:
            field_id: Unique field identifier
            sensor_type: Type of sensor data to retrieve

        Returns:
            Precision agriculture data
        """
        if not field_id:
            raise ValueError("field_id must not be empty")
        precision_data = self._request(
            f"/fields/{quote(field_id, safe='')}/precision",
            {"sensor_type": sensor_type},
        )
        if not isinstance(precision_data, dict):
            raise ValueError("Precision agriculture endpoint must return a JSON object")

        logger.info(f"Retrieved precision agriculture data for field {field_id}")
        return precision_data

    def optimize_irrigation(
        self, field_data: dict[str, Any], weather_forecast: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Optimize irrigation schedule based on field and weather data.

        Args:
            field_data: Field sensor data
            weather_forecast: Weather forecast data

        Returns:
            Irrigation optimization results
        """
        # Simple irrigation optimization
        current_moisture = sum(field_data["sensors"]["soil_moisture"]["values"]) / len(
            field_data["sensors"]["soil_moisture"]["values"]
        )
        forecast_precip = sum(d["precipitation"] for d in weather_forecast)

        # Determine irrigation needs
        if current_moisture < 0.3:
            irrigation_needed = True
            irrigation_amount = 25.0  # mm
        elif current_moisture < 0.35 and forecast_precip < 20:
            irrigation_needed = True
            irrigation_amount = 15.0  # mm
        else:
            irrigation_needed = False
            irrigation_amount = 0.0

        optimization = {
            "irrigation_needed": irrigation_needed,
            "recommended_amount": irrigation_amount,
            "current_moisture": current_moisture,
            "forecast_precipitation": forecast_precip,
            "schedule": self._generate_irrigation_schedule(
                irrigation_needed, irrigation_amount
            ),
            "efficiency_estimate": 0.85,
        }

        logger.info(
            f"Irrigation optimization completed: {'needed' if irrigation_needed else 'not needed'}"
        )
        return optimization

    def _generate_irrigation_schedule(
        self, irrigation_needed: bool, amount: float
    ) -> list[dict[str, Any]]:
        """Generate irrigation schedule."""
        if not irrigation_needed:
            return []

        from datetime import datetime, timedelta

        schedule = []
        base_time = datetime.now()

        # Schedule irrigation for early morning
        for i in range(3):  # 3 days
            schedule.append(
                {
                    "date": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "time": "06:00",
                    "duration_minutes": int(amount * 2),  # Rough conversion
                    "amount_mm": amount / 3,
                }
            )

        return schedule


# Convenience functions
def create_agricultural_api(
    config: AgriculturalConfig | None = None,
) -> AgriculturalAPI:
    """Create a new AgriculturalAPI instance."""
    return AgriculturalAPI(config)


def get_crop_recommendations(
    location: dict[str, float], soil_data: dict[str, Any]
) -> list[str]:
    """Get crop recommendations for a location."""
    api = AgriculturalAPI()

    weather_data = api.get_weather_forecast(location, days=7)
    recommendations = []
    for crop in api.config.crop_types or []:
        analysis = api.analyze_crop_yield(crop, location, soil_data, weather_data)
        if analysis["predicted_yield"] > 7.0:
            recommendations.append(crop)

    return recommendations
