"""
FireRiskAssessor: Del Norte County fire risk assessment and monitoring.

This module provides comprehensive fire risk assessment capabilities for
Del Norte County, integrating real California fire data sources including
CAL FIRE, weather monitoring, and fuel moisture measurements.
"""

import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import h3
from shapely.geometry import shape

logger = logging.getLogger(__name__)


class FireRiskAssessor:
    """
    Fire risk assessment system for Del Norte County.

    Comprehensive fire risk analysis for Del Norte County's forested areas,
    wildland-urban interface zones, and critical infrastructure protection.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data_integrator: Any,
        spatial_processor: Any,
        output_dir: Path,
    ):
        """Initialize fire risk assessor."""
        self.config = config
        self.data_integrator = data_integrator
        self.spatial_processor = spatial_processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get fire risk configuration
        self.fire_config = config.get("analyses", {}).get("fire_risk", {})
        self.h3_resolution = config.get("spatial", {}).get("h3_resolution", 8)

        self.last_analysis_time: datetime | None = None

        logger.info("FireRiskAssessor initialized for Del Norte County")

    def run_analysis(
        self, temporal_range: tuple[str, str] | None = None
    ) -> dict[str, Any]:
        """Run comprehensive fire risk analysis."""
        logger.info("🔥 Starting fire risk analysis for Del Norte County...")

        start_time = datetime.now()
        results = {
            "analysis_type": "fire_risk",
            "location": "del_norte_county",
            "timestamp": start_time.isoformat(),
            "temporal_range": temporal_range,
            "config": self.fire_config,
        }

        try:
            # Acquire fire data
            fire_data = self._acquire_fire_data(temporal_range)
            results["data_acquisition"] = fire_data

            # Fire weather analysis
            weather_analysis = self._analyze_fire_weather(fire_data)
            results["fire_weather_analysis"] = weather_analysis

            # Historical fire analysis
            historical_analysis = self._analyze_historical_fires(fire_data)
            results["historical_fire_analysis"] = historical_analysis

            # Fuel assessment
            fuel_analysis = self._assess_fuel_conditions(fire_data)
            results["fuel_analysis"] = fuel_analysis

            # WUI risk assessment
            wui_analysis = self._assess_wui_risk(fire_data)
            results["wui_analysis"] = wui_analysis

            # Integrated risk assessment
            risk_assessment = self._generate_risk_assessment(results)
            results["risk_assessment"] = risk_assessment

            # Spatial data preparation
            spatial_data = self._prepare_spatial_data(results)
            results["spatial_data"] = spatial_data

            processing_time = datetime.now() - start_time
            results["processing_time"] = str(processing_time)
            results["status"] = "success"

            self._save_analysis_results(results)
            self.last_analysis_time = datetime.now()

            logger.info(f"✅ Fire risk analysis completed in {processing_time}")

        except Exception as e:
            logger.error(f"❌ Fire risk analysis failed: {e}")
            results["status"] = "error"
            results["error_message"] = str(e)
            results["processing_time"] = str(datetime.now() - start_time)

        return results

    def _acquire_fire_data(
        self, temporal_range: tuple[str, str] | None
    ) -> dict[str, Any]:
        """Acquire fire-related data from multiple sources."""
        bounds = self.config.get("location", {}).get("bounds", {})
        bbox = (
            bounds.get("west"),
            bounds.get("south"),
            bounds.get("east"),
            bounds.get("north"),
        )

        fire_data: dict[str, Any] = {
            "bbox": bbox,
            "temporal_range": temporal_range,
            "data_sources": {},
        }

        # Historical fire perimeters
        try:
            fire_perimeters = self.data_integrator.calfire_client.get_fire_perimeters(
                bbox=bbox, start_year=1950, include_metadata=True
            )
            fire_data["data_sources"]["fire_perimeters"] = fire_perimeters
        except Exception as e:
            logger.warning(f"Error acquiring fire perimeters: {e}")

        weather = self.data_integrator.noaa_client.get_weather_data(
            station_id=self.fire_config.get("weather_station", "KCEC")
        )
        if not weather.get("success"):
            raise RuntimeError(
                f"NOAA weather acquisition failed: {weather.get('error')}"
            )
        required_weather = ("temperature", "humidity", "wind_speed", "timestamp")
        if any(weather.get(field) is None for field in required_weather):
            raise RuntimeError(
                "NOAA weather response is missing fire-weather measurements"
            )
        temperature_c = float(weather["temperature"])
        wind_speed_mph = float(weather["wind_speed"]) * 2.236936
        weather_data = {
            "data_source": "NOAA weather observations",
            "measurements": [
                {
                    "date": weather["timestamp"],
                    "station_id": self.fire_config.get("weather_station", "KCEC"),
                    "temperature_f": temperature_c * 9 / 5 + 32,
                    "relative_humidity": float(weather["humidity"]),
                    "wind_speed_mph": wind_speed_mph,
                    "fire_weather_index": self._calculate_fwi(
                        temperature_c, float(weather["humidity"]), wind_speed_mph
                    ),
                }
            ],
        }
        fire_data["data_sources"]["fire_weather"] = weather_data

        fire_data["data_sources"]["fuel_moisture"] = self._load_fuel_moisture_data()

        return fire_data

    def _calculate_fwi(
        self, temp_c: float, humidity: float, wind_speed: float
    ) -> float:
        """Calculate simplified fire weather index."""
        # Simplified FWI calculation
        temp_f = temp_c * 9 / 5 + 32
        fwi = (temp_f - humidity) + (wind_speed * 0.5)
        return max(0, fwi)

    def _load_fuel_moisture_data(self) -> dict[str, Any]:
        """Load configured fuel-moisture observations from an empirical source."""
        source = self.fire_config.get("fuel_moisture_source")
        if not source:
            raise RuntimeError(
                "Fuel-moisture analysis requires fire_risk.fuel_moisture_source"
            )
        if callable(source):
            payload = source()
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Fuel-moisture source does not exist: {path}")
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() in {".csv", ".tsv"}:
                frame = pd.read_csv(
                    path, sep="\t" if path.suffix.lower() == ".tsv" else ","
                )
                payload = {"measurements": frame.to_dict(orient="records")}
            else:
                raise ValueError(
                    f"Unsupported fuel-moisture source format: {path.suffix}"
                )
        if isinstance(payload, list):
            payload = {"measurements": payload}
        if not isinstance(payload, dict) or not payload.get("measurements"):
            raise ValueError("Fuel-moisture source returned no measurements")
        return payload

    def _analyze_fire_weather(self, fire_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze fire weather conditions."""
        weather_data = fire_data["data_sources"].get("fire_weather", {})
        measurements = weather_data.get("measurements", [])

        if not measurements:
            return {"status": "no_data"}

        df = pd.DataFrame(measurements)
        df["date"] = pd.to_datetime(df["date"])

        # Fire weather thresholds from config
        fire_weather_config = self.fire_config.get("fire_weather", {})
        critical_temp = fire_weather_config.get(
            "critical_temperature", 80
        )  # Fahrenheit
        critical_humidity = fire_weather_config.get("critical_humidity", 15)  # percent
        critical_wind = fire_weather_config.get("critical_wind_speed", 25)  # mph

        analysis = {
            "summary_statistics": {
                "mean_temperature": df["temperature_f"].mean(),
                "mean_humidity": df["relative_humidity"].mean(),
                "mean_wind_speed": df["wind_speed_mph"].mean(),
                "mean_fwi": df["fire_weather_index"].mean(),
            },
            "critical_conditions": {
                "high_temp_days": (df["temperature_f"] > critical_temp).sum(),
                "low_humidity_days": (
                    df["relative_humidity"] < critical_humidity
                ).sum(),
                "high_wind_days": (df["wind_speed_mph"] > critical_wind).sum(),
                "extreme_fire_weather_days": (
                    (df["temperature_f"] > critical_temp)
                    & (df["relative_humidity"] < critical_humidity)
                ).sum(),
            },
            "fire_danger_distribution": {
                "low": (df["fire_weather_index"] < 20).sum(),
                "moderate": (
                    (df["fire_weather_index"] >= 20) & (df["fire_weather_index"] < 40)
                ).sum(),
                "high": (
                    (df["fire_weather_index"] >= 40) & (df["fire_weather_index"] < 60)
                ).sum(),
                "very_high": (
                    (df["fire_weather_index"] >= 60) & (df["fire_weather_index"] < 80)
                ).sum(),
                "extreme": (df["fire_weather_index"] >= 80).sum(),
            },
        }

        return analysis

    def _analyze_historical_fires(self, fire_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze historical fire patterns."""
        fire_perimeters = fire_data["data_sources"].get("fire_perimeters", {})
        features = fire_perimeters.get("features", [])
        if not features:
            return {"status": "no_data", "fire_statistics": {"total_fires": 0}}

        def property_value(properties: dict[str, Any], *names: str) -> Any:
            for name in names:
                value = properties.get(name)
                if value not in (None, ""):
                    return value
            return None

        acres = []
        years = []
        causes: dict[str, int] = {}
        months: dict[str, int] = {}
        for feature in features:
            properties = feature.get("properties") or {}
            raw_acres = property_value(properties, "GIS_ACRES", "AcresBurned", "acres")
            try:
                if raw_acres is not None:
                    acres.append(float(raw_acres))
            except (TypeError, ValueError):
                logger.warning("Ignoring non-numeric fire area: %r", raw_acres)
            raw_year = property_value(properties, "Fire_Year", "Year", "year")
            try:
                if raw_year is not None:
                    years.append(int(raw_year))
            except (TypeError, ValueError):
                pass
            cause = property_value(properties, "Cause", "cause")
            if cause:
                causes[str(cause)] = causes.get(str(cause), 0) + 1
            date_value = property_value(properties, "Date", "Started", "start_date")
            if date_value:
                try:
                    month = pd.to_datetime(date_value).strftime("%B")
                    months[month] = months.get(month, 0) + 1
                except (TypeError, ValueError):
                    pass

        cause_total = sum(causes.values())
        return {
            "data_source": "CAL FIRE perimeter observations",
            "fire_statistics": {
                "total_fires": len(features),
                "total_acres_burned": sum(acres),
                "average_fire_size_acres": sum(acres) / len(acres) if acres else None,
                "largest_fire_acres": max(acres) if acres else None,
                "fires_by_decade": {
                    f"{decade}s": sum(
                        1 for year in years if (year // 10) * 10 == decade
                    )
                    for decade in sorted({(year // 10) * 10 for year in years})
                },
            },
            "fire_causes": (
                {cause: count / cause_total for cause, count in causes.items()}
                if cause_total
                else {}
            ),
            "fires_by_month": months,
        }

    def _assess_fuel_conditions(self, fire_data: dict[str, Any]) -> dict[str, Any]:
        """Assess current fuel moisture and loading conditions."""
        fuel_data = fire_data["data_sources"].get("fuel_moisture", {})
        measurements = fuel_data.get("measurements", [])

        if not measurements:
            return {"status": "no_data"}

        df = pd.DataFrame(measurements)
        df["date"] = pd.to_datetime(df["date"])

        # Get most recent measurements
        recent_data = df[df["date"] >= (datetime.now() - timedelta(days=7))]

        fuel_assessment = {
            "current_conditions": {
                "mean_live_moisture": recent_data["live_fuel_moisture_percent"].mean(),
                "mean_dead_moisture": recent_data["dead_fuel_moisture_percent"].mean(),
                "sites_below_critical_live": (
                    recent_data["live_fuel_moisture_percent"] < 60
                ).sum(),
                "sites_below_critical_dead": (
                    recent_data["dead_fuel_moisture_percent"] < 8
                ).sum(),
            },
            "trend_analysis": {
                "moisture_trend": (
                    "Decreasing"
                    if recent_data["live_fuel_moisture_percent"].mean()
                    < df["live_fuel_moisture_percent"].mean()
                    else "Stable"
                ),
                "days_below_critical": (
                    recent_data["live_fuel_moisture_percent"] < 60
                ).sum(),
            },
            "fuel_loading": {
                "forest_types": {
                    "Douglas Fir": {"fuel_load_tons_acre": 25, "fire_risk": "High"},
                    "Mixed Conifer": {"fuel_load_tons_acre": 22, "fire_risk": "High"},
                    "Oak Woodland": {
                        "fuel_load_tons_acre": 12,
                        "fire_risk": "Moderate",
                    },
                }
            },
        }

        return fuel_assessment

    def _assess_wui_risk(self, fire_data: dict[str, Any]) -> dict[str, Any]:
        """Assess wildland-urban interface fire risk."""
        source = self.fire_config.get("wui_source")
        if not source:
            return {"status": "no_data", "message": "No WUI dataset configured"}
        if callable(source):
            payload = source()
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"WUI source does not exist: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("WUI source must return a JSON object")
        return payload

    def _generate_risk_assessment(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate integrated fire risk assessment."""
        weather_analysis = analysis_results.get("fire_weather_analysis", {})
        fuel_analysis = analysis_results.get("fuel_analysis", {})
        wui_analysis = analysis_results.get("wui_analysis", {})

        risk_components = {}

        # Weather risk component
        critical_conditions = weather_analysis.get("critical_conditions", {})
        extreme_days = critical_conditions.get("extreme_fire_weather_days", 0)
        total_days = 90  # Assuming 90-day analysis period
        weather_risk = min(extreme_days / (total_days * 0.1), 1.0)  # Normalize
        risk_components["weather_risk"] = weather_risk

        # Fuel moisture risk
        current_conditions = fuel_analysis.get("current_conditions", {})
        live_moisture = current_conditions.get("mean_live_moisture", 100)
        fuel_risk = max(0, (80 - live_moisture) / 40)  # Normalize to 0-1
        risk_components["fuel_risk"] = fuel_risk

        # WUI risk
        high_risk_structures = wui_analysis.get("structure_vulnerability", {}).get(
            "high_risk_structures", 0
        )
        total_structures = wui_analysis.get("structure_vulnerability", {}).get(
            "total_structures_in_wui", 1
        )
        wui_risk = high_risk_structures / total_structures
        risk_components["wui_risk"] = wui_risk

        # Calculate overall risk
        weights = {"weather_risk": 0.3, "fuel_risk": 0.4, "wui_risk": 0.3}
        overall_risk = sum(
            risk_components[comp] * weights[comp] for comp in weights.keys()
        )

        risk_assessment = {
            "overall_risk_score": overall_risk,
            "risk_components": risk_components,
            "risk_level": (
                "High"
                if overall_risk > 0.7
                else "Moderate"
                if overall_risk > 0.4
                else "Low"
            ),
            "priority_areas": [],
            "recommendations": [],
        }

        # Generate recommendations
        if weather_risk > 0.5:
            risk_assessment["recommendations"].append(
                "Enhanced fire weather monitoring"
            )
        if fuel_risk > 0.5:
            risk_assessment["recommendations"].append("Fuel reduction treatments")
        if wui_risk > 0.5:
            risk_assessment["recommendations"].append(
                "WUI defensible space enforcement"
            )

        return risk_assessment

    def _prepare_spatial_data(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Prepare spatial data for integration."""
        spatial_data = {
            "h3_resolution": self.h3_resolution,
            "h3_cells": {},
            "data_type": "fire_risk",
        }

        risk_assessment = analysis_results.get("risk_assessment", {})
        overall_risk = risk_assessment.get("overall_risk_score", 0)

        perimeter_data = (
            analysis_results.get("data_acquisition", {})
            .get("data_sources", {})
            .get("fire_perimeters", {})
        )
        features = perimeter_data.get("features", [])
        areas = []
        for feature in features:
            properties = feature.get("properties") or {}
            try:
                areas.append(
                    float(properties.get("GIS_ACRES", properties.get("AcresBurned", 0)))
                )
            except (TypeError, ValueError):
                areas.append(0.0)
        maximum_area = max(areas, default=0.0)
        for index, feature in enumerate(features):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            centroid = shape(geometry).centroid
            h3_cell = h3.latlng_to_cell(centroid.y, centroid.x, self.h3_resolution)
            area_score = areas[index] / maximum_area if maximum_area else 0.0
            spatial_data["h3_cells"][h3_cell] = {
                "fire_risk_score": max(overall_risk, area_score),
                "data_quality": "empirical",
                "last_updated": datetime.now().isoformat(),
            }

        return spatial_data

    def _save_analysis_results(self, results: dict[str, Any]) -> None:
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"fire_risk_analysis_{timestamp}.json"

        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Fire risk analysis results saved to: {results_file}")

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring system status."""
        return {
            "monitor_type": "fire_risk",
            "location": "del_norte_county",
            "last_analysis": (
                self.last_analysis_time.isoformat() if self.last_analysis_time else None
            ),
            "configuration": self.fire_config,
            "monitoring_active": True,
        }
