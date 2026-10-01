"""
Situational awareness module.

Provides common operating picture, sensor integration,
threat assessment, and real-time dashboard capabilities.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classifications."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"


class DataSource(Enum):
    """Types of data sources."""

    SENSOR = "sensor"
    FIELD_REPORT = "field_report"
    SATELLITE = "satellite"
    SOCIAL_MEDIA = "social_media"
    WEATHER = "weather"
    MODEL = "model"


@dataclass
class SensoryInput:
    """Represents incoming sensor data."""

    source_id: str
    source_type: DataSource
    timestamp: datetime
    location: dict[str, float] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8


@dataclass
class LayerConfig:
    """Configuration for a COP layer."""

    layer_id: str
    name: str
    source: str
    visible: bool = True
    refresh_rate_seconds: int = 60
    symbology: dict[str, Any] = field(default_factory=dict)


class SituationalAwareness:
    """
    Maintain situational awareness through common operating picture,
    sensor fusion, and real-time threat assessment.
    """

    def __init__(
        self,
        data_sources: list[str] | None = None,
        fusion_algorithms: list[str] | None = None,
        update_interval: int = 60,
    ):
        """
        Initialize situational awareness system.

        Args:
            data_sources: Data sources to integrate
            fusion_algorithms: Descriptive label for the fusion approach used by
                downstream reporting. Not executed algorithmically; ``fuse_data``
                always performs a confidence-weighted average.
            update_interval: Update interval in seconds
        """
        self.data_sources = data_sources or ["sensors", "field_reports", "satellite"]
        self.fusion_algorithms = fusion_algorithms or ["kalman", "bayesian"]
        self.update_interval = update_interval
        self._sensor_data: dict[str, SensoryInput] = {}
        self._layers: dict[str, LayerConfig] = {}
        self._current_threat_level = ThreatLevel.LOW
        logger.info(
            f"Initialized SituationalAwareness with {len(self.data_sources)} sources"
        )

    def integrate_sensors(
        self,
        sensor_network: dict[str, Any],
        data_types: list[str],
        sampling_rate: str = "continuous",
    ) -> dict[str, Any]:
        """
        Integrate sensor network data.

        Args:
            sensor_network: Sensor network configuration
            data_types: Types of data to collect
            sampling_rate: Data sampling rate

        Returns:
            Integration status and data summary
        """
        sensors = sensor_network.get("sensors", [])

        sensors_out: list[dict[str, Any]] = []
        integration: dict[str, Any] = {
            "sensor_count": len(sensors),
            "data_types": data_types,
            "sampling_rate": sampling_rate,
            "integration_status": "active",
            "sensors": sensors_out,
            "timestamp": datetime.now().isoformat(),
        }

        for sensor in sensors:
            sensor_id = sensor.get("id", f"sensor_{len(self._sensor_data)}")

            # Create sensor input record
            sensor_input = SensoryInput(
                source_id=sensor_id,
                source_type=DataSource.SENSOR,
                timestamp=datetime.now(),
                location=sensor.get("location"),
                data=sensor.get("readings", {}),
                confidence=sensor.get("confidence", 0.8),
            )
            self._sensor_data[sensor_id] = sensor_input

            sensors_out.append(
                {
                    "sensor_id": sensor_id,
                    "type": sensor.get("type", "unknown"),
                    "location": sensor.get("location"),
                    "status": "connected",
                    "last_reading": sensor_input.data,
                }
            )

        logger.info(f"Integrated {len(sensors)} sensors")
        return integration

    def build_cop(
        self,
        layers: list[dict[str, Any]],
        extent: dict[str, Any],
        symbology: dict[str, Any],
        refresh_rate: int = 30,
    ) -> dict[str, Any]:
        """
        Build common operating picture.

        Args:
            layers: Map layers to include
            extent: Map extent
            symbology: Symbology definitions
            refresh_rate: Refresh rate in seconds

        Returns:
            COP configuration
        """
        layers_out: list[dict[str, Any]] = []
        cop: dict[str, Any] = {
            "cop_id": f"cop_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "extent": extent,
            "refresh_rate_seconds": refresh_rate,
            "layers": layers_out,
            "status": "active",
        }

        for layer_data in layers:
            layer = LayerConfig(
                layer_id=layer_data.get("id", f"layer_{len(self._layers)}"),
                name=layer_data.get("name", "Layer"),
                source=layer_data.get("source", ""),
                visible=layer_data.get("visible", True),
                refresh_rate_seconds=layer_data.get("refresh", refresh_rate),
                symbology=symbology.get(str(layer_data.get("type")), {})
                if layer_data.get("type") is not None
                else {},
            )
            self._layers[layer.layer_id] = layer

            layers_out.append(
                {
                    "layer_id": layer.layer_id,
                    "name": layer.name,
                    "type": layer_data.get("type"),
                    "visible": layer.visible,
                    "z_order": layer_data.get("z_order", 0),
                }
            )

        logger.info(f"Built COP with {len(layers)} layers")
        return cop

    def assess_threat(
        self,
        hazard: dict[str, Any],
        affected_area: dict[str, Any],
        assets_at_risk: list[dict[str, Any]],
        projection_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Assess current threat level.

        Args:
            hazard: Hazard information
            affected_area: Area affected
            assets_at_risk: Assets in affected area
            projection_hours: Hours to project forward

        Returns:
            Threat assessment
        """
        # Calculate threat score based on hazard characteristics
        hazard_intensity = hazard.get("intensity", 0.5)
        hazard_speed = hazard.get("speed", 0)
        population_at_risk = sum(a.get("population", 0) for a in assets_at_risk)

        # Simple threat scoring
        threat_score = (
            hazard_intensity * 0.4
            + min(hazard_speed / 50, 1.0) * 0.2
            + min(population_at_risk / 100000, 1.0) * 0.4
        )

        # Determine threat level
        if threat_score >= 0.8:
            level = ThreatLevel.CATASTROPHIC
        elif threat_score >= 0.6:
            level = ThreatLevel.EXTREME
        elif threat_score >= 0.4:
            level = ThreatLevel.HIGH
        elif threat_score >= 0.2:
            level = ThreatLevel.MODERATE
        else:
            level = ThreatLevel.LOW

        self._current_threat_level = level

        assessment = {
            "assessment_id": f"threat_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "hazard": {
                "type": hazard.get("type", "unknown"),
                "intensity": hazard_intensity,
                "speed_kmh": hazard_speed,
                "direction": hazard.get("direction", "unknown"),
            },
            "threat_level": level.value,
            "threat_score": round(threat_score, 2),
            "affected_area": {
                "area_sq_km": affected_area.get("area_sq_km", 0),
                "geometry": affected_area.get("geometry"),
            },
            "assets_at_risk": {
                "count": len(assets_at_risk),
                "population": population_at_risk,
                "critical_infrastructure": [
                    a for a in assets_at_risk if a.get("critical", False)
                ],
            },
            "projection": {
                "hours": projection_hours,
                "expected_expansion": "increasing" if hazard_speed > 10 else "stable",
                "confidence": 0.7,
            },
            "recommendations": self._generate_recommendations(level, hazard),
        }

        logger.info(f"Threat assessment: {level.value} (score: {threat_score:.2f})")
        return assessment

    def _generate_recommendations(
        self, level: ThreatLevel, hazard: dict[str, Any]
    ) -> list[str]:
        """Generate action recommendations based on threat level."""
        recommendations = {
            ThreatLevel.LOW: ["Continue normal monitoring", "Review emergency plans"],
            ThreatLevel.MODERATE: [
                "Increase monitoring frequency",
                "Alert emergency personnel",
                "Prepare evacuation resources",
            ],
            ThreatLevel.HIGH: [
                "Activate EOC",
                "Issue public warnings",
                "Pre-position resources",
                "Consider evacuation warnings",
            ],
            ThreatLevel.EXTREME: [
                "Full EOC activation",
                "Issue evacuation orders",
                "Deploy all available resources",
                "Request mutual aid",
            ],
            ThreatLevel.CATASTROPHIC: [
                "Declare state of emergency",
                "Mass evacuation",
                "Request federal assistance",
                "Activate all mutual aid agreements",
            ],
        }
        return recommendations.get(level, [])

    def fuse_data(
        self,
        sources: list[dict[str, Any]],
        fusion_method: str = "weighted_average",
        confidence_weighting: bool = True,
    ) -> dict[str, Any]:
        """
        Fuse data from multiple sources.

        Args:
            sources: Data sources to fuse
            fusion_method: Fusion algorithm to use. Only
                ``"weighted_average"`` is supported; any other value raises
                ``ValueError``.
            confidence_weighting: Weight by source confidence

        Returns:
            Fused data product
        """
        if not sources:
            raise ValueError("fuse_data requires at least one source; got none")
        if fusion_method != "weighted_average":
            raise ValueError(
                f"Unsupported fusion_method {fusion_method!r}: only "
                "'weighted_average' is implemented"
            )

        fused_data_out: dict[str, Any] = {}
        fused: dict[str, Any] = {
            "fusion_method": fusion_method,
            "source_count": len(sources),
            "timestamp": datetime.now().isoformat(),
            "fused_data": fused_data_out,
            "confidence": 0,
        }

        # Collect all data fields
        all_fields = set()
        for source in sources:
            all_fields.update(source.get("data", {}).keys())

        # Fuse each field
        total_confidence = 0.0
        fused_field_count = 0
        for field_name in all_fields:
            values: list[float] = []
            weights: list[float] = []

            for source in sources:
                if field_name in source.get("data", {}):
                    value = source["data"][field_name]
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                        # Source confidence clamped to [0, 1]; equal weights
                        # when confidence weighting is disabled.
                        confidence = (
                            min(1.0, max(0.0, float(source.get("confidence", 0.5))))
                            if confidence_weighting
                            else 1.0
                        )
                        weights.append(confidence)

            if values and weights:
                # Weighted average
                total_weight = sum(weights)
                if total_weight > 0:
                    fused_value = (
                        sum(v * w for v, w in zip(values, weights)) / total_weight
                    )
                    fused_data_out[field_name] = round(float(fused_value), 2)
                    # Confidence of a fused field = mean confidence of the
                    # sources that contributed to it.
                    total_confidence += sum(weights) / len(weights)
                    fused_field_count += 1

        fused["confidence"] = (
            round(min(1.0, total_confidence / fused_field_count), 2)
            if fused_field_count
            else 0
        )

        logger.debug(f"Fused data from {len(sources)} sources")
        return fused

    def generate_dashboard(
        self,
        widgets: list[dict[str, Any]],
        layout: str = "standard",
        update_frequency: int = 30,
    ) -> dict[str, Any]:
        """
        Generate real-time dashboard.

        Args:
            widgets: Dashboard widgets
            layout: Dashboard layout
            update_frequency: Update frequency in seconds

        Returns:
            Dashboard configuration
        """
        widgets_out: list[dict[str, Any]] = []
        dashboard: dict[str, Any] = {
            "dashboard_id": f"dash_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "layout": layout,
            "update_frequency_seconds": update_frequency,
            "created_at": datetime.now().isoformat(),
            "widgets": widgets_out,
            "status": "active",
        }

        for widget in widgets:
            widget_config: dict[str, Any] = {
                "widget_id": widget.get("id", f"widget_{len(widgets_out)}"),
                "type": widget.get("type", "text"),
                "title": widget.get("title", "Widget"),
                "position": widget.get("position", {"row": 0, "col": 0}),
                "size": widget.get("size", {"width": 1, "height": 1}),
                "data_source": widget.get("data_source"),
                "refresh_rate": widget.get("refresh", update_frequency),
            }
            widgets_out.append(widget_config)

        logger.info(f"Generated dashboard with {len(widgets)} widgets")
        return dashboard

    def get_current_threat_level(self) -> str:
        """Get current threat level."""
        return self._current_threat_level.value
