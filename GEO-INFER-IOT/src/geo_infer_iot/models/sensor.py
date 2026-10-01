"""
Sensor Data Models

This module defines comprehensive data models for IoT sensors, sensor networks,
and related metadata using Pydantic for validation and type safety.
"""

import logging
import math
from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import h3

# Optional imports for enhanced functionality
try:
    from geo_infer_space.osc_geo.utils.spatial_operations import (
        CoordinateTransform,  # noqa: F401
    )  # noqa: F401

    HAS_SPATIAL_OPS = True
except ImportError:
    HAS_SPATIAL_OPS = False

logger = logging.getLogger(__name__)


class Location(BaseModel):
    """Geographic location model."""

    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude in decimal degrees"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude in decimal degrees"
    )
    h3_index: str | None = Field(None, description="H3 hexagonal index")
    h3_resolution: int = Field(8, ge=0, le=15, description="H3 resolution level")
    elevation_meters: float | None = Field(
        None, description="Elevation above sea level"
    )
    coordinate_system: str = Field("WGS84", description="Coordinate reference system")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Auto-generate H3 index if coordinates provided but not h3_index
        if (
            self.latitude is not None
            and self.longitude is not None
            and not self.h3_index
        ):
            self.h3_index = h3.latlng_to_cell(
                self.latitude, self.longitude, self.h3_resolution
            )

    @field_validator("h3_index")
    def validate_h3_index(cls, v: str | None) -> str | None:
        """Validate H3 index format."""
        if v and not h3.is_valid_cell(v):
            raise ValueError(f"Invalid H3 index: {v}")
        return v


class SensorCapabilities(BaseModel):
    """Sensor measurement capabilities."""

    measured_variables: list[str] = Field(
        ..., description="Variables this sensor can measure"
    )
    measurement_range: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Measurement ranges for each variable"
    )
    accuracy: dict[str, float] = Field(
        default_factory=dict, description="Accuracy specifications"
    )
    precision: dict[str, float] = Field(
        default_factory=dict, description="Precision specifications"
    )
    sampling_rate_hz: float | None = Field(None, description="Sampling frequency in Hz")
    power_consumption_watts: float | None = Field(None, description="Power consumption")
    battery_life_hours: float | None = Field(None, description="Expected battery life")


class SensorCalibration(BaseModel):
    """Sensor calibration information."""

    last_calibration: datetime | None = Field(
        None, description="Last calibration timestamp"
    )
    calibration_method: str | None = Field(None, description="Calibration method used")
    calibration_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Calibration parameters and coefficients"
    )
    next_calibration_due: datetime | None = Field(
        None, description="Next calibration due date"
    )
    calibration_certificate: str | None = Field(
        None, description="Calibration certificate ID"
    )


class Sensor(BaseModel):
    """Complete sensor data model."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    name: str | None = Field(None, description="Human-readable sensor name")
    description: str | None = Field(None, description="Sensor description")

    # Network and type information
    network_id: str = Field(..., description="ID of the sensor network")
    sensor_type: str = Field(
        ..., description="Type of sensor (e.g., temperature, humidity)"
    )
    manufacturer: str | None = Field(None, description="Sensor manufacturer")
    model: str | None = Field(None, description="Sensor model")

    # Location information
    location: Location = Field(..., description="Geographic location")

    # Capabilities and specifications
    capabilities: SensorCapabilities = Field(
        default_factory=lambda: SensorCapabilities(
            measured_variables=[],
            measurement_range={},
            accuracy={},
            precision={},
            sampling_rate_hz=None,
            power_consumption_watts=None,
            battery_life_hours=None,
        ),
        description="Sensor measurement capabilities",
    )

    # Status and operational information
    status: str = Field(
        "active", description="Sensor status (active, inactive, maintenance, error)"
    )
    operational_since: datetime | None = Field(
        None, description="When sensor became operational"
    )
    last_communication: datetime | None = Field(
        None, description="Last communication timestamp"
    )

    # Calibration information
    calibration: SensorCalibration = Field(
        default_factory=lambda: SensorCalibration(
            last_calibration=None,
            calibration_method=None,
            calibration_parameters={},
            next_calibration_due=None,
            calibration_certificate=None,
        ),
        description="Sensor calibration information",
    )

    # Metadata and additional properties
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    tags: list[str] = Field(
        default_factory=list, description="Sensor tags for categorization"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now, description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Record last update timestamp"
    )

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate sensor status."""
        valid_statuses = [
            "active",
            "inactive",
            "maintenance",
            "error",
            "decommissioned",
        ]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {valid_statuses}")
        return v

    def update_location(self, latitude: float, longitude: float, **kwargs: Any) -> None:
        """Update sensor location and related fields."""
        self.location.latitude = latitude
        self.location.longitude = longitude

        # Update H3 index
        if kwargs.get("h3_resolution"):
            self.location.h3_resolution = kwargs["h3_resolution"]
        self.location.h3_index = h3.latlng_to_cell(
            latitude, longitude, self.location.h3_resolution
        )

        self.updated_at = datetime.now()

    def add_capability(self, variable: str, **kwargs: Any) -> None:
        """Add a measurement capability to the sensor."""
        if variable not in self.capabilities.measured_variables:
            self.capabilities.measured_variables.append(variable)

        # Update measurement range if provided
        if "min_value" in kwargs or "max_value" in kwargs:
            if variable not in self.capabilities.measurement_range:
                self.capabilities.measurement_range[variable] = {}
            if "min_value" in kwargs:
                self.capabilities.measurement_range[variable]["min"] = kwargs[
                    "min_value"
                ]
            if "max_value" in kwargs:
                self.capabilities.measurement_range[variable]["max"] = kwargs[
                    "max_value"
                ]

        self.updated_at = datetime.now()

    def update_calibration(self, calibration_data: dict[str, Any]) -> None:
        """Update sensor calibration information."""
        if "last_calibration" in calibration_data:
            self.calibration.last_calibration = calibration_data["last_calibration"]
        if "calibration_method" in calibration_data:
            self.calibration.calibration_method = calibration_data["calibration_method"]
        if "calibration_parameters" in calibration_data:
            self.calibration.calibration_parameters.update(
                calibration_data["calibration_parameters"]
            )
        if "next_calibration_due" in calibration_data:
            self.calibration.next_calibration_due = calibration_data[
                "next_calibration_due"
            ]

        self.updated_at = datetime.now()

    def get_health_score(self) -> float:
        """Calculate overall sensor health score."""
        score = 1.0

        # Factor in calibration status
        if self.calibration.last_calibration:
            days_since_calibration = (
                datetime.now() - self.calibration.last_calibration
            ).days
            if days_since_calibration > 365:  # Over a year
                score *= 0.7
            elif days_since_calibration > 180:  # Over 6 months
                score *= 0.85

        # Factor in communication status
        if self.last_communication:
            minutes_since_communication = (
                datetime.now() - self.last_communication
            ).total_seconds() / 60
            if minutes_since_communication > 60:  # Over an hour
                score *= 0.8
            elif minutes_since_communication > 15:  # Over 15 minutes
                score *= 0.9

        # Factor in operational status
        if self.status != "active":
            score *= 0.5

        return max(0.0, min(1.0, score))


class SensorNetwork(BaseModel):
    """Sensor network data model."""

    network_id: str = Field(..., description="Unique network identifier")
    name: str = Field(..., description="Human-readable network name")
    description: str | None = Field(None, description="Network description")

    # Network topology and configuration
    protocol: str = Field(..., description="Communication protocol (MQTT, CoAP, etc.)")
    topology: str = Field(
        "mesh", description="Network topology (mesh, star, hierarchical)"
    )

    # Spatial coverage
    spatial_bounds: dict[str, float] = Field(
        ..., description="Geographic bounds: lat_min, lat_max, lon_min, lon_max"
    )
    h3_resolution: int = Field(8, description="Default H3 resolution for the network")

    # Sensor types and capabilities
    sensor_types: list[str] = Field(..., description="Types of sensors in this network")
    expected_sensor_count: int | None = Field(
        None, description="Expected number of sensors"
    )

    # Operational information
    status: str = Field("active", description="Network status")
    operational_since: datetime | None = Field(
        None, description="When network became operational"
    )
    coordinator_contact: str | None = Field(
        None, description="Network coordinator contact info"
    )

    # Configuration and metadata
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Network configuration"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Network tags")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("protocol")
    def validate_protocol(cls, v: str) -> str:
        """Validate communication protocol."""
        valid_protocols = [
            "MQTT",
            "CoAP",
            "LoRaWAN",
            "HTTP",
            "WebSocket",
            "Bluetooth",
            "Zigbee",
        ]
        if v.upper() not in valid_protocols:
            raise ValueError(
                f"Invalid protocol '{v}'. Must be one of: {valid_protocols}"
            )
        return v.upper()

    @field_validator("topology")
    def validate_topology(cls, v: str) -> str:
        """Validate network topology."""
        valid_topologies = ["mesh", "star", "hierarchical", "bus", "ring"]
        if v.lower() not in valid_topologies:
            raise ValueError(
                f"Invalid topology '{v}'. Must be one of: {valid_topologies}"
            )
        return v.lower()

    @field_validator("spatial_bounds")
    def validate_spatial_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate spatial bounds."""
        required_keys = ["lat_min", "lat_max", "lon_min", "lon_max"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required spatial bound: {key}")

        # Validate coordinate ranges
        if not (-90 <= v["lat_min"] <= v["lat_max"] <= 90):
            raise ValueError("Invalid latitude bounds")
        if not (-180 <= v["lon_min"] <= v["lon_max"] <= 180):
            raise ValueError("Invalid longitude bounds")

        return v

    def get_coverage_area(self) -> float:
        """Calculate coverage area in square kilometers.

        Uses a spherical approximation: latitude spans a constant
        ~111.32 km per degree, while the longitude span is scaled by the
        cosine of the mean latitude to account for meridian convergence.
        """
        lat_range = self.spatial_bounds["lat_max"] - self.spatial_bounds["lat_min"]
        lon_range = self.spatial_bounds["lon_max"] - self.spatial_bounds["lon_min"]
        mean_lat = math.radians(
            (self.spatial_bounds["lat_max"] + self.spatial_bounds["lat_min"]) / 2
        )

        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(mean_lat)
        return lat_range * km_per_deg_lat * lon_range * km_per_deg_lon

    def is_location_covered(self, latitude: float, longitude: float) -> bool:
        """Check if a location is within the network's spatial bounds."""
        return (
            self.spatial_bounds["lat_min"] <= latitude <= self.spatial_bounds["lat_max"]
            and self.spatial_bounds["lon_min"]
            <= longitude
            <= self.spatial_bounds["lon_max"]
        )

    def get_h3_cells(self) -> list[str]:
        """Get H3 cells covering the network's spatial bounds."""
        bounds = self.spatial_bounds
        polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [bounds["lon_min"], bounds["lat_min"]],
                    [bounds["lon_max"], bounds["lat_min"]],
                    [bounds["lon_max"], bounds["lat_max"]],
                    [bounds["lon_min"], bounds["lat_max"]],
                    [bounds["lon_min"], bounds["lat_min"]],
                ]
            ],
        }
        return list(h3.geo_to_cells(polygon, self.h3_resolution))


class SensorDeployment(BaseModel):
    """Sensor deployment and installation information."""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    sensor_id: str = Field(..., description="Associated sensor ID")

    # Deployment details
    deployment_date: datetime = Field(..., description="When sensor was deployed")
    deployed_by: str | None = Field(None, description="Who deployed the sensor")
    deployment_method: str = Field("manual", description="Deployment method")

    # Installation location details
    installation_height_meters: float | None = Field(
        None, description="Installation height"
    )
    mounting_type: str | None = Field(None, description="How sensor is mounted")
    environmental_conditions: dict[str, Any] = Field(
        default_factory=dict, description="Environmental conditions at deployment site"
    )

    # Power and connectivity
    power_source: str = Field("battery", description="Power source type")
    connectivity_method: str = Field("wireless", description="Connectivity method")
    signal_strength: float | None = Field(
        None, description="Signal strength at deployment"
    )

    # Documentation
    deployment_notes: str | None = Field(
        None, description="Deployment notes and observations"
    )
    site_photos: list[str] = Field(default_factory=list, description="Site photo URLs")
    documentation_urls: list[str] = Field(
        default_factory=list, description="Documentation URLs"
    )

    # Status and maintenance
    deployment_status: str = Field("active", description="Deployment status")
    last_maintenance: datetime | None = Field(
        None, description="Last maintenance visit"
    )
    next_maintenance_due: datetime | None = Field(
        None, description="Next maintenance due"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("deployment_method")
    def validate_deployment_method(cls, v: str) -> str:
        """Validate deployment method."""
        valid_methods = ["manual", "automated", "aerial", "vehicle", "stationary"]
        if v not in valid_methods:
            raise ValueError(
                f"Invalid deployment method '{v}'. Must be one of: {valid_methods}"
            )
        return v

    @field_validator("power_source")
    def validate_power_source(cls, v: str) -> str:
        """Validate power source."""
        valid_sources = ["battery", "solar", "mains", "wind", "thermal"]
        if v not in valid_sources:
            raise ValueError(
                f"Invalid power source '{v}'. Must be one of: {valid_sources}"
            )
        return v
