"""
Sensor Registry Module

This module manages the registration and metadata of IoT sensor networks and devices,
integrating with H3 spatial indexing for efficient spatial queries.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import math


from geo_infer_iot.models.timestamps import utc_now
from geo_infer_time.core.timestamps import normalize_timestamp
from geo_infer_iot.models.spatial import sensor_cell

logger = logging.getLogger(__name__)


@dataclass
class SensorMetadata:
    """Metadata for an individual sensor."""

    sensor_id: str
    network_id: str
    sensor_type: str
    latitude: float
    longitude: float
    h3_index: str = ""
    h3_resolution: int = 8
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=utc_now)
    last_seen: datetime | None = None

    def __post_init__(self) -> None:
        self.registered_at = normalize_timestamp(self.registered_at)
        if self.last_seen is not None:
            self.last_seen = normalize_timestamp(self.last_seen)
        expected = sensor_cell(self.latitude, self.longitude, self.h3_resolution)
        if self.h3_index and self.h3_index != expected:
            raise ValueError("h3_index must match sensor coordinates and resolution")
        self.h3_index = expected


@dataclass
class SensorNetworkRecord:
    """Registry record for a sensor network with spatial bounds.

    Renamed from ``SensorNetwork`` to avoid clashing with the validated
    ``geo_infer_iot.models.sensor.SensorNetwork`` pydantic model.
    """

    network_id: str
    name: str
    protocol: str
    spatial_bounds: dict[str, Any]
    sensor_types: list[str]
    sensor_count: int = 0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.created_at = normalize_timestamp(self.created_at)


class SensorRegistry:
    """
    Registry for managing IoT sensor networks and individual sensors.

    Provides capabilities for:
    - Registering sensor networks and individual sensors
    - Spatial queries using H3 indexing
    - Sensor metadata management
    - Network topology tracking
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.networks: dict[str, SensorNetworkRecord] = {}
        self.sensors: dict[str, SensorMetadata] = {}
        self.h3_spatial_index: dict[str, set[str]] = {}  # h3_index -> sensor_ids

        logger.info("Sensor Registry initialized")

    def register_network(self, **kwargs: Any) -> SensorNetworkRecord:
        """Register a new sensor network."""
        network_id = kwargs.get("network_id") or str(uuid.uuid4())

        network = SensorNetworkRecord(
            network_id=network_id,
            name=kwargs["name"],
            protocol=kwargs["protocol"],
            spatial_bounds=kwargs["spatial_bounds"],
            sensor_types=kwargs["sensor_types"],
        )

        self.networks[network_id] = network
        logger.info(f"Registered sensor network: {network.name}")
        return network

    def register_sensor(self, sensor_info: dict) -> SensorMetadata:
        """Register an individual sensor."""
        sensor = SensorMetadata(**sensor_info)
        if sensor.sensor_id in self.sensors:
            raise ValueError(f"Sensor '{sensor.sensor_id}' is already registered")

        self.sensors[sensor.sensor_id] = sensor

        # Add to spatial index
        if sensor.h3_index not in self.h3_spatial_index:
            self.h3_spatial_index[sensor.h3_index] = set()
        self.h3_spatial_index[sensor.h3_index].add(sensor.sensor_id)

        # Update network sensor count
        if sensor.network_id in self.networks:
            self.networks[sensor.network_id].sensor_count += 1

        logger.info(f"Registered sensor: {sensor.sensor_id}")
        return sensor

    def get_sensors_in_h3_cell(self, h3_index: str) -> list[SensorMetadata]:
        """Get all sensors in a specific H3 cell."""
        sensor_ids = self.h3_spatial_index.get(h3_index, set())
        return [self.sensors[sid] for sid in sensor_ids if sid in self.sensors]

    def get_sensors_by_type(self, sensor_type: str) -> list[SensorMetadata]:
        """Get all sensors of a specific type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]

    def get_sensors_in_area(
        self, bounds: dict, h3_resolution: int = 8
    ) -> list[SensorMetadata]:
        """Return exact inclusive WGS84 bounds matches in registration order.

        ``lon_min > lon_max`` denotes a box crossing the antimeridian. The
        retained H3 resolution hint does not change coordinate membership.
        An exact scan avoids centroid-cover gaps, tiny-box misses and mixed
        resolution false negatives while bounding work by the sensor count.
        """
        sensor_cell(0.0, 0.0, h3_resolution)
        lat_min, lat_max = float(bounds["lat_min"]), float(bounds["lat_max"])
        lon_min, lon_max = float(bounds["lon_min"]), float(bounds["lon_max"])
        if not all(
            math.isfinite(value) for value in (lat_min, lat_max, lon_min, lon_max)
        ):
            raise ValueError("Bounds must contain finite coordinates")
        if not -90 <= lat_min <= lat_max <= 90 or not (
            -180 <= lon_min <= 180 and -180 <= lon_max <= 180
        ):
            raise ValueError("Bounds must contain valid WGS84 coordinates")

        def inside(sensor: SensorMetadata) -> bool:
            longitude_matches = (
                (lon_min <= sensor.longitude <= lon_max)
                if lon_min <= lon_max
                else (sensor.longitude >= lon_min or sensor.longitude <= lon_max)
            )
            return lat_min <= sensor.latitude <= lat_max and longitude_matches

        return [sensor for sensor in self.sensors.values() if inside(sensor)]
