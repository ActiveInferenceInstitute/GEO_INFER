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

import h3

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
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime | None = None

    def __post_init__(self) -> None:
        if not self.h3_index:
            self.h3_index = h3.latlng_to_cell(
                self.latitude, self.longitude, self.h3_resolution
            )


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
    created_at: datetime = field(default_factory=datetime.now)


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
        """Get sensors within geographic bounds using H3 spatial indexing.

        Builds a polygon from the bounding box, maps it to the H3 cells
        covering it at ``h3_resolution``, and collects sensors registered
        in those cells plus their ring-1 neighbors (boundary tolerance for
        sensors whose cell center falls just outside the polygon). An exact
        bbox check on the candidate set keeps the result identical to a
        brute-force scan, without scanning the whole registry.

        The documented fallback is a direct bbox scan, used only when the
        H3 spatial index is empty (nothing has been indexed yet).

        Args:
            bounds: Mapping with ``lat_min``, ``lat_max``, ``lon_min``,
                ``lon_max``.
            h3_resolution: H3 resolution used to discretize the bbox.

        Returns:
            Sensors whose coordinates lie within the bounds.
        """
        lat_min = float(bounds["lat_min"])
        lat_max = float(bounds["lat_max"])
        lon_min = float(bounds["lon_min"])
        lon_max = float(bounds["lon_max"])

        def within_bounds(sensor: SensorMetadata) -> bool:
            return (
                lat_min <= sensor.latitude <= lat_max
                and lon_min <= sensor.longitude <= lon_max
            )

        if not self.h3_spatial_index:
            # Fallback: no sensors indexed yet, so the H3 grid cannot
            # contribute candidates; scan the registry directly.
            return [s for s in self.sensors.values() if within_bounds(s)]

        polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        }
        cells = h3.geo_to_cells(polygon, h3_resolution)
        candidate_cells: set[str] = set()
        for cell in cells:
            candidate_cells.update(h3.grid_disk(cell, 1))

        matched: list[SensorMetadata] = []
        seen: set[str] = set()
        for cell in candidate_cells:
            for sensor_id in self.h3_spatial_index.get(cell, ()):
                if sensor_id in seen or sensor_id not in self.sensors:
                    continue
                sensor = self.sensors[sensor_id]
                if within_bounds(sensor):
                    seen.add(sensor_id)
                    matched.append(sensor)
        return matched
