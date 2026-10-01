#!/usr/bin/env python3
"""
GEO-INFER-IOT Example: Smart Sensor Network Management

Thin orchestration over the real GEO-INFER-IOT public API:
- IoTSystem wires registry + ingestion + quality control
- SensorRegistry registers networks and sensors with H3 indexing
- IoTDataIngestion ingests measurements with spatial indexing
- QualityController validates measurements before ingestion

The example runs fully offline: measurements are simulated locally and no
broker or external service is contacted.
"""

import logging
from datetime import datetime, timedelta, UTC
from typing import Any

from geo_infer_iot import IoTSystem, QualityController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("smart_sensor_network")

# Small deployment around a farm field
CENTER_LAT, CENTER_LON = 40.7128, -74.0060


def build_network_config() -> dict[str, Any]:
    """Configuration for the demo network."""
    return {
        "spatial": {"default_resolution": 8},
        "sensor_networks": {},
    }


def register_deployment(system: IoTSystem) -> list[dict[str, Any]]:
    """Register one network and three sensors through the IoT system."""
    result = system.register_network(
        network_id="SMART_CITY_001",
        name="Downtown Environmental Monitoring",
        protocol="mqtt",
        spatial_bounds={
            "lat_min": CENTER_LAT - 0.05,
            "lat_max": CENTER_LAT + 0.05,
            "lon_min": CENTER_LON - 0.05,
            "lon_max": CENTER_LON + 0.05,
        },
        sensor_types=["temperature", "humidity"],
    )
    logger.info("Network registration: %s", result)

    sensors: list[dict[str, Any]] = []
    for i in range(3):
        sensor_data = {
            "sensor_id": f"SENSOR_{i:03d}",
            "network_id": "SMART_CITY_001",
            "sensor_type": "temperature" if i % 2 == 0 else "humidity",
            "latitude": CENTER_LAT + 0.01 * i,
            "longitude": CENTER_LON + 0.01 * i,
            "status": "active",
        }
        registration = system.register_sensor(sensor_data)
        logger.info("Sensor registration: %s", registration)
        sensors.append(sensor_data)

    return sensors


def simulate_measurements(
    sensors: list[dict[str, Any]], count: int = 6
) -> list[dict[str, Any]]:
    """Build deterministic simulated readings for the registered sensors."""
    now = datetime.now(UTC)
    measurements: list[dict[str, Any]] = []
    for i in range(count):
        sensor = sensors[i % len(sensors)]
        measurements.append(
            {
                "sensor_id": sensor["sensor_id"],
                "variable": sensor["sensor_type"],
                "value": 18.0 + 0.5 * i,
                "unit": "celsius"
                if sensor["sensor_type"] == "temperature"
                else "percent",
                "latitude": sensor["latitude"],
                "longitude": sensor["longitude"],
                "timestamp": (now - timedelta(minutes=5 * i)).isoformat(),
            }
        )
    return measurements


def main() -> None:
    print("=" * 60)
    print("GEO-INFER-IOT Smart Sensor Network demo")
    print("=" * 60)

    config = build_network_config()
    system = IoTSystem(config)

    init_result = system.initialize()
    print(f"Initialization: success={init_result['success']}")

    sensors = register_deployment(system)

    # Validate readings through the quality controller before ingestion
    quality = QualityController(config)
    measurements = simulate_measurements(sensors)
    accepted = []
    for reading in measurements:
        check = quality.validate_measurement(reading)
        if check.passed:
            accepted.append(reading)
        else:
            logger.warning(
                "Rejected reading %s: %s", reading["sensor_id"], check.issues
            )

    # Ingest the accepted readings through the real ingestion engine
    async def ingest_all() -> int:
        ingested = 0
        for reading in accepted:
            if await system.ingestion.ingest_measurement(reading):
                ingested += 1
        return ingested

    import asyncio

    ingested = asyncio.run(ingest_all())
    print(f"Ingested {ingested}/{len(measurements)} measurements")

    stats = system.ingestion.get_measurement_statistics()
    print("Ingestion statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    status = system.get_system_status()
    print(
        f"System status: {status['status']} ({status['sensors']} sensors, "
        f"{status['networks']} networks)"
    )

    exported = system.export_system_state("smart_sensor_network_state.json")
    print(f"State export: success={exported['success']} path={exported['export_path']}")

    print("=" * 60)
    print(
        f"Done. Registry holds {len(system.registry.sensors)} sensors in "
        f"{len(system.registry.networks)} network(s)."
    )


if __name__ == "__main__":
    main()
