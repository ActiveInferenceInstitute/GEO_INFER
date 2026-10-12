#!/usr/bin/env python3
"""GEO-INFER-IOT module orchestrator.

Runs one documented end-to-end IoT operation on synthetic telemetry (no
network): register a synthetic sensor network in the ``SensorRegistry``,
run a telemetry stream through ``QualityController`` batch validation, and
fuse the individually-passed temperature readings to an uninstrumented
target location with ``SpatialDataFusion``. All work goes through the real
``geo_infer_iot`` public API.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_iot import QualityController, SensorRegistry, SpatialDataFusion

    rng = np.random.default_rng(42)

    registry = SensorRegistry()
    registry.register_network(
        network_id="net_synthetic_01",
        name="Synthetic Valley Field Mesh",
        protocol="mqtt",
        spatial_bounds={
            "min_lat": 44.88,
            "max_lat": 44.96,
            "min_lon": -123.14,
            "max_lon": -123.04,
        },
        sensor_types=["temperature", "soil_moisture"],
    )

    station_locations = [
        ("temp_01", 44.90, -123.12),
        ("temp_02", 44.93, -123.10),
        ("temp_03", 44.91, -123.07),
        ("temp_04", 44.95, -123.06),
        ("soil_01", 44.89, -123.09),
    ]
    for sensor_id, latitude, longitude in station_locations:
        registry.register_sensor(
            {
                "sensor_id": sensor_id,
                "network_id": "net_synthetic_01",
                "sensor_type": (
                    "temperature" if sensor_id.startswith("temp") else "soil_moisture"
                ),
                "latitude": latitude,
                "longitude": longitude,
                "h3_resolution": 8,
            }
        )

    # Synthetic telemetry stream: 15-minute samples over 2 hours from each
    # temperature station, plus one stuck-sensor spike that QC must flag.
    now = datetime.now()
    measurements: list[dict[str, Any]] = []
    for step in range(8):
        timestamp = now - timedelta(minutes=15 * (8 - step))
        for sensor_id, latitude, longitude in station_locations:
            if not sensor_id.startswith("temp"):
                continue
            value = (
                14.0
                + 0.4 * step
                + float(rng.normal(0.0, 0.3))
                + 0.01 * (latitude - 44.9)
            )
            if sensor_id == "temp_03" and step == 7:
                value = 999.0  # hardware spike the quality gate should catch
            measurements.append(
                {
                    "sensor_id": sensor_id,
                    "variable": "temperature",
                    "value": round(value, 3),
                    "unit": "celsius",
                    "timestamp": timestamp.isoformat(),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

    controller = QualityController()
    quality = controller.validate_batch(measurements)
    failed_sensors = sorted(
        {entry["sensor_id"] for entry in quality["results"] if not entry["passed"]}
    )

    # ``validate_batch`` verdicts are index-aligned with the input list, so
    # drop only the individually rejected readings and fuse the rest.
    valid_readings = [
        measurement
        for measurement, verdict in zip(measurements, quality["results"])
        if verdict["passed"]
    ]
    fusion = SpatialDataFusion().fuse_sensor_data(
        valid_readings,
        target_variable="temperature",
        target_location=(44.92, -123.08),
    )

    temperature_sensors = registry.get_sensors_by_type("temperature")
    return {
        "operation": "sensor_stream_qc_and_spatial_fusion",
        "networks_registered": len(registry.networks),
        "sensors_registered": len(registry.sensors),
        "temperature_stations": len(temperature_sensors),
        "distinct_h3_cells": len(
            {sensor.h3_index for sensor in registry.sensors.values()}
        ),
        "measurements_ingested": len(measurements),
        "qc_pass_rate": round(float(quality["pass_rate"]), 4),
        "qc_failed_measurements": quality["failed_measurements"],
        "qc_flagged_sensors": failed_sensors,
        "readings_after_qc": len(valid_readings),
        "fused_temperature_celsius": round(
            float(fusion.get("fused_value", fusion.get("value", 0.0))), 4
        )
        if "error" not in fusion
        else None,
        "fusion_uncertainty": round(float(fusion.get("uncertainty", 0.0)), 4)
        if "error" not in fusion
        else None,
        "fusion_measurement_count": fusion.get("measurement_count"),
        "fusion_error": fusion.get("error"),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("IOT", _operation))
