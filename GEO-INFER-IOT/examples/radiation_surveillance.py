#!/usr/bin/env python3
"""
Radiation Surveillance Network Example

Thin orchestration over ``geo_infer_iot.RadiationMonitoringSystem``:
- Ingests a batch of radiation measurements with quality control and
  anomaly detection driven by an explicit empirical baseline
- Runs Bayesian spatial inference over the ingested measurements
- Reports system metrics

The example runs fully offline: measurements are simulated locally and no
public radiation API is contacted.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, UTC
from typing import Any

from geo_infer_iot import RadiationMonitoringSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("radiation_surveillance")

# Empirical background radiation baseline (uSv/h). Anomaly detection in
# RadiationMonitoringSystem requires these keys to be configured explicitly.
BACKGROUND_RADIATION = 0.10
NOISE_LEVEL = 0.02

CENTER_LAT, CENTER_LON = 37.7749, -122.4194  # San Francisco Bay Area


def build_config() -> dict[str, Any]:
    """Configuration for the radiation surveillance system."""
    return {
        "spatial": {"h3_resolution": 6},
        "sensor_networks": {
            "bay_area_network": {
                "name": "Bay Area Radiation Network",
                "protocol": "mqtt",
                "coverage": "regional",
            },
        },
        "radiation_baseline": {
            "background_radiation": BACKGROUND_RADIATION,
            "noise_level": NOISE_LEVEL,
        },
        "quality_control": {
            "sensor_validation": {
                "min_radiation": 0.0,
                "max_radiation": 100.0,
            },
        },
        "anomaly_detection": {
            "statistical": {
                "threshold_mild": 2.0,
            },
        },
        "bayesian_inference": {
            "covariance": {
                "function": "matern_52",
                "length_scale": 50_000.0,
                "noise_variance": 0.01,
            },
            "confidence_levels": [0.68, 0.95],
        },
    }


def simulate_measurements(num_sensors: int = 12) -> list[dict[str, Any]]:
    """Deterministic simulated gamma dose-rate readings around the bay area.

    Most sensors sit near the background level; a handful are elevated so
    the anomaly detector has something to flag.
    """
    now = datetime.now(UTC)
    measurements: list[dict[str, Any]] = []
    for i in range(num_sensors):
        lat = CENTER_LAT + 0.02 * math.cos(i)
        lon = CENTER_LON + 0.02 * math.sin(i)
        # Every fourth sensor reads 5 sigma above background
        value = BACKGROUND_RADIATION + (
            5.0 * NOISE_LEVEL if i % 4 == 0 else 0.5 * NOISE_LEVEL
        )
        measurements.append(
            {
                "sensor_id": f"RAD_{i:03d}",
                "variable": "gamma_radiation",
                "value": round(value, 4),
                "unit": "uSv/h",
                "latitude": lat,
                "longitude": lon,
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
            }
        )
    return measurements


async def main() -> None:
    print("=" * 60)
    print("GEO-INFER-IOT Radiation Surveillance demo")
    print("=" * 60)

    system = RadiationMonitoringSystem(build_config())
    system.setup_spatial_inference("gamma_radiation")

    measurements = simulate_measurements()
    results = await system.process_measurements(measurements)

    print(f"Processed: {results['processed']}  Failed: {results['failed']}")
    print(f"Unique H3 cells: {len(results['spatial_cells'])}")
    print(f"Anomalies detected: {len(results['anomalies'])}")
    for anomaly in results["anomalies"]:
        print(
            f"  - {anomaly['sensor_id']} at "
            f"({anomaly['location'][0]:.4f}, {anomaly['location'][1]:.4f}): "
            f"{anomaly['value']} uSv/h"
        )

    inference = await system.perform_spatial_inference("gamma_radiation")
    if inference:
        print(
            "Spatial inference produced predictions over "
            f"{len(inference.get('grid_coordinates', []))} grid points"
        )
    else:
        logger.warning("Spatial inference returned no results")

    metrics = system.get_system_metrics()
    print(
        f"Metrics: processed={metrics['measurements_processed']} "
        f"anomalies={metrics['anomalies_detected']} "
        f"error_rate={metrics['error_rate']:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
