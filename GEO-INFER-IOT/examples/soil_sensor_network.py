#!/usr/bin/env python3
"""
Soil Sensor Network Integration Example

This example demonstrates how to use GEO-INFER-IOT to integrate a network of
soil moisture sensors with H3 spatial indexing (from GEO-INFER-SPACE) and
Bayesian spatial inference (from GEO-INFER-BAYES) for real-time soil monitoring.

Key features demonstrated:
- IoT sensor data ingestion from MQTT
- H3 spatial indexing for efficient spatial queries
- Bayesian spatial interpolation for continuous soil moisture maps
- Real-time visualization and monitoring
- Integration with existing H3-OSC capabilities from GEO-INFER-SPACE

Usage:
    python soil_sensor_network.py --config config/soil_network.yaml
"""

import asyncio
import logging
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import h3

# GEO-INFER imports. geo_infer_space/geo_infer_bayes are required workspace
# dependencies of geo_infer_iot, so a plain import failing means the
# installation is broken and the example must not pretend to work.
from geo_infer_iot import IoTSystem, BayesianSpatialInference

# Standard dependencies
try:
    import paho.mqtt.client as mqtt
    import yaml

    HAS_DEPS = True
except ImportError as e:
    logging.error(f"Missing dependencies: {e}")
    HAS_DEPS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("soil_sensor_network")


class SoilSensorNetwork:
    """
    Soil sensor network integration with H3 spatial indexing and Bayesian inference.

    This class demonstrates the integration of:
    - MQTT-based soil sensor data ingestion
    - H3 spatial indexing for efficient spatial operations
    - Bayesian spatial interpolation for continuous soil moisture mapping
    - Real-time visualization and monitoring
    """

    def __init__(self, config: dict):
        self.config = config
        self.sensors = {}
        self.measurements = []
        self.spatial_index = {}
        self.current_map = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Initialize GEO-INFER components
        self.iot_system = IoTSystem(config)
        self.spatial_inference = None

        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to soil sensor topics
            topics = self.config.get("mqtt", {}).get("topics", [])
            for topic in topics:
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Parse sensor data
            data = json.loads(msg.payload.decode())
            sensor_id = data.get("sensor_id")
            timestamp = data.get("timestamp", datetime.now().isoformat())

            # Extract sensor measurements
            measurement = {
                "sensor_id": sensor_id,
                "timestamp": timestamp,
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "soil_moisture": data.get("soil_moisture"),
                "temperature": data.get("temperature"),
                "ph": data.get("ph", None),
                "conductivity": data.get("conductivity", None),
            }

            # Add H3 spatial index
            if measurement["latitude"] and measurement["longitude"]:
                h3_index = h3.latlng_to_cell(
                    measurement["latitude"],
                    measurement["longitude"],
                    self.config.get("spatial", {}).get("h3_resolution", 8),
                )
                measurement["h3_index"] = h3_index

            # Store measurement
            self.measurements.append(measurement)

            # Register sensor if new
            if sensor_id not in self.sensors:
                self._register_sensor(sensor_id, measurement)

            # Update spatial inference if enough data
            if len(self.measurements) >= 10:  # Minimum data threshold
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._update_spatial_inference(), self._loop
                    )
                else:
                    logger.warning(
                        "No event loop captured; skipping spatial inference update"
                    )

            logger.info(f"Processed measurement from sensor {sensor_id}")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _register_sensor(self, sensor_id: str, measurement: dict):
        """Register a new sensor in the network."""
        sensor_info = {
            "sensor_id": sensor_id,
            "latitude": measurement["latitude"],
            "longitude": measurement["longitude"],
            "h3_index": measurement["h3_index"],
            "first_seen": measurement["timestamp"],
            "last_seen": measurement["timestamp"],
            "measurement_count": 1,
        }

        self.sensors[sensor_id] = sensor_info

        # Register with GEO-INFER-IOT system
        self.iot_system.registry.register_sensor(sensor_info)
        logger.info(f"Registered new sensor: {sensor_id}")

    async def _update_spatial_inference(self):
        """Update Bayesian spatial inference with new measurements."""
        try:
            # Prepare data for Bayesian inference
            recent_data = self._get_recent_measurements(
                hours=self.config.get("inference", {}).get("temporal_window_hours", 1)
            )

            if len(recent_data) < 5:  # Minimum data for inference
                return

            # Setup spatial inference if not already done
            if self.spatial_inference is None:
                self.spatial_inference = BayesianSpatialInference(
                    variable="soil_moisture",
                    spatial_resolution=self.config.get("spatial", {}).get(
                        "h3_resolution", 8
                    ),
                    temporal_window="1h",
                )

            # Define priors based on domain knowledge
            priors = {
                "mean_function": self.config.get("inference", {}).get(
                    "mean_function", "constant"
                ),
                "covariance": self.config.get("inference", {}).get(
                    "covariance", "matern_52"
                ),
                "noise_variance": self.config.get("inference", {}).get(
                    "noise_variance", 0.01
                ),
                "length_scale": self.config.get("inference", {}).get(
                    "length_scale", 1000
                ),  # meters
            }
            # Perform Bayesian spatial inference (synchronous GP backend)
            posterior = self.spatial_inference.infer_spatial_distribution(
                sensor_data=recent_data.to_dict("records"),
                priors=priors,
                update_interval="15min",
            )

            # Generate updated spatial map
            self.current_map = self.spatial_inference.get_posterior_map(
                confidence_intervals=[0.8, 0.95]
            )

            logger.info("Updated spatial inference with Bayesian methods")

        except Exception as e:
            logger.error(f"Error updating spatial inference: {e}")

    def _get_recent_measurements(self, hours: int = 1) -> pd.DataFrame:
        """Get recent measurements within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = []
        for measurement in self.measurements:
            if datetime.fromisoformat(measurement["timestamp"]) > cutoff_time:
                recent.append(measurement)

        return pd.DataFrame(recent)

    def generate_visualization(self, output_file: str = "soil_network_map.html"):
        """Generate interactive visualization of the sensor network and soil moisture map."""
        try:
            # Create base map
            center_lat = np.mean([s["latitude"] for s in self.sensors.values()])
            center_lon = np.mean([s["longitude"] for s in self.sensors.values()])

            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap"
            )

            # Add sensor locations
            sensor_cluster = MarkerCluster(name="Soil Sensors")

            for sensor_id, sensor in self.sensors.items():
                # Get latest measurement for this sensor
                latest = self._get_latest_measurement(sensor_id)

                popup_text = f"""
                <b>Sensor ID:</b> {sensor_id}<br>
                <b>Location:</b> {sensor["latitude"]:.4f}, {sensor["longitude"]:.4f}<br>
                <b>H3 Index:</b> {sensor["h3_index"]}<br>
                <b>Soil Moisture:</b> {latest.get("soil_moisture", "N/A")}%<br>
                <b>Temperature:</b> {latest.get("temperature", "N/A")}°C<br>
                <b>Last Update:</b> {latest.get("timestamp", "N/A")}
                """

                # Color code by soil moisture level
                moisture = latest.get("soil_moisture", 50)
                if moisture < 30:
                    color = "red"  # Dry
                elif moisture < 60:
                    color = "orange"  # Moderate
                else:
                    color = "green"  # Moist

                folium.Marker(
                    location=[sensor["latitude"], sensor["longitude"]],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Sensor {sensor_id}",
                    icon=folium.Icon(color=color, icon="tint"),
                ).add_to(sensor_cluster)

            sensor_cluster.add_to(m)

            # Add H3 grid visualization
            self._add_h3_grid_overlay(m)

            # Add soil moisture heatmap
            self._add_soil_moisture_heatmap(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            m.save(output_file)
            logger.info(f"Visualization saved to {output_file}")

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")

    def _get_latest_measurement(self, sensor_id: str) -> dict:
        """Get the latest measurement for a specific sensor."""
        sensor_measurements = [
            m for m in self.measurements if m["sensor_id"] == sensor_id
        ]

        if sensor_measurements:
            return max(sensor_measurements, key=lambda x: x["timestamp"])
        return {}

    def _add_h3_grid_overlay(self, map_obj):
        """Add H3 grid overlay to the map."""
        try:
            # Get unique H3 cells from sensor locations
            h3_cells = set(sensor["h3_index"] for sensor in self.sensors.values())

            # Create H3 cell layer
            for h3_index in h3_cells:
                # Get cell boundary
                boundary = h3.cell_to_boundary(h3_index)

                # Calculate average soil moisture for this cell
                cell_measurements = [
                    m for m in self.measurements if m.get("h3_index") == h3_index
                ]

                if cell_measurements:
                    avg_moisture = np.mean(
                        [
                            m["soil_moisture"]
                            for m in cell_measurements
                            if m.get("soil_moisture") is not None
                        ]
                    )

                    # Color based on moisture level
                    if avg_moisture < 30:
                        color = "#ff4444"  # Red for dry
                        opacity = 0.6
                    elif avg_moisture < 60:
                        color = "#ffaa44"  # Orange for moderate
                        opacity = 0.5
                    else:
                        color = "#44ff44"  # Green for moist
                        opacity = 0.4

                    folium.Polygon(
                        locations=boundary,
                        color=color,
                        weight=2,
                        opacity=opacity,
                        fillOpacity=0.3,
                        popup=f"H3 Cell: {h3_index}<br>Avg Moisture: {avg_moisture:.1f}%",
                    ).add_to(map_obj)

        except Exception as e:
            logger.error(f"Error adding H3 grid overlay: {e}")

    def _add_soil_moisture_heatmap(self, map_obj):
        """Add soil moisture heatmap to the map."""
        try:
            # Prepare heatmap data
            heat_data = []
            for measurement in self.measurements[-100:]:  # Use recent measurements
                if (
                    measurement.get("latitude")
                    and measurement.get("longitude")
                    and measurement.get("soil_moisture")
                ):
                    heat_data.append(
                        [
                            measurement["latitude"],
                            measurement["longitude"],
                            measurement["soil_moisture"] / 100.0,  # Normalize to 0-1
                        ]
                    )

            if heat_data:
                HeatMap(
                    heat_data,
                    name="Soil Moisture Heatmap",
                    radius=20,
                    blur=15,
                    gradient={
                        0.2: "blue",
                        0.4: "cyan",
                        0.6: "lime",
                        0.8: "yellow",
                        1: "red",
                    },
                ).add_to(map_obj)

        except Exception as e:
            logger.error(f"Error adding soil moisture heatmap: {e}")

    async def start_monitoring(self):
        """Start the soil sensor network monitoring."""
        logger.info("Starting soil sensor network monitoring...")

        # Connect to MQTT broker
        mqtt_config = self.config.get("mqtt", {})
        broker_host = mqtt_config.get("broker_host", "localhost")
        broker_port = mqtt_config.get("broker_port", 1883)

        # Capture the running loop so MQTT thread callbacks can schedule
        # coroutines back onto it.
        self._loop = asyncio.get_running_loop()

        try:
            self.mqtt_client.connect(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()

            logger.info(f"Connected to MQTT broker at {broker_host}:{broker_port}")
            # Start periodic tasks
            while True:
                await asyncio.sleep(60)  # Update every minute

                # Generate periodic visualizations
                if len(self.measurements) > 0:
                    self.generate_visualization()

                # Print status
                logger.info(
                    f"Status: {len(self.sensors)} sensors, "
                    f"{len(self.measurements)} measurements"
                )

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration for the soil sensor network."""
    return {
        "mqtt": {
            "broker_host": "localhost",
            "broker_port": 1883,
            "topics": ["sensors/+/soil_moisture", "sensors/+/data"],
        },
        "spatial": {"h3_resolution": 8, "coordinate_system": "WGS84"},
        "inference": {
            "temporal_window_hours": 1,
            "mean_function": "constant",
            "covariance": "matern_52",
            "noise_variance": 0.01,
            "length_scale": 1000,
        },
        "visualization": {
            "update_interval_seconds": 60,
            "output_file": "soil_network_map.html",
        },
    }


def simulate_sensor_data():
    """Generate simulated sensor data for testing."""
    import random

    # Define a test area (e.g., around a farm)
    center_lat, center_lon = 40.7128, -74.0060  # New York area

    # Generate random sensor locations
    sensors = []
    for i in range(10):
        sensor_id = f"soil_sensor_{i:03d}"
        lat = center_lat + random.uniform(-0.01, 0.01)
        lon = center_lon + random.uniform(-0.01, 0.01)

        sensors.append({"sensor_id": sensor_id, "latitude": lat, "longitude": lon})

    logger.info(f"Generated {len(sensors)} simulated sensors")

    # You could publish this data to MQTT for testing
    return sensors


async def main():
    """Main function to run the soil sensor network example."""
    logger.info("Starting GEO-INFER-IOT Soil Sensor Network Example")

    # Generate simulated data for demonstration
    sensors = simulate_sensor_data()

    # Create simple visualization
    config = {"spatial": {"h3_resolution": 8}}
    network = SoilSensorNetwork(config)

    # Add simulated sensors
    for sensor in sensors:
        h3_index = h3.latlng_to_cell(sensor["latitude"], sensor["longitude"], 8)
        sensor["h3_index"] = h3_index
        network.sensors[sensor["sensor_id"]] = sensor

    # Generate visualization
    network.generate_visualization("demo_soil_network.html")

    logger.info("Demo completed. Check demo_soil_network.html for visualization.")


if __name__ == "__main__":
    asyncio.run(main())
