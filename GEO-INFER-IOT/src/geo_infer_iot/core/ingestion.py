"""
IoT Data Ingestion Module

This module handles the ingestion and processing of IoT sensor data streams,
integrating with GEO-INFER-SPACE for H3 spatial indexing and GEO-INFER-BAYES
for Bayesian spatial inference.

Key features:
- Multi-protocol IoT data ingestion (MQTT, CoAP, LoRaWAN, HTTP)
- Real-time H3 spatial indexing
- H3 neighbourhood context from GEO-INFER-SPACE
- Bayesian spatial inference for converting point measurements to surfaces
- Quality control and data validation
"""

import asyncio
import logging
import json
from datetime import datetime, UTC
from typing import Any, cast
from dataclasses import dataclass, field
from collections import defaultdict
import time

# Core dependencies
import aiomqtt as asyncio_mqtt
import h3
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

# GEO-INFER-SPACE and GEO-INFER-BAYES are required workspace dependencies for
# this module. Import their current public paths directly so API drift is an
# explicit import failure instead of a silent fallback.
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_space.utils.h3_utils import (
    get_h3_neighbors,
    h3_resolution_stats,
)
from geo_infer_bayes import GaussianProcess, SpatialCovariance  # type: ignore[import-untyped]

from geo_infer_iot.models.measurement import normalize_timestamp


class SpatialOperations:
    """IoT-facing adapter over the current GEO-INFER-SPACE indexing API."""

    def __init__(self) -> None:
        self.indexer = SpatialIndexingInterface(backend="h3")

    def latlon_to_meters(
        self, latitude: float, longitude: float
    ) -> tuple[float, float]:
        """Convert latitude/longitude to a deterministic local metric approximation."""
        return longitude * 111_320.0, latitude * 110_540.0


logger = logging.getLogger(__name__)


@dataclass
class SensorMeasurement:
    """Data class for sensor measurements with spatial context."""

    sensor_id: str
    timestamp: datetime
    variable: str
    value: float
    unit: str
    latitude: float
    longitude: float
    h3_index: str | None = None
    h3_resolution: int = 8
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the timestamp to aware UTC and compute the H3 index.

        Raises:
            ValueError: If ``timestamp`` is timezone-naive.
        """
        self.timestamp = normalize_timestamp(self.timestamp)
        if self.h3_index is None and self.latitude and self.longitude:
            self.h3_index = h3.latlng_to_cell(
                self.latitude, self.longitude, self.h3_resolution
            )


@dataclass
class SpatialInferenceConfig:
    """Configuration for Bayesian spatial inference."""

    variable: str
    h3_resolution: int = 8
    temporal_window_hours: float = 1.0
    spatial_range_km: float = 10.0
    covariance_function: str = "matern_52"
    mean_function: str = "constant"
    length_scale: float = 1000.0  # meters
    noise_variance: float = 0.01
    update_interval_minutes: int = 15
    confidence_levels: list[float] = field(default_factory=lambda: [0.68, 0.95])


class IoTDataIngestion:
    """
    IoT data ingestion engine with spatial indexing and Bayesian inference.

    This class handles real-time ingestion of IoT sensor data, performs
    H3 spatial indexing, and integrates with Bayesian inference for
    converting point measurements to continuous spatial distributions.
    """

    def __init__(self, registry: Any, config: dict[str, Any] | None = None):
        self.registry = registry
        self.config = config or {}

        # Data storage
        self.measurements: list[SensorMeasurement] = []
        self.spatial_index: dict[str, list[SensorMeasurement]] = defaultdict(list)

        # Spatial inference configuration
        self.inference_configs: dict[str, SpatialInferenceConfig] = {}
        self.spatial_models: dict[str, Any] = {}

        # GEO-INFER-SPACE indexing and local metric projection
        self.spatial_ops = SpatialOperations()

        # Protocol handlers
        self.protocol_handlers: dict[str, Any] = {}
        self._setup_protocol_handlers()

        # Processing state
        self.is_processing = False
        self.processing_tasks: list[asyncio.Task[Any]] = []
        # Observability: last ingest duration (ms) and last error, for
        # performance monitoring and caller-side error discrimination.
        self.last_ingest_latency_ms: float = 0.0
        self.last_ingest_error: str | None = None

        logger.info("IoT Data Ingestion engine initialized")

    def _setup_protocol_handlers(self) -> None:
        """Setup handlers for different IoT protocols."""
        self.protocol_handlers["mqtt"] = self._handle_mqtt
        self.protocol_handlers["async_mqtt"] = self._handle_async_mqtt

    async def ingest_measurement(self, measurement: dict | SensorMeasurement) -> bool:
        """
        Ingest a single sensor measurement.

        Args:
            measurement: Either a dictionary with measurement data or SensorMeasurement object

        Returns:
            bool: True if measurement was successfully ingested and processed
        """
        start = time.perf_counter()
        try:
            # Convert to SensorMeasurement if needed
            if isinstance(measurement, dict):
                measurement = self._dict_to_measurement(measurement)

            # Validate measurement
            if not self._validate_measurement(measurement):
                logger.warning(
                    f"Invalid measurement from sensor {measurement.sensor_id}"
                )
                self.last_ingest_error = (
                    f"invalid measurement from sensor {measurement.sensor_id}"
                )
                return False

            # Add H3 spatial index
            self._add_spatial_index(measurement)

            # Store measurement
            self.measurements.append(measurement)
            if measurement.h3_index is not None:
                self.spatial_index[measurement.h3_index].append(measurement)

            if (
                hasattr(self.registry, "sensors")
                and measurement.sensor_id in self.registry.sensors
            ):
                self.registry.sensors[
                    measurement.sensor_id
                ].last_seen = measurement.timestamp

            # Trigger spatial inference update if configured
            if measurement.variable in self.inference_configs:
                await self._update_spatial_inference(measurement.variable)

            logger.debug(
                f"Ingested measurement: {measurement.sensor_id} -> {measurement.value}"
            )
            self.last_ingest_error = None
            return True

        except (ValueError, KeyError, TypeError) as e:
            # Expected input errors from parsing/validation: report as a
            # failed ingest, not a crash.
            self.last_ingest_error = f"{type(e).__name__}: {e}"
            logger.error(f"Error ingesting measurement: {e}")
            return False
        finally:
            self.last_ingest_latency_ms = (time.perf_counter() - start) * 1000.0

    def _dict_to_measurement(self, data: dict) -> SensorMeasurement:
        """Convert dictionary to SensorMeasurement object.

        The timestamp must be timezone-aware (ISO-8601 with an offset or an
        aware datetime) and is normalized to UTC; a missing timestamp defaults
        to the current UTC time.

        Raises:
            ValueError: If the timestamp is timezone-naive or not ISO-8601.
        """
        raw_timestamp = data.get("timestamp")
        return SensorMeasurement(
            sensor_id=data["sensor_id"],
            timestamp=(
                datetime.now(UTC)
                if raw_timestamp is None
                else normalize_timestamp(raw_timestamp)
            ),
            variable=data["variable"],
            value=float(data["value"]),
            unit=data.get("unit", ""),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            h3_resolution=data.get("h3_resolution", 8),
            quality_flags=data.get("quality_flags", []),
            metadata=data.get("metadata", {}),
        )

    def _validate_measurement(self, measurement: SensorMeasurement) -> bool:
        """Validate sensor measurement data."""
        # Basic validation
        if not measurement.sensor_id or not measurement.variable:
            return False

        # Coordinate validation
        if not (-90 <= measurement.latitude <= 90):
            return False
        if not (-180 <= measurement.longitude <= 180):
            return False

        # Value validation (basic range check)
        if not isinstance(measurement.value, (int, float)) or np.isnan(
            measurement.value
        ):
            return False

        return True

    def _add_spatial_index(self, measurement: SensorMeasurement) -> None:
        """Add the H3 index plus neighbourhood and resolution context."""
        # Basic H3 indexing
        if not measurement.h3_index:
            measurement.h3_index = h3.latlng_to_cell(
                measurement.latitude, measurement.longitude, measurement.h3_resolution
            )

        # Spatial context from GEO-INFER-SPACE
        if measurement.h3_index is not None:
            try:
                # Get neighbor cells for spatial context
                neighbors = get_h3_neighbors(measurement.h3_index, ring_size=1)
                measurement.metadata["h3_neighbors"] = neighbors

                # Calculate H3 resolution statistics
                stats = h3_resolution_stats(measurement.h3_resolution)
                measurement.metadata["h3_stats"] = stats

            except Exception as e:
                logger.warning(f"Error in enhanced spatial indexing: {e}")

    def setup_spatial_inference(
        self, config: dict[str, Any] | SpatialInferenceConfig | Any
    ) -> None:
        """
        Setup Bayesian spatial inference for a specific variable.

        Args:
            config: Configuration for spatial inference
        """
        if isinstance(config, dict):
            config_obj = SpatialInferenceConfig(**config)
        elif not isinstance(config, SpatialInferenceConfig):
            config_obj = SpatialInferenceConfig(
                variable=config.variable,
                h3_resolution=getattr(config, "h3_resolution", 8),
                temporal_window_hours=getattr(config, "temporal_window_hours", 1.0),
                spatial_range_km=getattr(config, "spatial_range_km", 10.0),
                covariance_function=getattr(config, "covariance_function", "matern_52"),
                mean_function=getattr(config, "mean_function", "constant"),
                length_scale=getattr(config, "length_scale", 1000.0),
                noise_variance=getattr(config, "noise_variance", 0.01),
                update_interval_minutes=getattr(config, "update_interval_minutes", 15),
                confidence_levels=getattr(config, "confidence_levels", [0.68, 0.95]),
            )
        else:
            config_obj = config

        self.inference_configs[config_obj.variable] = config_obj

        # Initialize Gaussian Process model
        try:
            # Setup covariance function
            if config_obj.covariance_function == "matern_52":
                cov_func = SpatialCovariance.matern_52(
                    length_scale=config_obj.length_scale, variance=1.0
                )
            elif config_obj.covariance_function == "rbf":
                cov_func = SpatialCovariance.rbf(
                    length_scale=config_obj.length_scale, variance=1.0
                )
            else:
                cov_func = SpatialCovariance.matern_52(
                    length_scale=config_obj.length_scale, variance=1.0
                )

            # Initialize Gaussian Process
            gp_model = GaussianProcess(
                covariance_function=cov_func,
                mean_function=config_obj.mean_function,
                noise_variance=config_obj.noise_variance,
            )

            self.spatial_models[config_obj.variable] = gp_model

            logger.info(f"Setup spatial inference for variable: {config_obj.variable}")

        except Exception as e:
            logger.error(f"Error setting up spatial inference: {e}")

    async def _update_spatial_inference(self, variable: str) -> None:
        """Update Bayesian spatial inference for a variable."""
        if variable not in self.spatial_models:
            return

        try:
            cfg = self.inference_configs[variable]
            model = self.spatial_models[variable]

            # Get recent measurements for this variable
            recent_data = self._get_recent_measurements(
                variable=variable, hours=cfg.temporal_window_hours
            )

            if len(recent_data) < 3:  # Need minimum data for inference
                return

            # Prepare spatial coordinates (convert to meters)
            coords_list = []
            values_list = []
            h3_indices_list: list[str] = []

            for measurement in recent_data:
                # Convert lat/lon to local coordinate system
                x, y = self.spatial_ops.latlon_to_meters(
                    measurement.latitude, measurement.longitude
                )

                coords_list.append([x, y])
                values_list.append(measurement.value)
                if measurement.h3_index is not None:
                    h3_indices_list.append(measurement.h3_index)

            coords = np.array(coords_list)
            values = np.array(values_list)

            # Perform Bayesian inference
            # geo_infer_bayes.GaussianProcess exposes synchronous fit/predict;
            # call them directly so inference actually runs.
            model.fit(coords, values)

            # Generate predictions on H3 grid
            prediction_grid = self._generate_h3_prediction_grid(
                h3_indices_list, cfg.h3_resolution
            )

            predictions = model.predict(prediction_grid, return_std=True)

            # Store results
            self._store_spatial_predictions(variable, predictions, prediction_grid, cfg)

            logger.info(
                f"Updated spatial inference for {variable}: {len(recent_data)} measurements"
            )

        except Exception as e:
            logger.error(f"Error updating spatial inference for {variable}: {e}")

    def _get_recent_measurements(
        self, variable: str, hours: float
    ) -> list[SensorMeasurement]:
        """Get recent measurements for a specific variable.

        Timestamps are normalized to UTC at ingestion time, so naive
        datetimes are not expected here; measurements constructed directly
        with naive timestamps are still compared as UTC for consistency.
        """
        cutoff_time = datetime.now(UTC) - pd.Timedelta(hours=hours)

        recent = [
            m
            for m in self.measurements
            if (
                m.variable == variable
                and (
                    m.timestamp
                    if m.timestamp.tzinfo is not None
                    else m.timestamp.replace(tzinfo=UTC)
                )
                > cutoff_time
            )
        ]

        return recent

    def _generate_h3_prediction_grid(
        self, measurement_h3_indices: list[str], resolution: int
    ) -> np.ndarray:
        """Generate H3 grid for spatial predictions."""
        # Get unique H3 cells and their neighbors for prediction
        h3_cells = set(measurement_h3_indices)

        # Add neighbor cells for smoother interpolation
        for h3_index in list(h3_cells):
            h3_cells.update(get_h3_neighbors(h3_index, ring_size=2))

        # Convert H3 cells to coordinates
        grid_coords = []
        for h3_index in h3_cells:
            lat, lon = h3.cell_to_latlng(h3_index)

            x, y = self.spatial_ops.latlon_to_meters(lat, lon)

            grid_coords.append([x, y])

        return np.array(grid_coords)

    def _store_spatial_predictions(
        self,
        variable: str,
        predictions: dict,
        grid_coords: np.ndarray,
        config: SpatialInferenceConfig,
    ) -> None:
        """Store spatial prediction results."""
        # This would typically store to a database or cache
        # For now, we'll store in memory
        if not hasattr(self, "spatial_predictions"):
            self.spatial_predictions = {}

        self.spatial_predictions[variable] = {
            "predictions": predictions,
            "grid_coords": grid_coords,
            "config": config,
            "timestamp": datetime.now(UTC),
        }

    async def start_stream_processing(self) -> None:
        """Start real-time stream processing."""
        if self.is_processing:
            logger.warning("Stream processing already running")
            return

        self.is_processing = True
        logger.info("Starting IoT stream processing")

        # Start protocol handlers
        for handler in self.protocol_handlers.values():
            task = asyncio.create_task(handler())
            self.processing_tasks.append(task)

        # Start periodic spatial inference updates
        task = asyncio.create_task(self._periodic_spatial_updates())
        self.processing_tasks.append(task)

    async def stop_stream_processing(self) -> None:
        """Stop stream processing."""
        self.is_processing = False

        # Cancel all tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()

        logger.info("Stopped IoT stream processing")

    async def _periodic_spatial_updates(self) -> None:
        """Periodically update spatial inference models."""
        while self.is_processing:
            try:
                for variable in self.inference_configs:
                    await self._update_spatial_inference(variable)

                # Wait for next update cycle
                update_interval = (
                    min(
                        config.update_interval_minutes
                        for config in self.inference_configs.values()
                    )
                    if self.inference_configs
                    else 15
                )

                await asyncio.sleep(update_interval * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic spatial updates: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _handle_mqtt(self) -> None:
        """Handle synchronous MQTT protocol using paho-mqtt client.

        Connects to the configured MQTT broker, subscribes to sensor topics,
        and ingests incoming measurements into the spatial index.
        Runs in a thread-executor to avoid blocking the event loop.
        """
        broker_config = self.config.get("mqtt", {})
        host = broker_config.get("host", "localhost")
        port = broker_config.get("port", 1883)
        username = broker_config.get("username")
        password = broker_config.get("password")
        topics = broker_config.get("topics", ["geo_infer/sensors/#"])
        keepalive = broker_config.get("keepalive", 60)

        client = mqtt.Client(client_id=f"geo_infer_iot_{id(self)}")

        if username:
            client.username_pw_set(username, password)

        # Capture the ingestor's event loop up front: paho callbacks run in a
        # worker thread where asyncio.get_event_loop() would not return this loop.
        loop = asyncio.get_running_loop()

        def on_connect(
            mqttc: mqtt.Client, userdata: object, flags: dict, rc: int
        ) -> None:
            if rc == 0:
                logger.info("MQTT connected to %s:%d", host, port)
                for topic in topics:
                    mqttc.subscribe(topic)
                    logger.debug("MQTT subscribed to topic: %s", topic)
            else:
                logger.error("MQTT connection failed with code %d", rc)

        def on_message(
            mqttc: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage
        ) -> None:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                asyncio.run_coroutine_threadsafe(
                    self.ingest_measurement(payload),
                    loop,
                )
            except json.JSONDecodeError:
                logger.warning("MQTT received non-JSON payload on topic %s", msg.topic)
            except Exception as exc:
                logger.error("MQTT on_message error: %s", exc)

        def on_disconnect(mqttc: mqtt.Client, userdata: object, rc: int) -> None:
            if rc != 0:
                logger.warning("MQTT unexpected disconnect (rc=%d); will reconnect", rc)

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        try:
            await loop.run_in_executor(
                None,
                lambda: client.connect(host, port, keepalive),
            )
            logger.info("MQTT client starting loop")
            while self.is_processing:
                await loop.run_in_executor(None, lambda: client.loop(timeout=1.0))
        except ConnectionRefusedError:
            logger.error("MQTT broker %s:%d refused connection", host, port)
        except Exception as exc:
            logger.error("MQTT handler error: %s", exc)
        finally:
            client.disconnect()
            logger.info("MQTT client disconnected")

    async def _handle_async_mqtt(self) -> None:
        """Handle MQTT protocol using asyncio-mqtt (coroutine-native).

        Connects with aiomqtt for non-blocking message consumption.
        """

        broker_config = self.config.get("mqtt", {})
        host = broker_config.get("host", "localhost")
        port = broker_config.get("port", 1883)
        username = broker_config.get("username")
        password = broker_config.get("password")
        topics = broker_config.get("topics", ["geo_infer/sensors/#"])

        credentials: dict = {}
        if username:
            credentials = {"username": username, "password": password or ""}

        try:
            async with asyncio_mqtt.Client(
                hostname=host,
                port=port,
                **credentials,
            ) as client:
                logger.info("Async MQTT connected to %s:%d", host, port)
                for topic in topics:
                    await client.subscribe(topic)
                    logger.debug("Async MQTT subscribed to topic: %s", topic)

                async with client.messages as messages:  # type: ignore[attr-defined]
                    async for msg in messages:
                        if not self.is_processing:
                            break
                        try:
                            payload = json.loads(msg.payload.decode("utf-8"))
                            await self.ingest_measurement(payload)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Async MQTT non-JSON payload on %s", msg.topic
                            )
                        except Exception as exc:
                            logger.error("Async MQTT message error: %s", exc)
        except asyncio_mqtt.MqttError as exc:
            logger.error("Async MQTT connection error: %s", exc)
        except Exception as exc:
            logger.error("Async MQTT handler error: %s", exc)

    def get_spatial_distribution(
        self, variable: str, confidence_level: float = 0.95
    ) -> dict | None:
        """
        Get current spatial distribution for a variable.

        Args:
            variable: Variable name
            confidence_level: Confidence level for uncertainty bounds

        Returns:
            Dictionary with spatial distribution data or None if not available
        """
        if (
            not hasattr(self, "spatial_predictions")
            or variable not in self.spatial_predictions
        ):
            return None

        prediction_data = self.spatial_predictions[variable]

        # Format results for API consumption
        res_timestamp = prediction_data["timestamp"]
        ts_str = (
            res_timestamp.isoformat()
            if hasattr(res_timestamp, "isoformat")
            else str(res_timestamp)
        )
        cfg = prediction_data["config"]
        h3_res = getattr(cfg, "h3_resolution", 8)
        grid_coords_raw = prediction_data["grid_coords"]
        if hasattr(grid_coords_raw, "tolist"):
            grid_coords_list = grid_coords_raw.tolist()
        elif isinstance(grid_coords_raw, (list, tuple)):
            grid_coords_list = list(grid_coords_raw)
        else:
            grid_coords_list = []
        result = {
            "variable": variable,
            "timestamp": ts_str,
            "h3_resolution": h3_res,
            "confidence_level": confidence_level,
            "predictions": prediction_data["predictions"],
            "grid_coordinates": grid_coords_list,
        }

        return result

    def get_measurement_statistics(self) -> dict:
        """Get statistics about ingested measurements."""
        if not self.measurements:
            return {}

        # Basic statistics
        total_measurements = len(self.measurements)
        unique_sensors = len(set(m.sensor_id for m in self.measurements))
        unique_variables = len(set(m.variable for m in self.measurements))
        unique_h3_cells = len(set(m.h3_index for m in self.measurements if m.h3_index))

        # Time range
        timestamps = [m.timestamp for m in self.measurements]
        min_time = min(timestamps)
        max_time = max(timestamps)

        return {
            "total_measurements": total_measurements,
            "unique_sensors": unique_sensors,
            "unique_variables": unique_variables,
            "unique_h3_cells": unique_h3_cells,
            "time_range": {"start": min_time.isoformat(), "end": max_time.isoformat()},
            "spatial_inference_enabled": len(self.inference_configs) > 0,
            "variables_with_inference": list(self.inference_configs.keys()),
        }


class RadiationMonitoringSystem:
    """
    Specialized IoT system for radiation monitoring with enhanced logging and testing.

    This class provides a simplified interface for radiation monitoring applications,
    integrating IoT data ingestion, spatial analysis, and Bayesian inference with
    comprehensive logging and quality assurance.
    """

    def __init__(self, config: dict[str, Any], logger: Any | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        from .registry import SensorRegistry

        self.registry = SensorRegistry(config.get("sensor_networks", {}))
        self.ingestion = IoTDataIngestion(self.registry, config)

        # Performance tracking
        self.start_time = time.time()
        self.metrics: dict[str, Any] = {
            "measurements_processed": 0,
            "spatial_inferences": 0,
            "anomalies_detected": 0,
            "errors_encountered": 0,
        }

        # Quality control
        self.quality_thresholds = config.get("quality_control", {})

        self.logger.info(
            "RadiationMonitoringSystem initialized",
            extra={
                "config_keys": list(config.keys()),
                "sensor_networks": list(config.get("sensor_networks", {}).keys()),
            },
        )

    async def process_measurements(self, measurements: list[dict]) -> dict:
        """Process a batch of measurements with full logging."""
        self.logger.info(
            "Starting measurement processing",
            extra={
                "measurement_count": len(measurements),
                "operation": "batch_processing",
            },
        )

        start_time = time.time()
        results: dict[str, Any] = {
            "processed": 0,
            "failed": 0,
            "spatial_cells": set(),
            "anomalies": [],
            "quality_issues": [],
        }

        for measurement in measurements:
            try:
                # Convert to SensorMeasurement object
                sensor_measurement = self.ingestion._dict_to_measurement(measurement)

                # Quality control
                quality_result = self._quality_control(sensor_measurement)
                if not quality_result["passed"]:
                    quality_issues = cast(list[Any], results["quality_issues"])
                    quality_issues.append(
                        {
                            "sensor_id": sensor_measurement.sensor_id,
                            "issues": quality_result["issues"],
                        }
                    )

                # Anomaly detection
                if self._is_anomaly(sensor_measurement):
                    anomalies_list = cast(list[Any], results["anomalies"])
                    anomalies_list.append(
                        {
                            "sensor_id": sensor_measurement.sensor_id,
                            "location": [
                                sensor_measurement.latitude,
                                sensor_measurement.longitude,
                            ],
                            "value": sensor_measurement.value,
                            "h3_index": sensor_measurement.h3_index,
                        }
                    )
                    self.metrics["anomalies_detected"] += 1

                # Ingest measurement
                success = await self.ingestion.ingest_measurement(sensor_measurement)
                if success:
                    results["processed"] += 1
                    spatial_cells_set = cast(set, results["spatial_cells"])
                    spatial_cells_set.add(sensor_measurement.h3_index)
                    self.metrics["measurements_processed"] += 1
                else:
                    results["failed"] += 1
                    self.metrics["errors_encountered"] += 1

            except Exception as e:
                self.logger.error(
                    "Error processing measurement",
                    extra={
                        "error": str(e),
                        "measurement_id": measurement.get("sensor_id", "unknown"),
                        "operation": "measurement_processing",
                    },
                )
                results["failed"] += 1
                self.metrics["errors_encountered"] += 1

        processing_time = time.time() - start_time
        spatial_cells_set = cast(set, results["spatial_cells"])
        results["spatial_cells"] = list(spatial_cells_set)
        results["processing_time"] = processing_time

        self.logger.info(
            "Measurement processing complete",
            extra={
                "processed": results["processed"],
                "failed": results["failed"],
                "unique_h3_cells": len(results["spatial_cells"]),
                "anomalies_detected": len(results["anomalies"]),
                "processing_time_seconds": processing_time,
                "operation": "batch_processing",
            },
        )

        return results

    def _quality_control(self, measurement: SensorMeasurement) -> dict:
        """Perform quality control on a measurement."""
        validation = self.quality_thresholds.get("sensor_validation", {})

        issues = []

        # Check radiation value range
        min_rad = validation.get("min_radiation", 0.0)
        max_rad = validation.get("max_radiation", 100.0)
        if not (min_rad <= measurement.value <= max_rad):
            issues.append(
                f"Radiation value {measurement.value} outside range [{min_rad}, {max_rad}]"
            )

        # Check coordinate validity
        if not (-90 <= measurement.latitude <= 90):
            issues.append(f"Invalid latitude: {measurement.latitude}")
        if not (-180 <= measurement.longitude <= 180):
            issues.append(f"Invalid longitude: {measurement.longitude}")

        # Check timestamp validity
        now = datetime.now(UTC)
        time_diff = abs((now - measurement.timestamp).total_seconds())
        if time_diff > 24 * 3600:  # More than 24 hours old
            issues.append(f"Timestamp too old: {time_diff} seconds")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "quality_score": (
                1.0 if len(issues) == 0 else max(0, 1.0 - len(issues) * 0.2)
            ),
        }

    def _is_anomaly(self, measurement: SensorMeasurement) -> bool:
        """Simple anomaly detection based on statistical thresholds."""
        anomaly_config = self.config.get("anomaly_detection", {}).get("statistical", {})

        # Get the configured empirical baseline for anomaly detection.
        baseline_config = self.config.get("radiation_baseline", {})
        if (
            "background_radiation" not in baseline_config
            or "noise_level" not in baseline_config
        ):
            raise ValueError(
                "radiation_baseline must define background_radiation and noise_level"
            )
        background = baseline_config["background_radiation"]
        noise = baseline_config["noise_level"]
        if noise <= 0:
            raise ValueError("radiation_baseline.noise_level must be greater than zero")

        # Calculate z-score
        z_score = abs(measurement.value - background) / noise

        # Check against thresholds
        mild_threshold = float(anomaly_config.get("threshold_mild", 2.0))
        return bool(z_score >= mild_threshold)

    def setup_spatial_inference(self, variable: str = "gamma_radiation") -> None:
        """Setup Bayesian spatial inference for radiation monitoring."""
        bayes_config = self.config.get("bayesian_inference", {})

        inference_config = SpatialInferenceConfig(
            variable=variable,
            h3_resolution=self.config.get("spatial", {}).get("h3_resolution", 5),
            temporal_window_hours=1.0,
            spatial_range_km=bayes_config.get("covariance", {}).get(
                "length_scale", 50000
            )
            / 1000,
            covariance_function=bayes_config.get("covariance", {}).get(
                "function", "matern_52"
            ),
            length_scale=bayes_config.get("covariance", {}).get("length_scale", 50000),
            noise_variance=bayes_config.get("covariance", {}).get(
                "noise_variance", 0.01
            ),
            confidence_levels=bayes_config.get("confidence_levels", [0.68, 0.95]),
        )

        self.ingestion.setup_spatial_inference(inference_config)

        self.logger.info(
            "Spatial inference configured",
            extra={
                "variable": variable,
                "h3_resolution": inference_config.h3_resolution,
                "covariance_function": inference_config.covariance_function,
                "length_scale": inference_config.length_scale,
                "operation": "spatial_inference_setup",
            },
        )

    async def perform_spatial_inference(
        self, variable: str = "gamma_radiation"
    ) -> dict:
        """Perform Bayesian spatial inference on collected measurements."""
        self.logger.info(
            "Starting spatial inference",
            extra={"variable": variable, "operation": "spatial_inference"},
        )

        start_time = time.time()

        # Trigger spatial inference update
        await self.ingestion._update_spatial_inference(variable)

        # Get results
        results = self.ingestion.get_spatial_distribution(variable)

        inference_time = time.time() - start_time
        self.metrics["spatial_inferences"] += 1

        if results:
            self.logger.info(
                "Spatial inference complete",
                extra={
                    "variable": variable,
                    "prediction_cells": len(results.get("predictions", [])),
                    "mean_prediction": (
                        np.mean(results.get("predictions", []))
                        if results.get("predictions")
                        else 0
                    ),
                    "inference_time_seconds": inference_time,
                    "operation": "spatial_inference",
                },
            )
        else:
            self.logger.warning(
                "Spatial inference returned no results",
                extra={"variable": variable, "operation": "spatial_inference"},
            )

        return results or {}

    def get_system_metrics(self) -> dict:
        """Get comprehensive system performance metrics."""
        runtime = time.time() - self.start_time

        metrics = {
            "runtime_seconds": runtime,
            "measurements_per_second": self.metrics["measurements_processed"]
            / max(runtime, 1),
            "error_rate": self.metrics["errors_encountered"]
            / max(self.metrics["measurements_processed"], 1),
            "anomaly_rate": self.metrics["anomalies_detected"]
            / max(self.metrics["measurements_processed"], 1),
            **self.metrics,
        }

        self.logger.info("System metrics collected", extra=metrics)
        return metrics

    def validate_system_health(self) -> dict:
        """Validate overall system health for testing purposes."""
        metrics = self.get_system_metrics()

        health_checks = {
            "measurements_processing": metrics["measurements_processed"] > 0,
            "error_rate_acceptable": metrics["error_rate"]
            < 0.1,  # Less than 10% errors
            "performance_acceptable": metrics["measurements_per_second"]
            > 10,  # At least 10/sec
            "spatial_inference_working": metrics["spatial_inferences"] > 0,
        }

        overall_health = all(health_checks.values())

        health_result = {
            "overall_healthy": overall_health,
            "checks": health_checks,
            "metrics": metrics,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.logger.info(
            "System health validation",
            extra={
                "overall_healthy": overall_health,
                "failed_checks": [k for k, v in health_checks.items() if not v],
                "operation": "health_check",
            },
        )

        return health_result


class GlobalRadiationMonitor:
    """Global-scale radiation monitoring orchestrator over RadiationMonitoringSystem."""

    def __init__(self, config: dict[str, Any], logger: Any | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.radiation_system = RadiationMonitoringSystem(config, logger)

    async def run_monitoring_cycle(
        self, measurements: list[dict] | None = None
    ) -> dict:
        """Run a complete monitoring cycle."""
        self.logger.info(
            "Starting global monitoring cycle", extra={"operation": "monitoring_cycle"}
        )

        if measurements is None:
            source = self.config.get("measurement_source")
            if not callable(source):
                raise RuntimeError(
                    "run_monitoring_cycle requires measurements or a callable measurement_source"
                )
            measurements = source()
        if not isinstance(measurements, list) or not measurements:
            raise ValueError("measurements must be a non-empty list")

        # Setup spatial inference
        self.radiation_system.setup_spatial_inference()

        # Process measurements
        processing_results = await self.radiation_system.process_measurements(
            measurements
        )

        # Perform spatial inference
        inference_results = await self.radiation_system.perform_spatial_inference()

        # Get system metrics
        system_metrics = self.radiation_system.get_system_metrics()

        # Validate system health
        health_status = self.radiation_system.validate_system_health()

        cycle_results = {
            "processing": processing_results,
            "inference": inference_results,
            "metrics": system_metrics,
            "health": health_status,
        }

        self.logger.info(
            "Global monitoring cycle complete",
            extra={
                "sensors_processed": processing_results["processed"],
                "anomalies_detected": len(processing_results["anomalies"]),
                "system_healthy": health_status["overall_healthy"],
                "operation": "monitoring_cycle",
            },
        )

        return cycle_results
