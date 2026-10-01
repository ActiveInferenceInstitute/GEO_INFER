"""
Sensor API Module

This module provides REST API endpoints for sensor management and data access.
Integrates with FastAPI for high-performance web services.
"""

import logging
from typing import Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
import h3

# Optional imports for enhanced functionality
try:
    from geo_infer_iot.core.registry import SensorRegistry
    from geo_infer_iot.core.ingestion import IoTDataIngestion

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False

logger = logging.getLogger(__name__)


class SensorAPI:
    """
    REST API for sensor management and data access.

    Provides endpoints for:
    - Sensor registration and management
    - Real-time sensor data ingestion
    - Historical data queries with spatial filtering
    - Sensor network status monitoring
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.app = FastAPI(title="GEO-INFER-IOT Sensor API", version="1.0.0")

        # Initialize core components if available
        self.registry: Any | None = None
        self.ingestion: Any | None = None
        if HAS_CORE_MODULES:
            self.registry = SensorRegistry(config)
            self.ingestion = IoTDataIngestion(self.registry, config)
        else:
            self.registry = None
            self.ingestion = None

        # Setup API routes
        self._setup_routes()

        logger.info("SensorAPI initialized")

    def _setup_routes(self) -> None:
        """Setup API routes and endpoints."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """API root endpoint with service information."""
            return {
                "service": "GEO-INFER-IOT Sensor API",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "endpoints": ["/sensors", "/measurements", "/networks", "/health"],
            }

        @self.app.get("/sensors")
        async def list_sensors(
            sensor_type: str | None = Query(None, description="Filter by sensor type"),
            network_id: str | None = Query(None, description="Filter by network ID"),
            h3_index: str | None = Query(None, description="Filter by H3 index"),
            limit: int = Query(100, description="Maximum number of sensors to return"),
            offset: int = Query(0, description="Offset for pagination"),
        ) -> dict[str, Any]:
            """List sensors with optional filtering."""
            if self.registry is None:
                raise HTTPException(
                    status_code=503, detail="Sensor registry not available"
                )

            try:
                sensors = []

                # Get all sensors or apply filters
                if sensor_type or network_id or h3_index:
                    # Apply filters (simplified implementation)
                    for sensor in self.registry.sensors.values():
                        if sensor_type and sensor.sensor_type != sensor_type:
                            continue
                        if network_id and sensor.network_id != network_id:
                            continue
                        if h3_index and sensor.h3_index != h3_index:
                            continue
                        sensors.append(sensor)
                else:
                    sensors = list(self.registry.sensors.values())

                # Apply pagination
                total_sensors = len(sensors)
                paginated_sensors = sensors[offset : offset + limit]

                return {
                    "sensors": [
                        {
                            "sensor_id": s.sensor_id,
                            "network_id": s.network_id,
                            "sensor_type": s.sensor_type,
                            "latitude": s.latitude,
                            "longitude": s.longitude,
                            "h3_index": s.h3_index,
                            "status": s.status,
                            "registered_at": s.registered_at.isoformat(),
                            "last_seen": s.last_seen.isoformat()
                            if s.last_seen
                            else None,
                        }
                        for s in paginated_sensors
                    ],
                    "total_count": total_sensors,
                    "returned_count": len(paginated_sensors),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": offset + limit < total_sensors,
                    },
                }

            except Exception as e:
                logger.error(f"Error listing sensors: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error retrieving sensors: {str(e)}"
                ) from e

        @self.app.get("/sensors/{sensor_id}")
        async def get_sensor(sensor_id: str) -> dict[str, Any]:
            """Get detailed information about a specific sensor."""
            if self.registry is None:
                raise HTTPException(
                    status_code=503, detail="Sensor registry not available"
                )

            sensor = self.registry.sensors.get(sensor_id)
            if not sensor:
                raise HTTPException(
                    status_code=404, detail=f"Sensor {sensor_id} not found"
                )

            return {
                "sensor_id": sensor.sensor_id,
                "network_id": sensor.network_id,
                "sensor_type": sensor.sensor_type,
                "location": {
                    "latitude": sensor.latitude,
                    "longitude": sensor.longitude,
                    "h3_index": sensor.h3_index,
                },
                "status": sensor.status,
                "metadata": sensor.metadata,
                "registered_at": sensor.registered_at.isoformat(),
                "last_seen": sensor.last_seen.isoformat() if sensor.last_seen else None,
            }

        @self.app.post("/sensors")
        async def register_sensor(sensor_data: dict) -> dict[str, Any]:
            """Register a new sensor."""
            if self.registry is None:
                raise HTTPException(
                    status_code=503, detail="Sensor registry not available"
                )

            try:
                sensor = self.registry.register_sensor(sensor_data)
                return {
                    "message": "Sensor registered successfully",
                    "sensor_id": sensor.sensor_id,
                    "status": "registered",
                }
            except Exception as e:
                logger.error(f"Error registering sensor: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Error registering sensor: {str(e)}"
                ) from e

        @self.app.get("/measurements")
        async def query_measurements(
            sensor_id: str | None = Query(None, description="Filter by sensor ID"),
            variable: str | None = Query(None, description="Filter by variable type"),
            start_time: datetime | None = Query(
                None, description="Start time for query"
            ),
            end_time: datetime | None = Query(None, description="End time for query"),
            h3_index: str | None = Query(
                None, description="H3 cell to filter measurements by location"
            ),
            h3_resolution: int = Query(
                8, description="H3 resolution used to match measurements to h3_index"
            ),
            limit: int = Query(1000, description="Maximum measurements to return"),
        ) -> dict[str, Any]:
            """Query sensor measurements with temporal and spatial filtering."""
            if self.ingestion is None:
                raise HTTPException(
                    status_code=503, detail="Data ingestion not available"
                )

            if h3_index is not None and not h3.is_valid_cell(h3_index):
                raise HTTPException(
                    status_code=400, detail=f"Invalid H3 index: {h3_index}"
                )

            try:
                # Filter measurements based on query parameters
                filtered_measurements = []

                for measurement in self.ingestion.measurements:
                    # Time filtering
                    if start_time and measurement.timestamp < start_time:
                        continue
                    if end_time and measurement.timestamp > end_time:
                        continue

                    # Sensor filtering
                    if sensor_id and measurement.sensor_id != sensor_id:
                        continue

                    # Variable filtering
                    if variable and measurement.variable != variable:
                        continue

                    # Spatial filtering: match each measurement's location to the
                    # requested H3 cell at the requested resolution
                    if h3_index is not None:
                        measurement_cell = h3.latlng_to_cell(
                            measurement.latitude,
                            measurement.longitude,
                            h3_resolution,
                        )
                        if measurement_cell != h3_index:
                            continue

                    filtered_measurements.append(measurement)

                # Apply limit
                limited_measurements = (
                    filtered_measurements[-limit:]
                    if len(filtered_measurements) > limit
                    else filtered_measurements
                )

                return {
                    "measurements": [
                        {
                            "sensor_id": m.sensor_id,
                            "timestamp": m.timestamp.isoformat(),
                            "variable": m.variable,
                            "value": m.value,
                            "unit": m.unit,
                            "location": {
                                "latitude": m.latitude,
                                "longitude": m.longitude,
                                "h3_index": m.h3_index,
                            },
                            "quality_flags": m.quality_flags,
                            "metadata": m.metadata,
                        }
                        for m in limited_measurements
                    ],
                    "total_count": len(filtered_measurements),
                    "returned_count": len(limited_measurements),
                    "query_parameters": {
                        "sensor_id": sensor_id,
                        "variable": variable,
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None,
                        "h3_index": h3_index,
                        "h3_resolution": h3_resolution,
                    },
                }

            except Exception as e:
                logger.error(f"Error querying measurements: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error querying measurements: {str(e)}"
                ) from e

        @self.app.post("/measurements")
        async def submit_measurements(measurements: list[dict]) -> dict[str, Any]:
            """Submit new sensor measurements."""
            if self.ingestion is None:
                raise HTTPException(
                    status_code=503, detail="Data ingestion not available"
                )

            try:
                processed_count = 0
                failed_count = 0

                for measurement in measurements:
                    success = await self.ingestion.ingest_measurement(measurement)
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1

                return {
                    "message": f"Processed {processed_count} measurements, {failed_count} failed",
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "total_submitted": len(measurements),
                }

            except Exception as e:
                logger.error(f"Error submitting measurements: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing measurements: {str(e)}"
                ) from e

        @self.app.get("/networks")
        async def list_networks() -> dict[str, Any]:
            """List all sensor networks."""
            if self.registry is None:
                raise HTTPException(
                    status_code=503, detail="Sensor registry not available"
                )

            networks = []
            for network in self.registry.networks.values():
                networks.append(
                    {
                        "network_id": network.network_id,
                        "name": network.name,
                        "protocol": network.protocol,
                        "spatial_bounds": network.spatial_bounds,
                        "sensor_types": network.sensor_types,
                        "sensor_count": network.sensor_count,
                        "created_at": network.created_at.isoformat(),
                    }
                )

            return {"networks": networks, "total_networks": len(networks)}

        @self.app.get("/health")
        async def health_check() -> dict[str, Any]:
            """Health check endpoint."""
            health_status: dict[str, Any] = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {},
            }

            # Check core services
            if self.registry:
                health_status["services"]["registry"] = "available"
            else:
                health_status["services"]["registry"] = "unavailable"

            if self.ingestion:
                health_status["services"]["ingestion"] = "available"
                # Get ingestion statistics
                stats = self.ingestion.get_measurement_statistics()
                health_status["ingestion_stats"] = stats
            else:
                health_status["services"]["ingestion"] = "unavailable"

            return health_status

        @self.app.get("/spatial/{h3_index}/sensors")
        async def get_sensors_in_h3_cell(h3_index: str) -> dict[str, Any]:
            """Get all sensors in a specific H3 cell."""
            if self.registry is None:
                raise HTTPException(
                    status_code=503, detail="Sensor registry not available"
                )

            sensors = self.registry.get_sensors_in_h3_cell(h3_index)

            return {
                "h3_index": h3_index,
                "sensor_count": len(sensors),
                "sensors": [
                    {
                        "sensor_id": s.sensor_id,
                        "sensor_type": s.sensor_type,
                        "latitude": s.latitude,
                        "longitude": s.longitude,
                        "status": s.status,
                    }
                    for s in sensors
                ],
            }

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
        """Run the API server."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, **kwargs)
