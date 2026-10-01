"""
Streaming API Module

This module provides WebSocket and streaming API endpoints for real-time
sensor data streaming and live monitoring capabilities.
"""

import logging
import asyncio
import dataclasses
import json
from typing import Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Optional imports for enhanced functionality
try:
    from geo_infer_iot.core.ingestion import IoTDataIngestion

    HAS_INGESTION = True
except ImportError:
    HAS_INGESTION = False

logger = logging.getLogger(__name__)


class StreamingAPI:
    """
    WebSocket and streaming API for real-time sensor data.

    Provides real-time data streaming capabilities including:
    - Live sensor data feeds via WebSocket
    - Real-time spatial inference updates
    - Live anomaly detection alerts
    - Streaming data export capabilities
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        ingestion: IoTDataIngestion | None = None,
    ) -> None:
        self.config = config or {}
        self.app = FastAPI(title="GEO-INFER-IOT Streaming API", version="1.0.0")

        # WebSocket connection management
        self.active_connections: set[WebSocket] = set()
        self.sensor_subscriptions: dict[str, set[WebSocket]] = {}
        self.spatial_subscriptions: dict[str, set[WebSocket]] = {}

        # Share the caller's ingestion instance when one is provided so
        # measurements submitted through the SensorAPI reach stream
        # subscribers; otherwise own a private instance.
        if ingestion is not None:
            self.ingestion: IoTDataIngestion | None = ingestion
        elif HAS_INGESTION:
            self.ingestion = IoTDataIngestion(None, self.config)
        else:
            self.ingestion = None

        # Setup API routes
        self._setup_routes()

        logger.info("StreamingAPI initialized")

    def _setup_routes(self) -> None:
        """Setup API routes and WebSocket endpoints."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """API root endpoint."""
            return {
                "service": "GEO-INFER-IOT Streaming API",
                "version": "1.0.0",
                "status": "operational",
                "websocket_endpoint": "/ws/sensor-stream",
                "timestamp": datetime.now().isoformat(),
            }

        @self.app.websocket("/ws/sensor-stream")
        async def sensor_stream_websocket(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time sensor data streaming."""
            await websocket.accept()

            # Add to active connections
            self.active_connections.add(websocket)

            try:
                # Wait for subscription message
                subscription_data = await websocket.receive_text()
                subscription = json.loads(subscription_data)

                sensor_ids = subscription.get("sensor_ids", [])
                h3_indices = subscription.get("h3_indices", [])

                # Register subscriptions
                for sensor_id in sensor_ids:
                    if sensor_id not in self.sensor_subscriptions:
                        self.sensor_subscriptions[sensor_id] = set()
                    self.sensor_subscriptions[sensor_id].add(websocket)

                for h3_index in h3_indices:
                    if h3_index not in self.spatial_subscriptions:
                        self.spatial_subscriptions[h3_index] = set()
                    self.spatial_subscriptions[h3_index].add(websocket)

                # Send confirmation
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "subscription_confirmed",
                            "sensor_ids": sensor_ids,
                            "h3_indices": h3_indices,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                )

                # Keep connection alive, forwarding measurements ingested
                # since the last tick to this connection's subscriptions.
                ingestion = self.ingestion
                cursor: int | None = None
                if ingestion is not None:
                    cursor = len(ingestion.measurements)

                while True:
                    if ingestion is not None and cursor is not None:
                        for measurement in ingestion.measurements[cursor:]:
                            await self._forward_to_socket(
                                websocket, measurement, sensor_ids, h3_indices
                            )
                        cursor = len(ingestion.measurements)

                    # Example: Send periodic heartbeat
                    await asyncio.sleep(1)

                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    )

            except WebSocketDisconnect:
                # Remove from all subscriptions
                for sensor_connections in self.sensor_subscriptions.values():
                    sensor_connections.discard(websocket)

                for spatial_connections in self.spatial_subscriptions.values():
                    spatial_connections.discard(websocket)

                self.active_connections.discard(websocket)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.active_connections.discard(websocket)

        @self.app.get("/streams")
        async def list_streams() -> dict[str, Any]:
            """List available data streams."""
            streams = {
                "sensor_data_stream": {
                    "type": "sensor_measurements",
                    "format": "json",
                    "real_time": True,
                    "websocket_endpoint": "/ws/sensor-stream",
                    "description": "Real-time sensor measurement stream",
                },
                "spatial_inference_stream": {
                    "type": "spatial_predictions",
                    "format": "json",
                    "real_time": True,
                    "description": (
                        "Spatial inference results delivered over the sensor "
                        "stream WebSocket by broadcast_spatial_inference"
                    ),
                },
            }

            return {
                "streams": streams,
                "total_streams": len(streams),
                "active_connections": len(self.active_connections),
            }

        @self.app.get("/subscriptions")
        async def get_subscriptions() -> dict[str, Any]:
            """Get current subscription status."""
            return {
                "active_connections": len(self.active_connections),
                "sensor_subscriptions": {
                    sensor_id: len(connections)
                    for sensor_id, connections in self.sensor_subscriptions.items()
                },
                "spatial_subscriptions": {
                    h3_index: len(connections)
                    for h3_index, connections in self.spatial_subscriptions.items()
                },
                "timestamp": datetime.now().isoformat(),
            }

    async def broadcast_measurement(self, measurement: dict) -> None:
        """Broadcast a new measurement to subscribed clients.

        Sends directly on each subscribed WebSocket; subscribers whose send
        fails are removed from all subscription sets.
        """
        sensor_id = measurement.get("sensor_id")
        h3_index = measurement.get("h3_index")

        if sensor_id in self.sensor_subscriptions:
            message = {
                "type": "sensor_measurement",
                "data": measurement,
                "timestamp": datetime.now().isoformat(),
            }
            await self._send_to_all(self.sensor_subscriptions[sensor_id], message)

        if h3_index in self.spatial_subscriptions:
            message = {
                "type": "spatial_measurement",
                "data": measurement,
                "h3_index": h3_index,
                "timestamp": datetime.now().isoformat(),
            }
            await self._send_to_all(self.spatial_subscriptions[h3_index], message)

    async def broadcast_spatial_inference(self, inference_result: dict) -> None:
        """Broadcast spatial inference results to all connected clients."""
        message = {
            "type": "spatial_inference",
            "data": inference_result,
            "timestamp": datetime.now().isoformat(),
        }
        await self._send_to_all(self.active_connections, message)

    async def _send_to_all(self, connections: set[WebSocket], message: dict) -> None:
        """Send a message to every connection, dropping the disconnected."""
        payload = json.dumps(message)
        disconnected: set[WebSocket] = set()
        for websocket in list(connections):
            try:
                await websocket.send_text(payload)
            except Exception:
                disconnected.add(websocket)

        for websocket in disconnected:
            connections.discard(websocket)
            self.active_connections.discard(websocket)
            for subs in self.sensor_subscriptions.values():
                subs.discard(websocket)
            for subs in self.spatial_subscriptions.values():
                subs.discard(websocket)

    @staticmethod
    def _measurement_payload(measurement: Any) -> dict:
        """Convert an ingested measurement to a JSON-serializable dict."""
        data = dataclasses.asdict(measurement)
        timestamp = data.get("timestamp")
        if isinstance(timestamp, datetime):
            data["timestamp"] = timestamp.isoformat()
        return data

    async def _forward_to_socket(
        self,
        websocket: WebSocket,
        measurement: Any,
        sensor_ids: list[str],
        h3_indices: list[str],
    ) -> None:
        """Forward one ingested measurement to a single subscribed socket."""
        sensor_id = getattr(measurement, "sensor_id", None)
        h3_index = getattr(measurement, "h3_index", None)
        if sensor_id in sensor_ids:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "sensor_measurement",
                        "data": self._measurement_payload(measurement),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )
        if h3_index in h3_indices:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "spatial_measurement",
                        "data": self._measurement_payload(measurement),
                        "h3_index": h3_index,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
