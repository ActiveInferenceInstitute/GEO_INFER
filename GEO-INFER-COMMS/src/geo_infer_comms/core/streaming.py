"""
Real-time data streaming system for GEO-INFER-COMMS.

This module implements comprehensive streaming infrastructure for real-time
data transmission, including geospatial data streams, sensor data feeds,
and live event streaming with geospatial filtering and optimization.
"""

from __future__ import annotations
import json
import logging
import threading
import time
import queue
from typing import Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field

from geo_infer_comms.models.message import StreamRequest, StreamResponse
from geo_infer_comms.models.spatial import (
    GeospatialMetadata,
    GeospatialPoint,
    SpatialFilter,
)
from geo_infer_comms.utils.validation import validate_stream_config


class DataStream:
    """
    Individual data stream with geospatial capabilities.

    Represents a single stream of data with geospatial filtering,
    buffering, and real-time delivery capabilities.
    """

    def __init__(
        self,
        stream_id: str,
        config: StreamRequest,
        buffer_size: int = 1000,
        enable_compression: bool = True,
    ):
        self.stream_id = stream_id
        self.config = config
        self.buffer_size = buffer_size
        self.enable_compression = enable_compression

        # Stream state
        self.is_active = False
        self.subscribers: set[str] = set()
        self.data_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.spatial_filter: SpatialFilter | None = None

        # Geospatial context
        if config.geospatial_filter:
            self.spatial_filter = SpatialFilter.from_dict(config.geospatial_filter)

        # Performance tracking
        self.data_points_sent = 0
        self.bytes_transferred = 0
        self.created_at = datetime.now(UTC)

        self.logger = logging.getLogger(__name__)

    def add_data_point(
        self, data: Any, geospatial_context: GeospatialMetadata | None = None
    ) -> bool:
        """Add a data point to the stream."""
        if not self.is_active:
            return False

        # Check geospatial filtering
        if self.spatial_filter and geospatial_context:
            if not self.spatial_filter.matches_location(geospatial_context.location):
                return False  # Data point filtered out

        # Create data point with metadata
        data_point = {
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
            "geospatial_context": (
                geospatial_context.to_dict() if geospatial_context else None
            ),
            "stream_id": self.stream_id,
        }

        try:
            # Add to buffer (may block if full)
            self.data_buffer.put(data_point, timeout=1.0)

            # Update metrics
            self.data_points_sent += 1
            self.bytes_transferred += len(json.dumps(data_point).encode("utf-8"))

            return True

        except queue.Full:
            self.logger.warning(f"Stream buffer full for {self.stream_id}")
            return False

    def get_data_points(
        self, count: int = 1, timeout: float = 1.0
    ) -> list[dict[str, Any]]:
        """Get data points from the stream buffer."""
        data_points = []

        for _ in range(count):
            try:
                data_point = self.data_buffer.get(timeout=timeout)
                data_points.append(data_point)
            except queue.Empty:
                break

        return data_points

    def get_stats(self) -> dict[str, Any]:
        """Get stream statistics."""
        return {
            "stream_id": self.stream_id,
            "is_active": self.is_active,
            "subscribers": len(self.subscribers),
            "buffer_size": self.data_buffer.qsize(),
            "data_points_sent": self.data_points_sent,
            "bytes_transferred": self.bytes_transferred,
            "created_at": self.created_at.isoformat(),
            "stream_type": self.config.stream_type,
            "geospatial_filter": (
                self.spatial_filter.to_dict() if self.spatial_filter else None
            ),
        }


class StreamManager:
    """
    Central stream management system.

    Manages multiple data streams with geospatial filtering,
    subscriber management, and real-time delivery optimization.
    """

    def __init__(
        self,
        max_streams: int = 1000,
        default_buffer_size: int = 1000,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        self.max_streams = max_streams
        self.default_buffer_size = default_buffer_size
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Stream storage
        self.streams: dict[str, DataStream] = {}
        self.stream_subscribers: dict[str, set[str]] = {}  # stream_id -> subscriber_ids

        # Spatial indexing for streams
        self.spatial_streams: dict[str, list[str]] = {}  # location_key -> stream_ids

        # Threading
        self._lock = threading.RLock()
        self._streaming_thread: threading.Thread | None = None
        self._running = False

        # Performance tracking
        self.metrics = StreamMetrics()

        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start the stream manager."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._streaming_thread = threading.Thread(
                target=self._process_streams, daemon=True
            )
            self._streaming_thread.start()

            self.logger.info("Stream manager started")

    def stop(self) -> None:
        """Stop the stream manager."""
        with self._lock:
            self._running = False
            if self._streaming_thread:
                self._streaming_thread.join(timeout=5.0)
            self.logger.info("Stream manager stopped")

    def create_stream(self, request: StreamRequest, creator_id: str) -> StreamResponse:
        """Create a new data stream."""
        # Validate request
        if not validate_stream_config(request.model_dump()):
            raise ValueError("Invalid stream configuration")

        # Check stream limit
        if len(self.streams) >= self.max_streams:
            raise ValueError(f"Maximum number of streams ({self.max_streams}) reached")

        # Create stream response
        stream = StreamResponse(
            name=request.name,
            stream_type=request.stream_type,
            geospatial_filter=request.geospatial_filter,
        )

        # Create data stream
        data_stream = DataStream(
            stream_id=stream.stream_id,
            config=request,
            buffer_size=self.default_buffer_size,
        )

        with self._lock:
            self.streams[stream.stream_id] = data_stream
            self.stream_subscribers[stream.stream_id] = set()

            # Add to spatial index if geospatial filter exists
            if request.geospatial_filter:
                self._add_stream_to_spatial_index(
                    stream.stream_id, request.geospatial_filter
                )

        self.metrics.streams_created += 1
        self.logger.info(f"Stream created: {stream.stream_id} by {creator_id}")
        return stream

    def get_stream(self, stream_id: str) -> DataStream | None:
        """Get a specific stream by ID."""
        with self._lock:
            return self.streams.get(stream_id)

    def get_streams(
        self, stream_type: str | None = None, limit: int = 100
    ) -> list[DataStream]:
        """Get streams with optional filtering."""
        with self._lock:
            streams = list(self.streams.values())

        # Apply filters
        if stream_type:
            streams = [s for s in streams if s.config.stream_type == stream_type]

        # Limit results
        return streams[:limit]

    def subscribe_to_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """Subscribe to a stream."""
        stream = self.streams.get(stream_id)
        if not stream:
            return False

        with self._lock:
            if subscriber_id not in self.stream_subscribers[stream_id]:
                self.stream_subscribers[stream_id].add(subscriber_id)
                stream.subscribers.add(subscriber_id)

        self.logger.info(f"Subscriber {subscriber_id} subscribed to stream {stream_id}")
        return True

    def unsubscribe_from_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """Unsubscribe from a stream."""
        with self._lock:
            if stream_id in self.stream_subscribers:
                self.stream_subscribers[stream_id].discard(subscriber_id)

                if stream_id in self.streams:
                    self.streams[stream_id].subscribers.discard(subscriber_id)

        self.logger.info(
            f"Subscriber {subscriber_id} unsubscribed from stream {stream_id}"
        )
        return True

    def publish_to_stream(
        self,
        stream_id: str,
        data: Any,
        geospatial_context: GeospatialMetadata | None = None,
    ) -> bool:
        """Publish data to a stream."""
        stream = self.streams.get(stream_id)
        if not stream:
            return False

        return stream.add_data_point(data, geospatial_context)

    def get_streams_by_location(
        self, location: GeospatialPoint, radius_km: float = 1.0
    ) -> list[DataStream]:
        """Get streams near a specific location."""
        nearby_streams = []

        # Find location keys within radius
        for location_key, stream_ids in self.spatial_streams.items():
            try:
                lon, lat = map(float, location_key.split(","))
                key_location = GeospatialPoint(longitude=lon, latitude=lat)

                if location.distance_to(key_location) <= (radius_km * 1000):
                    for stream_id in stream_ids:
                        stream = self.streams.get(stream_id)
                        if stream:
                            nearby_streams.append(stream)
            except (ValueError, IndexError):
                continue

        return nearby_streams

    def get_stream_statistics(self) -> dict[str, Any]:
        """Get comprehensive stream statistics."""
        with self._lock:
            active_streams = len([s for s in self.streams.values() if s.is_active])
            total_subscribers = sum(
                len(subs) for subs in self.stream_subscribers.values()
            )

            return {
                "total_streams": len(self.streams),
                "active_streams": active_streams,
                "total_subscribers": total_subscribers,
                "spatial_index_size": len(self.spatial_streams),
                "metrics": self.metrics.to_dict(),
            }

    def _process_streams(self) -> None:
        """Background thread to process stream data delivery."""
        while self._running:
            try:
                # Process each active stream
                for stream_id, stream in list(self.streams.items()):
                    if stream.is_active and not stream.data_buffer.empty():
                        self._deliver_stream_data(stream_id, stream)

                # Brief pause
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing streams: {e}")
                time.sleep(1.0)

    def _deliver_stream_data(self, stream_id: str, stream: DataStream) -> None:
        """Drain a stream's in-memory buffer and account for delivery.

        This is an in-memory buffer simulation: data points are pulled from
        the stream's queue and counted in metrics, but are not pushed to
        subscriber transports. Wire this method to a live transport (e.g.
        the WebSocket manager in ``api.websocket_api``) to distribute data
        externally; the streaming protocols in this module are likewise
        registration stubs, not network clients.
        """
        try:
            # Get data points from buffer
            data_points = stream.get_data_points(count=10)  # Batch delivery

            if not data_points:
                return

            self.metrics.data_points_delivered += len(data_points)

        except Exception as e:
            self.logger.error(f"Error delivering stream data for {stream_id}: {e}")

    def _add_stream_to_spatial_index(
        self, stream_id: str, geospatial_filter: dict[str, Any]
    ) -> None:
        """Add stream to spatial index."""
        # Extract location from geospatial filter (simplified)
        if geospatial_filter.get("filter_type") == "bounds":
            bounds_data = geospatial_filter.get("parameters", {}).get("bounds", {})
            if bounds_data:
                center_lon = (
                    bounds_data["min_longitude"] + bounds_data["max_longitude"]
                ) / 2
                center_lat = (
                    bounds_data["min_latitude"] + bounds_data["max_latitude"]
                ) / 2
                location_key = f"{center_lon:.3f},{center_lat:.3f}"

                if location_key not in self.spatial_streams:
                    self.spatial_streams[location_key] = []
                self.spatial_streams[location_key].append(stream_id)


@dataclass
class StreamMetrics:
    """Metrics for streaming system performance."""

    streams_created: int = 0
    streams_deleted: int = 0
    data_points_sent: int = 0
    data_points_delivered: int = 0
    bytes_transferred: int = 0
    subscribers_connected: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "streams_created": self.streams_created,
            "streams_deleted": self.streams_deleted,
            "data_points_sent": self.data_points_sent,
            "data_points_delivered": self.data_points_delivered,
            "delivery_success_rate": (
                self.data_points_delivered / max(self.data_points_sent, 1) * 100
            ),
            "bytes_transferred": self.bytes_transferred,
            "subscribers_connected": self.subscribers_connected,
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.streams_created = 0
        self.streams_deleted = 0
        self.data_points_sent = 0
        self.data_points_delivered = 0
        self.bytes_transferred = 0
        self.subscribers_connected = 0
        self.start_time = datetime.now(UTC)


class GeospatialDataStream:
    """
    Specialized stream for geospatial data with advanced spatial features.

    Handles geospatial data streams with spatial indexing, filtering,
    and real-time geospatial analysis capabilities.
    """

    def __init__(
        self,
        stream_id: str,
        geospatial_config: dict[str, Any],
        spatial_resolution: float = 0.001,  # degrees
    ):
        self.stream_id = stream_id
        self.geospatial_config = geospatial_config
        self.spatial_resolution = spatial_resolution

        # Spatial data structures
        self.spatial_data: dict[str, dict[str, Any]] = {}  # location_key -> data
        self.temporal_data: dict[
            str, list[dict[str, Any]]
        ] = {}  # location_key -> time_series
        self.spatial_aggregations: dict[
            str, dict[str, Any]
        ] = {}  # aggregation_type -> results

        # Real-time analysis
        self.hotspots: list[dict[str, Any]] = []
        self.anomalies: list[dict[str, Any]] = []
        self.patterns: list[dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    def add_geospatial_data(
        self, location: GeospatialPoint, data: Any, timestamp: datetime | None = None
    ) -> None:
        """Add geospatial data point to the stream."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        location_key = self._generate_location_key(location)

        # Store data
        if location_key not in self.spatial_data:
            self.spatial_data[location_key] = {}

        data_point = {
            "location": {
                "longitude": location.longitude,
                "latitude": location.latitude,
            },
            "data": data,
            "timestamp": timestamp.isoformat(),
        }

        self.spatial_data[location_key] = data_point

        # Add to temporal series
        if location_key not in self.temporal_data:
            self.temporal_data[location_key] = []

        self.temporal_data[location_key].append(data_point)

        # Keep only recent data (last 1000 points per location)
        if len(self.temporal_data[location_key]) > 1000:
            self.temporal_data[location_key] = self.temporal_data[location_key][-1000:]

        # Update aggregations
        self._update_aggregations(location_key, data_point)

        # Detect hotspots and anomalies
        self._detect_spatial_patterns()

    def get_data_at_location(
        self, location: GeospatialPoint, radius_km: float = 0.1
    ) -> list[dict[str, Any]]:
        """Get data points near a location."""
        nearby_data = []

        for location_key, data_point in self.spatial_data.items():
            try:
                lon, lat = map(float, location_key.split(","))
                key_location = GeospatialPoint(longitude=lon, latitude=lat)

                if location.distance_to(key_location) <= (radius_km * 1000):
                    nearby_data.append(data_point)
            except (ValueError, IndexError):
                continue

        return nearby_data

    def get_temporal_series(
        self, location: GeospatialPoint, time_window: timedelta | None = None
    ) -> list[dict[str, Any]]:
        """Get temporal data series for a location."""
        location_key = self._generate_location_key(location)
        series = self.temporal_data.get(location_key, [])

        if time_window:
            cutoff_time = datetime.now(UTC) - time_window
            series = [
                point
                for point in series
                if datetime.fromisoformat(point["timestamp"]) >= cutoff_time
            ]

        return series

    def _generate_location_key(self, location: GeospatialPoint) -> str:
        """Generate location key for spatial indexing."""
        # Round to spatial resolution for grouping nearby points
        lon_rounded = (
            round(location.longitude / self.spatial_resolution)
            * self.spatial_resolution
        )
        lat_rounded = (
            round(location.latitude / self.spatial_resolution) * self.spatial_resolution
        )
        return f"{lon_rounded:.6f},{lat_rounded:.6f}"

    def _update_aggregations(
        self, location_key: str, data_point: dict[str, Any]
    ) -> None:
        """Update spatial aggregations.

        Non-numeric data payloads cannot be aggregated numerically and are
        skipped rather than raising ``TypeError`` in the hot ingest path.
        """
        data_value = data_point.get("data", 0)

        if isinstance(data_value, bool) or not isinstance(data_value, (int, float)):
            return

        if location_key not in self.spatial_aggregations:
            self.spatial_aggregations[location_key] = {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "avg": 0.0,
            }

        agg = self.spatial_aggregations[location_key]
        agg["count"] += 1
        agg["sum"] += data_value
        agg["min"] = min(agg["min"], data_value)
        agg["max"] = max(agg["max"], data_value)
        agg["avg"] = agg["sum"] / agg["count"]

    def _detect_spatial_patterns(self) -> None:
        """Detect spatial patterns like hotspots and anomalies."""
        # Simple hotspot detection (high density areas)
        location_counts = {}
        for location_key in self.spatial_data.keys():
            # Parse location
            try:
                lon, lat = map(float, location_key.split(","))
                location_counts[location_key] = len(
                    self.temporal_data.get(location_key, [])
                )
            except (ValueError, IndexError):
                continue

        # Find hotspots (locations with high activity)
        threshold = max(location_counts.values()) * 0.8 if location_counts else 0
        self.hotspots = [
            {"location_key": loc, "count": count}
            for loc, count in location_counts.items()
            if count >= threshold
        ]

        # Simple anomaly detection (values significantly different from average)
        anomalies = []
        for location_key, agg in self.spatial_aggregations.items():
            if agg["count"] > 10:  # Only consider locations with sufficient data
                avg_value = agg["avg"]
                current_value = self.spatial_data[location_key].get("data", 0)

                if isinstance(current_value, bool) or not isinstance(
                    current_value, (int, float)
                ):
                    continue

                # Simple anomaly detection (values > 2 standard deviations)
                if abs(current_value - avg_value) > (
                    avg_value * 0.5
                ):  # 50% deviation threshold
                    anomalies.append(
                        {
                            "location_key": location_key,
                            "current_value": current_value,
                            "average": avg_value,
                            "deviation": abs(current_value - avg_value)
                            / avg_value
                            * 100,
                        }
                    )

        self.anomalies = anomalies[:10]  # Keep top 10 anomalies


class StreamingProtocolManager:
    """
    Protocol manager for different streaming protocols.

    Supports multiple streaming protocols (WebSocket, MQTT, HTTP/2 Server Push, etc.)
    with unified interface and geospatial capabilities.
    """

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.protocols: dict[str, StreamingProtocol] = {}

        # Protocol implementations
        self.protocols["websocket"] = WebSocketStreamingProtocol(stream_manager)
        self.protocols["mqtt"] = MQTTStreamingProtocol(stream_manager)
        self.protocols["sse"] = ServerSentEventsProtocol(stream_manager)

        self.logger = logging.getLogger(__name__)

    def get_protocol(self, protocol_name: str) -> StreamingProtocol | None:
        """Get a streaming protocol implementation."""
        return self.protocols.get(protocol_name)

    def list_available_protocols(self) -> list[str]:
        """List all available streaming protocols."""
        return list(self.protocols.keys())

    def register_protocol(self, name: str, protocol: StreamingProtocol) -> None:
        """Register a new streaming protocol."""
        self.protocols[name] = protocol
        self.logger.info(f"Streaming protocol registered: {name}")


class StreamingProtocol:
    """Base class for streaming protocols."""

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.logger = logging.getLogger(__name__)

    async def start_streaming(self, stream_id: str) -> bool:
        """Start streaming for a stream."""
        raise RuntimeError("Streaming subclasses must implement start_streaming")

    async def stop_streaming(self, stream_id: str) -> bool:
        """Stop streaming for a stream."""
        raise RuntimeError("Streaming subclasses must implement stop_streaming")

    def get_protocol_stats(self) -> dict[str, Any]:
        """Get protocol-specific statistics."""
        return {"protocol": self.__class__.__name__}


class WebSocketStreamingProtocol(StreamingProtocol):
    """WebSocket streaming protocol implementation."""

    def __init__(self, stream_manager: StreamManager):
        super().__init__(stream_manager)
        self.active_connections: dict[str, set[str]] = {}  # stream_id -> connection_ids

    async def start_streaming(self, stream_id: str) -> bool:
        """Start WebSocket streaming for a stream."""
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = set()

        # In a real implementation, would set up WebSocket connections
        self.logger.info(f"WebSocket streaming started for stream: {stream_id}")
        return True

    async def stop_streaming(self, stream_id: str) -> bool:
        """Stop WebSocket streaming for a stream."""
        if stream_id in self.active_connections:
            del self.active_connections[stream_id]

        self.logger.info(f"WebSocket streaming stopped for stream: {stream_id}")
        return True


class MQTTStreamingProtocol(StreamingProtocol):
    """MQTT streaming protocol implementation."""

    def __init__(self, stream_manager: StreamManager):
        super().__init__(stream_manager)
        self.mqtt_topics: dict[str, str] = {}  # stream_id -> mqtt_topic

    async def start_streaming(self, stream_id: str) -> bool:
        """Start MQTT streaming for a stream."""
        topic = f"geoinfer/streams/{stream_id}"
        self.mqtt_topics[stream_id] = topic

        # In a real implementation, would connect to MQTT broker
        self.logger.info(
            f"MQTT streaming started for stream: {stream_id} on topic: {topic}"
        )
        return True

    async def stop_streaming(self, stream_id: str) -> bool:
        """Stop MQTT streaming for a stream."""
        if stream_id in self.mqtt_topics:
            del self.mqtt_topics[stream_id]

        self.logger.info(f"MQTT streaming stopped for stream: {stream_id}")
        return True


class ServerSentEventsProtocol(StreamingProtocol):
    """Server-Sent Events streaming protocol implementation."""

    def __init__(self, stream_manager: StreamManager):
        super().__init__(stream_manager)
        self.sse_clients: dict[str, list[dict[str, Any]]] = {}  # stream_id -> clients

    async def start_streaming(self, stream_id: str) -> bool:
        """Start Server-Sent Events streaming for a stream."""
        if stream_id not in self.sse_clients:
            self.sse_clients[stream_id] = []

        # In a real implementation, would set up SSE connections
        self.logger.info(f"SSE streaming started for stream: {stream_id}")
        return True

    async def stop_streaming(self, stream_id: str) -> bool:
        """Stop Server-Sent Events streaming for a stream."""
        if stream_id in self.sse_clients:
            del self.sse_clients[stream_id]

        self.logger.info(f"SSE streaming stopped for stream: {stream_id}")
        return True


class StreamingAnalytics:
    """
    Analytics and monitoring for streaming system.

    Provides insights into streaming performance, data patterns,
    and geospatial data analysis for streaming applications.
    """

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.streaming_history: list[dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    def record_streaming_event(
        self, stream_id: str, event_type: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record a streaming event for analytics."""
        event = {
            "stream_id": stream_id,
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details or {},
        }

        self.streaming_history.append(event)

        # Keep only recent history
        if len(self.streaming_history) > 5000:
            self.streaming_history = self.streaming_history[-5000:]

    def get_streaming_analytics(self, stream_id: str) -> dict[str, Any]:
        """Get analytics for a specific stream."""
        stream_events = [
            event for event in self.streaming_history if event["stream_id"] == stream_id
        ]

        if not stream_events:
            return {"message": "No analytics data available for stream"}

        # Analyze events
        event_types: dict[str, int] = {}
        for event in stream_events:
            event_type = event["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "stream_id": stream_id,
            "total_events": len(stream_events),
            "event_types": event_types,
            "time_range": {
                "start": min(e["timestamp"] for e in stream_events),
                "end": max(e["timestamp"] for e in stream_events),
            },
        }

    def get_system_streaming_analytics(self) -> dict[str, Any]:
        """Get system-wide streaming analytics."""
        if not self.streaming_history:
            return {"message": "No streaming history available"}

        # Analyze all streaming events
        stream_activity: dict[str, int] = {}
        for event in self.streaming_history:
            stream_id = event["stream_id"]
            stream_activity[stream_id] = stream_activity.get(stream_id, 0) + 1

        return {
            "total_streaming_events": len(self.streaming_history),
            "active_streams": len(set(e["stream_id"] for e in self.streaming_history)),
            "most_active_streams": sorted(
                stream_activity.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "time_range": {
                "start": min(e["timestamp"] for e in self.streaming_history),
                "end": max(e["timestamp"] for e in self.streaming_history),
            },
        }


class StreamingOrchestrator:
    """
    High-level streaming orchestrator.

    Coordinates multiple streaming components and provides
    unified streaming capabilities for complex geospatial applications.
    """

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.protocol_manager = StreamingProtocolManager(stream_manager)
        self.geospatial_streams: dict[str, GeospatialDataStream] = {}
        self.analytics = StreamingAnalytics(stream_manager)

        self.logger = logging.getLogger(__name__)

    def create_geospatial_stream(
        self, stream_id: str, geospatial_config: dict[str, Any]
    ) -> GeospatialDataStream:
        """Create a specialized geospatial data stream."""
        geospatial_stream = GeospatialDataStream(
            stream_id=stream_id, geospatial_config=geospatial_config
        )

        self.geospatial_streams[stream_id] = geospatial_stream
        self.logger.info(f"Geospatial stream created: {stream_id}")
        return geospatial_stream

    def stream_geospatial_data(
        self,
        stream_id: str,
        location: GeospatialPoint,
        data: Any,
        protocol: str = "websocket",
    ) -> bool:
        """Stream geospatial data through specified protocol."""
        # Add to geospatial stream if exists
        if stream_id in self.geospatial_streams:
            self.geospatial_streams[stream_id].add_geospatial_data(location, data)

        # Stream through protocol manager
        protocol_impl = self.protocol_manager.get_protocol(protocol)
        if protocol_impl:
            # In a real implementation, would stream through the protocol
            self.analytics.record_streaming_event(
                stream_id,
                "data_streamed",
                {
                    "protocol": protocol,
                    "location": {
                        "longitude": location.longitude,
                        "latitude": location.latitude,
                    },
                },
            )

        return True

    def get_streaming_insights(self) -> dict[str, Any]:
        """Get comprehensive streaming insights."""
        return {
            "stream_manager_stats": self.stream_manager.get_stream_statistics(),
            "protocol_stats": {
                name: protocol.get_protocol_stats()
                for name, protocol in self.protocol_manager.protocols.items()
            },
            "geospatial_stream_count": len(self.geospatial_streams),
            "analytics": self.analytics.get_system_streaming_analytics(),
        }
