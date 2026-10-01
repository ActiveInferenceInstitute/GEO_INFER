"""
Real-time stream processing for GEO-INFER-TIME.

This module provides capabilities for processing temporal data streams
in real-time with sliding windows, tumbling windows, session windows,
aggregation, late data handling, bounded watermarking, automated anomaly
alerts, and stream ingest adapters for WebSocket and Kafka sources.
"""

import logging
import math
from bisect import bisect_right
from collections import deque
from contextlib import aclosing
from datetime import datetime, timedelta
from typing import Any
from collections.abc import Callable

import numpy as np
from geo_infer_time.core.stream_ingest import (
    KafkaIngestAdapter as KafkaIngestAdapter,
    ReplayIngestAdapter as ReplayIngestAdapter,
    StreamIngestAdapter as StreamIngestAdapter,
    WebSocketIngestAdapter as WebSocketIngestAdapter,
    normalize_timestamp,
)

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Real-time stream processor for temporal data.

    Provides sliding, tumbling, and session windowing with aggregation,
    bounded watermarking, late data handling, automated sliding-window
    anomaly alert handlers, and WebSocket/Kafka stream ingest adapters.
    """

    def __init__(
        self,
        window_size: timedelta,
        slide_interval: timedelta | None = None,
        aggregation_func: Callable[[list[float]], float] | None = None,
        watermark_delay: timedelta | None = None,
        max_buffer_points: int = 10000,
        max_history_windows: int = 1000,
    ) -> None:
        """
        Initialize the stream processor.

        Args:
            window_size: Size of the processing window
            slide_interval: Interval for sliding windows (if None, uses window_size)
            aggregation_func: Optional aggregation function
            watermark_delay: Optional bounded watermarking delay (allowed lateness).
                The watermark advances to `max_timestamp - watermark_delay`.
            max_buffer_points: Capacity of each active and late-data buffer.
                Exhaustion raises BufferError before accepting another point.
            max_history_windows: Number of processed summaries to retain.
        """
        if not isinstance(window_size, timedelta):
            raise TypeError("window_size must be a timedelta")
        if window_size <= timedelta(0):
            raise ValueError("window_size must be greater than zero")
        if slide_interval is not None and not isinstance(slide_interval, timedelta):
            raise TypeError("slide_interval must be a timedelta")
        if slide_interval is not None and slide_interval <= timedelta(0):
            raise ValueError("slide_interval must be greater than zero")
        if watermark_delay is not None and not isinstance(watermark_delay, timedelta):
            raise TypeError("watermark_delay must be a timedelta")
        if watermark_delay is not None and watermark_delay < timedelta(0):
            raise ValueError("watermark_delay must be non-negative")
        if aggregation_func is not None and not callable(aggregation_func):
            raise TypeError("aggregation_func must be callable")

        for name, limit in (
            ("max_buffer_points", max_buffer_points),
            ("max_history_windows", max_history_windows),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"{name} must be an integer")
            if limit <= 0:
                raise ValueError(f"{name} must be positive")
        self.max_buffer_points = max_buffer_points
        self.max_history_windows = max_history_windows
        self.window_size = window_size
        self.slide_interval = (
            slide_interval if slide_interval is not None else window_size
        )
        self.aggregation_func = (
            aggregation_func if aggregation_func is not None else np.mean
        )
        self.watermark_delay = watermark_delay

        self.buffer: deque[dict[str, Any]] = deque()
        self.windows: list[dict[str, Any]] = []
        self._max_timestamp: datetime | None = None
        self._watermark: datetime | None = None
        self._late_data: list[dict[str, Any]] = []
        self._event_handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._anomaly_alert_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._stats: dict[str, int] = {
            "total_points": 0,
            "total_windows": 0,
            "late_arrivals": 0,
            "events_detected": 0,
            "anomaly_alerts": 0,
        }

    def add_data_point(
        self,
        timestamp: datetime,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a data point to the stream.

        Args:
            timestamp: Data point timestamp
            value: Data point value
            metadata: Optional metadata
        """
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if isinstance(value, bool):
            raise TypeError("value must be a finite number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("value must be a finite number") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("value must be finite")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        timestamp = normalize_timestamp(timestamp)
        point = {
            "timestamp": timestamp,
            "value": numeric_value,
            "metadata": dict(metadata) if metadata is not None else {},
        }

        # Evict by event time before checking capacity. Buffer ordering is stable
        # for equal timestamps and never depends on arrival order.
        newest = (
            max(timestamp, self._max_timestamp) if self._max_timestamp else timestamp
        )
        cutoff_time = newest - self.window_size
        retained = [item for item in self.buffer if item["timestamp"] >= cutoff_time]
        is_late = self._watermark is not None and timestamp < self._watermark
        target_size = len(self._late_data) if is_late else len(retained)
        if (
            is_late or timestamp >= cutoff_time
        ) and target_size >= self.max_buffer_points:
            raise BufferError(
                "Stream capacity reached; drain late data or advance event time"
            )
        self.buffer = deque(retained)

        # Check for late data (arrived strictly before current watermark)
        if self._watermark is not None and timestamp < self._watermark:
            self._late_data.append(point)
            self._stats["late_arrivals"] += 1
            logger.debug(
                "Late data point received: %s (watermark: %s)",
                timestamp.isoformat(),
                self._watermark.isoformat(),
            )
        elif timestamp >= cutoff_time:
            index = bisect_right(
                self.buffer, timestamp, key=lambda item: item["timestamp"]
            )
            self.buffer.insert(index, point)

        self._stats["total_points"] += 1

        # Track max timestamp observed
        if self._max_timestamp is None or timestamp > self._max_timestamp:
            self._max_timestamp = timestamp

        # Update watermark with bounded delay if configured
        if self.watermark_delay is not None:
            new_watermark = self._max_timestamp - self.watermark_delay
            if self._watermark is None or new_watermark > self._watermark:
                self._watermark = new_watermark
        else:
            if self._watermark is None or timestamp > self._watermark:
                self._watermark = timestamp

    async def ingest_adapter_stream(
        self,
        adapter: StreamIngestAdapter,
        max_messages: int | None = None,
        auto_process_windows: bool = False,
        **stream_kwargs: Any,
    ) -> int:
        """
        Ingest data continuously from a StreamIngestAdapter.

        Args:
            adapter: WebSocket, Kafka, replay, or custom StreamIngestAdapter.
            max_messages: Optional maximum messages to consume.
            auto_process_windows: If True, calls process_window() after each point.
            **stream_kwargs: Passed to adapter.stream_data().

        Returns:
            Number of points ingested.
        """
        if not isinstance(adapter, StreamIngestAdapter):
            raise TypeError("adapter must be an instance of StreamIngestAdapter")
        if max_messages is not None:
            if isinstance(max_messages, bool) or not isinstance(max_messages, int):
                raise TypeError("max_messages must be an integer")
            if max_messages < 0:
                raise ValueError("max_messages must be non-negative")
            if max_messages == 0:
                return 0

        ingested_count = 0
        async with aclosing(
            adapter.stream_data(max_messages=max_messages, **stream_kwargs)
        ) as records:
            async for record in records:
                ts, val, meta = adapter.parse_record(record)
                self.add_data_point(ts, val, meta)
                if auto_process_windows:
                    self.process_window()
                await adapter.acknowledge(record)
                ingested_count += 1
                if max_messages is not None and ingested_count >= max_messages:
                    break
        return ingested_count

    async def ingest_websocket_stream(
        self,
        url: str = "ws://localhost:8765",
        max_messages: int | None = None,
        adapter: WebSocketIngestAdapter | None = None,
        **kwargs: Any,
    ) -> int:
        """Ingest a real WebSocket source, optionally using a configured adapter.

        When supplied, the adapter owns its connection configuration; URL and
        transport kwargs are ignored. This method owns iteration cleanup.
        """
        if adapter is None:
            adapter = WebSocketIngestAdapter({"url": url, **kwargs})
        return await self.ingest_adapter_stream(adapter, max_messages=max_messages)

    async def ingest_kafka_stream(
        self,
        topic: str = "geo_infer_temporal_events",
        bootstrap_servers: str | list[str] | None = None,
        max_messages: int | None = None,
        adapter: KafkaIngestAdapter | None = None,
        **kwargs: Any,
    ) -> int:
        """Ingest and acknowledge Kafka records after successful processing.

        A supplied adapter owns its topic, servers and transport configuration.
        This method owns iteration cleanup.
        """
        if adapter is None:
            config: dict[str, Any] = {"topic": topic, **kwargs}
            if bootstrap_servers is not None:
                config["bootstrap_servers"] = bootstrap_servers
            adapter = KafkaIngestAdapter(config)
        return await self.ingest_adapter_stream(adapter, max_messages=max_messages)

    def process_window(self) -> dict[str, Any] | None:
        """
        Process the current window and return aggregated result.

        Returns:
            Aggregated window result or None if window is empty
        """
        if not self.buffer:
            return None

        window_data = list(self.buffer)
        values = [point["value"] for point in window_data]

        result = {
            "window_start": window_data[0]["timestamp"].isoformat(),
            "window_end": window_data[-1]["timestamp"].isoformat(),
            "count": len(window_data),
            "aggregated_value": float(self.aggregation_func(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }

        self.windows.append(result)
        del self.windows[: -self.max_history_windows]
        self._stats["total_windows"] += 1
        return result

    def get_recent_windows(self, count: int = 10) -> list[dict[str, Any]]:
        """
        Get recent processing windows.

        Args:
            count: Number of recent windows to return

        Returns:
            List of recent window results
        """
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []
        return self.windows[-count:]

    def process_tumbling_windows(self) -> list[dict[str, Any]]:
        """
        Process data using non-overlapping tumbling windows.

        Divides the buffered data into fixed-size, non-overlapping windows
        and aggregates each independently.

        Returns:
            List of tumbling window results.
        """
        if not self.buffer:
            return []

        all_points = sorted(self.buffer, key=lambda p: p["timestamp"])
        if not all_points:
            return []

        window_start = all_points[0]["timestamp"]
        results: list[dict[str, Any]] = []
        current_window: list[dict[str, Any]] = []

        for point in all_points:
            if point["timestamp"] >= window_start + self.window_size:
                # Emit current window
                if current_window:
                    results.append(self._aggregate_points(current_window))
                # Start new window aligned to boundary
                intervals = (point["timestamp"] - window_start) // self.window_size
                window_start += intervals * self.window_size
                current_window = []
            current_window.append(point)

        # Emit last window if non-empty
        if current_window:
            results.append(self._aggregate_points(current_window))

        return results

    def process_sliding_windows(self) -> list[dict[str, Any]]:
        """
        Process data using overlapping sliding windows.

        Slides a window of fixed size across the data at the configured
        slide interval, producing an aggregated result for each position.

        Returns:
            List of sliding window results.
        """
        if not self.buffer:
            return []

        all_points = sorted(self.buffer, key=lambda p: p["timestamp"])
        if not all_points:
            return []

        results: list[dict[str, Any]] = []
        window_start = all_points[0]["timestamp"]
        stream_end = all_points[-1]["timestamp"]

        while window_start <= stream_end:
            window_end = window_start + self.window_size
            window_points = [
                p for p in all_points if window_start <= p["timestamp"] < window_end
            ]

            if window_points:
                results.append(self._aggregate_points(window_points))

            window_start += self.slide_interval

        return results

    def process_session_windows(
        self,
        session_gap: timedelta,
    ) -> list[dict[str, Any]]:
        """
        Process data using session windows.

        Groups data points that arrive within a specified gap of each other
        into sessions. A new session starts when no data arrives for longer
        than session_gap.

        Args:
            session_gap: Maximum inactivity gap before starting a new session.

        Returns:
            List of session window results with session duration and bounds.
        """
        if not isinstance(session_gap, timedelta):
            raise TypeError("session_gap must be a timedelta")
        if session_gap < timedelta(0):
            raise ValueError("session_gap must be non-negative")

        if not self.buffer:
            return []

        all_points = sorted(self.buffer, key=lambda p: p["timestamp"])
        if not all_points:
            return []

        results: list[dict[str, Any]] = []
        current_session: list[dict[str, Any]] = [all_points[0]]

        for i in range(1, len(all_points)):
            gap = all_points[i]["timestamp"] - all_points[i - 1]["timestamp"]
            if gap > session_gap:
                # End current session, start new one
                res = self._aggregate_points(current_session)
                res["session_duration_seconds"] = (
                    current_session[-1]["timestamp"] - current_session[0]["timestamp"]
                ).total_seconds()
                results.append(res)
                current_session = []
            current_session.append(all_points[i])

        # Emit last session
        if current_session:
            res = self._aggregate_points(current_session)
            res["session_duration_seconds"] = (
                current_session[-1]["timestamp"] - current_session[0]["timestamp"]
            ).total_seconds()
            results.append(res)

        return results

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle (e.g. 'threshold_breach',
                'anomaly', 'trend_change', 'anomaly_alert')
            handler: Callback function that receives the event dict.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")

        self._event_handlers[event_type] = handler
        logger.info("Registered event handler for '%s'", event_type)

    def register_anomaly_alert_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Register an automated anomaly alert handler.

        Args:
            handler: Callback function receiving automated anomaly alert dictionaries.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")
        if handler not in self._anomaly_alert_handlers:
            self._anomaly_alert_handlers.append(handler)
        logger.info("Registered automated anomaly alert handler")

    def process_sliding_window_anomaly_alerts(
        self,
        z_threshold: float = 3.0,
        min_window_points: int = 3,
        auto_notify: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Slide across buffered windows and compute automated anomaly alerts.

        For each sliding window evaluation with sufficient data points,
        calculates statistical baseline metrics (mean, std, z-scores) and
        identifies points exceeding the alert threshold. Emits structured
        anomaly alert events to all registered alert handlers.

        Args:
            z_threshold: Z-score threshold for anomaly detection.
            min_window_points: Minimum data points required in a window.
            auto_notify: Whether to trigger registered anomaly alert handlers.

        Returns:
            List of generated anomaly alert dictionaries.
        """
        if isinstance(z_threshold, bool) or not isinstance(z_threshold, (int, float)):
            raise TypeError("z_threshold must be a number")
        if not math.isfinite(z_threshold) or z_threshold <= 0:
            raise ValueError("z_threshold must be finite and greater than zero")
        if isinstance(min_window_points, bool) or not isinstance(
            min_window_points, int
        ):
            raise TypeError("min_window_points must be an integer")
        if min_window_points < 2:
            raise ValueError("min_window_points must be at least 2")

        if len(self.buffer) < min_window_points:
            return []

        all_points = sorted(self.buffer, key=lambda p: p["timestamp"])
        window_start = all_points[0]["timestamp"]
        stream_end = all_points[-1]["timestamp"]
        alerts: list[dict[str, Any]] = []

        while window_start <= stream_end:
            window_end = window_start + self.window_size
            window_pts = [
                p for p in all_points if window_start <= p["timestamp"] < window_end
            ]

            if len(window_pts) >= min_window_points:
                vals = np.array([p["value"] for p in window_pts])
                mean = float(np.mean(vals))
                std = float(np.std(vals))

                if std > 1e-10:
                    for pt in window_pts:
                        z_val = abs(pt["value"] - mean) / std
                        if z_val > z_threshold:
                            alert = {
                                "type": "anomaly_alert",
                                "method": "sliding_window_zscore",
                                "window_start": window_start.isoformat(),
                                "window_end": window_end.isoformat(),
                                "timestamp": pt["timestamp"].isoformat(),
                                "value": pt["value"],
                                "z_score": round(float(z_val), 4),
                                "threshold": float(z_threshold),
                                "window_mean": round(mean, 4),
                                "window_std": round(std, 4),
                                "window_points_count": len(window_pts),
                                "metadata": pt.get("metadata", {}),
                            }
                            alerts.append(alert)
                            self._stats["anomaly_alerts"] += 1
                            self._stats["events_detected"] += 1

                            if auto_notify:
                                for hdl in self._anomaly_alert_handlers:
                                    try:
                                        hdl(alert)
                                    except Exception as err:
                                        logger.error(
                                            "Error executing anomaly alert handler: %s",
                                            err,
                                        )

            window_start += self.slide_interval

        return alerts

    def detect_threshold_events(
        self,
        upper_threshold: float | None = None,
        lower_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect threshold breach events in the current buffer.

        Args:
            upper_threshold: Upper value threshold.
            lower_threshold: Lower value threshold.

        Returns:
            List of detected threshold events.
        """
        for name, threshold in (
            ("upper_threshold", upper_threshold),
            ("lower_threshold", lower_threshold),
        ):
            if threshold is not None:
                if isinstance(threshold, bool):
                    raise TypeError(f"{name} must be a finite number")
                try:
                    numeric_threshold = float(threshold)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"{name} must be a finite number") from exc
                if not math.isfinite(numeric_threshold):
                    raise ValueError(f"{name} must be finite")

        upper_thresh = float(upper_threshold) if upper_threshold is not None else None
        lower_thresh = float(lower_threshold) if lower_threshold is not None else None

        if (
            upper_thresh is not None
            and lower_thresh is not None
            and lower_thresh > upper_thresh
        ):
            raise ValueError("lower_threshold cannot exceed upper_threshold")

        events: list[dict[str, Any]] = []

        for point in self.buffer:
            value = point["value"]
            ts = point["timestamp"]

            if upper_thresh is not None and value > upper_thresh:
                event = {
                    "type": "threshold_breach",
                    "direction": "upper",
                    "timestamp": ts.isoformat(),
                    "value": value,
                    "threshold": upper_thresh,
                }
                events.append(event)
                self._stats["events_detected"] += 1

                if "threshold_breach" in self._event_handlers:
                    self._event_handlers["threshold_breach"](event)

            if lower_thresh is not None and value < lower_thresh:
                event = {
                    "type": "threshold_breach",
                    "direction": "lower",
                    "timestamp": ts.isoformat(),
                    "value": value,
                    "threshold": lower_thresh,
                }
                events.append(event)
                self._stats["events_detected"] += 1

                if "threshold_breach" in self._event_handlers:
                    self._event_handlers["threshold_breach"](event)

        return events

    def detect_anomalies_zscore(
        self,
        z_threshold: float = 3.0,
    ) -> list[dict[str, Any]]:
        """
        Detect anomalous data points using z-score on the current buffer.

        Args:
            z_threshold: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomaly events.
        """
        if isinstance(z_threshold, bool) or not isinstance(z_threshold, (int, float)):
            raise TypeError("z_threshold must be a number")
        if not math.isfinite(z_threshold) or z_threshold <= 0:
            raise ValueError("z_threshold must be finite and greater than zero")

        if len(self.buffer) < 3:
            return []

        values = np.array([p["value"] for p in self.buffer])
        mean = float(np.mean(values))
        std = float(np.std(values))

        if std < 1e-10:
            return []

        anomalies: list[dict[str, Any]] = []

        for point in self.buffer:
            z_score = abs(point["value"] - mean) / std
            if z_score > z_threshold:
                event = {
                    "type": "anomaly",
                    "method": "zscore",
                    "timestamp": point["timestamp"].isoformat(),
                    "value": point["value"],
                    "z_score": round(float(z_score), 4),
                    "mean": round(float(mean), 4),
                    "std": round(float(std), 4),
                }
                anomalies.append(event)
                self._stats["events_detected"] += 1

                if "anomaly" in self._event_handlers:
                    self._event_handlers["anomaly"](event)

        return anomalies

    def get_watermark(self) -> datetime | None:
        """
        Get the current watermark timestamp.

        The watermark represents the completeness boundary timestamp;
        data points arriving earlier than the watermark are treated as late data.

        Returns:
            Current watermark or None if no data received.
        """
        return self._watermark

    def get_late_data(self) -> list[dict[str, Any]]:
        """
        Get all late-arriving data points.

        Returns:
            List of data points that arrived after the watermark.
        """
        return list(self._late_data)

    def flush_late_data(self) -> list[dict[str, Any]]:
        """
        Flush late data and return it, clearing the late buffer.

        Returns:
            List of late data points that were cleared.
        """
        flushed = list(self._late_data)
        self._late_data.clear()
        return flushed

    def get_buffer_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current buffer state.

        Returns:
            Dictionary with buffer statistics.
        """
        if not self.buffer:
            return {
                "size": 0,
                "window_start": None,
                "window_end": None,
                "watermark": self._watermark.isoformat() if self._watermark else None,
                "watermark_delay_seconds": self.watermark_delay.total_seconds()
                if self.watermark_delay
                else None,
            }

        values = [p["value"] for p in self.buffer]
        return {
            "size": len(self.buffer),
            "window_start": self.buffer[0]["timestamp"].isoformat(),
            "window_end": self.buffer[-1]["timestamp"].isoformat(),
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "watermark": self._watermark.isoformat() if self._watermark else None,
            "watermark_delay_seconds": self.watermark_delay.total_seconds()
            if self.watermark_delay
            else None,
            "late_data_count": len(self._late_data),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get stream processing statistics.

        Returns:
            Dictionary of cumulative statistics.
        """
        return dict(self._stats)

    def reset(self) -> None:
        """Reset the stream processor, clearing all buffers and state."""
        self.buffer.clear()
        self.windows.clear()
        self._max_timestamp = None
        self._watermark = None
        self._late_data.clear()
        self._stats = {
            "total_points": 0,
            "total_windows": 0,
            "late_arrivals": 0,
            "events_detected": 0,
            "anomaly_alerts": 0,
        }
        logger.info("StreamProcessor reset")

    def _aggregate_points(
        self,
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Aggregate a list of data points into a window result.

        Args:
            points: List of data point dicts.

        Returns:
            Aggregated window result dict.
        """
        values = [p["value"] for p in points]
        return {
            "window_start": points[0]["timestamp"].isoformat(),
            "window_end": points[-1]["timestamp"].isoformat(),
            "count": len(points),
            "aggregated_value": float(self.aggregation_func(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
