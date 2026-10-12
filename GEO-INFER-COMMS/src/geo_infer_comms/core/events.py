"""
Event-driven communication system for GEO-INFER-COMMS.

This module implements comprehensive event-driven communication including
event publishing, subscription management, event processing, and real-time
coordination with geospatial context and filtering capabilities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, cast
from collections.abc import Callable
import logging
import threading
import time
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
import uuid
import queue

from geo_infer_comms.models.message import (
    EventPublishRequest,
    EventPublishResponse,
    EventSubscriptionRequest,
    EventSubscriptionResponse,
    MessagePriority,
)
from geo_infer_comms.models.spatial import GeospatialPoint, SpatialIndex
from geo_infer_comms.utils.validation import validate_event_type, validate_url


def _extract_geospatial_coord(ctx: Any, key: str) -> float | None:
    """Extract a latitude/longitude from geospatial context.

    Accepts either a flat dict ``{key: value}``, a dict with a nested
    ``location`` dict, or an object exposing the coordinate as an attribute.
    Returns ``None`` when the coordinate is not available.
    """
    if isinstance(ctx, dict):
        value = ctx.get(key)
        if value is None and isinstance(ctx.get("location"), dict):
            value = ctx["location"].get(key)
        return cast(float | None, value)
    return cast(float | None, getattr(ctx, key, None))


def _event_location_from_context(ctx: Any) -> GeospatialPoint | None:
    """Extract a :class:`GeospatialPoint` from an event geospatial context.

    Accepts both a flat dict ``{"latitude": ..., "longitude": ...}`` and a
    nested ``{"location": {"latitude": ..., "longitude": ...}}`` shape (the
    same shapes :func:`_extract_geospatial_coord` accepts), or an object
    exposing the coordinates as attributes.

    Returns ``None`` when the context carries no coordinate data at all.

    Raises:
        ValueError: When coordinates are present but incomplete or not
            convertible to floats. Raised before any state is mutated so a
            malformed context can never leave events/metrics inconsistent.
    """
    if ctx is None:
        return None
    latitude = _extract_geospatial_coord(ctx, "latitude")
    longitude = _extract_geospatial_coord(ctx, "longitude")
    if latitude is None and longitude is None:
        return None
    if latitude is None or longitude is None:
        raise ValueError(
            "geo_infer_comms.events: geospatial context supplies only one of "
            "latitude/longitude; both are required for spatial indexing"
        )
    try:
        return GeospatialPoint(longitude=float(longitude), latitude=float(latitude))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"geo_infer_comms.events: invalid coordinates in geospatial "
            f"context ({latitude!r}, {longitude!r}): {exc}"
        ) from exc


class EventManager:
    """
    Central event management system.

    Handles event publishing, subscription management, and real-time
    event processing with geospatial filtering and priority handling.
    """

    def __init__(
        self,
        max_events: int = 10000,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        self.max_events = max_events
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Event storage and processing
        self.events: dict[str, EventPublishResponse] = {}
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_events)
        self.spatial_index = SpatialIndex()

        # Subscription management
        self.subscriptions: dict[str, EventSubscriptionResponse] = {}
        self.subscriber_callbacks: dict[str, list[Callable]] = {}
        self.subscription_callbacks: dict[str, Callable] = {}
        self.event_type_subscribers: dict[str, set[str]] = {}

        # Event processing
        self.event_processors: dict[str, EventProcessor] = {}
        self.event_history: list[EventPublishResponse] = []

        # Threading and concurrency
        self._lock = threading.RLock()
        self._processing_thread: threading.Thread | None = None
        self._running = False

        # Metrics and monitoring
        self.metrics = EventMetrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Register default event processors
        self._register_default_processors()

    def start(self) -> None:
        """Start the event manager."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._processing_thread = threading.Thread(
                target=self._process_events, daemon=True
            )
            self._processing_thread.start()
            self.logger.info("Event manager started")

    def stop(self) -> None:
        """Stop the event manager."""
        with self._lock:
            self._running = False
            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)
            self.logger.info("Event manager stopped")

    def publish_event(self, request: EventPublishRequest) -> EventPublishResponse:
        """
        Publish an event to the system.

        Args:
            request: Event publication request

        Returns:
            Event publication response

        Raises:
            ValueError: If event request is invalid
        """
        if not self._running:
            raise RuntimeError("Event manager is not running")

        # Validate event type
        if not validate_event_type(request.event_type):
            raise ValueError(f"Invalid event type: {request.event_type}")

        # Validate and extract the location from the geospatial context BEFORE
        # any state mutation, so malformed contexts cannot leave events,
        # metrics, or the spatial index inconsistent.
        location = _event_location_from_context(request.geospatial_context)

        # Create event response
        event = EventPublishResponse(
            event_type=request.event_type,
            payload=request.payload,
            source=request.source,
            target_channels=request.target_channels,
            priority=request.priority,
            geospatial_context=request.geospatial_context,
        )

        # Store event
        with self._lock:
            self.events[event.event_id] = event

            # Add to spatial index if a valid location was provided. Both the
            # nested {"location": {...}} and flat {"latitude", "longitude"}
            # context shapes are indexed.
            if location is not None:
                self.spatial_index.insert(location, event.event_id)
            # Queue for processing
            priority_value = self._get_priority_value(request.priority)
            self.event_queue.put((priority_value, event.event_id, event))

        self.metrics.events_published += 1
        self.logger.info(f"Event published: {event.event_id}")
        return event

    def subscribe_to_events(
        self,
        subscriber_id: str,
        request: EventSubscriptionRequest,
        callback: Callable[[EventPublishResponse], None],
    ) -> str:
        """
        Subscribe to events with filtering.

        Args:
            subscriber_id: Unique subscriber identifier
            request: Event subscription configuration
            callback: Function to call when events are received

        Returns:
            Subscription ID for unsubscribing
        """
        subscription = EventSubscriptionResponse(
            event_types=request.event_types,
            filter_criteria=request.filter_criteria,
            delivery_mode=request.delivery_mode,
            callback_url=request.callback_url,
        )

        subscription_id = f"evt_sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            self.subscriptions[subscription_id] = subscription

            # Add to subscriber callbacks
            if subscriber_id not in self.subscriber_callbacks:
                self.subscriber_callbacks[subscriber_id] = []
            self.subscriber_callbacks[subscriber_id].append(callback)
            self.subscription_callbacks[subscription_id] = callback
            # Update event type mappings
            for event_type in request.event_types:
                if event_type not in self.event_type_subscribers:
                    self.event_type_subscribers[event_type] = set()
                self.event_type_subscribers[event_type].add(subscriber_id)

        self.logger.info(
            f"Event subscription created: {subscription_id} for {subscriber_id}"
        )
        return subscription_id

    def unsubscribe_from_events(self, subscriber_id: str, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscriber_id: Subscriber identifier
            subscription_id: Subscription to remove

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            if subscription_id not in self.subscriptions:
                return False

            # Remove the callback registered for this exact subscription
            callback = self.subscription_callbacks.pop(subscription_id, None)
            if callback is not None and subscriber_id in self.subscriber_callbacks:
                callbacks = self.subscriber_callbacks[subscriber_id]
                if callback in callbacks:
                    callbacks.remove(callback)
                if not callbacks:
                    del self.subscriber_callbacks[subscriber_id]
            # Remove subscription
            del self.subscriptions[subscription_id]

        self.logger.info(f"Event subscription removed: {subscription_id}")
        return True

    def register_event_processor(
        self, event_type: str, processor: EventProcessor
    ) -> None:
        """Register an event processor for a specific event type."""
        self.event_processors[event_type] = processor
        self.logger.info(f"Event processor registered for type: {event_type}")

    def get_events(
        self,
        event_type: str | None = None,
        source: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[EventPublishResponse]:
        """
        Get events with filtering.

        Args:
            event_type: Filter by event type
            source: Filter by event source
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            events = list(self.events.values())

        # Apply filters
        filtered_events = events

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if source:
            filtered_events = [e for e in filtered_events if e.source == source]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered_events[:limit]

    def get_event_statistics(self) -> dict[str, Any]:
        """Get event system statistics."""
        with self._lock:
            event_type_counts: dict[str, int] = {}
            for event in self.events.values():
                event_type_counts[event.event_type] = (
                    event_type_counts.get(event.event_type, 0) + 1
                )

            return {
                "total_events": len(self.events),
                "total_subscriptions": len(self.subscriptions),
                "event_types": list(set(e.event_type for e in self.events.values())),
                "event_type_counts": event_type_counts,
                "metrics": self.metrics.to_dict(),
            }

    def _process_events(self) -> None:
        """Background thread to process events."""
        while self._running:
            try:
                # Get next event from queue (with timeout)
                try:
                    priority, event_id, event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process event
                self._handle_event(event)

                # Mark task as done
                self.event_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                time.sleep(1.0)  # Brief pause before retrying

    def _handle_event(self, event: EventPublishResponse) -> None:
        """Handle event processing and delivery."""
        try:
            # Process with registered processor
            processor = self.event_processors.get(event.event_type)
            if processor:
                processor.process_event(event)

            # Deliver to subscribers
            self._deliver_event_to_subscribers(event)

            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > 1000:  # Keep recent events
                self.event_history = self.event_history[-1000:]

            self.metrics.events_processed += 1

        except Exception as e:
            self.logger.error(f"Error handling event {event.event_id}: {e}")
            self.metrics.processing_failures += 1

    def _deliver_event_to_subscribers(self, event: EventPublishResponse) -> None:
        """Deliver event to matching subscribers."""
        # Find subscribers for this event type
        subscriber_ids = self.event_type_subscribers.get(event.event_type, set())

        for subscriber_id in subscriber_ids:
            # Get subscriber callbacks
            callbacks = self.subscriber_callbacks.get(subscriber_id, [])

            for callback in callbacks:
                try:
                    # In a real implementation, would use threading or async
                    callback(event)
                except Exception as e:
                    self.logger.error(
                        f"Error delivering event to subscriber {subscriber_id}: {e}"
                    )

    def _get_priority_value(self, priority: MessagePriority) -> int:
        """Convert priority to queue priority value (lower = higher priority)."""
        priority_map = {
            MessagePriority.URGENT: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 4,
        }
        return priority_map.get(priority, 3)

    def _register_default_processors(self) -> None:
        """Register default event processors."""
        self.event_processors.update(
            {
                "data_update": DataUpdateProcessor(),
                "system_alert": SystemAlertProcessor(),
                "user_action": UserActionProcessor(),
                "sensor_trigger": SensorTriggerProcessor(),
                "geospatial_change": GeospatialChangeProcessor(),
            }
        )


@dataclass
class EventMetrics:
    """Metrics for event system performance."""

    events_published: int = 0
    events_processed: int = 0
    events_delivered: int = 0
    processing_failures: int = 0
    subscriptions_created: int = 0
    subscriptions_removed: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = datetime.now(UTC) - self.start_time
        return {
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "events_delivered": self.events_delivered,
            "processing_failures": self.processing_failures,
            "subscriptions_created": self.subscriptions_created,
            "subscriptions_removed": self.subscriptions_removed,
            "processing_success_rate": (
                self.events_processed / max(self.events_published, 1) * 100
            ),
            "uptime_seconds": uptime.total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.events_published = 0
        self.events_processed = 0
        self.events_delivered = 0
        self.processing_failures = 0
        self.subscriptions_created = 0
        self.subscriptions_removed = 0
        self.start_time = datetime.now(UTC)


class EventProcessor(ABC):
    """Base class for event processors."""

    @abstractmethod
    def process_event(self, event: EventPublishResponse) -> None:
        """
        Process an event.

        Args:
            event: Event to process
        """
        raise RuntimeError("Event subclasses must implement process_event")


class DataUpdateProcessor(EventProcessor):
    """Processor for data update events."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def process_event(self, event: EventPublishResponse) -> None:
        """Process data update events."""
        payload = event.payload
        # In a real implementation, would handle data updates
        self.logger.info(
            f"Processing data update: {payload.get('dataset_id', 'unknown')}"
        )


class SystemAlertProcessor(EventProcessor):
    """Processor for system alert events."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def process_event(self, event: EventPublishResponse) -> None:
        """Process system alert events."""
        payload = event.payload
        alert_level = payload.get("alert_level", "info")
        # In a real implementation, would handle system alerts
        self.logger.info(
            f"Processing system alert ({alert_level}): {payload.get('message', '')}"
        )


class UserActionProcessor(EventProcessor):
    """Processor for user action events."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def process_event(self, event: EventPublishResponse) -> None:
        """Process user action events."""
        payload = event.payload
        action_type = payload.get("action_type", "unknown")
        # In a real implementation, would handle user actions
        self.logger.info(f"Processing user action: {action_type}")


class SensorTriggerProcessor(EventProcessor):
    """Processor for sensor trigger events."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def process_event(self, event: EventPublishResponse) -> None:
        """Process sensor trigger events."""
        payload = event.payload
        sensor_id = payload.get("sensor_id", "unknown")
        trigger_value = payload.get("trigger_value")
        # In a real implementation, would handle sensor triggers
        self.logger.info(f"Processing sensor trigger: {sensor_id} = {trigger_value}")


class GeospatialChangeProcessor(EventProcessor):
    """Processor for geospatial change events."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def process_event(self, event: EventPublishResponse) -> None:
        """Process geospatial change events."""
        payload = event.payload
        change_type = payload.get("change_type", "unknown")
        location = payload.get("location", {})
        # In a real implementation, would handle geospatial changes
        self.logger.info(f"Processing geospatial change ({change_type}): {location}")


class EventFilter:
    """Advanced event filtering capabilities."""

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.custom_filters: dict[str, Callable] = {}

        self.logger = logging.getLogger(__name__)

    def register_filter(self, filter_name: str, filter_func: Callable) -> None:
        """Register a custom event filter."""
        self.custom_filters[filter_name] = filter_func
        self.logger.info(f"Registered custom filter: {filter_name}")

    def apply_filters(
        self, event: EventPublishResponse, filters: list[dict[str, Any]]
    ) -> bool:
        """Apply multiple filters to an event."""
        for filter_config in filters:
            if not self._apply_single_filter(event, filter_config):
                return False
        return True

    def _apply_single_filter(
        self, event: EventPublishResponse, filter_config: dict[str, Any]
    ) -> bool:
        """Apply a single filter to an event."""
        filter_type = filter_config.get("type", "basic")

        if filter_type == "basic":
            return self._apply_basic_filter(event, filter_config)
        elif filter_type == "geospatial":
            return self._apply_geospatial_filter(event, filter_config)
        elif filter_type == "temporal":
            return self._apply_temporal_filter(event, filter_config)
        elif filter_type == "custom":
            return self._apply_custom_filter(event, filter_config)
        else:
            self.logger.warning(f"Unknown filter type: {filter_type}")
            return True

    def _apply_basic_filter(
        self, event: EventPublishResponse, filter_config: dict[str, Any]
    ) -> bool:
        """Apply basic event filters."""
        # Filter by event type
        allowed_types = filter_config.get("event_types", [])
        if allowed_types and event.event_type not in allowed_types:
            return False

        # Filter by source
        allowed_sources = filter_config.get("sources", [])
        if allowed_sources and event.source not in allowed_sources:
            return False

        # Filter by priority
        min_priority = filter_config.get("min_priority")
        if min_priority:
            priority_values = {"low": 1, "normal": 2, "high": 3, "urgent": 4}
            event_priority = priority_values.get(event.priority.value, 2)
            if event_priority < priority_values.get(min_priority, 1):
                return False

        return True

    def _apply_geospatial_filter(
        self, event: EventPublishResponse, filter_config: dict[str, Any]
    ) -> bool:
        """Apply geospatial filters to events using bounding-box and radius checks.

        Supported filter_config keys:
        - ``require_location`` (bool): return False when event has no context
        - ``bbox`` (dict): {min_lon, max_lon, min_lat, max_lat}
        - ``radius_km`` + ``center`` (dict {lat, lon}): circular area filter
        """
        if not event.geospatial_context:
            return not filter_config.get("require_location", False)

        ctx = event.geospatial_context
        lat, lon = (
            _extract_geospatial_coord(ctx, "latitude"),
            _extract_geospatial_coord(ctx, "longitude"),
        )

        # Bounding-box check
        bbox = filter_config.get("bbox")
        if bbox:
            if not (
                bbox.get("min_lon", -180) <= lon <= bbox.get("max_lon", 180)
                and bbox.get("min_lat", -90) <= lat <= bbox.get("max_lat", 90)
            ):
                return False

        # Circular radius check via the shared haversine implementation
        radius_km = filter_config.get("radius_km")
        center = filter_config.get("center")
        if radius_km is not None and center:
            point = GeospatialPoint(longitude=lon, latitude=lat)
            center_point = GeospatialPoint(
                longitude=cast(float, center["lon"]),
                latitude=cast(float, center["lat"]),
            )
            dist_km = point.distance_to(center_point) / 1000.0
            if dist_km > radius_km:
                return False

        return True

    def _apply_temporal_filter(
        self, event: EventPublishResponse, filter_config: dict[str, Any]
    ) -> bool:
        """Apply temporal filters to events."""
        # Filter by time range
        start_time = filter_config.get("start_time")
        end_time = filter_config.get("end_time")

        if start_time and event.timestamp < start_time:
            return False

        if end_time and event.timestamp > end_time:
            return False

        # Filter by recurring patterns (daily, weekly, etc.)
        schedule = filter_config.get("schedule")
        if schedule:
            # In a real implementation, would check against schedule
            return True

        return True

    def _apply_custom_filter(
        self, event: EventPublishResponse, filter_config: dict[str, Any]
    ) -> bool:
        """Apply custom filters."""
        filter_name = filter_config.get("filter_name")
        if filter_name in self.custom_filters:
            return cast(bool, self.custom_filters[filter_name](event, filter_config))
        else:
            self.logger.warning(f"Custom filter not found: {filter_name}")
            return True


class EventScheduler:
    """
    Advanced event scheduling and timing system.

    Provides sophisticated scheduling capabilities for events including
    recurring events, delayed delivery, and time-based triggering.
    """

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.scheduled_events: dict[str, ScheduledEvent] = {}
        self.recurring_events: dict[str, RecurringEvent] = {}

        self._scheduler_thread: threading.Thread | None = None
        self._running = False

        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start the event scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._process_scheduled_events, daemon=True
        )
        self._scheduler_thread.start()
        self.logger.info("Event scheduler started")

    def stop(self) -> None:
        """Stop the event scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self.logger.info("Event scheduler stopped")

    def schedule_event(
        self,
        event_request: EventPublishRequest,
        schedule_time: datetime,
        schedule_id: str | None = None,
    ) -> str:
        """Schedule an event for future publication."""
        if schedule_id is None:
            schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"

        scheduled_event = ScheduledEvent(
            schedule_id=schedule_id,
            event_request=event_request,
            schedule_time=schedule_time,
            status="scheduled",
        )

        self.scheduled_events[schedule_id] = scheduled_event

        self.logger.info(f"Event scheduled: {schedule_id} at {schedule_time}")
        return schedule_id

    def schedule_recurring_event(
        self,
        event_request: EventPublishRequest,
        schedule_config: dict[str, Any],
        recurring_id: str | None = None,
    ) -> str:
        """Schedule a recurring event."""
        if recurring_id is None:
            recurring_id = f"recurring_{uuid.uuid4().hex[:8]}"

        recurring_event = RecurringEvent(
            recurring_id=recurring_id,
            event_request=event_request,
            schedule_config=schedule_config,
            status="active",
        )

        self.recurring_events[recurring_id] = recurring_event

        self.logger.info(f"Recurring event scheduled: {recurring_id}")
        return recurring_id

    def cancel_scheduled_event(self, schedule_id: str) -> bool:
        """Cancel a scheduled event."""
        if schedule_id in self.scheduled_events:
            self.scheduled_events[schedule_id].status = "cancelled"
            self.logger.info(f"Scheduled event cancelled: {schedule_id}")
            return True
        return False

    def cancel_recurring_event(self, recurring_id: str) -> bool:
        """Cancel a recurring event."""
        if recurring_id in self.recurring_events:
            self.recurring_events[recurring_id].status = "cancelled"
            self.logger.info(f"Recurring event cancelled: {recurring_id}")
            return True
        return False

    def _process_scheduled_events(self) -> None:
        """Background thread to process scheduled events."""
        while self._running:
            try:
                current_time = datetime.now(UTC)

                # Process scheduled events
                for schedule_id, scheduled_event in list(self.scheduled_events.items()):
                    if (
                        scheduled_event.status == "scheduled"
                        and scheduled_event.schedule_time <= current_time
                    ):
                        try:
                            self.event_manager.publish_event(
                                scheduled_event.event_request
                            )
                            scheduled_event.status = "completed"
                        except Exception as e:
                            self.logger.error(
                                f"Failed to publish scheduled event {schedule_id}: {e}"
                            )
                            scheduled_event.status = "failed"

                # Process recurring events
                for recurring_id, recurring_event in list(
                    self.recurring_events.items()
                ):
                    if recurring_event.status == "active":
                        if recurring_event._should_trigger_recurring(current_time):
                            try:
                                self.event_manager.publish_event(
                                    recurring_event.event_request
                                )
                                recurring_event.last_triggered = current_time
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to publish recurring event {recurring_id}: {e}"
                                )

                # Brief pause before next check
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error processing scheduled events: {e}")
                time.sleep(5.0)  # Longer pause on error


@dataclass
class ScheduledEvent:
    """Represents a scheduled event."""

    schedule_id: str
    event_request: EventPublishRequest
    schedule_time: datetime
    status: str = "scheduled"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RecurringEvent:
    """Represents a recurring event."""

    recurring_id: str
    event_request: EventPublishRequest
    schedule_config: dict[str, Any]
    status: str = "active"
    last_triggered: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def _should_trigger_recurring(self, current_time: datetime) -> bool:
        """Check if recurring event should be triggered."""
        schedule_config = self.schedule_config

        # Check interval-based recurrence
        interval_seconds = schedule_config.get("interval_seconds")
        if interval_seconds and self.last_triggered:
            next_trigger = self.last_triggered + timedelta(seconds=interval_seconds)
            return current_time >= next_trigger

        # Check cron-like schedule (simplified)
        cron_schedule = schedule_config.get("cron_schedule")
        if cron_schedule:
            # In a real implementation, would parse cron and check
            return True

        return False


class EventWebhookManager:
    """
    Webhook management for external event delivery.

    Handles webhook registration, validation, and delivery for
    external systems that need to receive events.
    """

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.webhooks: dict[str, WebhookConfig] = {}
        self.webhook_history: list[WebhookDelivery] = []

        self.logger = logging.getLogger(__name__)

    def register_webhook(self, webhook_id: str, config: WebhookConfig) -> bool:
        """Register a webhook for event delivery."""
        if not validate_url(config.url):
            raise ValueError(f"Invalid webhook URL: {config.url}")

        self.webhooks[webhook_id] = config
        self.logger.info(f"Webhook registered: {webhook_id}")
        return True

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self.logger.info(f"Webhook unregistered: {webhook_id}")
            return True
        return False

    def deliver_to_webhook(self, webhook_id: str, event: EventPublishResponse) -> bool:
        """Deliver event to a specific webhook."""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return False

        try:
            # In a real implementation, would make HTTP request to webhook URL
            # For now, just log the delivery
            self.logger.info(f"Webhook delivery to {webhook_id}: {event.event_id}")

            # Record delivery
            delivery = WebhookDelivery(
                webhook_id=webhook_id,
                event_id=event.event_id,
                status="delivered",
                timestamp=datetime.now(UTC),
            )

            self.webhook_history.append(delivery)

            return True

        except Exception as e:
            self.logger.error(f"Webhook delivery failed for {webhook_id}: {e}")

            # Record failed delivery
            delivery = WebhookDelivery(
                webhook_id=webhook_id,
                event_id=event.event_id,
                status="failed",
                error=str(e),
                timestamp=datetime.now(UTC),
            )

            self.webhook_history.append(delivery)
            return False


@dataclass
class WebhookConfig:
    """Configuration for a webhook."""

    url: str
    event_types: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    secret: str | None = None
    retry_policy: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    webhook_id: str
    event_id: str
    status: str
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
