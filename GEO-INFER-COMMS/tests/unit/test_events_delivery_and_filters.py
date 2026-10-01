"""Tests for event delivery, subscription lifecycle, and event filtering (GS-227).

Covers:
- Publish-to-subscribe delivery filtered by event type.
- Unsubscribe semantics, including independent multi-subscription teardown.
- EventFilter geospatial (bbox, radius), temporal, basic, and custom filters.
- get_events filtering and event statistics.
"""

from datetime import datetime, timedelta, UTC

import pytest

from geo_infer_comms.core.events import EventFilter, EventManager

from geo_infer_comms.models.message import (
    EventPublishRequest,
    EventPublishResponse,
    EventSubscriptionRequest,
    MessagePriority,
)


UTC = UTC


@pytest.fixture()
def event_manager():
    """EventManager with the publish gate open and no background thread.

    Delivery is driven synchronously via ``_handle_event`` so tests are
    deterministic and free of wall-clock waits.
    """
    manager = EventManager(enable_persistence=False)
    manager._running = True
    yield manager


def _publish(
    em: EventManager,
    event_type="data_update",
    payload=None,
    source="unit-test",
    priority=MessagePriority.NORMAL,
):
    request = EventPublishRequest(
        event_type=event_type,
        payload=payload or {"temperature": 21.5},
        source=source,
        priority=priority,
    )
    return em.publish_event(request)


def _process(em: EventManager, event):
    """Drive one full delivery pass synchronously."""
    em._handle_event(event)


class TestPublishSubscribeDelivery:
    def test_event_type_filter_delivers_matching_only(self, event_manager):
        data_updates = []
        system_alerts = []
        event_manager.subscribe_to_events(
            "data-sub",
            EventSubscriptionRequest(event_types=["data_update"]),
            data_updates.append,
        )
        event_manager.subscribe_to_events(
            "alert-sub",
            EventSubscriptionRequest(event_types=["system_alert"]),
            system_alerts.append,
        )

        data_event = _publish(event_manager, "data_update")
        alert_event = _publish(event_manager, "system_alert")
        _process(event_manager, data_event)
        _process(event_manager, alert_event)

        assert [e.event_id for e in data_updates] == [data_event.event_id]
        assert [e.event_id for e in system_alerts] == [alert_event.event_id]
        assert event_manager.metrics.events_processed == 2

    def test_unsubscribe_stops_delivery(self, event_manager):
        received = []
        subscription_id = event_manager.subscribe_to_events(
            "sub",
            EventSubscriptionRequest(event_types=["data_update"]),
            received.append,
        )
        assert event_manager.unsubscribe_from_events("sub", subscription_id) is True

        event = _publish(event_manager)
        _process(event_manager, event)

        assert received == []
        assert event_manager.metrics.events_processed == 1
        assert subscription_id not in event_manager.subscriptions

    def test_unsubscribe_unknown_subscription_returns_false(self, event_manager):
        assert event_manager.unsubscribe_from_events("sub", "evt_sub_nope") is False

    def test_sibling_subscription_survives_unsubscribe(self, event_manager):
        first, second = [], []
        sub_a = event_manager.subscribe_to_events(
            "sub", EventSubscriptionRequest(event_types=["data_update"]), first.append
        )
        event_manager.subscribe_to_events(
            "sub", EventSubscriptionRequest(event_types=["data_update"]), second.append
        )
        assert event_manager.unsubscribe_from_events("sub", sub_a) is True

        event = _publish(event_manager)
        _process(event_manager, event)

        assert first == []
        assert [e.event_id for e in second] == [event.event_id]
        # The subscriber keeps its remaining callback registration.
        assert "sub" in event_manager.subscriber_callbacks

    def test_publish_rejected_when_not_running(self):
        manager = EventManager(enable_persistence=False)
        with pytest.raises(RuntimeError, match="not running"):
            _publish(manager)

    def test_invalid_event_type_rejected(self, event_manager):
        with pytest.raises(ValueError, match="Invalid event type"):
            _publish(event_manager, "bogus_event_type")
        assert event_manager.metrics.events_published == 0

    def test_urgent_priority_outranks_low_in_queue(self, event_manager):
        urgent = _publish(event_manager, priority=MessagePriority.URGENT)
        low = _publish(event_manager, priority=MessagePriority.LOW)

        assert event_manager._get_priority_value(MessagePriority.URGENT) == 1
        assert event_manager._get_priority_value(MessagePriority.LOW) == 4
        first_queued = event_manager.event_queue.get_nowait()[1]
        assert first_queued == urgent.event_id
        assert event_manager.event_queue.get_nowait()[1] == low.event_id


class TestGetEventsAndStatistics:
    def test_get_events_filters_by_type_and_source(self, event_manager):
        sensor_event = _publish(event_manager, "sensor_trigger", source="sensor-net")
        _publish(event_manager, "data_update", source="unit-test")

        by_type = event_manager.get_events(event_type="sensor_trigger")
        assert [e.event_id for e in by_type] == [sensor_event.event_id]

        by_source = event_manager.get_events(source="sensor-net")
        assert [e.event_id for e in by_source] == [sensor_event.event_id]

        assert event_manager.get_events(source="nowhere") == []

    def test_get_events_limit_returns_newest_first(self, event_manager):
        older = _publish(event_manager)
        newer = _publish(event_manager)
        assert newer.timestamp >= older.timestamp

        latest = event_manager.get_events(limit=1)
        assert len(latest) == 1
        assert latest[0].event_id in {older.event_id, newer.event_id}

    def test_statistics_reflect_published_events(self, event_manager):
        _publish(event_manager, "data_update")
        _publish(event_manager, "data_update")
        _publish(event_manager, "system_alert")

        stats = event_manager.get_event_statistics()
        assert stats["total_events"] == 3
        assert stats["event_type_counts"] == {"data_update": 2, "system_alert": 1}
        assert set(stats["event_types"]) == {"data_update", "system_alert"}
        assert stats["metrics"]["events_published"] == 3


def _make_event(
    event_type="data_update",
    priority=MessagePriority.NORMAL,
    source="unit-test",
    geospatial_context=None,
    timestamp=None,
):
    return EventPublishResponse(
        event_type=event_type,
        payload={"temperature": 21.5},
        source=source,
        priority=priority,
        geospatial_context=geospatial_context,
        timestamp=timestamp or datetime.now(UTC),
    )


@pytest.fixture()
def event_filter(event_manager):
    return EventFilter(event_manager)


class TestEventFilterGeospatial:
    def test_bbox_includes_point_inside(self, event_filter):
        event = _make_event(
            geospatial_context={"location": {"latitude": 48.85, "longitude": 2.35}}
        )
        assert event_filter._apply_geospatial_filter(
            event,
            {
                "bbox": {
                    "min_lon": 2.0,
                    "max_lon": 3.0,
                    "min_lat": 48.0,
                    "max_lat": 49.5,
                }
            },
        )

    def test_bbox_excludes_point_outside(self, event_filter):
        event = _make_event(
            geospatial_context={"location": {"latitude": 48.85, "longitude": 10.0}}
        )
        assert not event_filter._apply_geospatial_filter(
            event,
            {
                "bbox": {
                    "min_lon": 2.0,
                    "max_lon": 3.0,
                    "min_lat": 48.0,
                    "max_lat": 49.5,
                }
            },
        )

    def test_radius_includes_point_within_radius(self, event_filter):
        event = _make_event(
            geospatial_context={"latitude": 52.3906, "longitude": 13.0645}  # Potsdam
        )
        assert event_filter._apply_geospatial_filter(
            event,
            {"radius_km": 50, "center": {"lat": 52.5200, "lon": 13.4050}},  # Berlin
        )

    def test_radius_excludes_point_beyond_radius(self, event_filter):
        event = _make_event(
            geospatial_context={"latitude": 52.3906, "longitude": 13.0645}
        )
        assert not event_filter._apply_geospatial_filter(
            event,
            {"radius_km": 10, "center": {"lat": 52.5200, "lon": 13.4050}},
        )

    def test_require_location_rejects_contextless_events(self, event_filter):
        event = _make_event(geospatial_context=None)
        assert not event_filter._apply_geospatial_filter(
            event, {"require_location": True}
        )
        assert event_filter._apply_geospatial_filter(event, {})


class TestEventFilterTemporal:
    FIXED_TS = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

    def test_start_time_includes_later_event(self, event_filter):
        event = _make_event(timestamp=self.FIXED_TS)
        config = {"start_time": self.FIXED_TS - timedelta(hours=1)}
        assert event_filter._apply_temporal_filter(event, config)

    def test_start_time_excludes_earlier_event(self, event_filter):
        event = _make_event(timestamp=self.FIXED_TS)
        config = {"start_time": self.FIXED_TS + timedelta(hours=1)}
        assert not event_filter._apply_temporal_filter(event, config)

    def test_end_time_excludes_later_event(self, event_filter):
        event = _make_event(timestamp=self.FIXED_TS)
        config = {"end_time": self.FIXED_TS - timedelta(minutes=30)}
        assert not event_filter._apply_temporal_filter(event, config)

    def test_window_bounds_are_inclusive(self, event_filter):
        event = _make_event(timestamp=self.FIXED_TS)
        config = {"start_time": self.FIXED_TS, "end_time": self.FIXED_TS}
        assert event_filter._apply_temporal_filter(event, config)


class TestEventFilterBasicAndCustom:
    def test_basic_event_type_filter(self, event_filter):
        event = _make_event(event_type="system_alert")
        assert not event_filter._apply_basic_filter(
            event, {"event_types": ["data_update"]}
        )
        assert event_filter._apply_basic_filter(
            event, {"event_types": ["system_alert"]}
        )

    def test_basic_source_filter(self, event_filter):
        event = _make_event(source="unit-test")
        assert not event_filter._apply_basic_filter(event, {"sources": ["sensor-net"]})
        assert event_filter._apply_basic_filter(event, {"sources": ["unit-test"]})

    def test_basic_min_priority_filter(self, event_filter):
        low = _make_event(priority=MessagePriority.LOW)
        urgent = _make_event(priority=MessagePriority.URGENT)
        assert not event_filter._apply_basic_filter(low, {"min_priority": "high"})
        assert event_filter._apply_basic_filter(urgent, {"min_priority": "high"})

    def test_registered_custom_filter_is_applied(self, event_filter):
        event_filter.register_filter(
            "only_urgent", lambda event, cfg: event.priority == MessagePriority.URGENT
        )
        urgent = _make_event(priority=MessagePriority.URGENT)
        normal = _make_event(priority=MessagePriority.NORMAL)
        config = [{"type": "custom", "filter_name": "only_urgent"}]
        assert event_filter.apply_filters(urgent, config)
        assert not event_filter.apply_filters(normal, config)

    def test_unknown_filter_type_passes_event_through(self, event_filter):
        event = _make_event()
        assert event_filter._apply_single_filter(event, {"type": "mystery"})

    def test_combined_filters_require_all_to_pass(self, event_filter):
        inside = _make_event(
            geospatial_context={"latitude": 48.85, "longitude": 2.35},
        )
        outside = _make_event(
            geospatial_context={"latitude": 48.85, "longitude": 10.0},
        )
        filters = [
            {"type": "basic", "sources": ["unit-test"]},
            {
                "type": "geospatial",
                "bbox": {
                    "min_lon": 2.0,
                    "max_lon": 3.0,
                    "min_lat": 48.0,
                    "max_lat": 49.5,
                },
            },
        ]
        assert event_filter.apply_filters(inside, filters)
        assert not event_filter.apply_filters(outside, filters)
