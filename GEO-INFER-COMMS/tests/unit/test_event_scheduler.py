"""Tests for EventScheduler recurrence and scheduled publication (GS-227).

The recurring-trigger decision is exercised with explicit clock arguments
(a clock stub), and one scheduler pass is driven deterministically by
patching the module's ``time.sleep`` with a rendezvous barrier — no
wall-clock sleeps, no polling.
"""

from datetime import datetime, timedelta, UTC
from threading import Event
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from geo_infer_comms.core import events as events_module
from geo_infer_comms.core.events import EventManager, EventScheduler, RecurringEvent
from geo_infer_comms.models.message import EventPublishRequest


UTC = UTC


@pytest.fixture()
def event_manager():
    """EventManager with the publish gate open; no background thread."""
    manager = EventManager(enable_persistence=False)
    manager._running = True
    yield manager


@pytest.fixture()
def scheduler(event_manager):
    return EventScheduler(event_manager)


def _request(event_type="data_update", payload=None):
    return EventPublishRequest(
        event_type=event_type,
        payload=payload or {"temperature": 21.5},
        source="scheduler-test",
    )


def _recurring(schedule_config, last_triggered=None):
    return RecurringEvent(
        recurring_id="r1",
        event_request=_request(),
        schedule_config=schedule_config,
        last_triggered=last_triggered,
    )


@contextmanager
def _single_scheduler_pass(monkeypatch, scheduler):
    """Run exactly one iteration of the scheduler loop, deterministically.

    The patched ``time.sleep`` signals when the pass has completed its
    publication work, then parks the thread until the caller releases it.
    """
    entered = Event()
    release = Event()

    def fake_sleep(_seconds):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(events_module, "time", SimpleNamespace(sleep=fake_sleep))
    scheduler.start()
    assert entered.wait(timeout=5), "scheduler thread never reached its pause point"

    yield  # publication state from the single pass is now stable

    release.set()
    scheduler.stop()


class TestRecurringTriggerDecision:
    def test_interval_without_last_trigger_is_never_due(self):
        recurring = _recurring({"interval_seconds": 60})
        assert not recurring._should_trigger_recurring(datetime.now(UTC))

    def test_interval_due_once_elapsed(self):
        now = datetime.now(UTC)
        recurring = _recurring(
            {"interval_seconds": 60}, last_triggered=now - timedelta(seconds=90)
        )
        assert recurring._should_trigger_recurring(now)

    def test_interval_not_due_before_elapsed(self):
        now = datetime.now(UTC)
        recurring = _recurring(
            {"interval_seconds": 60}, last_triggered=now - timedelta(seconds=10)
        )
        assert not recurring._should_trigger_recurring(now)

    def test_cron_schedule_is_always_due(self):
        recurring = _recurring({"cron_schedule": "0 * * * *"})
        assert recurring._should_trigger_recurring(datetime.now(UTC))

    def test_empty_schedule_config_is_never_due(self):
        recurring = _recurring({})
        assert not recurring._should_trigger_recurring(datetime.now(UTC))


class TestScheduledPublication:
    def test_due_scheduled_event_is_published_and_cancelled_one_is_not(
        self, monkeypatch, event_manager, scheduler
    ):
        past = datetime.now(UTC) - timedelta(seconds=10)
        kept_id = scheduler.schedule_event(_request(), past)
        cancelled_id = scheduler.schedule_event(_request(), past)
        assert scheduler.cancel_scheduled_event(cancelled_id) is True

        with _single_scheduler_pass(monkeypatch, scheduler):
            assert event_manager.metrics.events_published == 1
            (published,) = event_manager.events.values()
            assert published.event_type == "data_update"
            assert published.source == "scheduler-test"
            assert scheduler.scheduled_events[kept_id].status == "completed"
            assert scheduler.scheduled_events[cancelled_id].status == "cancelled"

    def test_future_scheduled_event_is_not_published(
        self, monkeypatch, event_manager, scheduler
    ):
        future = datetime.now(UTC) + timedelta(hours=1)
        schedule_id = scheduler.schedule_event(_request(), future)

        with _single_scheduler_pass(monkeypatch, scheduler):
            assert event_manager.metrics.events_published == 0
            assert scheduler.scheduled_events[schedule_id].status == "scheduled"

    def test_due_recurring_event_publishes_and_stamps_last_triggered(
        self, monkeypatch, event_manager, scheduler
    ):
        recurring_id = scheduler.schedule_recurring_event(
            _request(), {"cron_schedule": "0 * * * *"}
        )

        with _single_scheduler_pass(monkeypatch, scheduler):
            assert event_manager.metrics.events_published == 1
            assert scheduler.recurring_events[recurring_id].last_triggered is not None

    def test_failed_publication_marks_scheduled_event_failed(
        self, monkeypatch, event_manager, scheduler
    ):
        past = datetime.now(UTC) - timedelta(seconds=10)
        schedule_id = scheduler.schedule_event(
            _request(event_type="bogus_event_type"),
            past,  # rejected by the manager
        )

        with _single_scheduler_pass(monkeypatch, scheduler):
            assert event_manager.metrics.events_published == 0
            assert scheduler.scheduled_events[schedule_id].status == "failed"


class TestSchedulerBookkeeping:
    def test_schedule_event_returns_id_and_records_entry(self, scheduler):
        schedule_id = scheduler.schedule_event(
            _request(), datetime.now(UTC) + timedelta(minutes=5), schedule_id="sched-1"
        )
        assert schedule_id == "sched-1"
        assert scheduler.scheduled_events["sched-1"].status == "scheduled"

    def test_cancel_unknown_scheduled_event_returns_false(self, scheduler):
        assert scheduler.cancel_scheduled_event("sched-nope") is False

    def test_cancel_recurring_event(self, scheduler):
        recurring_id = scheduler.schedule_recurring_event(_request(), {})
        assert scheduler.cancel_recurring_event(recurring_id) is True
        assert scheduler.recurring_events[recurring_id].status == "cancelled"
        assert scheduler.cancel_recurring_event("recurring-nope") is False

    def test_start_is_idempotent(self, scheduler):
        scheduler.start()
        thread = scheduler._scheduler_thread
        scheduler.start()
        assert scheduler._scheduler_thread is thread
        scheduler.stop()
