"""Lifecycle regression tests for EnhancedLogger's async processor (GS19-15).

stop() must drain queued entries, never resurrect the worker, and route
post-stop log() calls synchronously; the atexit helper drains through a
weak reference without keeping loggers alive.
"""

from __future__ import annotations

import gc
import weakref

from geo_infer_log import EnhancedLogger, LogEntry, _shutdown_logger_at_exit


class _Recorder:
    """Instance-level stand-in for _write_log_entry capturing every entry."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    def __call__(self, entry: LogEntry) -> None:
        self.entries.append(entry)


def _async_logger(name: str) -> tuple[EnhancedLogger, _Recorder]:
    logger = EnhancedLogger(
        name,
        {"async_logging": True, "outputs": {"console": {"enabled": False}}},
    )
    recorder = _Recorder()
    logger._write_log_entry = recorder  # type: ignore[method-assign]
    return logger, recorder


def test_stop_drains_every_queued_entry() -> None:
    logger, recorder = _async_logger("lifecycle-drain")
    total = 50
    for index in range(total):
        logger.log("INFO", f"op_{index}", f"message {index}")
    logger.stop(timeout=5.0)
    # The worker consumes entries concurrently while log() enqueues; the
    # union of consumed and drained entries must still be every entry.
    assert len(recorder.entries) == total
    assert logger.log_queue.qsize() == 0
    thread = logger._log_processor_thread
    assert thread is not None
    assert not thread.is_alive()


def test_log_after_stop_writes_synchronously_without_resurrect() -> None:
    logger, recorder = _async_logger("lifecycle-post-stop")
    logger.log("INFO", "during", "queued while running")
    logger.stop(timeout=5.0)
    before = len(recorder.entries)
    logger.log("INFO", "after", "written synchronously")
    assert len(recorder.entries) == before + 1
    assert recorder.entries[-1].operation == "after"
    # The stopped logger must not resurrect the background processor.
    assert logger.log_queue.qsize() == 0
    thread = logger._log_processor_thread
    assert thread is not None
    assert not thread.is_alive()
    logger.stop()
    assert not thread.is_alive()


def test_stop_idempotent_on_sync_logger() -> None:
    logger = EnhancedLogger(
        "lifecycle-sync",
        {"async_logging": False, "outputs": {"console": {"enabled": False}}},
    )
    logger.stop()
    assert logger._log_processor_stopped is True
    assert logger._log_processor_thread is None
    logger.stop()
    logger.log("INFO", "after_sync_stop", "still written synchronously")


def test_shutdown_helper_drains_through_weakref() -> None:
    logger, recorder = _async_logger("lifecycle-atexit")
    for index in range(5):
        logger.log("INFO", f"burst_{index}", f"message {index}")
    _shutdown_logger_at_exit(weakref.ref(logger))
    assert len(recorder.entries) == 5


def test_shutdown_helper_ignores_dead_reference() -> None:
    logger, recorder = _async_logger("lifecycle-atexit-dead")
    for index in range(5):
        logger.log("INFO", f"burst_{index}", f"message {index}")
    # A running worker thread holds a strong reference to the logger via
    # its bound target; only after stop() joins the thread can the logger
    # actually become unreachable.
    logger.stop(timeout=5.0)
    ref = weakref.ref(logger)
    del logger
    gc.collect()
    assert ref() is None
    _shutdown_logger_at_exit(ref)
