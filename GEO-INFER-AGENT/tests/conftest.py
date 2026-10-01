"""Test fixtures for deterministic asyncio behavior under Python 3.12+."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def fresh_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Install a fresh event loop per test and close it afterwards.

    The unittest-style suites drive coroutines with ``run_until_complete``
    across several calls in one test (an agent started in one call keeps
    background tasks that later calls observe), so each test needs a single
    persistent loop rather than ``asyncio.run`` per call. Pending tasks are
    cancelled and async generators shut down before the loop is closed.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
