#!/usr/bin/env python3

"""
Unit tests for AgentInterface.start_agent task retention (GS-260).

The facade schedules registry.start_agent as a fire-and-forget task. The task
must be retained (strong reference) until completion so the coroutine cannot be
garbage-collected mid-flight, and any exception inside it must surface in the
module logger instead of vanishing unobserved.
"""

import asyncio
import unittest
from unittest import mock

from geo_infer_agent.api import interface as interface_module
from geo_infer_agent.api.interface import (
    AgentInterface,
    _on_start_task_done,
    _pending_start_tasks,
)
from geo_infer_agent.core.agent_registry import agent_registry


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStartAgentTaskRetention(unittest.TestCase):
    """Regression tests for retained start tasks and logged start failures."""

    def setUp(self) -> None:
        self.interface = AgentInterface()

    def tearDown(self) -> None:
        interface_module._pending_start_tasks.clear()
        for agent_id in list(agent_registry.agents):
            try:
                _run(agent_registry.stop_agent(agent_id))
            except KeyError:
                pass
            try:
                agent_registry.remove_agent(agent_id)
            except RuntimeError:
                continue

    def test_start_agent_becomes_running_within_tick(self) -> None:
        agent_id = _run(
            self.interface.create_agent("default", {}, agent_id="start-retain-1")
        )
        self.assertTrue(_run(self.interface.start_agent(agent_id)))
        _run(asyncio.sleep(0.2))
        self.assertTrue(agent_registry.is_agent_running(agent_id))
        self.assertEqual(len(_pending_start_tasks), 0)

    def test_start_failure_is_logged_and_task_discarded(self) -> None:
        agent_id = _run(
            self.interface.create_agent("default", {}, agent_id="start-fail-1")
        )

        async def exploding_start(_agent_id: str) -> None:
            raise RuntimeError("boom")

        with (
            mock.patch.object(
                agent_registry, "start_agent", side_effect=exploding_start
            ),
            self.assertLogs("geo_infer_agent.api.interface", level="ERROR") as logs,
        ):
            self.assertTrue(_run(self.interface.start_agent(agent_id)))
            _run(asyncio.sleep(0.2))

        self.assertEqual(len(_pending_start_tasks), 0)
        self.assertTrue(any("boom" in line for line in logs.output))


class TestOnStartTaskDone(unittest.TestCase):
    """Direct checks for the done-callback contract."""

    def test_callback_discards_task_and_logs_exception(self) -> None:
        interface_module._pending_start_tasks.clear()

        async def failing() -> None:
            raise RuntimeError("callback-boom")

        task = asyncio.get_event_loop().create_task(failing())
        _pending_start_tasks.add(task)
        task.add_done_callback(_on_start_task_done)
        with self.assertLogs("geo_infer_agent.api.interface", level="ERROR") as logs:
            _run(asyncio.sleep(0.05))  # let the task fail and callback fire
        self.assertNotIn(task, _pending_start_tasks)
        self.assertTrue(any("callback-boom" in line for line in logs.output))

    def test_callback_silent_for_clean_completion(self) -> None:
        interface_module._pending_start_tasks.clear()

        async def clean() -> None:
            return None

        task = asyncio.get_event_loop().create_task(clean())
        _pending_start_tasks.add(task)
        task.add_done_callback(_on_start_task_done)
        _run(asyncio.sleep(0.05))
        self.assertNotIn(task, _pending_start_tasks)
