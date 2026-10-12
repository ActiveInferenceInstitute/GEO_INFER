"""Retained facade start tasks finish observably, including failure/cancellation."""

import asyncio
from unittest import mock

import pytest

from geo_infer_agent.api import interface as interface_module
from geo_infer_agent.api.interface import (
    AgentInterface,
    _on_start_task_done,
    _pending_start_tasks,
)
from geo_infer_agent.core.agent_registry import agent_registry


@pytest.mark.asyncio(loop_scope="function")
async def test_start_agent_becomes_running_when_start_task_completes():
    interface = AgentInterface()
    agent_id = await interface.create_agent("default", {}, agent_id="start-retain-1")
    assert await interface.start_agent(agent_id)
    starts = tuple(_pending_start_tasks)
    assert len(starts) == 1
    await asyncio.wait_for(asyncio.gather(*starts), timeout=2)
    assert agent_registry.is_agent_running(agent_id)
    assert not _pending_start_tasks


@pytest.mark.asyncio(loop_scope="function")
async def test_blocked_start_is_retained_until_completion():
    interface = AgentInterface()
    agent_id = await interface.create_agent("default", {}, agent_id="start-blocked")
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_start(_agent_id):
        entered.set()
        await release.wait()

    with mock.patch.object(agent_registry, "start_agent", side_effect=blocked_start):
        assert await interface.start_agent(agent_id)
        await asyncio.wait_for(entered.wait(), timeout=2)
        starts = tuple(_pending_start_tasks)
        assert len(starts) == 1
        assert not starts[0].done()
        release.set()
        await asyncio.wait_for(starts[0], timeout=2)
    assert not _pending_start_tasks


@pytest.mark.asyncio(loop_scope="function")
async def test_start_failure_is_logged_and_task_discarded(caplog):
    interface = AgentInterface()
    agent_id = await interface.create_agent("default", {}, agent_id="start-fail-1")

    async def exploding_start(_agent_id):
        raise RuntimeError("boom")

    with (
        mock.patch.object(agent_registry, "start_agent", side_effect=exploding_start),
        caplog.at_level("ERROR", logger="geo_infer_agent.api.interface"),
    ):
        assert await interface.start_agent(agent_id)
        starts = tuple(_pending_start_tasks)
        results = await asyncio.wait_for(
            asyncio.gather(*starts, return_exceptions=True), timeout=2
        )
    assert isinstance(results[0], RuntimeError)
    assert not _pending_start_tasks
    assert "boom" in caplog.text


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.asyncio(loop_scope="function")
async def test_done_callback_discards_task_and_observes_exception(failure, caplog):
    callback_finished = asyncio.Event()

    async def complete():
        if failure:
            raise RuntimeError("callback-boom")

    def on_done(task):
        _on_start_task_done(task)
        callback_finished.set()

    task = asyncio.create_task(complete())
    _pending_start_tasks.add(task)
    task.add_done_callback(on_done)
    with caplog.at_level("ERROR", logger="geo_infer_agent.api.interface"):
        await asyncio.wait_for(callback_finished.wait(), timeout=2)
    assert task.done()
    assert task not in _pending_start_tasks
    assert ("callback-boom" in caplog.text) is failure


@pytest.mark.asyncio(loop_scope="function")
async def test_cancelled_start_callback_discards_without_error(caplog, managed_task):
    callback_finished = asyncio.Event()

    def on_done(task):
        _on_start_task_done(task)
        callback_finished.set()

    with caplog.at_level("ERROR", logger="geo_infer_agent.api.interface"):
        async with managed_task(asyncio.Event().wait()) as task:
            interface_module._pending_start_tasks.add(task)
            task.add_done_callback(on_done)
            task.cancel()
            await asyncio.wait_for(callback_finished.wait(), timeout=2)
    assert task.cancelled()
    assert task not in _pending_start_tasks
    assert not caplog.records
