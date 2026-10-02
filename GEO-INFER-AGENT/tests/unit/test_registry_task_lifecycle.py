"""A registry releases real completed tasks and observes task failures."""

import asyncio

import pytest

from geo_infer_agent.core.agent_registry import AgentRegistry


@pytest.mark.asyncio(loop_scope="function")
async def test_natural_completion_releases_registry_task():
    registry = AgentRegistry()
    agent_id = await registry.create_agent(
        "default", {"max_runtime": 0.001, "decision_frequency": 0}, agent_id="natural"
    )
    await registry.start_agent(agent_id)
    task = registry.agent_tasks[agent_id]
    await asyncio.wait_for(task, timeout=2)
    assert task.done()
    assert agent_id not in registry.agent_tasks
    assert not registry.is_agent_running(agent_id)
    assert not registry.get_agent(agent_id).running


@pytest.mark.asyncio(loop_scope="function")
async def test_explicit_stop_cancels_and_awaits_running_agent():
    registry = AgentRegistry()
    agent_id = await registry.create_agent("default", {}, agent_id="cancelled")
    await registry.start_agent(agent_id)
    task = registry.agent_tasks[agent_id]
    await asyncio.wait_for(registry.stop_agent(agent_id), timeout=2)
    assert task.cancelled()
    assert agent_id not in registry.agent_tasks
    assert not registry.is_agent_running(agent_id)


@pytest.mark.asyncio(loop_scope="function")
async def test_unhandled_run_failure_is_observed_and_released(monkeypatch, caplog):
    registry = AgentRegistry()
    agent_id = await registry.create_agent("default", {}, agent_id="failed")

    async def fail():
        raise RuntimeError("run failed")

    monkeypatch.setattr(registry.get_agent(agent_id), "run", fail)
    with caplog.at_level("ERROR", logger="geo_infer_agent.core.agent_registry"):
        await registry.start_agent(agent_id)
        task = registry.agent_tasks[agent_id]
        with pytest.raises(RuntimeError, match="run failed"):
            await asyncio.wait_for(task, timeout=2)
    assert agent_id not in registry.agent_tasks
    assert not registry.is_agent_running(agent_id)
    assert "run failed" in caplog.text


@pytest.mark.asyncio(loop_scope="function")
async def test_stale_completion_callback_preserves_restarted_agent(managed_task):
    registry = AgentRegistry()
    agent_id = await registry.create_agent("default", {}, agent_id="restarted")
    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    async with managed_task(asyncio.Event().wait()) as replacement:
        registry.agent_tasks[agent_id] = replacement
        registry.running_agents.add(agent_id)
        registry._on_agent_task_done(agent_id, completed)
        assert registry.agent_tasks[agent_id] is replacement
        assert registry.is_agent_running(agent_id)
        await registry.stop_agent(agent_id)


@pytest.mark.asyncio(loop_scope="function")
async def test_stop_completion_does_not_clear_concurrent_restart(
    monkeypatch, managed_task
):
    registry = AgentRegistry()
    agent_id = await registry.create_agent("default", {}, agent_id="stop-restart")
    agent = registry.get_agent(agent_id)
    entered = [asyncio.Event(), asyncio.Event()]
    generation = 0

    async def perceive():
        nonlocal generation
        current = generation
        generation += 1
        entered[current].set()
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(agent, "perceive", perceive)
    await registry.start_agent(agent_id)
    original = registry.agent_tasks[agent_id]
    await asyncio.wait_for(entered[0].wait(), timeout=2)

    async def restart_after_completion():
        try:
            await original
        except asyncio.CancelledError:
            pass
        await registry.start_agent(agent_id)
        await entered[1].wait()

    async with managed_task(restart_after_completion()) as restarter:
        # Register the restarter's task waiter before stop registers its own.
        # This real completion order exposes the old stop's stale cleanup.
        ready = asyncio.Event()
        asyncio.get_running_loop().call_soon(ready.set)
        await asyncio.wait_for(ready.wait(), timeout=2)
        await asyncio.wait_for(registry.stop_agent(agent_id), timeout=2)
        await asyncio.wait_for(restarter, timeout=2)
        replacement = registry.agent_tasks[agent_id]
        assert replacement is not original
        assert not replacement.done()
        assert registry.is_agent_running(agent_id)
        assert agent.running
        await asyncio.wait_for(registry.stop_agent(agent_id), timeout=2)
