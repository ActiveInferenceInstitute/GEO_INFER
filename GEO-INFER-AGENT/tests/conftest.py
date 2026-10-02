"""Function-scoped asyncio ownership and real service cleanup for AGENT tests."""

import asyncio
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio


@pytest.fixture
def managed_task():
    """Retain a background task and cancel/await it even if an assertion fails."""

    @asynccontextmanager
    async def manage(coroutine):
        task = asyncio.create_task(coroutine)
        try:
            yield task
        finally:
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.CancelledError:
                pass

    return manage


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def clean_agent_services(request, tmp_path, monkeypatch):
    """Stop owned tasks through their APIs, and fail on leaked background work.

    pytest-asyncio owns the loop. The registry and service singletons are shared
    by production imports, so cleanup covers both those objects and any fresh
    singleton a test creates. References are restored only after actual tasks
    finish; clearing a set or closing a loop does not count as shutdown.
    """
    from geo_infer_agent.api.interface import _pending_start_tasks
    from geo_infer_agent.api.messaging import MessagingService, messaging_service
    from geo_infer_agent.api.telemetry import TelemetryService, telemetry_service
    from geo_infer_agent.core.agent_registry import AgentRegistry, agent_registry

    monkeypatch.setenv("GEO_INFER_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    initial = {
        AgentRegistry: AgentRegistry._instance,
        MessagingService: MessagingService._instance,
        TelemetryService: TelemetryService._instance,
    }
    baseline = asyncio.all_tasks()
    yield

    starts = tuple(_pending_start_tasks)
    for task in starts:
        if not task.done():
            task.cancel()
    if starts:
        await asyncio.wait_for(asyncio.gather(*starts, return_exceptions=True), 2)
    assert not _pending_start_tasks, (
        "start tasks must leave the retention set on completion"
    )

    instances = [agent_registry, messaging_service, telemetry_service]
    instances.extend(initial.values())
    instances.extend(cls._instance for cls in initial)
    if request.instance is not None:
        instances.extend(
            getattr(request.instance, name, None) for name in ("registry", "service")
        )
    seen = set()
    for instance in instances:
        if instance is None or id(instance) in seen:
            continue
        seen.add(id(instance))
        if isinstance(instance, AgentRegistry):
            for agent_id in tuple(instance.agents):
                await asyncio.wait_for(instance.stop_agent(agent_id), 2)
                instance.remove_agent(agent_id)
            assert not instance.agent_tasks
            assert not instance.running_agents
        elif isinstance(instance, (MessagingService, TelemetryService)):
            await asyncio.wait_for(instance.stop(), 2)
            fields = (
                ("message_queues", "channels", "message_callbacks")
                if isinstance(instance, MessagingService)
                else ("metrics", "agent_health", "metric_callbacks")
            )
            for field in fields:
                getattr(instance, field).clear()
    for cls, instance in initial.items():
        cls._instance = instance

    current = asyncio.current_task()
    leaked = {task for task in asyncio.all_tasks() - baseline if task is not current}
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.wait_for(asyncio.gather(*leaked, return_exceptions=True), 2)
    assert not leaked, f"Test left unowned background tasks: {leaked}"
