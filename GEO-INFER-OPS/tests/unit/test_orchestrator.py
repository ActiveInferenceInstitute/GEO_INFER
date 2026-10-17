"""
Unit tests for orchestrator functionality.
"""

import pytest
import asyncio
from geo_infer_ops.core.orchestrator import Orchestrator, TaskStatus


class TestOrchestrator:
    """Test Orchestrator class."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create an orchestrator instance."""
        return Orchestrator(max_concurrent_tasks=5)

    def test_add_task(self, orchestrator: Orchestrator) -> None:
        """Test adding tasks."""

        def simple_task():
            return "result"

        task_id = orchestrator.add_task(
            name="test_task",
            func=simple_task,
        )

        assert task_id in orchestrator.tasks
        assert orchestrator.tasks[task_id].name == "test_task"

    def test_task_dependencies(self, orchestrator: Orchestrator) -> None:
        """Test task with dependencies."""

        def task1():
            return "task1_result"

        def task2():
            return "task2_result"

        task1_id = orchestrator.add_task(name="task1", func=task1)
        task2_id = orchestrator.add_task(
            name="task2", func=task2, dependencies=[task1_id]
        )

        assert task2_id in orchestrator.tasks
        assert orchestrator.tasks[task2_id].dependencies == [task1_id]

    def test_execute_workflow(self, orchestrator: Orchestrator) -> None:
        """Test workflow execution."""
        results = []

        def task1():
            results.append("task1")
            return "task1_result"

        def task2():
            results.append("task2")
            return "task2_result"

        orchestrator.add_task(name="task1", func=task1)
        orchestrator.add_task(name="task2", func=task2)

        workflow_result = asyncio.run(orchestrator.execute_workflow())

        assert workflow_result["completed_tasks"] == 2
        assert "task1" in results
        assert "task2" in results

    def test_execute_workflow_with_dependencies(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test workflow execution with dependencies."""
        execution_order = []

        def task1():
            execution_order.append(1)
            return "task1_result"

        def task2():
            execution_order.append(2)
            return "task2_result"

        task1_id = orchestrator.add_task(name="task1", func=task1)
        orchestrator.add_task(name="task2", func=task2, dependencies=[task1_id])

        asyncio.run(orchestrator.execute_workflow())

        # Task1 should execute before task2
        assert execution_order == [1, 2]

    def test_get_task_status(self, orchestrator: Orchestrator) -> None:
        """Test getting task status."""

        def simple_task():
            return "result"

        task_id = orchestrator.add_task(name="test_task", func=simple_task)

        status = orchestrator.get_task_status(task_id)

        assert status is not None
        assert status["name"] == "test_task"
        assert status["status"] == TaskStatus.PENDING.value

    def test_get_workflow_status(self, orchestrator: Orchestrator) -> None:
        """Test getting workflow status."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        orchestrator.add_task(name="task1", func=task1)
        orchestrator.add_task(name="task2", func=task2)

        status = orchestrator.get_workflow_status()

        assert status["total_tasks"] == 2
        assert "status_counts" in status
        assert "tasks" in status


class TestOrchestratorRetryAndDependencySemantics:
    """Focused tests for retry and dependency handling."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        return Orchestrator(max_concurrent_tasks=5, retry_delay_seconds=0)

    def test_retry_until_success(self, orchestrator: Orchestrator) -> None:
        """A flaky task is retried and completes after transient failures."""
        attempts = {"count": 0}

        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient failure")
            return "recovered"

        task_id = orchestrator.add_task(name="flaky", func=flaky, max_retries=3)
        result = asyncio.run(orchestrator.execute_workflow())

        assert result["completed_tasks"] == 1
        task = orchestrator.tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2
        assert task.result == "recovered"

    def test_retry_exhaustion_marks_failed(self, orchestrator: Orchestrator) -> None:
        """A permanently failing task exhausts retries and is marked FAILED."""
        attempts = {"count": 0}

        def always_fails() -> None:
            attempts["count"] += 1
            raise RuntimeError("permanent failure")

        task_id = orchestrator.add_task(
            name="failing", func=always_fails, max_retries=2
        )
        result = asyncio.run(orchestrator.execute_workflow())

        assert result["failed_tasks"] == 1
        task = orchestrator.tasks[task_id]
        assert task.status == TaskStatus.FAILED
        assert attempts["count"] == 3  # initial attempt + 2 retries
        assert "permanent failure" in (task.error or "")

    def test_failed_dependency_deadlocks_dependent(
        self, orchestrator: Orchestrator
    ) -> None:
        """A task depending on a permanently failed task is marked FAILED."""

        def always_fails() -> None:
            raise RuntimeError("upstream failure")

        def dependent() -> str:
            return "never runs"

        failing_id = orchestrator.add_task(
            name="failing", func=always_fails, max_retries=0
        )
        dependent_id = orchestrator.add_task(
            name="dependent", func=dependent, dependencies=[failing_id]
        )
        result = asyncio.run(orchestrator.execute_workflow())

        assert result["failed_tasks"] >= 1
        assert orchestrator.tasks[failing_id].status == TaskStatus.FAILED
        dependent = orchestrator.tasks[dependent_id]
        assert dependent.status == TaskStatus.FAILED
        assert "deadlock" in (dependent.error or "")

    def test_invalid_dependency_raises(self, orchestrator: Orchestrator) -> None:
        """A dependency referencing an unknown task id aborts the workflow."""
        orchestrator.add_task(
            name="orphan", func=lambda: "x", dependencies=["missing-task"]
        )
        with pytest.raises(ValueError, match="Invalid task dependencies"):
            asyncio.run(orchestrator.execute_workflow())

    def test_workflow_counters_count_each_task_once(
        self, orchestrator: Orchestrator
    ) -> None:
        """Completed tasks are counted once even across multiple scheduling waves."""

        def first() -> str:
            return "first"

        def second() -> str:
            return "second"

        first_id = orchestrator.add_task(name="first", func=first)
        orchestrator.add_task(name="second", func=second, dependencies=[first_id])

        result = asyncio.run(orchestrator.execute_workflow())

        assert result["completed_tasks"] == 2
        assert result["failed_tasks"] == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_concurrency_bounds_reject_nonpositive_or_noninteger_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        Orchestrator(max_concurrent_tasks=value)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), True])
def test_retry_delays_require_finite_nonnegative_values(delay):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Orchestrator(retry_delay_seconds=delay)


def test_graph_validation_precedes_all_effects_and_rejects_duplicate_ids():
    calls = []
    o = Orchestrator()
    o.add_task("independent", lambda: calls.append("ran"), task_id="ready")
    o.add_task("a", lambda: 1, task_id="a", dependencies=["b"])
    o.add_task("b", lambda: 1, task_id="b", dependencies=["a"])
    with pytest.raises(ValueError, match="cycle"):
        asyncio.run(o.execute_workflow())
    assert calls == []
    original = o.tasks["a"]
    with pytest.raises(ValueError, match="already exists"):
        o.add_task("replacement", lambda: 99, task_id="a")
    assert o.tasks["a"] is original
    assert all(t.status == TaskStatus.PENDING for t in o.tasks.values())


def test_input_snapshots_and_failed_dependency_accounting():
    o = Orchestrator(retry_delay_seconds=0)

    def fail():
        raise ValueError("failed upstream")

    o.add_task("bad", fail, task_id="bad", max_retries=0)
    dependencies = ["bad"]
    metadata = {"value": 0}
    blocked = o.add_task(
        "blocked", lambda value: value, dependencies=dependencies, metadata=metadata
    )
    dependencies.clear()
    metadata["value"] = 999
    report = asyncio.run(o.execute_workflow())
    assert o.tasks[blocked].dependencies == ["bad"]
    assert o.tasks[blocked].metadata == {"value": 0}
    o.get_task_status(blocked)["dependencies"].clear()
    o.get_workflow_status()["tasks"][blocked]["dependencies"].clear()
    assert o.tasks[blocked].dependencies == ["bad"]
    assert report["failed_tasks"] == 2
    assert report["completed_tasks"] == report["cancelled_tasks"] == 0


def test_native_results_falsey_summaries_and_recovered_error_state():
    o = Orchestrator(retry_delay_seconds=0)
    native = {"values": []}
    native_id = o.add_task("native", lambda: native)
    values = [0, False, [], {}, ""]
    ids = [o.add_task("value", lambda value=value: value) for value in values]
    attempts = []

    def recover():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("transient")
        return native

    recovered = o.add_task("retry", recover, max_retries=1)
    with pytest.raises(RuntimeError, match="not completed"):
        o.get_task_result(native_id)
    with pytest.raises(KeyError):
        o.get_task_result("unknown")
    report = asyncio.run(o.execute_workflow())
    assert o.get_task_result(native_id) is native
    assert o.get_task_result(recovered) is native
    assert o.tasks[recovered].error is None
    for task_id, value in zip(ids, values):
        assert report["task_results"][task_id]["result"] == str(value)
        assert o.get_task_result(task_id) is value


def test_async_callable_and_dependency_progress_do_not_wait_for_a_wave():
    async def scenario():
        o = Orchestrator(max_concurrent_tasks=2)
        released = asyncio.Event()
        order = []

        class AsyncCallable:
            async def __call__(self):
                order.append("short")
                return 7

        async def long():
            await released.wait()
            order.append("long")

        def dependent():
            order.append("dependent")
            released.set()

        short = o.add_task("short", AsyncCallable())
        long_id = o.add_task("long", long)
        dependent_id = o.add_task("dependent", dependent, dependencies=[short])
        o.get_task_status(dependent_id)["dependencies"].append(long_id)
        o.get_workflow_status()["tasks"][dependent_id]["dependencies"].append(long_id)
        result = await asyncio.wait_for(o.execute_workflow(), timeout=5)
        assert result["completed_tasks"] == 3
        assert o.get_task_result(short) == 7
        assert order.index("dependent") < order.index("long")

    asyncio.run(scenario())


def test_running_cancellation_stops_native_coroutine_and_counts_dependents():
    async def scenario():
        o = Orchestrator()
        started, finished = asyncio.Event(), asyncio.Event()

        async def blocking():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task_id = o.add_task("blocking", blocking)
        blocked = o.add_task(
            "dependent", lambda: "must not run", dependencies=[task_id]
        )
        workflow = asyncio.create_task(o.execute_workflow())
        await asyncio.wait_for(started.wait(), timeout=5)
        assert o.cancel_task(task_id)
        report = await asyncio.wait_for(workflow, timeout=5)
        assert finished.is_set()
        assert report["cancelled_tasks"] == 1 and report["failed_tasks"] == 1
        assert o.tasks[task_id].status == TaskStatus.CANCELLED
        assert o.tasks[blocked].status == TaskStatus.FAILED
        assert not o.cancel_task(task_id)

    asyncio.run(scenario())


def test_workflow_cancellation_cleans_workers_and_prevents_concurrent_mutation():
    async def scenario():
        o = Orchestrator()
        started, finished = asyncio.Event(), asyncio.Event()

        async def blocking():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task_id = o.add_task("blocking", blocking)
        pending = o.add_task("pending", lambda: None, dependencies=[task_id])
        workflow = asyncio.create_task(o.execute_workflow())
        await asyncio.wait_for(started.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="already executing"):
            await o.execute_workflow()
        with pytest.raises(RuntimeError, match="Cannot add"):
            o.add_task("late", lambda: None)
        with pytest.raises(RuntimeError, match="Cannot reset"):
            o.reset_workflow()
        workflow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await workflow
        assert finished.is_set()
        assert all(
            o.tasks[name].status == TaskStatus.CANCELLED for name in [task_id, pending]
        )
        assert o._running == {} and not o._executing
        o.reset_workflow()
        assert all(t.status == TaskStatus.PENDING for t in o.tasks.values())

    asyncio.run(scenario())
