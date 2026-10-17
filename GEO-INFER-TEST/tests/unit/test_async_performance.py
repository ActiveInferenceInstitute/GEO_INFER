"""Real awaited benchmark/load work and owned-task cancellation contracts."""

import asyncio
import inspect

import pytest

from geo_infer_test.core.performance_monitor import BenchmarkRunner, LoadTester


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "runner", [BenchmarkRunner(iterations=2, warmup=0), LoadTester()]
)
def test_sync_surfaces_reject_known_async_work_before_calling_it(runner):
    calls = []

    async def work():
        calls.append("called")

    class AsyncCallable:
        async def __call__(self):
            calls.append("called")

    for target in (work, AsyncCallable()):
        with pytest.raises(TypeError, match="run_async"):
            runner.run(target)
    assert calls == []


def test_sync_benchmark_rejects_and_closes_hidden_unstarted_coroutines():
    coroutines = []

    async def work():
        raise AssertionError("Async body must not run on the sync surface")

    def wrapper():
        coroutines.append(work())
        return coroutines[-1]

    with pytest.raises(TypeError, match="run_async"):
        BenchmarkRunner(iterations=2, warmup=0).run(wrapper)
    assert len(coroutines) == 1
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


def test_sync_load_counts_hidden_awaitables_as_errors_and_closes_coroutines():
    coroutines = []

    async def work():
        raise AssertionError("Async body must not run on the sync surface")

    def wrapper():
        result = work()
        coroutines.append(result)
        return result

    result = LoadTester(concurrency=2, total_requests=7).run(wrapper)
    assert result["completed"] == result["errors"] == 7
    assert result["error_rate"] == 1
    assert all(inspect.getcoroutinestate(c) == inspect.CORO_CLOSED for c in coroutines)


@pytest.mark.asyncio
async def test_sync_rejection_preserves_existing_future_and_suspended_coroutine():
    future = asyncio.get_running_loop().create_future()

    async def work():
        await asyncio.sleep(0)

    coroutine = work()
    coroutine.send(None)
    try:
        for value in (future, coroutine):
            with pytest.raises(TypeError, match="run_async"):
                BenchmarkRunner(iterations=1, warmup=0).run(lambda value=value: value)
        assert not future.cancelled()
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_SUSPENDED
    finally:
        coroutine.close()
        future.cancel()


@pytest.mark.asyncio
async def test_async_benchmark_awaits_warmup_and_every_measured_call():
    completed = []

    async def work(value):
        await asyncio.sleep(0)
        completed.append(value)

    result = await BenchmarkRunner(iterations=5, warmup=2).run_async(
        work, 42, label="awaited"
    )
    assert completed == [42] * 7
    assert result["label"] == "awaited"
    assert result["iterations"] == 5
    assert result["warmup"] == 2
    assert 0 <= result["min_s"] <= result["median_s"] <= result["max_s"]
    assert result["total_s"] >= result["max_s"]
    assert result["operations_per_second"] > 0


@pytest.mark.asyncio
async def test_async_benchmark_accepts_synchronous_completed_work():
    calls = []
    result = await BenchmarkRunner(iterations=3, warmup=1).run_async(calls.append, 1)
    assert calls == [1] * 4
    assert result["iterations"] == 3


@pytest.mark.asyncio
async def test_async_benchmark_propagates_failure_without_a_partial_success():
    calls = []

    async def work():
        await asyncio.sleep(0)
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("actual work failed")

    with pytest.raises(ValueError, match="actual work failed"):
        await BenchmarkRunner(iterations=5, warmup=1).run_async(work)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_load_awaits_requests_with_bounded_concurrency():
    active = 0
    peak = 0
    completed = []
    ready = asyncio.Event()
    release = asyncio.Event()

    async def work(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 3:
            ready.set()
        try:
            await release.wait()
            await asyncio.sleep(0)
            completed.append(value)
        finally:
            active -= 1

    task = asyncio.create_task(
        LoadTester(concurrency=3, total_requests=17).run_async(work, 42)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert active == peak == 3
        release.set()
        result = await task
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert active == 0
    assert peak == 3
    assert completed == [42] * 17
    assert result["completed"] == 17
    assert result["errors"] == 0
    assert result["throughput_rps"] > 0


@pytest.mark.asyncio
async def test_async_load_counts_empty_exception_messages_as_errors():
    calls = []

    async def work():
        await asyncio.sleep(0)
        calls.append(1)
        raise ValueError()

    result = await LoadTester(concurrency=3, total_requests=17).run_async(work)
    assert len(calls) == result["completed"] == result["errors"] == 17
    assert result["error_rate"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["benchmark", "load"])
async def test_cancelled_async_runs_await_all_owned_work_cleanup(kind):
    started = 0
    active = 0
    cleaned = 0
    concurrency = 1 if kind == "benchmark" else 3
    ready = asyncio.Event()
    never = asyncio.Event()

    async def work():
        nonlocal started, active, cleaned
        started += 1
        active += 1
        if active == concurrency:
            ready.set()
        try:
            await never.wait()
        finally:
            active -= 1
            cleaned += 1

    runner = (
        BenchmarkRunner(iterations=20, warmup=0)
        if kind == "benchmark"
        else LoadTester(concurrency=concurrency, total_requests=20)
    )
    task = asyncio.create_task(runner.run_async(work))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert active == 0
    assert started == cleaned == concurrency


@pytest.mark.asyncio
async def test_request_cancellation_propagates_and_cleans_other_load_workers():
    active = 0
    started = 0
    ready = asyncio.Event()
    never = asyncio.Event()

    async def work():
        nonlocal active, started
        started += 1
        own_number = started
        active += 1
        if active == 3:
            ready.set()
        try:
            await ready.wait()
            if own_number == 1:
                raise asyncio.CancelledError()
            await never.wait()
        finally:
            active -= 1

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            LoadTester(concurrency=3, total_requests=20).run_async(work), timeout=5
        )
    assert active == 0
    assert started == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["benchmark", "load"])
@pytest.mark.parametrize("after_cancel", ["return", "error"])
async def test_suppressed_cancellation_cannot_continue_measurements(kind, after_cancel):
    concurrency = 1 if kind == "benchmark" else 2
    ready = asyncio.Event()
    never = asyncio.Event()
    calls = []
    cleaned = []

    async def work():
        calls.append(1)
        if len(calls) == concurrency:
            ready.set()
        try:
            try:
                await never.wait()
            except asyncio.CancelledError:
                if after_cancel == "error":
                    raise ValueError("error after swallowed cancellation") from None
                return
        finally:
            await asyncio.sleep(0)
            cleaned.append(1)

    runner = (
        BenchmarkRunner(iterations=5, warmup=0)
        if kind == "benchmark"
        else LoadTester(concurrency=concurrency, total_requests=5)
    )
    task = asyncio.create_task(runner.run_async(work))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == len(cleaned) == concurrency


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["benchmark", "load"])
async def test_preexisting_caught_cancellation_does_not_cancel_a_new_run(kind):
    async def scenario():
        task = asyncio.current_task()
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        assert task.cancelling() == 1
        completed = []

        async def work():
            await asyncio.sleep(0)
            completed.append(1)

        runner = (
            BenchmarkRunner(iterations=3, warmup=0)
            if kind == "benchmark"
            else LoadTester(concurrency=2, total_requests=3)
        )
        result = await runner.run_async(work)
        assert len(completed) == 3
        assert task.cancelling() == 1
        return result

    result = await asyncio.create_task(scenario())
    assert result["iterations" if kind == "benchmark" else "completed"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["benchmark", "load"])
async def test_request_explicit_uncancel_preserves_native_completion_semantics(kind):
    completed = []

    async def work():
        task = asyncio.current_task()
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            task.uncancel()
        assert task.cancelling() == 0
        completed.append(1)

    runner = (
        BenchmarkRunner(iterations=3, warmup=0)
        if kind == "benchmark"
        else LoadTester(concurrency=2, total_requests=3)
    )
    result = await asyncio.create_task(runner.run_async(work))
    assert len(completed) == 3
    assert result["iterations" if kind == "benchmark" else "completed"] == 3


@pytest.mark.asyncio
async def test_worker_uncancel_does_not_erase_parent_load_cancellation():
    ready = asyncio.Event()
    never = asyncio.Event()
    calls = []

    async def work():
        calls.append(1)
        ready.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            asyncio.current_task().uncancel()

    task = asyncio.create_task(
        LoadTester(concurrency=1, total_requests=5).run_async(work)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["benchmark", "load"])
async def test_cleared_old_cancellation_cannot_hide_a_later_new_request(kind):
    ready = asyncio.Event()
    never = asyncio.Event()
    calls = []

    async def scenario():
        owner = asyncio.current_task()
        owner.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        assert owner.cancelling() == 1

        async def work():
            calls.append(1)
            if len(calls) == 1:
                owner.uncancel()
                return
            if len(calls) == 2:
                ready.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    return

        runner = (
            BenchmarkRunner(iterations=5, warmup=0)
            if kind == "benchmark"
            else LoadTester(concurrency=1, total_requests=5)
        )
        return await runner.run_async(work)

    task = asyncio.create_task(scenario())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 2
