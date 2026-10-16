"""Real local compositions and workflow lifecycle guarantees."""

from __future__ import annotations

import asyncio

import pytest

from geo_infer_examples import LocalAPIConnector, ModuleOrchestrator
from geo_infer_examples.models.integration_models import (
    WorkflowDefinition,
    WorkflowStep,
)


def workflow(steps, *, strategy="sequential", **kwargs):
    """Build an explicitly registered local workflow."""
    return WorkflowDefinition(
        "local",
        "Local composition",
        "Real operation composition",
        steps,
        execution_strategy=strategy,
        **kwargs,
    )


def orchestrator(operations, *, resilience=True):
    """Bind only explicitly supplied operations without remote services."""
    return ModuleOrchestrator(
        api_connector=LocalAPIConnector(operations), resilience_enabled=resilience
    )


@pytest.mark.parametrize("strategy", ["sequential", "parallel"])
async def test_data_space_time_energy_forecast(tmp_path, strategy):
    """Persisted sensor values reach real H3, temporal fitting, and forecasting APIs."""
    import pandas as pd
    import xarray as xr
    from geo_infer_data.core.storage import LocalFileBackend
    from geo_infer_data.models.schemas import DataLineage, DatasetMetadata
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_time.core.analysis import TemporalAnalyzer
    from geo_infer_time.models.timeseries import TimeSeries
    from geo_infer_energy import EnergyDemandForecaster

    backend = LocalFileBackend({"base_path": str(tmp_path)})

    async def persist(data):
        frame = pd.DataFrame(data["records"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        metadata = DatasetMetadata(
            title="Local sensors",
            lineage=DataLineage(
                source="sensor fixture", process="local composition", created_by="test"
            ),
        )
        identifier = await backend.store(frame, metadata)
        restored = await backend.retrieve(identifier, {})
        pd.testing.assert_frame_equal(frame, restored)
        return {
            "values": restored.value.tolist(),
            "timestamps": restored.timestamp.tolist(),
            "points": list(zip(restored.latitude, restored.longitude)),
            "dataset_id": identifier,
        }

    def index(data):
        indexer = SpatialIndexingInterface(backend="h3")
        return {
            "cells": [
                indexer.latlng_to_cell(lat, lon, resolution=9)
                for lat, lon in data["points"]
            ]
        }

    def trend(data):
        series = TimeSeries(
            data=pd.Series(data["values"]).to_numpy(),
            timestamps=pd.DatetimeIndex(data["timestamps"]),
        )
        return TemporalAnalyzer().detect_trend(series, method="linear")

    def forecast(data):
        assert data["slope"] == pytest.approx(1.0)
        assert len(set(data["indexed_cells"])) == 1
        demand = xr.DataArray(
            data["values"],
            dims="time",
            coords={"time": pd.DatetimeIndex(data["timestamps"]).tz_convert(None)},
        )
        return {
            "forecast": EnergyDemandForecaster()
            .forecast_demand(demand, forecast_years=1)
            .demand_forecast.values.tolist()
        }

    engine = orchestrator(
        {
            ("DATA", "/persist"): persist,
            ("SPACE", "/index"): index,
            ("TIME", "/trend"): trend,
            ("ENERGY", "/forecast"): forecast,
        }
    )
    steps = [
        WorkflowStep("persist", "DATA", "/persist"),
        WorkflowStep(
            "index",
            "SPACE",
            "/index",
            dependencies=["persist"],
            output_mapping={"indexed_cells": "cells"},
        ),
        WorkflowStep(
            "trend",
            "TIME",
            "/trend",
            dependencies=["index"],
            output_mapping={"slope": "slope_per_sample"},
        ),
        WorkflowStep("forecast", "ENERGY", "/forecast", dependencies=["trend"]),
    ]
    assert engine.register_workflow(workflow(list(reversed(steps)), strategy=strategy))
    records = [
        {
            "timestamp": f"2024-01-0{index + 1}T00:00:00Z",
            "value": 100.0 + index,
            "latitude": 37.7,
            "longitude": -122.4,
        }
        for index in range(4)
    ]
    result = await engine.execute_workflow("local", {"records": records})
    assert result.success
    assert result.data["forecast"]["forecast"] == pytest.approx([475.0])
    execution = engine.get_workflow_status(result.metadata["execution_id"])
    assert execution.performance_metrics["total_seconds"] > 0
    assert set(execution.performance_metrics["step_seconds"]) == {
        step.name for step in steps
    }


@pytest.mark.parametrize("strategy", ["sequential", "parallel", "conditional"])
async def test_local_inputs_are_independently_owned(strategy):
    """Local mutations cannot modify caller-owned nested data."""

    def mutate(data):
        data["nested"].append(2)
        return {"count": len(data["nested"])}

    engine = orchestrator({("DATA", "/mutate"): mutate})
    assert engine.register_workflow(
        workflow([WorkflowStep("mutate", "DATA", "/mutate")], strategy=strategy)
    )
    source = {"nested": [1]}
    result = await engine.execute_workflow("local", source)
    assert source == {"nested": [1]}
    assert result.data["mutate"] == {"count": 2}


@pytest.mark.parametrize(
    "steps",
    [
        [WorkflowStep("a", "DATA", "/ok"), WorkflowStep("a", "DATA", "/ok")],
        [
            WorkflowStep("a", "DATA", "/ok", dependencies=["b"]),
            WorkflowStep("b", "DATA", "/ok", dependencies=["a"]),
        ],
        [WorkflowStep("a", "DATA", "/ok", dependencies=["missing"])],
    ],
)
def test_invalid_dependency_graph_is_rejected_before_execution(steps):
    """Duplicate names, cycles, and absent dependencies cannot enter execution."""
    engine = orchestrator({("DATA", "/ok"): lambda data: data})
    assert not engine.register_workflow(workflow(steps))


@pytest.mark.parametrize(
    "policy, expected_calls", [(None, 1), ({"max_attempts": 2}, 2)]
)
async def test_required_failures_have_finite_explicit_retries(policy, expected_calls):
    """A required failure is never converted to an optional step or retried recursively."""
    calls = []

    def fail(data):
        calls.append(data)
        raise RuntimeError("operation failed")

    engine = orchestrator({("DATA", "/fail"): fail})
    definition = workflow([WorkflowStep("fail", "DATA", "/fail")], retry_policy=policy)
    assert engine.register_workflow(definition)
    with pytest.raises(RuntimeError, match="operation failed"):
        await engine.execute_workflow("local", {})
    assert len(calls) == expected_calls
    assert not definition.steps[0].optional
    assert all(item.status == "failed" for item in engine.active_executions.values())


async def test_explicit_retry_can_recover_without_changing_definition():
    """A declared second attempt reports real successful operation output."""
    calls = []

    def recover(data):
        calls.append(data)
        if len(calls) == 1:
            raise RuntimeError("first attempt failed")
        return {"value": 3}

    engine = orchestrator({("DATA", "/run"): recover})
    assert engine.register_workflow(
        workflow(
            [WorkflowStep("run", "DATA", "/run")], retry_policy={"max_attempts": 2}
        )
    )
    result = await engine.execute_workflow("local", {})
    assert len(calls) == 2
    assert result.data == {"run": {"value": 3}}
    assert result.metadata["recovery_attempts"] == 1


async def test_parallel_failure_cancels_peer_even_when_failure_is_second():
    """Required failure drains unfinished siblings before returning to the caller."""
    started, stopped = asyncio.Event(), asyncio.Event()

    async def wait(data):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def fail(data):
        await started.wait()
        raise RuntimeError("parallel failure")

    engine = orchestrator(
        {("DATA", "/wait"): wait, ("DATA", "/fail"): fail}, resilience=False
    )
    assert engine.register_workflow(
        workflow(
            [
                WorkflowStep("wait", "DATA", "/wait"),
                WorkflowStep("fail", "DATA", "/fail"),
            ],
            strategy="parallel",
        )
    )
    with pytest.raises(RuntimeError, match="parallel failure"):
        await asyncio.wait_for(engine.execute_workflow("local", {}), timeout=2)
    assert stopped.is_set()


@pytest.mark.parametrize("deadline", ["step", "workflow", "cancel"])
async def test_deadlines_and_cancellation_drain_async_operations(deadline):
    """Deadline or caller cancellation closes async work and records its actual status."""
    started, stopped = asyncio.Event(), asyncio.Event()

    async def wait(data):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    engine = orchestrator({("DATA", "/wait"): wait}, resilience=False)
    step = WorkflowStep(
        "wait", "DATA", "/wait", timeout=0.02 if deadline == "step" else 2
    )
    assert engine.register_workflow(
        workflow([step], timeout=0.02 if deadline == "workflow" else None)
    )
    task = asyncio.create_task(engine.execute_workflow("local", {}))
    await started.wait()
    if deadline == "cancel":
        task.cancel()
    with pytest.raises(
        asyncio.CancelledError if deadline == "cancel" else TimeoutError
    ):
        await task
    assert stopped.is_set()
    state = next(iter(engine.active_executions.values()))
    assert state.status == ("cancelled" if deadline == "cancel" else "failed")
    assert state.end_time is not None


async def test_concurrent_runs_have_distinct_execution_identities():
    """Overlapping invocations cannot overwrite each other's tracked results."""
    engine = orchestrator({("DATA", "/echo"): lambda data: data})
    assert engine.register_workflow(workflow([WorkflowStep("echo", "DATA", "/echo")]))
    results = await asyncio.gather(
        *(engine.execute_workflow("local", {"value": index}) for index in range(10))
    )
    assert len({item.metadata["execution_id"] for item in results}) == 10
    assert [item.data["echo"]["value"] for item in results] == list(range(10))


@pytest.mark.parametrize(
    "operation, error",
    [
        (lambda data: [], TypeError),
        (lambda data: (_ for _ in ()).throw(ValueError("bad data")), ValueError),
    ],
)
async def test_local_result_and_operation_errors_propagate(operation, error):
    """The connector cannot turn malformed results or domain errors into success."""
    engine = orchestrator({("DATA", "/run"): operation}, resilience=False)
    assert engine.register_workflow(workflow([WorkflowStep("run", "DATA", "/run")]))
    with pytest.raises(error):
        await engine.execute_workflow("local", {})


async def test_feedback_uses_latest_numeric_iterations_after_nine():
    """Convergence uses iterations ten and eleven rather than lexical ordering."""
    values = iter([*range(1, 10), 20, 20, 20])
    engine = orchestrator({("COG", "/update"): lambda data: {"estimate": next(values)}})
    assert engine.register_workflow(
        workflow(
            [
                WorkflowStep(
                    "update",
                    "COG",
                    "/update",
                    feedback_mapping={"estimate": "estimate"},
                )
            ],
            strategy="feedback_loop",
            max_iterations=12,
            convergence_threshold=0,
        )
    )
    result = await engine.execute_workflow("local", {})
    assert result.metadata["iterations"] == 11


async def test_event_workflow_composes_registered_operations():
    """A native local producer triggers its dependent operation once."""
    engine = orchestrator(
        {
            ("DATA", "/source"): lambda data: {"value": 4},
            ("COG", "/double"): lambda data: {"doubled": data["value"] * 2},
        }
    )
    assert engine.register_workflow(
        workflow(
            [
                WorkflowStep("source", "DATA", "/source", emits_events=["ready"]),
                WorkflowStep(
                    "double",
                    "COG",
                    "/double",
                    dependencies=["source"],
                    trigger_events=["ready"],
                ),
            ],
            strategy="event_driven",
        )
    )
    result = await engine.execute_workflow("local", {})
    assert result.data["double"] == {"doubled": 8}


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_iterations": 0},
        {"convergence_threshold": -1},
        {"retry_policy": {"max_attempts": 1000}},
        {"retry_policy": []},
        {"retry_policy": {"attempts": 2}},
        {"retry_policy": {"delay_seconds": float("inf")}},
    ],
)
def test_invalid_execution_budgets_are_rejected(options):
    """Invalid budgets cannot silently select defaults or unbounded retries."""
    engine = orchestrator({("DATA", "/echo"): lambda data: data})
    assert not engine.register_workflow(
        workflow([WorkflowStep("echo", "DATA", "/echo")], **options)
    )


@pytest.mark.parametrize("timeout", [0, -1, True, "unbounded"])
def test_invalid_default_deadline_is_rejected(tmp_path, timeout):
    """Local configuration cannot disable deadlines with an invalid default."""
    import json

    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"orchestrator": {"default_timeout": timeout}}))
    with pytest.raises(ValueError, match="default_timeout"):
        ModuleOrchestrator(config_path=str(config))


async def test_event_signal_is_retained_until_all_dependencies_finish():
    """An early event cannot bypass a second dependency or disappear before it completes."""
    engine = orchestrator(
        {
            ("DATA", "/a"): lambda data: {"a": 2},
            ("DATA", "/b"): lambda data: {"b": 3},
            ("COG", "/sum"): lambda data: {"value": data["a"] + data["b"]},
        }
    )
    steps = [
        WorkflowStep("a", "DATA", "/a", emits_events=["ready"]),
        WorkflowStep(
            "sum", "COG", "/sum", dependencies=["a", "b"], trigger_events=["ready"]
        ),
        WorkflowStep("b", "DATA", "/b"),
    ]
    assert engine.register_workflow(workflow(steps, strategy="event_driven"))
    result = await engine.execute_workflow("local", {})
    assert result.data["sum"] == {"value": 5}


async def test_untriggered_required_work_cannot_report_success():
    """An event workflow must execute every required step before succeeding."""
    engine = orchestrator({("DATA", "/echo"): lambda data: data}, resilience=False)
    assert engine.register_workflow(
        workflow(
            [WorkflowStep("never", "DATA", "/echo", trigger_events=["absent"])],
            strategy="event_driven",
        )
    )
    with pytest.raises(ValueError, match="not triggered"):
        await engine.execute_workflow("local", {})


async def test_inflight_definition_is_independent_of_later_edits():
    """Changing a stored definition cannot redirect an already admitted invocation."""
    started, release = asyncio.Event(), asyncio.Event()

    async def first(data):
        started.set()
        await release.wait()
        return {"value": 1}

    engine = orchestrator(
        {("DATA", "/first"): first, ("DATA", "/last"): lambda data: data}
    )
    definition = workflow(
        [
            WorkflowStep("first", "DATA", "/first"),
            WorkflowStep("last", "DATA", "/last", dependencies=["first"]),
        ]
    )
    assert engine.register_workflow(definition)
    task = asyncio.create_task(engine.execute_workflow("local", {}))
    await started.wait()
    definition.steps[1].endpoint = "/changed"
    release.set()
    result = await task
    assert result.data["last"] == {"value": 1}
