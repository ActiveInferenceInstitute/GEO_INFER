"""Integration coverage: the ``ModuleOrchestrator`` execution engine end-to-end.

The engine module docstring sanctions exactly one local execution seam:
monkeypatching or subclassing ``APIConnector`` with an object returning
HTTP-like responses (``status_code``, ``json()``). These tests inject such a
scripted connector and drive ``execute_workflow`` end-to-end through the two
strategies that depend on real step transport:

- a sequential happy path (step ordering, inter-step data flow, execution
  tracking), and
- a feedback_loop workflow that reaches convergence and exits converged
  instead of exhausting ``max_iterations``.

The convergence check itself is also pinned directly: malformed iteration
data must raise instead of silently reading as "never converged".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

import pytest

import geo_infer_examples.core.module_orchestrator as module_orchestrator
from geo_infer_examples.core.module_orchestrator import APIConnector, ModuleStatus
from geo_infer_examples.models.integration_models import (
    WorkflowDefinition,
    WorkflowStep,
)

pytestmark = [pytest.mark.integration]

_CONFIG_YAML = """\
modules:
  DATA:
    api_base: "https://data.example.invalid"
  COG:
    api_base: "https://cog.example.invalid"
"""


class _FakeResponse:
    """Minimal HTTP-like response object: ``status_code`` / ``json()`` / text."""

    def __init__(
        self, status_code: int, payload: dict | None = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload: dict = {} if payload is None else payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class _ScriptedAPIConnector(APIConnector):
    """Docstring-sanctioned seam: scripted responses plus call recording."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._handlers: dict[tuple[str, str], Callable[[dict], dict]] = {}

    def register(
        self, module: str, endpoint: str, handler: Callable[[dict], dict]
    ) -> None:
        self._handlers[(module, endpoint)] = handler

    def get(
        self, *, module: str, endpoint: str, timeout: int | None = None, **_: Any
    ) -> _FakeResponse:
        self.calls.append(("GET", module, endpoint))
        if endpoint == "/health":
            return _FakeResponse(200, {"capabilities": ["health"]})
        raise AssertionError(f"unexpected GET {module}{endpoint}")

    async def get_async(
        self, *, module: str, endpoint: str, timeout: int | None = None, **_: Any
    ) -> _FakeResponse:
        return self.get(module=module, endpoint=endpoint, timeout=timeout)

    async def post_async(
        self,
        *,
        module: str,
        endpoint: str,
        data: dict,
        timeout: int | None = None,
        **_: Any,
    ) -> _FakeResponse:
        self.calls.append(("POST", module, endpoint))
        handler = self._handlers.get((module, endpoint))
        if handler is None:
            raise AssertionError(f"unexpected POST {module}{endpoint}")
        return _FakeResponse(200, handler(data))


@pytest.fixture
def orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> module_orchestrator.ModuleOrchestrator:
    """Orchestrator whose transport is the scripted docstring-sanctioned seam."""
    config_path = tmp_path / "orchestrator_config.yaml"
    config_path.write_text(_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setattr(module_orchestrator, "APIConnector", _ScriptedAPIConnector)
    orchestrator = module_orchestrator.ModuleOrchestrator(config_path=str(config_path))
    # Both declared modules were health-checked through the injected transport.
    health = orchestrator.get_module_health()
    assert health["DATA"] is ModuleStatus.AVAILABLE
    assert health["COG"] is ModuleStatus.AVAILABLE
    return orchestrator


async def test_sequential_workflow_executes_steps_in_order_and_chains_data(
    orchestrator: module_orchestrator.ModuleOrchestrator,
) -> None:
    """A sequential workflow runs every step in order and chains step outputs."""
    connector = orchestrator.api_connector
    assert isinstance(connector, _ScriptedAPIConnector)
    connector.register("DATA", "/ingest", lambda data: {"records": 3})
    connector.register("DATA", "/enrich", lambda data: {"enriched": data["raw"] * 10})
    connector.register(
        "COG",
        "/report",
        lambda data: {"summary": f"{data['records']}:{data['enriched']}"},
    )

    workflow = WorkflowDefinition(
        id="gs19_sequential_pipeline",
        name="Sequential Pipeline",
        description="Engine execution probe: ordering and data chaining.",
        steps=[
            WorkflowStep(name="ingest", module="DATA", endpoint="/ingest"),
            WorkflowStep(
                name="enrich",
                module="DATA",
                endpoint="/enrich",
                dependencies=["ingest"],
                input_mapping={"raw": "records"},
            ),
            WorkflowStep(
                name="report",
                module="COG",
                endpoint="/report",
                dependencies=["enrich"],
            ),
        ],
        execution_strategy="sequential",
    )
    assert orchestrator.register_workflow(workflow) is True

    result = await orchestrator.execute_workflow(
        "gs19_sequential_pipeline", {"source": "synthetic"}
    )

    assert result.success is True
    assert list(result.data) == ["ingest", "enrich", "report"]
    assert result.data["ingest"] == {"records": 3}
    assert result.data["enrich"] == {"enriched": 30}
    assert result.data["report"] == {"summary": "3:30"}

    post_calls = [call for call in connector.calls if call[0] == "POST"]
    assert post_calls == [
        ("POST", "DATA", "/ingest"),
        ("POST", "DATA", "/enrich"),
        ("POST", "COG", "/report"),
    ]

    execution = orchestrator.get_workflow_status(result.metadata["execution_id"])
    assert execution is not None
    assert execution.status == "completed"
    assert execution.results == result.data
    assert execution.module_statuses == {
        "DATA": ModuleStatus.AVAILABLE,
        "COG": ModuleStatus.AVAILABLE,
    }
    assert result.metadata["workflow_id"] == "gs19_sequential_pipeline"


async def test_feedback_loop_reaches_convergence_and_exits_converged(
    orchestrator: module_orchestrator.ModuleOrchestrator,
) -> None:
    """A contracting belief update converges before exhausting max_iterations."""
    connector = orchestrator.api_connector
    assert isinstance(connector, _ScriptedAPIConnector)
    # Geometric contraction toward the fixed point 10.0 (factor 0.5 per step).
    connector.register(
        "COG",
        "/update",
        lambda data: {"estimate": 10.0 + (data["estimate"] - 10.0) * 0.5},
    )

    workflow = WorkflowDefinition(
        id="gs19_belief_convergence",
        name="Belief Convergence",
        description="Engine execution probe: feedback loop reaching convergence.",
        steps=[
            WorkflowStep(
                name="update_belief",
                module="COG",
                endpoint="/update",
                feedback_mapping={"estimate": "estimate"},
            )
        ],
        execution_strategy="feedback_loop",
        max_iterations=50,
        convergence_threshold=0.01,
    )
    assert orchestrator.register_workflow(workflow) is True

    result = await orchestrator.execute_workflow(
        "gs19_belief_convergence", {"estimate": 11.0}
    )

    assert result.success is True
    # 10.5 -> 10.25 -> 10.125 -> 10.0625: the relative change 0.0625/10.125
    # drops below the 0.01 threshold at the fourth iteration, so the loop
    # breaks there instead of running all 50 iterations.
    assert result.metadata["iterations"] == 4
    assert result.metadata["iterations"] < workflow.max_iterations
    assert list(result.data) == [f"iteration_{i}" for i in range(4)]

    estimates = [
        result.data[f"iteration_{i}"]["update_belief"]["estimate"] for i in range(4)
    ]
    assert estimates[-1] == pytest.approx(10.0625)
    final_change = abs(estimates[-1] - estimates[-2]) / abs(estimates[-2])
    assert final_change < workflow.convergence_threshold


class TestCheckConvergenceSemantics:
    """Documented convergence semantics preserved; malformed data now raises."""

    def test_fewer_than_two_iterations_is_not_converged(
        self, orchestrator: module_orchestrator.ModuleOrchestrator
    ) -> None:
        results = {"iteration_0": {"update_belief": {"estimate": 10.5}}}
        assert orchestrator._check_convergence(results, 0.01) is False

    def test_no_common_numeric_keys_is_not_converged(
        self, orchestrator: module_orchestrator.ModuleOrchestrator
    ) -> None:
        results = {
            "iteration_0": {"a": {"value": 1.0}},
            "iteration_1": {"b": {"other": 1.0}},
        }
        assert orchestrator._check_convergence(results, 0.01) is False

    def test_stabilized_series_converges(
        self, orchestrator: module_orchestrator.ModuleOrchestrator
    ) -> None:
        results = {
            "iteration_0": {"update_belief": {"estimate": 1.0}},
            "iteration_1": {"update_belief": {"estimate": 1.0001}},
        }
        assert orchestrator._check_convergence(results, 0.001) is True

    def test_malformed_numeric_leaf_raises_instead_of_silent_false(
        self, orchestrator: module_orchestrator.ModuleOrchestrator
    ) -> None:
        """An int leaf beyond float range must not read as "never converged"."""
        results = {
            "iteration_0": {"update_belief": {"estimate": 1}},
            "iteration_1": {"update_belief": {"estimate": 10**400}},
        }
        with pytest.raises(OverflowError):
            orchestrator._check_convergence(results, 0.01)
