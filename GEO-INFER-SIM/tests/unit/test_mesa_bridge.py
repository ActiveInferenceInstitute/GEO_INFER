"""
Unit tests for the Mesa-backed simulation bridge (SIM-01).

These tests exercise :class:`geo_infer_sim.core.mesa_bridge.MesaModelBridge`
against real Mesa models. They are skipped automatically when the optional
``mesa`` dependency is not installed, so the rest of the SIM test suite remains
runnable in a minimal environment.

The completed SIM-01 acceptance criteria covered here are:

* a real Mesa-backed scenario runs,
* state/metric history is recorded,
* JSON and DataFrame artifacts are exported,
* cancellation state is preserved (not overwritten to ``completed``), and
* error state is preserved (``failed``, not ``completed``).
"""

import json

import mesa
import pytest
from geo_infer_sim.core.mesa_bridge import HAS_MESA, MesaModelBridge
from geo_infer_sim.core.simulation_engine import SimulationConfig, SimulationState

# ---------------------------------------------------------------------------
# Mesa model fixtures
# ---------------------------------------------------------------------------


class WealthAgent(mesa.Agent):
    """Boltzmann-wealth agent: gives 1 unit to a random other agent per step."""

    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1

    def step(self):
        if self.wealth == 0:
            return
        other = self.random.choice(self.model.agents)
        if other is None or other is self:
            return
        self.wealth -= 1
        other.wealth += 1


class WealthModel(mesa.Model):
    """A simple, deterministic-seeded Mesa model used across the tests."""

    def __init__(self, n_agents=5, rng=42):
        super().__init__(rng=rng)
        WealthAgent.create_agents(model=self, n=n_agents)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "total_wealth": lambda m: sum(a.wealth for a in m.agents),
                "num_agents": lambda m: len(m.agents),
            },
            agent_reporters={"wealth": "wealth"},
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


class CancelModel(mesa.Model):
    """A model that asks its bridge to cancel after a fixed number of steps."""

    def __init__(self, cancel_after=3, rng=42):
        super().__init__(rng=rng)
        self._cancel_after = cancel_after
        self.bridge: MesaModelBridge | None = None  # set by the test before run()
        self.datacollector = mesa.DataCollector(
            model_reporters={"steps": lambda m: m.steps}
        )

    def step(self):
        self.datacollector.collect(self)
        if self.steps >= self._cancel_after and self.bridge is not None:
            self.bridge.cancel()


class BoomModel(mesa.Model):
    """A model whose step() raises once it has run a couple of steps."""

    def __init__(self, boom_at=2, rng=42):
        super().__init__(rng=rng)
        self._boom_at = boom_at

    def step(self):
        if self.steps >= self._boom_at:
            raise RuntimeError(f"intentional boom at step {self.steps}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMesaModelBridgeAvailability:
    """Sanity checks on the optional-dependency gating."""

    def test_has_mesa_flag_matches_import(self):
        assert HAS_MESA is True

    def test_bridge_requires_mesa_like_object(self):
        with pytest.raises(TypeError):
            MesaModelBridge(model=object())  # no step() attribute

    def test_bridge_rejects_none_model(self):
        with pytest.raises(TypeError):
            MesaModelBridge(model=None)


class TestMesaBridgeRun:
    """End-to-end run of a real Mesa model through the bridge."""

    @pytest.fixture
    def bridge(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=10.0, random_seed=42)
        return MesaModelBridge(model, config=cfg)

    def test_run_completes_and_advances_time(self, bridge):
        results = bridge.run()
        assert results["status"] == SimulationState.COMPLETED.value
        assert results["final_time"] == 10.0
        # 10 steps of time_step=1.0 -> 10 Mesa steps.
        assert results["mesa_steps"] == 10

    def test_state_history_is_recorded(self, bridge):
        bridge.initialize()
        results = bridge.run()
        # One entry per step + the initial snapshot = 11 entries.
        assert len(results["state_history"]) == 11
        first = results["state_history"][0]
        assert first["time"] == 0.0
        # The default state extractor pulls DataCollector model reporters.
        assert "total_wealth" in first["state"]
        assert "num_agents" in first["state"]
        # Wealth is conserved in the Boltzmann model.
        for entry in results["state_history"]:
            assert entry["state"]["total_wealth"] == 5

    def test_metric_history_is_recorded(self, bridge):
        results = bridge.run()
        assert "total_wealth" in results["metrics"]
        assert len(results["metrics"]["total_wealth"]) == 10
        assert all(v == 5.0 for v in results["metrics"]["total_wealth"])
        assert "num_agents" in results["metrics"]

    def test_custom_metric_extractors(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=5.0)
        bridge = MesaModelBridge(
            model,
            config=cfg,
            metric_extractors={
                "max_wealth": lambda m: max(a.wealth for a in m.agents),
            },
        )
        results = bridge.run()
        assert "max_wealth" in results["metrics"]
        assert len(results["metrics"]["max_wealth"]) == 5
        assert all(v >= 1 for v in results["metrics"]["max_wealth"])

    def test_custom_state_extractor(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=3.0)

        def extractor(m):
            return {"richest": max(a.wealth for a in m.agents)}

        bridge = MesaModelBridge(model, config=cfg, state_extractor=extractor)
        results = bridge.run()
        assert len(results["state_history"]) == 4
        for entry in results["state_history"]:
            assert set(entry["state"].keys()) == {"richest"}


class TestMesaBridgeExport:
    """JSON and DataFrame export artifacts."""

    @pytest.fixture
    def bridge(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=5.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.run()
        return bridge

    def test_export_json_round_trip(self, bridge):
        payload = bridge.export_results(format="json")
        assert isinstance(payload, str)
        data = json.loads(payload)
        assert set(data.keys()) == {"state_history", "metrics", "events"}
        assert len(data["state_history"]) == 6
        assert "total_wealth" in data["metrics"]
        # JSON-serializable: no numpy types leak through.
        json.loads(payload)  # would raise on numpy int64 etc.

    def test_export_dataframe(self, bridge):
        import pandas as pd

        df = bridge.export_results(format="dataframe")
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert "total_wealth" in df.columns
        assert len(df) == 6  # initial + 5 steps
        assert (df["total_wealth"] == 5).all()

    def test_export_dict(self, bridge):
        d = bridge.export_results(format="dict")
        assert d["status"] == SimulationState.COMPLETED.value
        assert d["current_time"] == 5.0
        assert "state_history" in d
        assert "metrics" in d

    def test_export_unsupported_format_rejected(self, bridge):
        with pytest.raises(ValueError, match="Unsupported export format"):
            bridge.export_results(format="csv")

    def test_save_checkpoint_json(self, bridge, tmp_path):
        checkpoint = tmp_path / "checkpoint.json"
        bridge.save_checkpoint(str(checkpoint))
        with open(checkpoint) as f:
            data = json.load(f)
        assert data["current_time"] == 5.0
        assert data["state"] == SimulationState.COMPLETED.value
        assert "total_wealth" in data["metrics"]


class TestMesaBridgeCancellation:
    """Cancellation state must be preserved, not overwritten to completed."""

    def test_cancel_from_model_step_preserves_cancelled_status(self):
        model = CancelModel(cancel_after=3, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=20.0)
        bridge = MesaModelBridge(model, config=cfg)
        model.bridge = bridge  # let the model trigger cancellation
        bridge.initialize()
        results = bridge.run()
        assert results["status"] == SimulationState.CANCELLED.value
        # The run must stop early, well before max_time=20.
        assert results["final_time"] < 20.0
        assert results["mesa_steps"] == 3
        # And the Mesa model's running flag should have been flipped by cancel().
        assert results["mesa_running"] is False

    def test_external_cancel_between_steps(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=20.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.initialize()
        bridge.step()
        bridge.step()
        bridge.cancel()
        assert bridge.state == SimulationState.CANCELLED
        results = bridge.run()
        # Re-running after cancel must keep the CANCELLED status (loop breaks
        # immediately at the top-of-loop guard).
        assert results["status"] == SimulationState.CANCELLED.value
        assert results["final_time"] == 2.0

    def test_model_self_stop_completes_normally(self):
        """A model flipping running=False on its own is a clean completion."""

        class SelfStopModel(mesa.Model):
            def __init__(self, rng=42):
                super().__init__(rng=rng)
                self.datacollector = mesa.DataCollector(
                    model_reporters={"steps": lambda m: m.steps}
                )

            def step(self):
                self.datacollector.collect(self)
                if self.steps >= 4:
                    self.running = False

        model = SelfStopModel(rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=100.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.initialize()
        results = bridge.run()
        assert results["status"] == SimulationState.COMPLETED.value
        assert results["mesa_steps"] == 4
        assert results["final_time"] == 4.0


class TestMesaBridgeErrors:
    """Error state must be preserved as FAILED, not masked as completed."""

    def test_model_step_exception_raises_and_marks_failed(self):
        model = BoomModel(boom_at=2, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=10.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.initialize()
        with pytest.raises(RuntimeError, match="intentional boom"):
            bridge.run()
        assert bridge.state == SimulationState.FAILED
        # History before the failure is preserved.
        assert len(bridge.state_history) >= 2

    def test_failed_state_not_overwritten_by_subsequent_run(self):
        model = BoomModel(boom_at=1, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=10.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.initialize()
        with pytest.raises(RuntimeError):
            bridge.run()
        assert bridge.state == SimulationState.FAILED
        # A second run() on a FAILED engine: step() should refuse to advance
        # because the state is no longer INITIALIZED/RUNNING.
        with pytest.raises(ValueError, match="Cannot step simulation"):
            bridge.run()

    def test_export_from_failed_run_preserves_status(self):
        model = BoomModel(boom_at=2, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=10.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.initialize()
        with pytest.raises(RuntimeError):
            bridge.run()
        d = bridge.export_results(format="dict")
        assert d["status"] == SimulationState.FAILED.value


class TestMesaBridgeMetricStatistics:
    """The inherited metric-statistics helper works on Mesa-collected metrics."""

    def test_get_metric_statistics(self):
        model = WealthModel(n_agents=5, rng=42)
        cfg = SimulationConfig(time_step=1.0, max_time=5.0)
        bridge = MesaModelBridge(model, config=cfg)
        bridge.run()
        stats = bridge.get_metric_statistics("total_wealth")
        assert stats["count"] == 5
        assert stats["mean"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["trend"] == "stable"
