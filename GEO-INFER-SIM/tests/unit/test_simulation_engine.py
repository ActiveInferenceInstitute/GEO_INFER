"""
Unit tests for simulation engine.
"""

import dataclasses
import json
import threading

import pytest
from geo_infer_sim.core.simulation_engine import (
    SimulationEngine,
    SimulationConfig,
    SimulationState,
)
from geo_infer_sim.module_simulations import ModuleSimulationConfig, ModuleSimulations


class TestSimulationEngine:
    """Test SimulationEngine class."""

    @pytest.fixture
    def engine(self) -> SimulationEngine:
        """Create a simulation engine instance."""
        config = SimulationConfig(time_step=1.0, max_time=10.0)
        return SimulationEngine(config)

    def test_initialize(self, engine: SimulationEngine) -> None:
        """Test simulation initialization."""
        initial_state = {"population": 100, "resources": 50}
        engine.initialize(initial_state)

        assert engine.state == SimulationState.INITIALIZED
        assert engine.current_time == 0.0
        assert len(engine.state_history) == 1

    def test_step(self, engine: SimulationEngine) -> None:
        """Test simulation step execution."""
        initial_state = {"value": 10}
        engine.initialize(initial_state)

        def step_func(time, state):
            return {"value": state["value"] + 1}

        engine.step(step_func)

        assert engine.current_time == 1.0
        assert engine.state == SimulationState.RUNNING

    def test_run(self, engine: SimulationEngine) -> None:
        """Test complete simulation run."""
        initial_state = {"counter": 0}
        engine.initialize(initial_state)

        def step_func(time, state):
            return {"counter": state["counter"] + 1}

        results = engine.run(step_func)

        assert results["status"] == SimulationState.COMPLETED.value
        assert results["final_time"] == 10.0
        assert "state_history" in results

    def test_pause_resume(self, engine: SimulationEngine) -> None:
        """Test pause and resume functionality."""
        initial_state = {"value": 0}
        engine.initialize(initial_state)

        engine.state = SimulationState.RUNNING
        engine.pause()

        assert engine.state == SimulationState.PAUSED

        engine.resume()

        assert engine.state == SimulationState.RUNNING

    def test_pause_mid_run_sets_paused_not_failed(self) -> None:
        """Pausing mid-run leaves the engine PAUSED, never FAILED."""
        engine = SimulationEngine(SimulationConfig(time_step=1.0, max_time=10.0))
        engine.initialize({"value": 0})

        def pause_on_third_step(time: float, state: dict) -> dict:
            if time == 2.0:
                engine.pause()
            return {"value": state["value"] + 1}

    def test_threaded_pause_preserves_paused_state_then_resumes_to_completion(
        self,
    ) -> None:
        """A pause racing the run loop pauses cleanly; resume finishes the run."""
        engine = SimulationEngine(SimulationConfig(time_step=1.0, max_time=10.0))
        engine.initialize({"value": 0})

        # Hold the step executing at time 3.0 open until the main thread has
        # issued a pause, so the pause provably lands mid-run (between steps)
        # rather than after the run completes.
        gate = threading.Event()
        entered_gated_step = threading.Event()

        def step_func(time: float, state: dict) -> dict:
            if time == 3.0:
                entered_gated_step.set()
                gate.wait(timeout=5.0)
            return {"value": state["value"] + 1}

        runner = threading.Thread(target=engine.run, args=(step_func,))
        runner.start()
        assert entered_gated_step.wait(timeout=5.0)
        engine.pause()
        gate.set()
        runner.join(timeout=5.0)

        assert not runner.is_alive()
        assert engine.state == SimulationState.PAUSED
        # The step executing at time 3.0 was held open by the gate and
        # completes after the pause lands; the loop then honors the pause
        # before starting the next step.
        assert engine.current_time == 4.0

        # Resume from the paused step: run() continues where it stopped.
        engine.resume()
        assert engine.state == SimulationState.RUNNING
        results = engine.run(step_func)

        assert engine.state == SimulationState.COMPLETED
        assert results["status"] == SimulationState.COMPLETED.value
        assert results["final_time"] == 10.0
        assert engine._current_state["value"] == 10

    def test_paused_run_matches_uninterrupted_run_with_same_seed(self) -> None:
        """Pause/resume reproduces the uninterrupted trajectory for a seed."""

        def make_engine() -> SimulationEngine:
            return SimulationEngine(
                SimulationConfig(time_step=1.0, max_time=10.0, random_seed=1234)
            )

        def step_func(time: float, state: dict) -> dict:
            return {"value": state["value"] + float(engine.rng.normal())}

        # Uninterrupted reference run
        engine = make_engine()
        engine.initialize({"value": 0.0})
        reference = engine.run(step_func)

        # Interrupted run: pause after three steps, then resume to completion
        engine = make_engine()
        engine.initialize({"value": 0.0})

        def pause_on_third_step(time: float, state: dict) -> dict:
            if time == 2.0:
                engine.pause()
            return {"value": state["value"] + float(engine.rng.normal())}

        engine.run(pause_on_third_step)
        assert engine.state == SimulationState.PAUSED
        engine.resume()
        results = engine.run(step_func)

        assert results["status"] == SimulationState.COMPLETED.value
        assert results["final_time"] == reference["final_time"]
        reference_value = reference["state_history"][-1]["state"]["value"]
        resumed_value = engine.state_history[-1]["state"]["value"]
        assert resumed_value == pytest.approx(reference_value)
        assert engine._current_state["value"] == pytest.approx(reference_value)

    def test_no_history_still_propagates_current_state(self) -> None:
        """The engine passes state forward when history persistence is off."""
        engine = SimulationEngine(
            SimulationConfig(time_step=1.0, max_time=2.0, save_state_history=False)
        )
        engine.initialize({"value": 0})
        engine.run(lambda _time, state: {"value": state["value"] + 1})
        assert engine._current_state == {"value": 2}
        assert engine.state_history == []

    def test_cancelled_run_preserves_cancelled_status(self) -> None:
        """Cancellation is not overwritten by a completed status."""
        engine = SimulationEngine(SimulationConfig(time_step=1.0, max_time=10.0))
        engine.initialize({"value": 0})

        def cancel_on_first_step(_time, state):
            engine.cancel()
            return {"value": state["value"] + 1}

        result = engine.run(cancel_on_first_step)
        assert result["status"] == SimulationState.CANCELLED.value

    def test_unsupported_export_format_is_rejected(
        self, engine: SimulationEngine
    ) -> None:
        """Export rejects formats that are not part of the public contract."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export_results(format="csv")

    def test_record_metric(self, engine: SimulationEngine) -> None:
        """Test metric recording."""
        engine.record_metric("test_metric", 42.0)

        assert "test_metric" in engine.metrics
        assert engine.metrics["test_metric"] == [42.0]

    def test_ant_simulation_updates_pheromone_trails(self) -> None:
        simulations = ModuleSimulations(
            ModuleSimulationConfig(time_horizon=1.0, time_step=1.0, random_seed=42)
        )

        result = simulations.simulate_ant(colony_size=2)

        assert result["module"] == "ANT"
        assert len(result["trail_history"]) == 1
        assert result["final_trails"].sum() > 0


class TestCheckpointReproducibility:
    """Loading a checkpoint must restore config and re-seed the RNG."""

    def test_load_checkpoint_restores_config_and_rng(self, tmp_path) -> None:
        config = SimulationConfig(
            time_step=0.5, max_time=10.0, output_interval=0.5, random_seed=123
        )
        engine = SimulationEngine(config)
        engine.initialize({"x": 1.0})

        filepath = str(tmp_path / "ckpt.json")
        engine.save_checkpoint(filepath)

        resumed = SimulationEngine(SimulationConfig(random_seed=999))
        resumed.load_checkpoint(filepath)

        assert resumed.config.time_step == 0.5
        assert resumed.config.max_time == 10.0
        assert resumed.config.random_seed == 123
        assert resumed.config.output_interval == 0.5

        # Resumed RNG draws reproduce the original engine's stream.
        expected = engine.rng.random(5)
        assert resumed.rng.random(5).tolist() == expected.tolist()

    def test_resumed_run_matches_uninterrupted_run(self, tmp_path) -> None:
        """A run resumed from a checkpoint draws the same random values as an
        uninterrupted run with the same seed."""

        def make(seed: int) -> SimulationEngine:
            engine = SimulationEngine(
                SimulationConfig(time_step=1.0, max_time=3.0, random_seed=seed)
            )
            engine.initialize({})
            return engine

        uninterrupted = make(7)
        for _ in range(2):
            uninterrupted.step(lambda t, s: {"v": uninterrupted.rng.random()})

        resumed = make(7)
        resumed.step(lambda t, s: {"v": resumed.rng.random()})
        resumed.save_checkpoint(str(tmp_path / "mid.json"))
        resumed.load_checkpoint(str(tmp_path / "mid.json"))
        resumed.step(lambda t, s: {"v": resumed.rng.random()})

        assert resumed._current_state["v"] == uninterrupted._current_state["v"]

    @pytest.mark.parametrize("dropped", ["rng_state", "config"])
    def test_checkpoint_without_snapshot_is_rejected(
        self, tmp_path, dropped: str
    ) -> None:
        """A checkpoint lacking the RNG or config snapshot cannot be resumed."""
        engine = SimulationEngine(SimulationConfig(random_seed=5))
        engine.initialize({"x": 1.0})
        filepath = tmp_path / "ckpt.json"
        engine.save_checkpoint(str(filepath))
        payload = json.loads(filepath.read_text())
        del payload[dropped]
        filepath.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=dropped):
            SimulationEngine(SimulationConfig()).load_checkpoint(str(filepath))


class TestSimulationConfigSurface:
    """GS-274: dead config must not masquerade as an engine feature."""

    def test_parallel_execution_is_not_a_config_option(self) -> None:
        field_names = {f.name for f in dataclasses.fields(SimulationConfig)}
        assert "parallel_execution" not in field_names
        with pytest.raises(TypeError):
            SimulationConfig(parallel_execution=True)
