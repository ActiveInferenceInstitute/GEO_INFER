---
name: geo-infer-sim
description: Agent-based and multi-paradigm simulation for geospatial workflows. Use when building agent-based models, cellular automata, or system-dynamics simulations, or running and comparing spatial planning scenarios.
prerequisites:
  required:
    - geo-infer-space
    - geo-infer-time
  recommended:
    - geo-infer-bayes
    - geo-infer-act
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-SIM

## Instructions

### Core Capabilities

- **Simulation engine**: `SimulationEngine` with a validated `SimulationConfig`, explicit state machine (INITIALIZED/RUNNING/PAUSED/COMPLETED/FAILED/CANCELLED), pause/resume/cancel, metrics, state history, and JSON checkpoints that restore both configuration and the exact RNG stream (`load_checkpoint` raises `ValueError` when a checkpoint lacks the `config` or `rng_state` snapshot)
- **Paradigms**: agent-based models (`AgentBasedModel`, `Agent`), system dynamics (`SystemDynamicsModel`), cellular automata (`CellularAutomata`)
- **Mesa bridge** (optional): `MesaModelBridge` wraps any `mesa.Model` with snapshot/metric collection; install with the `mesa` extra — everything else works without it
- **Scenario analysis**: `ScenarioManager` for scenario storage, comparison, and sequential or parallel batch execution with per-scenario error reporting
- **Module simulations**: `ModuleSimulations` runs toy numeric engine smoke-test models named after GEO-INFER modules

### Key Imports

```python
from geo_infer_sim import (
    SimulationEngine, SimulationConfig, AgentBasedModel, Agent,
    SystemDynamicsModel, CellularAutomata, ScenarioManager,
    ModuleSimulations, ModuleSimulationConfig,
)
from geo_infer_sim.core.mesa_bridge import MesaModelBridge
from geo_infer_sim.scenarios.scenario_manager import Scenario
```

### Examples

Run a deterministic engine simulation:

```python
from geo_infer_sim import SimulationEngine, SimulationConfig

engine = SimulationEngine(SimulationConfig(time_step=1.0, max_time=10.0, random_seed=42))
engine.initialize({"population": 100})
engine.run(lambda t, state: {"population": state["population"] + 1})
engine.record_metric("population", 101.0)
stats = engine.get_metric_statistics("population")
```

Run and compare scenarios:

```python
from geo_infer_sim import ScenarioManager

manager = ScenarioManager()
baseline = manager.create_scenario(
    name="baseline", initial_conditions={"population": 100}, parameters={"seed": 1}
)
policy = manager.create_scenario(
    name="policy", initial_conditions={"population": 100}, parameters={"seed": 1}
)

def my_simulation_func(scenario):
    engine = SimulationEngine(SimulationConfig(max_time=5.0, random_seed=scenario.parameters["seed"]))
    engine.initialize(scenario.initial_conditions)
    engine.run(lambda t, state: {"population": state["population"] * 1.01})
    return engine.get_state()

results = manager.run_scenarios(
    [baseline.scenario_id, policy.scenario_id], my_simulation_func, parallel=False
)
```

## Guidelines

- Deterministic-by-default: pass `random_seed` to `SimulationConfig`; use `np.random.default_rng(seed)` for your own stochastic functions rather than global NumPy seeding
- Mesa is optional (`pip install geo-infer-sim[mesa]`); `MesaModelBridge` raises a clear ImportError on construction without it
- ABM `spatial_bounds` is advisory metadata — agent positions are not clamped; `AgentBasedModel(neighbor_radius=...)` controls step() neighbor search
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SIM`

### Integrations

- None: this module has no runtime imports of other GEO-INFER modules.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SIM`
