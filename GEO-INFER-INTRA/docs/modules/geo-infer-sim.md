# GEO-INFER-SIM: Simulation

`GEO-INFER-SIM` owns the `geo_infer_sim` package under `GEO-INFER-SIM/src/`.
GEO-INFER-SIM: Simulation Environments for Geospatial Analysis

## Public import surface

These names are exported by the current owning package:

- `geo_infer_sim.SimulationEngine`
- `geo_infer_sim.SimulationConfig`
- `geo_infer_sim.AgentBasedModel`
- `geo_infer_sim.Agent`
- `geo_infer_sim.SystemDynamicsModel`
- `geo_infer_sim.CellularAutomata`
- `geo_infer_sim.ScenarioManager`
- `geo_infer_sim.ModuleSimulations`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_sim
from geo_infer_sim import SimulationEngine, SimulationConfig, AgentBasedModel, Agent
assert all(value is not None for value in (SimulationEngine, SimulationConfig, AgentBasedModel, Agent,))
assert geo_infer_sim.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module SIM --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-SIM/src/geo_infer_sim/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-SIM/README.md)
- [Module operating contract](../../../GEO-INFER-SIM/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-SIM/tests/integration/test_integration.py)
- [Regression: test_simulation_engine_integration.py](../../../GEO-INFER-SIM/tests/test_simulation_engine_integration.py)
- [Regression: test_abm.py](../../../GEO-INFER-SIM/tests/unit/test_abm.py)
- [Example source: basic_abm.py](../../../GEO-INFER-SIM/examples/basic_abm.py)
- [Example source: module_simulations_example.py](../../../GEO-INFER-SIM/examples/module_simulations_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
