# GEO-INFER-ANT: Ant Colony Optimization

`GEO-INFER-ANT` owns the `geo_infer_ant` package under `GEO-INFER-ANT/src/`.
GEO-INFER-ANT: Complex Adaptive Systems and Swarm Intelligence

## Public import surface

These names are exported by the current owning package:

- `geo_infer_ant.SwarmAgent`
- `geo_infer_ant.AgentPopulation`
- `geo_infer_ant.PheromoneSystem`
- `geo_infer_ant.DigitalStigmergy`
- `geo_infer_ant.AntColonyOptimization`
- `geo_infer_ant.ParticleSwarmOptimization`
- `geo_infer_ant.ArtificialBeeColony`
- `geo_infer_ant.EnvironmentalMonitoringSwarm`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_ant
from geo_infer_ant import SwarmAgent, AgentPopulation, PheromoneSystem, DigitalStigmergy
assert all(value is not None for value in (SwarmAgent, AgentPopulation, PheromoneSystem, DigitalStigmergy,))
assert geo_infer_ant.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ANT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ANT/src/geo_infer_ant/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ANT/README.md)
- [Module operating contract](../../../GEO-INFER-ANT/AGENTS.md)
- [Regression: test_swarm_integration.py](../../../GEO-INFER-ANT/tests/integration/test_swarm_integration.py)
- [Regression: test_performance.py](../../../GEO-INFER-ANT/tests/performance/test_performance.py)
- [Regression: test_algorithms.py](../../../GEO-INFER-ANT/tests/unit/test_algorithms.py)
- [Example source: swarm_intelligence_demo.py](../../../GEO-INFER-ANT/examples/swarm_intelligence_demo.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
