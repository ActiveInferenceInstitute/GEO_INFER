# GEO-INFER-APP: Application Framework

`GEO-INFER-APP` owns the `geo_infer_app` package under `GEO-INFER-APP/src/`.
GEO-INFER-APP

## Public import surface

These names are exported by the current owning package:

- `geo_infer_app.AgentInterface`
- `geo_infer_app.AgentState`
- `geo_infer_app.AgentType`
- `geo_infer_app.AgentFactory`
- `geo_infer_app.AgentVisualization`
- `geo_infer_app.AgentConfiguration`
- `geo_infer_app.AgentAPIClient`
- `geo_infer_app.AgentManager`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_app
from geo_infer_app import AgentInterface, AgentState, AgentType, AgentFactory
assert all(value is not None for value in (AgentInterface, AgentState, AgentType, AgentFactory,))
assert geo_infer_app.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module APP --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-APP/src/geo_infer_app/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-APP/README.md)
- [Module operating contract](../../../GEO-INFER-APP/AGENTS.md)
- [Regression: test_agent_pipeline.py](../../../GEO-INFER-APP/tests/integration/test_agent_pipeline.py)
- [Regression: test_bdi_interface.py](../../../GEO-INFER-APP/tests/unit/models/test_bdi_interface.py)
- [Regression: test_agent_api.py](../../../GEO-INFER-APP/tests/unit/test_agent_api.py)
- [Example source: bdi_agent_example.py](../../../GEO-INFER-APP/examples/agent_examples/bdi_agent_example.py)
- [Example source: agent_integration.py](../../../GEO-INFER-APP/examples/agent_integration.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
