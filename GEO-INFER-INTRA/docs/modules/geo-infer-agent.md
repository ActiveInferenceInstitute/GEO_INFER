# GEO-INFER-AGENT: Multi-Agent Systems

`GEO-INFER-AGENT` owns the `geo_infer_agent` package under `GEO-INFER-AGENT/src/`.
GEO-INFER-AGENT - Autonomous agent framework for geospatial applications

## Public import surface

These names are exported by the current owning package:

- `geo_infer_agent.BaseAgent`
- `geo_infer_agent.AgentState`
- `geo_infer_agent.AgentRegistry`
- `geo_infer_agent.BDIAgent`
- `geo_infer_agent.BDIState`
- `geo_infer_agent.ActiveInferenceAgent`
- `geo_infer_agent.ActiveInferenceState`
- `geo_infer_agent.GenerativeModel`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_agent
from geo_infer_agent import BaseAgent, AgentState, AgentRegistry, BDIAgent
assert all(value is not None for value in (BaseAgent, AgentState, AgentRegistry, BDIAgent,))
assert geo_infer_agent.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module AGENT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-AGENT/src/geo_infer_agent/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-AGENT/README.md)
- [Module operating contract](../../../GEO-INFER-AGENT/AGENTS.md)
- [Regression: test_bdi_lifecycle.py](../../../GEO-INFER-AGENT/tests/integration/test_bdi_lifecycle.py)
- [Regression: test_active_inference.py](../../../GEO-INFER-AGENT/tests/unit/models/test_active_inference.py)
- [Regression: test_bdi.py](../../../GEO-INFER-AGENT/tests/unit/models/test_bdi.py)
- [Example source: active_inference_geospatial.py](../../../GEO-INFER-AGENT/examples/active_inference_geospatial.py)
- [Example source: agent_examples.py](../../../GEO-INFER-AGENT/examples/agent_examples.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
