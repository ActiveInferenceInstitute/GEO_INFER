# GEO-INFER-TRANSPORT: Transportation Systems Module

`GEO-INFER-TRANSPORT` owns the `geo_infer_transport` package under `GEO-INFER-TRANSPORT/src/`.
GEO-INFER-TRANSPORT: Transportation Analysis Module

## Public import surface

These names are exported by the current owning package:

- `geo_infer_transport.TransportNetwork`
- `geo_infer_transport.RoutingEngine`
- `geo_infer_transport.TrafficAnalyzer`
- `geo_infer_transport.AccessibilityAnalyzer`
- `geo_infer_transport.TransitOptimizer`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_transport
from geo_infer_transport import TransportNetwork, RoutingEngine, TrafficAnalyzer, AccessibilityAnalyzer
assert all(value is not None for value in (TransportNetwork, RoutingEngine, TrafficAnalyzer, AccessibilityAnalyzer,))
assert geo_infer_transport.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module TRANSPORT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-TRANSPORT/src/geo_infer_transport/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-TRANSPORT/README.md)
- [Module operating contract](../../../GEO-INFER-TRANSPORT/AGENTS.md)
- [Regression: test_network_workflow.py](../../../GEO-INFER-TRANSPORT/tests/integration/test_network_workflow.py)
- [Regression: test_accessibility.py](../../../GEO-INFER-TRANSPORT/tests/test_accessibility.py)
- [Regression: test_network_routing.py](../../../GEO-INFER-TRANSPORT/tests/test_network_routing.py)
- [Example source: multimodal_analysis.py](../../../GEO-INFER-TRANSPORT/examples/multimodal_analysis.py)
- [Example source: traffic_simulation.py](../../../GEO-INFER-TRANSPORT/examples/traffic_simulation.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
