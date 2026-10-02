# GEO-INFER-COMMS: Communication Systems

`GEO-INFER-COMMS` owns the `geo_infer_comms` package under `GEO-INFER-COMMS/src/`.
GEO-INFER-COMMS: Geospatial Communications Infrastructure

## Public import surface

These names are exported by the current owning package:

- `geo_infer_comms.GeospatialCommunicationSystem`
- `geo_infer_comms.get_communication_system`
- `geo_infer_comms.configure_system`
- `geo_infer_comms.MessageBroker`
- `geo_infer_comms.MessageRouter`
- `geo_infer_comms.MessageFormatter`
- `geo_infer_comms.NotificationManager`
- `geo_infer_comms.NotificationMetrics`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_comms
from geo_infer_comms import GeospatialCommunicationSystem, get_communication_system, configure_system, MessageBroker
assert all(value is not None for value in (GeospatialCommunicationSystem, get_communication_system, configure_system, MessageBroker,))
assert geo_infer_comms.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module COMMS --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-COMMS/src/geo_infer_comms/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-COMMS/README.md)
- [Module operating contract](../../../GEO-INFER-COMMS/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-COMMS/tests/integration/test_integration.py)
- [Regression: test_websocket_broadcast.py](../../../GEO-INFER-COMMS/tests/integration/test_websocket_broadcast.py)
- [Regression: test_channels.py](../../../GEO-INFER-COMMS/tests/unit/test_channels.py)
- [Example source: basic_usage.py](../../../GEO-INFER-COMMS/examples/basic_usage.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
