# GEO-INFER-LOG: Logistics

`GEO-INFER-LOG` owns the `geo_infer_log` package under `GEO-INFER-LOG/src/`.
GEO-INFER-LOG: Geospatial Logistics Optimization Module

## Public import surface

These names are exported by the current owning package:

- `geo_infer_log.EnhancedLogger`
- `geo_infer_log.PerformanceMetrics`
- `geo_infer_log.LogAnalyzer`
- `geo_infer_log.SpatialLogContext`
- `geo_infer_log.get_logger`
- `geo_infer_log.LastMileRouter`
- `geo_infer_log.DeliveryScheduler`
- `geo_infer_log.ServiceAreaAnalyzer`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_log
from geo_infer_log import EnhancedLogger, PerformanceMetrics, LogAnalyzer, SpatialLogContext
assert all(value is not None for value in (EnhancedLogger, PerformanceMetrics, LogAnalyzer, SpatialLogContext,))
assert geo_infer_log.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module LOG --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-LOG/src/geo_infer_log/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-LOG/README.md)
- [Module operating contract](../../../GEO-INFER-LOG/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-LOG/tests/integration/test_integration.py)
- [Regression: test_routing.py](../../../GEO-INFER-LOG/tests/test_routing.py)
- [Regression: test_api_endpoints.py](../../../GEO-INFER-LOG/tests/unit/test_api_endpoints.py)
- [Example source: basic_routing_example.py](../../../GEO-INFER-LOG/examples/basic_routing_example.py)
- [Example source: last_mile_delivery.py](../../../GEO-INFER-LOG/examples/last_mile_delivery.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
