# GEO-INFER-OPS: Operations

`GEO-INFER-OPS` owns the `geo_infer_ops` package under `GEO-INFER-OPS/src/`.
GEO-INFER-OPS: Operations and infrastructure management for GEO-INFER framework.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_ops.setup_monitoring`
- `geo_infer_ops.load_config`
- `geo_infer_ops.get_config`
- `geo_infer_ops.setup_testing`
- `geo_infer_ops.Orchestrator`
- `geo_infer_ops.Task`
- `geo_infer_ops.TaskStatus`
- `geo_infer_ops.DeploymentManager`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_ops
from geo_infer_ops import setup_monitoring, load_config, get_config, setup_testing
assert all(value is not None for value in (setup_monitoring, load_config, get_config, setup_testing,))
assert geo_infer_ops.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module OPS --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-OPS/src/geo_infer_ops/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-OPS/README.md)
- [Module operating contract](../../../GEO-INFER-OPS/AGENTS.md)
- [Regression: test_deployment_integration.py](../../../GEO-INFER-OPS/tests/integration/test_deployment_integration.py)
- [Regression: test_acceptance_ops.py](../../../GEO-INFER-OPS/tests/test_acceptance_ops.py)
- [Regression: test_cache.py](../../../GEO-INFER-OPS/tests/test_cache.py)
- [Example source: demo_framework.py](../../../GEO-INFER-OPS/examples/demo_framework.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
