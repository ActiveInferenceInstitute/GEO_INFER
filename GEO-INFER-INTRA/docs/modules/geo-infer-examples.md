# GEO-INFER-EXAMPLES: Cross-Module Examples

`GEO-INFER-EXAMPLES` owns the `geo_infer_examples` package under `GEO-INFER-EXAMPLES/src/`.
GEO-INFER-EXAMPLES: Comprehensive demonstration framework for the GEO-INFER ecosystem.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_examples.core`
- `geo_infer_examples.models`
- `geo_infer_examples.ExecutionStrategy`
- `geo_infer_examples.ModuleOrchestrator`
- `geo_infer_examples.ModuleStatus`
- `geo_infer_examples.WorkflowDefinition`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_examples
from geo_infer_examples import core, models, ExecutionStrategy, ModuleOrchestrator
assert all(value is not None for value in (core, models, ExecutionStrategy, ModuleOrchestrator,))
assert geo_infer_examples.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-EXAMPLES/src/geo_infer_examples/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-EXAMPLES/README.md)
- [Module operating contract](../../../GEO-INFER-EXAMPLES/AGENTS.md)
- [Regression: test_example_orchestration.py](../../../GEO-INFER-EXAMPLES/tests/integration/test_example_orchestration.py)
- [Regression: test_module_orchestrator_execution.py](../../../GEO-INFER-EXAMPLES/tests/integration/test_module_orchestrator_execution.py)
- [Regression: test_orchestrator_scripts_execute.py](../../../GEO-INFER-EXAMPLES/tests/integration/test_orchestrator_scripts_execute.py)
- [Example source: run_example.py](../../../GEO-INFER-EXAMPLES/examples/agriculture_integration/precision_farming_system/scripts/run_example.py)
- [Example source: dashboard_app.py](../../../GEO-INFER-EXAMPLES/examples/area_study/scripts/dashboard_app.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
