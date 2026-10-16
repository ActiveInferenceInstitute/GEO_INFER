# GEO-INFER-EXAMPLES: Cross-Module Examples

`GEO-INFER-EXAMPLES` owns the `geo_infer_examples` package under `GEO-INFER-EXAMPLES/src/`.
GEO-INFER-EXAMPLES: Comprehensive demonstration framework for the GEO-INFER ecosystem.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_examples.LocalAPIConnector`
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

## Compose real local operations

Supply `LocalAPIConnector` to execute native module functions without a remote
server. Only explicitly registered `(module, endpoint)` callables are admitted.
Each takes an independently copied input dictionary and returns a dictionary,
synchronously or asynchronously. Native arrays and DataFrames can remain native
objects between local steps.

From the shared workspace environment:

```python
import asyncio
from geo_infer_examples import LocalAPIConnector, ModuleOrchestrator
from geo_infer_examples.models.integration_models import WorkflowDefinition, WorkflowStep
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

indexer = SpatialIndexingInterface(backend="h3")

def index_points(data):
    return {"cells": [indexer.latlng_to_cell(lat, lon, resolution=9)
                      for lat, lon in data["points"]]}

def summarize(data):
    return {"unique_cells": len(set(data["cells"]))}

engine = ModuleOrchestrator(api_connector=LocalAPIConnector({
    ("SPACE", "/index"): index_points,
    ("SPACE", "/summarize"): summarize,
}))
workflow = WorkflowDefinition(
    id="local_h3", name="Local H3", description="Index and summarize points",
    steps=[WorkflowStep("index", "SPACE", "/index"),
           WorkflowStep("summarize", "SPACE", "/summarize", dependencies=["index"])],
)
assert engine.register_workflow(workflow)
result = asyncio.run(engine.execute_workflow("local_h3", {
    "points": [(37.7, -122.4), (37.7, -122.4)],
}))
assert result.data["summarize"]["unique_cells"] == 1
```

The [local composition regression](../../../GEO-INFER-EXAMPLES/tests/integration/test_local_composition.py)
executes a larger DATA→SPACE→TIME→ENERGY chain: persist and retrieve sensor
observations, index real H3 cells, fit temporal trends, and forecast demand. Both
sequential and parallel scheduling preserve mapped outputs and dependency order.

Workflow definitions reject duplicate step names, absent dependencies, cycles,
and invalid budgets. Independent parallel operations are cancelled and awaited
when required work fails or the caller cancels. Execution identifiers are unique
across overlapping calls. Event and feedback strategies accept the same local
connector; feedback convergence uses the latest numeric iteration indices. Event
signals remain available until dependencies finish; untriggered required work
fails instead of reporting partial success. Each invocation copies its workflow
definition so later caller edits cannot redirect an in-flight attempt.

`WorkflowStep.timeout` bounds each async operation; `WorkflowDefinition.timeout`
bounds an execution attempt. Synchronous local callables run on the event loop
and can only be diagnosed as over-budget after returning. A hard termination
requirement needs an owned subprocess adapter. Cancellation does not undo an
operation's completed external side effects.

Required failures propagate. Retries require `retry_policy={"max_attempts": 2,
"delay_seconds": 0}` (two total attempts, at most ten supported); callers must
choose operations safe to repeat. Recovery never marks required steps optional.
A recovered result records its successful execution ID, `recovery_attempts`, and
`recovery_of`; the original failed execution remains inspectable. No retry policy
means one attempt. The default connector still requires an explicitly configured
transport, and bundled remote workflow definitions do not create services.

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
