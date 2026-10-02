# GEO-INFER-REQ: Requirements Management

`GEO-INFER-REQ` owns the `geo_infer_req` package under `GEO-INFER-REQ/src/`.
GEO-INFER-REQ: Requirements and Dependencies

## Public import surface

These names are exported by the current owning package:

- `geo_infer_req.RequirementsAnalyzer`
- `geo_infer_req.Requirement`
- `geo_infer_req.RequirementType`
- `geo_infer_req.RequirementStatus`
- `geo_infer_req.PriorityLevel`
- `geo_infer_req.DependencyGraph`
- `geo_infer_req.CompletenessReport`
- `geo_infer_req.TraceabilityManager`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_req
from geo_infer_req import RequirementsAnalyzer, Requirement, RequirementType, RequirementStatus
assert all(value is not None for value in (RequirementsAnalyzer, Requirement, RequirementType, RequirementStatus,))
assert geo_infer_req.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module REQ --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-REQ/src/geo_infer_req/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-REQ/README.md)
- [Module operating contract](../../../GEO-INFER-REQ/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-REQ/tests/integration/test_integration.py)
- [Regression: test_acceptance_req.py](../../../GEO-INFER-REQ/tests/unit/test_acceptance_req.py)
- [Regression: test_core.py](../../../GEO-INFER-REQ/tests/unit/test_core.py)
- [Example source: basic_requirements_example.py](../../../GEO-INFER-REQ/examples/basic_requirements_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
