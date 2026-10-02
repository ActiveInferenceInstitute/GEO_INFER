# GEO-INFER-TEST: Testing Framework

`GEO-INFER-TEST` owns the `geo_infer_test` package under `GEO-INFER-TEST/src/`.
GEO-INFER-TEST: Comprehensive Testing and Quality Assurance Module

## Public import surface

These names are exported by the current owning package:

- `geo_infer_test.GeoInferTestRunner`
- `geo_infer_test.TestConfiguration`
- `geo_infer_test.TestOutcome`
- `geo_infer_test.ValidationRule`
- `geo_infer_test.BaseValidator`
- `geo_infer_test.DataQualityValidator`
- `geo_infer_test.PerformanceValidator`
- `geo_infer_test.SpatialValidator`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_test
from geo_infer_test import GeoInferTestRunner, TestConfiguration, TestOutcome, ValidationRule
assert all(value is not None for value in (GeoInferTestRunner, TestConfiguration, TestOutcome, ValidationRule,))
assert geo_infer_test.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module TEST --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-TEST/src/geo_infer_test/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-TEST/README.md)
- [Module operating contract](../../../GEO-INFER-TEST/AGENTS.md)
- [Regression: test_act_agent_ant_coordination.py](../../../GEO-INFER-TEST/tests/integration/test_act_agent_ant_coordination.py)
- [Regression: test_ai_space_domain_integration.py](../../../GEO-INFER-TEST/tests/integration/test_ai_space_domain_integration.py)
- [Regression: test_cross_module.py](../../../GEO-INFER-TEST/tests/integration/test_cross_module.py)
- [Example source: basic_testing_example.py](../../../GEO-INFER-TEST/examples/basic_testing_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
