--

`GEO-INFER-ORG` owns the `geo_infer_org` package under `GEO-INFER-ORG/src/`.
GEO-INFER-ORG: Organizational Modeling

## Public import surface

These names are exported by the current owning package:

- `geo_infer_org.OrganizationModel`
- `geo_infer_org.OrgUnit`
- `geo_infer_org.Role`
- `geo_infer_org.Resource`
- `geo_infer_org.OrgStructureType`
- `geo_infer_org.RoleLevel`
- `geo_infer_org.OrgMetrics`
- `geo_infer_org.VotingEngine`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_org
from geo_infer_org import OrganizationModel, OrgUnit, Role, Resource
assert all(value is not None for value in (OrganizationModel, OrgUnit, Role, Resource,))
assert geo_infer_org.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ORG --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ORG/src/geo_infer_org/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ORG/README.md)
- [Module operating contract](../../../GEO-INFER-ORG/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-ORG/tests/integration/test_integration.py)
- [Regression: test_acceptance_org.py](../../../GEO-INFER-ORG/tests/unit/test_acceptance_org.py)
- [Regression: test_collaboration.py](../../../GEO-INFER-ORG/tests/unit/test_collaboration.py)
- [Example source: basic_governance_example.py](../../../GEO-INFER-ORG/examples/basic_governance_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
