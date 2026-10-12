# GEO-INFER-METAGOV: Meta-Governance & Organizational Governance Module

`GEO-INFER-METAGOV` owns the `geo_infer_metagov` package under `GEO-INFER-METAGOV/src/`.
GEO-INFER-METAGOV: Meta-Governance and Organizational Governance Methods

## Public import surface

These names are exported by the current owning package:

- `geo_infer_metagov.MultiLevelGovernanceFramework`
- `geo_infer_metagov.InstitutionalDesigner`
- `geo_infer_metagov.StakeholderGovernanceCoordinator`
- `geo_infer_metagov.PolycentricGovernanceSystem`
- `geo_infer_metagov.AdaptiveGovernanceSystem`
- `geo_infer_metagov.AccountabilityFramework`
- `geo_infer_metagov.ConflictResolver`
- `geo_infer_metagov.ConflictResolutionMethod`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_metagov
from geo_infer_metagov import MultiLevelGovernanceFramework, InstitutionalDesigner, StakeholderGovernanceCoordinator, PolycentricGovernanceSystem
assert all(value is not None for value in (MultiLevelGovernanceFramework, InstitutionalDesigner, StakeholderGovernanceCoordinator, PolycentricGovernanceSystem,))
assert geo_infer_metagov.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module METAGOV --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-METAGOV/src/geo_infer_metagov/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-METAGOV/README.md)
- [Module operating contract](../../../GEO-INFER-METAGOV/AGENTS.md)
- [Regression: test_norms_integration.py](../../../GEO-INFER-METAGOV/tests/integration/test_norms_integration.py)
- [Regression: test_org_integration.py](../../../GEO-INFER-METAGOV/tests/integration/test_org_integration.py)
- [Regression: test_sec_integration.py](../../../GEO-INFER-METAGOV/tests/integration/test_sec_integration.py)
- [Example source: advanced_integration_example.py](../../../GEO-INFER-METAGOV/examples/advanced_integration_example.py)
- [Example source: basic_example.py](../../../GEO-INFER-METAGOV/examples/basic_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
