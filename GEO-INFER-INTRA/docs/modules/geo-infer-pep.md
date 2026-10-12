# GEO-INFER-PEP: People & Communities

`GEO-INFER-PEP` owns the `geo_infer_pep` package under `GEO-INFER-PEP/src/`.
GEO-INFER-PEP: People, Engagement, Performance operations module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_pep.Address`
- `geo_infer_pep.Candidate`
- `geo_infer_pep.CandidateStatus`
- `geo_infer_pep.Compensation`
- `geo_infer_pep.Customer`
- `geo_infer_pep.Employee`
- `geo_infer_pep.EmploymentStatus`
- `geo_infer_pep.Gender`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_pep
from geo_infer_pep import Address, Candidate, CandidateStatus, Compensation
assert all(value is not None for value in (Address, Candidate, CandidateStatus, Compensation,))
assert geo_infer_pep.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module PEP --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-PEP/src/geo_infer_pep/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-PEP/README.md)
- [Module operating contract](../../../GEO-INFER-PEP/AGENTS.md)
- [Regression: test_pep_integration.py](../../../GEO-INFER-PEP/tests/integration/test_pep_integration.py)
- [Regression: test_api_error_handling.py](../../../GEO-INFER-PEP/tests/unit/test_api_error_handling.py)
- [Regression: test_crm.py](../../../GEO-INFER-PEP/tests/unit/test_crm.py)
- [Example source: basic_crm_example.py](../../../GEO-INFER-PEP/examples/basic_crm_example.py)
- [Example source: basic_hr_example.py](../../../GEO-INFER-PEP/examples/basic_hr_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
