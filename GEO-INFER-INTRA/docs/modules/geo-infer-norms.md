# GEO-INFER-NORMS: Norms & Standards

`GEO-INFER-NORMS` owns the `geo_infer_norms` package under `GEO-INFER-NORMS/src/`.
GEO-INFER-NORMS: Social-technical compliance modeling with deterministic and probabilistic aspects.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_norms.legal_frameworks`
- `geo_infer_norms.zoning_analysis`
- `geo_infer_norms.compliance_tracking`
- `geo_infer_norms.policy_impact`
- `geo_infer_norms.normative_inference`
- `geo_infer_norms.legal_entity`
- `geo_infer_norms.regulation`
- `geo_infer_norms.compliance_status`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_norms
from geo_infer_norms import legal_frameworks, zoning_analysis, compliance_tracking, policy_impact
assert all(value is not None for value in (legal_frameworks, zoning_analysis, compliance_tracking, policy_impact,))
assert geo_infer_norms.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module NORMS --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-NORMS/src/geo_infer_norms/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-NORMS/README.md)
- [Module operating contract](../../../GEO-INFER-NORMS/AGENTS.md)
- [Regression: test_norms_integration.py](../../../GEO-INFER-NORMS/tests/integration/test_norms_integration.py)
- [Regression: test_acceptance_norms.py](../../../GEO-INFER-NORMS/tests/unit/test_acceptance_norms.py)
- [Regression: test_api_compliance.py](../../../GEO-INFER-NORMS/tests/unit/test_api_compliance.py)
- [Example source: minimal_zoning_example.py](../../../GEO-INFER-NORMS/examples/minimal_zoning_example.py)
- [Example source: zoning_analysis_example.py](../../../GEO-INFER-NORMS/examples/zoning_analysis_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
