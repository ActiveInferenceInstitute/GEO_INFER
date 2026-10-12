# GEO-INFER-RISK: Risk Assessment

`GEO-INFER-RISK` owns the `geo_infer_risk` package under `GEO-INFER-RISK/src/`.
GEO-INFER-RISK: Geospatial Risk Analysis and Catastrophe Modeling Framework

## Public import surface

These names are exported by the current owning package:

- `geo_infer_risk.CRESCENT_CITY_GEO_INTEL_SCHEMA`
- `geo_infer_risk.CivicHazardDomain`
- `geo_infer_risk.CrescentCityAnchor`
- `geo_infer_risk.CrescentCityBounds`
- `geo_infer_risk.CrescentCityHazardIntel`
- `geo_infer_risk.MunicipalCodeSection`
- `geo_infer_risk.EnhancedRiskEngine`
- `geo_infer_risk.RiskModel`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_risk
from geo_infer_risk import CRESCENT_CITY_GEO_INTEL_SCHEMA, CivicHazardDomain, CrescentCityAnchor, CrescentCityBounds
assert all(value is not None for value in (CRESCENT_CITY_GEO_INTEL_SCHEMA, CivicHazardDomain, CrescentCityAnchor, CrescentCityBounds,))
assert geo_infer_risk.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module RISK --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-RISK/src/geo_infer_risk/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-RISK/README.md)
- [Module operating contract](../../../GEO-INFER-RISK/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-RISK/tests/integration/test_integration.py)
- [Regression: test_aal_exposure_years.py](../../../GEO-INFER-RISK/tests/unit/test_aal_exposure_years.py)
- [Regression: test_aep_pml_curve.py](../../../GEO-INFER-RISK/tests/unit/test_aep_pml_curve.py)
- [Example source: basic_risk_assessment.py](../../../GEO-INFER-RISK/examples/basic_risk_assessment.py)
- [Example source: comprehensive_risk_assessment.py](../../../GEO-INFER-RISK/examples/comprehensive_risk_assessment.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
