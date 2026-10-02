# GEO-INFER-AG: Agricultural Systems

`GEO-INFER-AG` owns the `geo_infer_ag` package under `GEO-INFER-AG/src/`.
GEO-INFER-AG: Agricultural geospatial inference and analysis module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_ag.AgriculturalAnalysis`
- `geo_infer_ag.FieldBoundaryManager`
- `geo_infer_ag.SeasonalAnalysis`
- `geo_infer_ag.SustainabilityAssessment`
- `geo_infer_ag.CropYieldModel`
- `geo_infer_ag.SoilHealthModel`
- `geo_infer_ag.WaterUsageModel`
- `geo_infer_ag.CarbonSequestrationModel`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_ag
from geo_infer_ag import AgriculturalAnalysis, FieldBoundaryManager, SeasonalAnalysis, SustainabilityAssessment
assert all(value is not None for value in (AgriculturalAnalysis, FieldBoundaryManager, SeasonalAnalysis, SustainabilityAssessment,))
assert geo_infer_ag.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module AG --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-AG/src/geo_infer_ag/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-AG/README.md)
- [Module operating contract](../../../GEO-INFER-AG/AGENTS.md)
- [Regression: test_agricultural_workflow.py](../../../GEO-INFER-AG/tests/integration/test_agricultural_workflow.py)
- [Regression: test_agricultural_api.py](../../../GEO-INFER-AG/tests/unit/api/test_agricultural_api.py)
- [Regression: test_resources.py](../../../GEO-INFER-AG/tests/unit/api/test_resources.py)
- [Example source: basic_agricultural_analysis.py](../../../GEO-INFER-AG/examples/basic_agricultural_analysis.py)
- [Example source: precision_agriculture.py](../../../GEO-INFER-AG/examples/precision_agriculture.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
