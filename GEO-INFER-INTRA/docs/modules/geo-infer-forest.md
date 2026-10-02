# GEO-INFER-FOREST: Forest Management Module

`GEO-INFER-FOREST` owns the `geo_infer_forest` package under `GEO-INFER-FOREST/src/`.
GEO-INFER-FOREST: Forest Management and Analysis Module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_forest.FireDangerRating`
- `geo_infer_forest.FireIncident`
- `geo_infer_forest.FireRiskAssessor`
- `geo_infer_forest.FireWeatherObservation`
- `geo_infer_forest.ForestHealthMonitor`
- `geo_infer_forest.ForestInventory`
- `geo_infer_forest.FuelType`
- `geo_infer_forest.CarbonSequestrationModeler`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_forest
from geo_infer_forest import FireDangerRating, FireIncident, FireRiskAssessor, FireWeatherObservation
assert all(value is not None for value in (FireDangerRating, FireIncident, FireRiskAssessor, FireWeatherObservation,))
assert geo_infer_forest.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module FOREST --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-FOREST/src/geo_infer_forest/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-FOREST/README.md)
- [Module operating contract](../../../GEO-INFER-FOREST/AGENTS.md)
- [Regression: test_forest_integration.py](../../../GEO-INFER-FOREST/tests/integration/test_forest_integration.py)
- [Regression: test_wildfire_risk.py](../../../GEO-INFER-FOREST/tests/test_wildfire_risk.py)
- [Regression: test_canopy_analysis.py](../../../GEO-INFER-FOREST/tests/unit/test_canopy_analysis.py)
- [Example source: __init__.py](../../../GEO-INFER-FOREST/examples/__init__.py)
- [Example source: basic_forest_analysis.py](../../../GEO-INFER-FOREST/examples/basic_forest_analysis.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
