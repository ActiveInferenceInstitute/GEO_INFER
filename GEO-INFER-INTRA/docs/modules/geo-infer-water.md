# GEO-INFER-WATER: Water Resource Module

`GEO-INFER-WATER` owns the `geo_infer_water` package under `GEO-INFER-WATER/src/`.
GEO-INFER-WATER: Water Resources Management Module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_water.HydrologicalModeler`
- `geo_infer_water.WaterQualityAssessor`
- `geo_infer_water.WaterSample`
- `geo_infer_water.WaterBodyType`
- `geo_infer_water.PollutantType`
- `geo_infer_water.WaterInfrastructurePlanner`
- `geo_infer_water.FloodDroughtAnalyzer`
- `geo_infer_water.WatershedDelineator`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_water
from geo_infer_water import HydrologicalModeler, WaterQualityAssessor, WaterSample, WaterBodyType
assert all(value is not None for value in (HydrologicalModeler, WaterQualityAssessor, WaterSample, WaterBodyType,))
assert geo_infer_water.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module WATER --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-WATER/src/geo_infer_water/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-WATER/README.md)
- [Module operating contract](../../../GEO-INFER-WATER/AGENTS.md)
- [Regression: test_water_integration.py](../../../GEO-INFER-WATER/tests/integration/test_water_integration.py)
- [Regression: test_flood_drought.py](../../../GEO-INFER-WATER/tests/unit/test_flood_drought.py)
- [Regression: test_green_ampt.py](../../../GEO-INFER-WATER/tests/unit/test_green_ampt.py)
- [Example source: __init__.py](../../../GEO-INFER-WATER/examples/__init__.py)
- [Example source: basic_water_analysis.py](../../../GEO-INFER-WATER/examples/basic_water_analysis.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
