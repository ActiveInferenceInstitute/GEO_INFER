# GEO-INFER-PLACE: Place-Based Analysis

`GEO-INFER-PLACE` owns the `geo_infer_place` package under `GEO-INFER-PLACE/src/`.
GEO-INFER-PLACE: Place-Based Geospatial Analysis Framework

## Public import surface

These names are exported by the current owning package:

- `geo_infer_place.PlaceInterface`
- `geo_infer_place.PlaceDataManager`
- `geo_infer_place.PlaceTemporalAnalyzer`
- `geo_infer_place.InteractiveVisualizationEngine`
- `geo_infer_place.CascadianAgriculturalH3Backend`
- `geo_infer_place.BaseAnalysisModule`
- `geo_infer_place.CachedAPIWrapper`
- `geo_infer_place.ForestHealthMonitor`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_place
from geo_infer_place import PlaceInterface, PlaceDataManager, PlaceTemporalAnalyzer, InteractiveVisualizationEngine
assert all(value is not None for value in (PlaceInterface, PlaceDataManager, PlaceTemporalAnalyzer, InteractiveVisualizationEngine,))
assert geo_infer_place.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module PLACE --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-PLACE/src/geo_infer_place/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-PLACE/README.md)
- [Module operating contract](../../../GEO-INFER-PLACE/AGENTS.md)
- [Regression: test_cascadia_geojson_pipeline.py](../../../GEO-INFER-PLACE/tests/integration/test_cascadia_geojson_pipeline.py)
- [Regression: test_cascadia_hydrography.py](../../../GEO-INFER-PLACE/tests/integration/test_cascadia_hydrography.py)
- [Regression: test_location_configs.py](../../../GEO-INFER-PLACE/tests/integration/test_location_configs.py)
- [Example source: del_norte_county_demo.py](../../../GEO-INFER-PLACE/examples/del_norte_county_demo.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
