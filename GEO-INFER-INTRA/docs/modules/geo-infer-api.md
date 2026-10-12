# GEO-INFER-API: API Management System

`GEO-INFER-API` owns the `geo_infer_api` package under `GEO-INFER-API/src/`.
GEO-INFER-API package.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_api.main_app`
- `geo_infer_api.Settings`
- `geo_infer_api.get_settings`
- `geo_infer_api.Feature`
- `geo_infer_api.FeatureCollection`
- `geo_infer_api.GeoJSONType`
- `geo_infer_api.Geometry`
- `geo_infer_api.LineString`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_api
from geo_infer_api import Settings, get_settings, Feature, FeatureCollection
assert all(value is not None for value in (Settings, get_settings, Feature, FeatureCollection,))
assert geo_infer_api.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module API --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-API/src/geo_infer_api/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-API/README.md)
- [Module operating contract](../../../GEO-INFER-API/AGENTS.md)
- [Regression: test_api_integration.py](../../../GEO-INFER-API/tests/integration/test_api_integration.py)
- [Regression: test_algorithms_router.py](../../../GEO-INFER-API/tests/unit/test_algorithms_router.py)
- [Regression: test_config.py](../../../GEO-INFER-API/tests/unit/test_config.py)
- [Example source: python_client.py](../../../GEO-INFER-API/examples/python_client.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
