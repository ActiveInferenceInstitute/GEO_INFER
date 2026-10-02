# GEO-INFER-MARINE: Marine and Coastal Module

`GEO-INFER-MARINE` owns the `geo_infer_marine` package under `GEO-INFER-MARINE/src/`.
GEO-INFER-MARINE: Marine and Oceanographic Analysis Module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_marine.OceanographicDataProcessor`
- `geo_infer_marine.CoastalAnalyzer`
- `geo_infer_marine.MarineEcosystemModeler`
- `geo_infer_marine.MarineHabitatType`
- `geo_infer_marine.SpeciesData`
- `geo_infer_marine.SeaLevelAnalyzer`
- `geo_infer_marine.MarineSpatialPlanner`
- `geo_infer_marine.OceanCurrentModeler`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_marine
from geo_infer_marine import OceanographicDataProcessor, CoastalAnalyzer, MarineEcosystemModeler, MarineHabitatType
assert all(value is not None for value in (OceanographicDataProcessor, CoastalAnalyzer, MarineEcosystemModeler, MarineHabitatType,))
assert geo_infer_marine.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module MARINE --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-MARINE/src/geo_infer_marine/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-MARINE/README.md)
- [Module operating contract](../../../GEO-INFER-MARINE/AGENTS.md)
- [Regression: test_marine_integration.py](../../../GEO-INFER-MARINE/tests/integration/test_marine_integration.py)
- [Regression: test_biodiversity.py](../../../GEO-INFER-MARINE/tests/unit/test_biodiversity.py)
- [Regression: test_coastal_analysis.py](../../../GEO-INFER-MARINE/tests/unit/test_coastal_analysis.py)
- [Example source: __init__.py](../../../GEO-INFER-MARINE/examples/__init__.py)
- [Example source: basic_marine_analysis.py](../../../GEO-INFER-MARINE/examples/basic_marine_analysis.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
