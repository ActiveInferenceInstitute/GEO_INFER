# GEO-INFER-ART: Artificial Intelligence Art

`GEO-INFER-ART` owns the `geo_infer_art` package under `GEO-INFER-ART/src/`.
GEO-INFER-ART: Art production and aesthetics with geospatial dimensions.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_art.GeoArt`
- `geo_infer_art.MapStyle`
- `geo_infer_art.StyleTransfer`
- `geo_infer_art.ColorPalette`
- `geo_infer_art.GenerativeMap`
- `geo_infer_art.ProceduralArt`
- `geo_infer_art.PlaceArt`
- `geo_infer_art.CulturalMap`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_art
from geo_infer_art import GeoArt, MapStyle, StyleTransfer, ColorPalette
assert all(value is not None for value in (GeoArt, MapStyle, StyleTransfer, ColorPalette,))
assert geo_infer_art.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ART --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ART/src/geo_infer_art/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ART/README.md)
- [Module operating contract](../../../GEO-INFER-ART/AGENTS.md)
- [Regression: test_art_generation_workflow.py](../../../GEO-INFER-ART/tests/integration/test_art_generation_workflow.py)
- [Regression: test_generative_terrain.py](../../../GEO-INFER-ART/tests/test_generative_terrain.py)
- [Regression: test_algorithm_registry_security.py](../../../GEO-INFER-ART/tests/unit/test_algorithm_registry_security.py)
- [Example source: artistic_map_generation.py](../../../GEO-INFER-ART/examples/artistic_map_generation.py)
- [Example source: run_all_examples.py](../../../GEO-INFER-ART/examples/run_all_examples.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
