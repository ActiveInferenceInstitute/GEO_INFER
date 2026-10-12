# GEO-INFER-COG: Cognitive Modeling

`GEO-INFER-COG` owns the `geo_infer_cog` package under `GEO-INFER-COG/src/`.
GEO-INFER-COG: Cognitive Geospatial Processing

## Public import surface

These names are exported by the current owning package:

- `geo_infer_cog.CognitiveProcessingEngine`
- `geo_infer_cog.SpatialPerceptionModel`
- `geo_infer_cog.SpatialReasoningEngine`
- `geo_infer_cog.SpatialMemoryModel`
- `geo_infer_cog.SpatialLanguageProcessor`
- `geo_infer_cog.HumanCenteredVisualizer`
- `geo_infer_cog.SpatialDecisionSupport`
- `geo_infer_cog.validate_spatial_data`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_cog
from geo_infer_cog import CognitiveProcessingEngine, SpatialPerceptionModel, SpatialReasoningEngine, SpatialMemoryModel
assert all(value is not None for value in (CognitiveProcessingEngine, SpatialPerceptionModel, SpatialReasoningEngine, SpatialMemoryModel,))
assert geo_infer_cog.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module COG --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-COG/src/geo_infer_cog/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-COG/README.md)
- [Module operating contract](../../../GEO-INFER-COG/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-COG/tests/integration/test_integration.py)
- [Regression: test_acceptance_cog.py](../../../GEO-INFER-COG/tests/unit/test_acceptance_cog.py)
- [Regression: test_attention.py](../../../GEO-INFER-COG/tests/unit/test_attention.py)
- [Example source: __init__.py](../../../GEO-INFER-COG/examples/__init__.py)
- [Example source: cognitive_processing_demo.py](../../../GEO-INFER-COG/examples/cognitive_processing_demo.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
