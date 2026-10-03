# GEO-INFER-INTRA: Knowledge Integration

`GEO-INFER-INTRA` owns the `geo_infer_intra` package under `GEO-INFER-INTRA/src/`.
GEO-INFER-INTRA - Reproducible documentation previews and configuration utilities for the GEO-INFER ecosystem.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_intra.MODULE_PROFILES`
- `geo_infer_intra.SpatialPreviewArtifacts`
- `geo_infer_intra.generate_all_module_previews`
- `geo_infer_intra.generate_module_preview_suite`
- `geo_infer_intra.render_leaflet_html`
- `geo_infer_intra.render_png_card`
- `geo_infer_intra.render_svg_card`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_intra
from geo_infer_intra import MODULE_PROFILES, SpatialPreviewArtifacts, generate_all_module_previews, generate_module_preview_suite
assert all(value is not None for value in (MODULE_PROFILES, SpatialPreviewArtifacts, generate_all_module_previews, generate_module_preview_suite,))
assert geo_infer_intra.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module INTRA --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

The system lane exercises three small public workflows with six identified
observations: real loopback HTTP ingestion, a fresh reader of local Parquet,
SPACE H3 indexing/alignment, and TIME aggregation with exact UTC nanoseconds.
Its analytical matrices distinguish absent observations from observed zero and
preserve caller state order, source/unit metadata and owned nested provenance.
The DATA-to-API workflow sends those actual stored
rows through `geo_infer_api.app.create_app()` using ASGI requests, checking
creation, retrieval, bounding-box selection, conflict, deletion and missing-item
responses. The API currently uses its declared process-local polygon store;
these tests establish local endpoint behavior, not a deployed service or a
persistent API database. Run them with:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --category system --timeout 600 --show-failures
```

## Source and examples

- [Owning package](../../../GEO-INFER-INTRA/src/geo_infer_intra/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-INTRA/README.md)
- [Module operating contract](../../../GEO-INFER-INTRA/AGENTS.md)
- [Regression: test_module_integration.py](../../../GEO-INFER-INTRA/tests/integration/test_module_integration.py)
- [Regression: test_geospatial_performance.py](../../../GEO-INFER-INTRA/tests/performance/test_geospatial_performance.py)
- [Regression: test_system_integration.py](../../../GEO-INFER-INTRA/tests/system/test_system_integration.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
